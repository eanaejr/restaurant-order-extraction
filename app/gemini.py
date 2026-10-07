"""Gemini call: structured output, strict validation, retry + model fallback."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from google import genai
from pydantic import ValidationError

from app.config import Settings
from app.menu import Menu
from app.schemas import GeminiOrder

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "system_prompt.md"

MAX_ATTEMPTS = 2  # one retry per model
RETRY_BACKOFF_S = 0.5
QUOTA_BACKOFF_S = 2.0  # longer pause while rate-limited (429)
QUOTA_RETRY_AFTER_S = 60
# Per-call HTTP timeout in milliseconds (the SDK's timeout is in ms and has
# no default); a stalled call becomes a transient error instead of a hang.
CALL_TIMEOUT_MS = 60_000

AUTH_CODES = {401, 403}  # retrying or switching models cannot fix these


class GeminiError(RuntimeError):
    """The model chain could not produce a valid order (transient failures)."""

    status_code = 502  # upstream (Gemini) problem
    retry_after_s: int | None = None


class GeminiAuthError(GeminiError):
    """Google rejected our API key — retrying or switching cannot help."""

    status_code = 500  # our configuration, not an outage


class GeminiQuotaError(GeminiError):
    """Every model in the chain answered 429 — free-tier quota exhausted."""

    status_code = 503
    retry_after_s = QUOTA_RETRY_AFTER_S


def simulate_failure(kind: str) -> GeminiError:
    """Fault injection (GEMINI_SIMULATE_FAILURE): the error a real failure of
    `kind` would produce — deterministic demos without a real outage."""

    if kind == "auth":
        return GeminiAuthError(
            "Simulated failure (GEMINI_SIMULATE_FAILURE=auth): "
            "Gemini rejected the API key — check GEMINI_API_KEY "
            "(https://aistudio.google.com/apikey)."
        )
    if kind == "quota":
        return GeminiQuotaError(
            "Simulated failure (GEMINI_SIMULATE_FAILURE=quota): "
            "free-tier rate limit reached (429) on all models. "
            f"Try again in about {QUOTA_RETRY_AFTER_S} seconds."
        )
    if kind == "transient":
        return GeminiError(
            "Simulated failure (GEMINI_SIMULATE_FAILURE=transient): "
            "Gemini is unreachable — the whole model chain failed "
            "(network/5xx). The calling assistant should tell the guest "
            "about technical difficulties."
        )
    raise ValueError(f"Unknown simulated failure kind: {kind!r}")


def load_system_prompt(path: Path = SYSTEM_PROMPT_FILE) -> str:
    """Read the system prompt from app/prompts/system_prompt.md."""

    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(f"System prompt file not found: {path}") from exc


def build_user_content(order_text: str, menu: Menu) -> str:
    """What the model sees as the user message: menu + the guest utterance."""

    return (
        "Menu (JSON):\n"
        f"{menu.as_prompt_text()}\n\n"
        "Guest utterance:\n"
        f"{order_text.strip()}"
    )


def build_config(settings: Settings, system_prompt: str) -> dict[str, object]:
    """Generation config for the call (pure, so tests can assert exactly
    what is — and is not — sent)."""

    config: dict[str, object] = {
        "system_instruction": system_prompt,
        "response_mime_type": "application/json",
        "response_schema": GeminiOrder,
        # the determinism/latency/cost knob on Gemini 3.x
        "thinking_config": {"thinking_level": settings.thinking_level},
    }
    # Only for legacy 2.5 models — the 3.x backend ignores temperature.
    if settings.temperature is not None:
        config["temperature"] = settings.temperature
    return config


def _validate_result(result: GeminiOrder) -> GeminiOrder:
    """Strict post-validation: the schema alone is not trusted."""

    for item in result.items:
        if not item.id.strip() or item.quantity < 1:
            raise ValueError(f"invalid model item: {item!r}")
    for item in result.unavailable:
        if not item.text.strip() or item.quantity < 1:
            raise ValueError(f"invalid model item: {item!r}")
    for suggestion in result.suggestions:
        if not suggestion.strip():
            raise ValueError(f"invalid model suggestion: {suggestion!r}")
    return result


def _extract_once(client: genai.Client, model: str, settings: Settings,
                  system_prompt: str, order_text: str, menu: Menu) -> GeminiOrder:
    """One attempt on one model: a Gemini call, parse and strict validation."""

    response = client.models.generate_content(
        model=model,
        contents=build_user_content(order_text, menu),
        config=build_config(settings, system_prompt),
    )
    raw = response.text
    if not raw or not raw.strip():
        raise ValueError("model returned an empty response")
    try:
        result = GeminiOrder.model_validate_json(raw)
    except ValidationError as exc:
        raise ValueError(f"model returned invalid JSON: {exc}") from exc
    return _validate_result(result)


def _failure_kind(exc: Exception) -> str:
    """'auth' (401/403 — nothing helps, stop), 'quota' (429 — pause, retry)
    or 'transient' (network/5xx/bad answer — a retry can help)."""

    code = getattr(exc, "code", None)
    if code in AUTH_CODES:
        return "auth"
    if code == 429:
        return "quota"
    return "transient"


def extract_order(order_text: str, menu: Menu, settings: Settings,
                  client: genai.Client | None = None,
                  system_prompt: str | None = None) -> GeminiOrder:
    """Turn a guest utterance into a validated GeminiOrder; raises
    GeminiAuthError / GeminiQuotaError / GeminiError (500 / 503 / 502)."""

    system_prompt = system_prompt if system_prompt is not None else load_system_prompt()
    if client is None:
        if not settings.api_key:
            raise GeminiAuthError(
                "GEMINI_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey and export it."
            )
        client = genai.Client(
            api_key=settings.api_key,
            http_options={"timeout": CALL_TIMEOUT_MS},
        )

    models = [settings.model, *settings.fallback_models]
    last_error: Exception | None = None
    quota_only = True  # did every failure say 429?
    for model in models:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                result = _extract_once(client, model, settings, system_prompt, order_text, menu)
                if model != settings.model:
                    logger.info("Order extracted with fallback model %s.", model)
                return result
            except Exception as exc:
                kind = _failure_kind(exc)
                if kind == "auth":
                    raise GeminiAuthError(
                        "Gemini rejected the API key — check GEMINI_API_KEY "
                        "(https://aistudio.google.com/apikey): "
                        f"{exc}"
                    ) from exc
                if kind != "quota":
                    quota_only = False
                last_error = exc
                logger.warning(
                    "Model %s attempt %d/%d failed (%s): %s",
                    model, attempt, MAX_ATTEMPTS, kind, exc,
                )
                if attempt < MAX_ATTEMPTS:
                    time.sleep(QUOTA_BACKOFF_S if kind == "quota" else RETRY_BACKOFF_S)

    if quota_only:
        raise GeminiQuotaError(
            f"Free-tier rate limit reached (429) on all models ({', '.join(models)}). "
            f"Try again in about {QUOTA_RETRY_AFTER_S} seconds."
        )
    chain = ", ".join(models)
    raise GeminiError(
        f"Gemini could not produce a valid order after {MAX_ATTEMPTS} attempt(s) "
        f"each on {chain} (last error {type(last_error).__name__}: {last_error})."
    )
