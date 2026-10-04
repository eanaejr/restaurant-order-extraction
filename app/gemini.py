"""Gemini call: turns a guest utterance into a structured order.

Design notes (the whole story is in the README):

- Structured output: `response_mime_type="application/json"` +
  `response_schema=GeminiOrder` — the model must answer in exactly that
  JSON shape, and the field descriptions in app/schemas.py carry the
  semantics. Per Google's best practice the JSON format is NOT duplicated
  in the prompt itself.
- `thinking_level="low"`: on Gemini 3.x models this is the recommended knob
  for determinism, latency and cost. The backend ignores temperature /
  top_p / top_k there, so we never send them for 3.x (GEMINI_TEMPERATURE
  exists only for legacy 2.5 models).
- Never trust the model: after parsing we strictly validate the result
  (quantities >= 1, non-empty ids/texts). Failures are classified (see
  _failure_kind): a rejected API key stops immediately (retrying cannot
  fix it), a 429 rate limit gets a longer pause, anything transient gets
  one retry per model and then the next free model from the fallback chain
  (a "503 high demand" on the newest models is a normal, transient
  situation). Only after the whole chain fails does the endpoint report
  the failure — 502 transient, 503 with Retry-After when only quota-limited.
"""

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
RETRY_BACKOFF_S = 0.5  # pause between attempts on the same model
QUOTA_BACKOFF_S = 2.0  # longer pause while being rate-limited (429)
QUOTA_RETRY_AFTER_S = 60  # advertised to the caller via the Retry-After header

# HTTP codes from the Gemini API that retrying or switching models cannot fix.
AUTH_CODES = {401, 403}


class GeminiError(RuntimeError):
    """The model chain could not produce a valid order (transient failures)."""

    status_code = 502  # bad gateway: the upstream (Gemini) is the problem
    retry_after_s: int | None = None


class GeminiAuthError(GeminiError):
    """Google rejected our API key — retrying or switching cannot help."""

    status_code = 500  # our configuration is the problem, not an outage


class GeminiQuotaError(GeminiError):
    """Every model in the chain answered 429 — free-tier quota exhausted."""

    status_code = 503  # try again later
    retry_after_s = QUOTA_RETRY_AFTER_S


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
    """Generation config for the call (a plain dict — the SDK converts it).

    Kept as a separate pure function so tests can assert exactly which
    parameters are (not) being sent — e.g. that no temperature is sent for
    Gemini 3.x models.
    """

    config: dict[str, object] = {
        "system_instruction": system_prompt,
        # Structured output: the model must answer in the GeminiOrder shape.
        "response_mime_type": "application/json",
        "response_schema": GeminiOrder,
        # The determinism/latency/cost knob for Gemini 3.x.
        "thinking_config": {"thinking_level": settings.thinking_level},
    }
    # Only for legacy models (2.5 family); the 3.x backend ignores temperature.
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
    """Classify a failure to decide what can still help.

    'auth'      — Google rejected the key (401/403): stop immediately,
                  retrying or switching models cannot fix it.
    'quota'     — 429 rate limit: pause longer, then retry / next model;
                  if every model is quota-limited, tell the caller to wait.
    'transient' — network errors, 5xx, or our own validation rejecting the
                  model's answer: a retry (same or next model) can fix it.
    """

    code = getattr(exc, "code", None)
    if code in AUTH_CODES:
        return "auth"
    if code == 429:
        return "quota"
    return "transient"


def extract_order(order_text: str, menu: Menu, settings: Settings,
                  client: genai.Client | None = None,
                  system_prompt: str | None = None) -> GeminiOrder:
    """Turn a guest utterance into a validated GeminiOrder.

    Raises GeminiAuthError (key rejected), GeminiQuotaError (all models
    rate-limited) or GeminiError (the whole chain failed) — the endpoint
    maps them to 500 / 503 / 502 respectively.
    """

    system_prompt = system_prompt if system_prompt is not None else load_system_prompt()
    if client is None:
        if not settings.api_key:
            raise GeminiAuthError(
                "GEMINI_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey and export it."
            )
        client = genai.Client(api_key=settings.api_key)

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
            except Exception as exc:  # noqa: BLE001 — classified below
                kind = _failure_kind(exc)
                if kind == "auth":
                    # Retrying or switching models cannot fix a rejected key.
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
