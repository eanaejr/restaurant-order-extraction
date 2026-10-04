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
  (quantities >= 1, non-empty ids/texts). One retry on any model/API
  failure, then the endpoint reports 502.
"""

from __future__ import annotations

import logging
from pathlib import Path

from google import genai
from pydantic import ValidationError

from app.config import Settings
from app.menu import Menu
from app.schemas import GeminiOrder

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "system_prompt.md"

MAX_ATTEMPTS = 2  # one retry


class GeminiError(RuntimeError):
    """The model could not be reached or returned an unusable answer."""


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


def _extract_once(client: genai.Client, settings: Settings,
                  system_prompt: str, order_text: str, menu: Menu) -> GeminiOrder:
    """One attempt: one Gemini call, parse and strictly validate."""

    response = client.models.generate_content(
        model=settings.model,
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


def extract_order(order_text: str, menu: Menu, settings: Settings,
                  client: genai.Client | None = None,
                  system_prompt: str | None = None) -> GeminiOrder:
    """Turn a guest utterance into a validated GeminiOrder.

    Raises GeminiError after MAX_ATTEMPTS failed attempts (API error,
    unparsable answer or invalid content).
    """

    system_prompt = system_prompt if system_prompt is not None else load_system_prompt()
    if client is None:
        if not settings.api_key:
            raise GeminiError(
                "GEMINI_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey and export it."
            )
        client = genai.Client(api_key=settings.api_key)

    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return _extract_once(client, settings, system_prompt, order_text, menu)
        except Exception as exc:  # noqa: BLE001 — one retry on any model failure
            last_error = exc
            logger.warning("Gemini attempt %d/%d failed: %s", attempt, MAX_ATTEMPTS, exc)

    raise GeminiError(
        f"Gemini could not produce a valid order after {MAX_ATTEMPTS} attempts "
        f"({type(last_error).__name__}: {last_error})."
    )
