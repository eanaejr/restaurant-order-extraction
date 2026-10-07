"""Application configuration, read from environment variables.

The Gemini API key comes only from GEMINI_API_KEY — never hardcoded, never
committed. For local development a gitignored `.env` file at the project root
may populate the variables; real environment variables always win
(override=False). No temperature by default: the Gemini 3.x backend ignores
it — determinism comes from thinking_level + response_schema."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# "minimal" is rejected by gemini-3.7/3.8-flash, hence the "low" default.
THINKING_LEVELS = ("minimal", "low", "medium", "high")

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_THINKING_LEVEL = "low"
# The newest models occasionally answer 503 "high demand".
DEFAULT_FALLBACK_MODELS = ("gemini-3.5-flash-lite", "gemini-3.7-flash")

SIMULATED_FAILURES = ("auth", "quota", "transient")  # fault injection, off by default


class ConfigError(RuntimeError):
    """Raised at startup when configuration values are invalid."""


@dataclass(frozen=True)
class Settings:
    """Snapshot of the environment configuration."""

    model: str
    thinking_level: str
    temperature: float | None  # only sent for legacy (non-3.x) models
    api_key: str | None  # None -> POST /order answers 500
    fallback_models: tuple[str, ...] = ()
    simulate_failure: str | None = None


def load_settings() -> Settings:
    """Read settings from the environment; raises ConfigError on values that
    would only produce confusing API errors later. A missing API key is not
    an error here — POST /order answers a clear 500 (see app/main.py)."""

    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=False)

    model = os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_MODEL

    thinking_level = os.getenv("GEMINI_THINKING_LEVEL", "").strip() or DEFAULT_THINKING_LEVEL
    if thinking_level not in THINKING_LEVELS:
        allowed = ", ".join(THINKING_LEVELS)
        raise ConfigError(
            f"GEMINI_THINKING_LEVEL={thinking_level!r} is invalid. "
            f"Allowed values: {allowed} (default: {DEFAULT_THINKING_LEVEL})."
        )

    temperature_raw = os.getenv("GEMINI_TEMPERATURE", "").strip()
    temperature: float | None = None
    if temperature_raw:
        try:
            temperature = float(temperature_raw)
        except ValueError as exc:
            raise ConfigError(
                f"GEMINI_TEMPERATURE={temperature_raw!r} is not a number."
            ) from exc
        if not 0.0 <= temperature <= 2.0:
            raise ConfigError(
                "GEMINI_TEMPERATURE must be between 0 and 2, "
                f"got {temperature}."
            )
        if _is_gemini_3(model):
            temperature = None  # not an error — the 3.x backend ignores it anyway

    simulate_failure = os.getenv("GEMINI_SIMULATE_FAILURE", "").strip().lower() or None
    if simulate_failure is not None and simulate_failure not in SIMULATED_FAILURES:
        allowed = ", ".join(SIMULATED_FAILURES)
        raise ConfigError(
            f"GEMINI_SIMULATE_FAILURE={simulate_failure!r} is invalid. "
            f"Allowed values: {allowed} (or unset to disable)."
        )

    api_key = os.getenv("GEMINI_API_KEY", "").strip() or None

    # unset -> defaults; explicitly set to "" -> fallback disabled
    fallback_raw = os.getenv("GEMINI_FALLBACK_MODELS")
    if fallback_raw is None:
        fallback_models = DEFAULT_FALLBACK_MODELS
    else:
        fallback_models = tuple(m.strip() for m in fallback_raw.split(",") if m.strip())
        fallback_models = tuple(
            m for m in fallback_models if m and m != model
        )

    return Settings(
        model=model,
        thinking_level=thinking_level,
        temperature=temperature,
        api_key=api_key,
        fallback_models=fallback_models,
        simulate_failure=simulate_failure,
    )


def _is_gemini_3(model: str) -> bool:
    """True for Gemini 3.x model ids (e.g. gemini-3.8-flash)."""

    return model.strip().lower().startswith("gemini-3")
