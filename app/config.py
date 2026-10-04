"""Application configuration, read from environment variables.

The Gemini API key is never hardcoded nor committed to the repository — it is
read from the GEMINI_API_KEY environment variable, as the task requires.

For local development, a `.env` file at the project root (gitignored) can
conveniently populate those variables; real environment variables always win
(override=False), so production deployments are unaffected either way.

Why there is no temperature by default: on the current Gemini 3.x models
(gemini-3.8-flash and siblings) the backend *ignores* temperature / top_p /
top_k. Determinism is instead controlled via thinking_level + response_schema,
which is exactly what this service does. GEMINI_TEMPERATURE exists only for
legacy Gemini 2.5 models, should you point GEMINI_MODEL at one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# The .env file for local development, at the project root (gitignored).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# "minimal" is supported by some models (3.5/3.6-flash) but rejected with an
# API error by gemini-3.7-flash and gemini-3.8-flash, hence the "low" default.
THINKING_LEVELS = ("minimal", "low", "medium", "high")

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_THINKING_LEVEL = "low"


class ConfigError(RuntimeError):
    """Raised at startup when configuration values are invalid."""


@dataclass(frozen=True)
class Settings:
    """Snapshot of the environment configuration."""

    model: str
    thinking_level: str
    temperature: float | None  # only sent for legacy (non gemini-3.x) models
    api_key: str | None  # None means "not set" -> endpoint answers 500


def load_settings() -> Settings:
    """Read settings from the environment.

    Raises ConfigError for values that would produce confusing API errors
    later (unknown thinking level, malformed temperature). A missing API key
    is deliberately NOT an error here: the service still starts, and POST
    /order returns a clear 500 instead (see app/main.py).
    """

    # Local development convenience: populate the environment from a .env
    # file at the project root if it exists. Real environment variables
    # always win (override=False).
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
            # Not an error: document the behaviour instead of failing.
            # The backend would silently ignore it anyway.
            temperature = None

    api_key = os.getenv("GEMINI_API_KEY", "").strip() or None

    return Settings(
        model=model,
        thinking_level=thinking_level,
        temperature=temperature,
        api_key=api_key,
    )


def _is_gemini_3(model: str) -> bool:
    """True for Gemini 3.x model ids (e.g. gemini-3.8-flash)."""

    return model.strip().lower().startswith("gemini-3")
