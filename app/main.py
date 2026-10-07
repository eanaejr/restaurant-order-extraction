"""FastAPI service: one endpoint, POST /order, that extracts a structured
order from a guest utterance using a free Gemini model."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app import gemini
from app.config import Settings, load_settings
from app.menu import Menu, load_menu
from app.schemas import (
    GeminiOrder,
    OrderItem,
    OrderRequest,
    OrderResponse,
    UnavailableItem,
)

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """App factory: the lifespan re-reads configuration on every start."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fail fast at boot; a missing API key is deliberately not fatal
        # (POST /order answers a clear 500 instead).
        app.state.settings = load_settings()
        app.state.menu = load_menu()
        app.state.system_prompt = gemini.load_system_prompt()
        settings: Settings = app.state.settings
        logger.info(
            "Order service ready (model=%s, thinking_level=%s, api_key=%s)",
            settings.model,
            settings.thinking_level,
            "set" if settings.api_key else "NOT SET — POST /order will answer 500",
        )
        yield

    app = FastAPI(
        title="Restaurant order extraction service",
        description=(
            "Turns a guest utterance into a structured order: "
            "items on the menu plus items that are not on the menu."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict:
        """Liveness + configuration snapshot (never the key itself)."""

        settings: Settings = app.state.settings
        return {
            "status": "ok",
            "model": settings.model,
            "fallback_models": list(settings.fallback_models),
            "thinking_level": settings.thinking_level,
            "api_key_set": bool(settings.api_key),
            **({"simulate_failure": settings.simulate_failure}
               if settings.simulate_failure else {}),
        }

    @app.post("/order", response_model=OrderResponse)
    def place_order(request: OrderRequest) -> OrderResponse:
        """Extract the order from the guest utterance.

        Note: plain `def` on purpose — FastAPI runs sync endpoints in its
        threadpool, so the synchronous Gemini call cannot block the event loop.
        """

        settings: Settings = app.state.settings
        menu: Menu = app.state.menu

        try:
            # Fault injection for demos (GEMINI_SIMULATE_FAILURE) — see README.
            if settings.simulate_failure:
                raise gemini.simulate_failure(settings.simulate_failure)

            if not settings.api_key:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "GEMINI_API_KEY is not set. Get a free key at "
                        "https://aistudio.google.com/apikey, export it and retry."
                    ),
                )

            result = gemini.extract_order(
                request.text,
                menu,
                settings,
                system_prompt=app.state.system_prompt,
            )
        except gemini.GeminiError as exc:
            retry_after = exc.retry_after_s
            raise HTTPException(
                status_code=exc.status_code,
                detail=str(exc),
                headers={"Retry-After": str(retry_after)} if retry_after else None,
            ) from exc

        return normalize_order(result, menu)

    return app


def normalize_order(result: GeminiOrder, menu: Menu) -> OrderResponse:
    """Never trust the model: unknown ids land in `unavailable` (kept, never
    dropped or replaced), duplicates merge, unknown suggested ids are dropped."""

    items: dict[str, int] = {}
    unavailable: dict[str, tuple[str, int]] = {}

    for item in result.items:
        if item.id in menu:
            items[item.id] = items.get(item.id, 0) + item.quantity
        else:
            text, quantity = unavailable.get(item.id.lower(), (item.id, 0))
            unavailable[item.id.lower()] = (text, quantity + item.quantity)

    for entry in result.unavailable:
        key = entry.text.strip().lower()
        text, quantity = unavailable.get(key, (entry.text.strip(), 0))
        unavailable[key] = (text, quantity + entry.quantity)

    suggestions = list(dict.fromkeys(s for s in result.suggestions if s in menu))

    return OrderResponse(
        items=[
            OrderItem(id=item_id, quantity=quantity)
            for item_id, quantity in items.items()
        ],
        unavailable=[
            UnavailableItem(text=text, quantity=quantity)
            for text, quantity in unavailable.values()
        ],
        suggestions=suggestions,
    )


app = create_app()
