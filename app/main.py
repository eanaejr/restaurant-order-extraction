"""FastAPI service: one endpoint, POST /order, that extracts a structured
order from a guest utterance using a free Gemini model.

The service is a stateless component of the phone assistant:

    guest (phone) <-> AI assistant (STT/TTS, conversation, session state)
                            |  POST /order {"text": "..."}
                            v
                     this service (extraction only)
                            |
                            v
              {"items": [...], "unavailable": [...]}

It never talks to the guest or the restaurant itself: the response goes back
to the calling assistant. `unavailable` is the signal for that assistant to
continue the conversation ("we don't have hamburgers, would you like...?").
"""

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
    """App factory: the lifespan re-reads configuration on every start,
    which keeps tests (and redeploys) simple."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fail fast on broken configuration or menu — at boot, not per request.
        # A missing API key is deliberately NOT fatal: the service starts and
        # POST /order returns a clear 500 instead.
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
        """Liveness + configuration snapshot (the key itself is never exposed)."""

        settings: Settings = app.state.settings
        return {
            "status": "ok",
            "model": settings.model,
            "thinking_level": settings.thinking_level,
            "api_key_set": bool(settings.api_key),
        }

    @app.post("/order", response_model=OrderResponse)
    def place_order(request: OrderRequest) -> OrderResponse:
        """Extract the order from the guest utterance.

        Note: plain `def` on purpose — FastAPI runs sync endpoints in its
        threadpool, so the synchronous Gemini call cannot block the event loop.
        """

        settings: Settings = app.state.settings
        menu: Menu = app.state.menu

        if not settings.api_key:
            raise HTTPException(
                status_code=500,
                detail=(
                    "GEMINI_API_KEY is not set. Get a free key at "
                    "https://aistudio.google.com/apikey, export it and retry."
                ),
            )

        try:
            result = gemini.extract_order(
                request.text,
                menu,
                settings,
                system_prompt=app.state.system_prompt,
            )
        except gemini.GeminiError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        return normalize_order(result, menu)

    return app


def normalize_order(result: GeminiOrder, menu: Menu) -> OrderResponse:
    """Post-processing that never trusts the model:

    - an item whose id is not on the menu lands in `unavailable` (the request
      is kept — never dropped, never silently replaced by something similar),
    - duplicate ids and duplicate unavailable texts are merged.
    """

    items: dict[str, int] = {}
    unavailable: dict[str, tuple[str, int]] = {}

    for item in result.items:
        if item.id in menu:
            items[item.id] = items.get(item.id, 0) + item.quantity
        else:
            # The model produced an id that is not on the menu. Keep the
            # request in unavailable so the caller can tell the guest.
            text, quantity = unavailable.get(item.id.lower(), (item.id, 0))
            unavailable[item.id.lower()] = (text, quantity + item.quantity)

    for entry in result.unavailable:
        key = entry.text.strip().lower()
        text, quantity = unavailable.get(key, (entry.text.strip(), 0))
        unavailable[key] = (text, quantity + entry.quantity)

    return OrderResponse(
        items=[
            OrderItem(id=item_id, quantity=quantity)
            for item_id, quantity in items.items()
        ],
        unavailable=[
            UnavailableItem(text=text, quantity=quantity)
            for text, quantity in unavailable.values()
        ],
    )


app = create_app()
