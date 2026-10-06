"""Pydantic models for the /order endpoint and for the Gemini response.

The Gemini* models double as the structured-output schema (response_schema):
the model is forced to answer in exactly that JSON shape. Per Google's
best practice, field descriptions live here in the schema and the JSON
format is NOT duplicated inside the prompt; the model learns the meaning of
each field from these descriptions.

Numeric constraints are deliberately NOT part of the Gemini-facing schema
(Gemini supports only a subset of JSON Schema); they are enforced afterwards
in Python — never trust the model.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------
# Public API models
# --------------------------------------------------------------------------


class OrderRequest(BaseModel):
    """Body of POST /order."""

    text: str = Field(
        description="The guest utterance, e.g. 'dvije margarite i jednu colu'."
    )

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty or whitespace-only")
        return value


class OrderItem(BaseModel):
    """Something the guest ordered that exists on the menu."""

    id: str = Field(description="Menu item id, exactly as in jelovnik.json.")
    quantity: int = Field(ge=1, description="How many the guest ordered.")


class UnavailableItem(BaseModel):
    """Something the guest asked for that is not on the menu."""

    text: str = Field(description="What the guest asked for, as literally as said.")
    quantity: int = Field(ge=1, description="How many the guest asked for.")


class OrderResponse(BaseModel):
    """Response of POST /order — the format proposed in the task, plus the
    `suggestions` extension (the reason is explained in the README)."""

    items: list[OrderItem] = Field(default_factory=list, description="Ordered items that exist on the menu.")
    unavailable: list[UnavailableItem] = Field(default_factory=list, description="Requested items that do not exist on the menu.")
    suggestions: list[str] = Field(default_factory=list, description="Menu ids suggested when the guest asked what is available (e.g. meat-free options); empty otherwise.")


# --------------------------------------------------------------------------
# Gemini structured-output schema (same shape as OrderResponse)
# --------------------------------------------------------------------------


class GeminiItem(BaseModel):
    id: str = Field(description="Menu item id, must be taken verbatim from the provided menu.")
    quantity: int = Field(description="Ordered quantity as a positive integer (at least 1). Use 1 when the guest does not specify a number.")


class GeminiUnavailable(BaseModel):
    text: str = Field(description="What the guest asked for, in its base (nominative singular) form, e.g. 'hamburger' from 'dva hamburgera'.")
    quantity: int = Field(description="Requested quantity as a positive integer (at least 1). Use 1 when the guest does not specify a number.")


class GeminiOrder(BaseModel):
    items: list[GeminiItem] = Field(default_factory=list, description="Everything the guest ordered that matches a menu item by id.")
    unavailable: list[GeminiUnavailable] = Field(default_factory=list, description="Everything the guest asked for that is NOT on the menu. Never substitute a similar menu item, never drop such requests.")
    suggestions: list[str] = Field(default_factory=list, description="Menu ids to suggest when the guest asks what is available (e.g. 'imate li nešto bez mesa?' -> the meat-free food ids). Only ids from the provided menu; food rather than drinks for food questions; empty when the guest did not ask for suggestions.")
