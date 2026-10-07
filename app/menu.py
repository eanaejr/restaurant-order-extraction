"""Loading and indexing of the restaurant menu (jelovnik.json)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

MENU_FILE = Path(__file__).resolve().parent.parent / "jelovnik.json"


class MenuItem(BaseModel):
    """One menu entry, exactly as in jelovnik.json."""

    id: str = Field(description="Unique menu item identifier used in orders.")
    naziv: str = Field(description="Full item name as printed on the menu.")
    kategorija: str = Field(description="Menu category (pizza, salata, piće).")
    cijena_eur: float = Field(description="Price in euros.")
    bez_mesa: bool = Field(description="True if the item contains no meat.")


class Menu:
    """The loaded menu plus a fast lookup by item id."""

    def __init__(self, items: list[MenuItem]) -> None:
        self.items = items
        self.by_id: dict[str, MenuItem] = {}
        for item in items:
            if item.id in self.by_id:
                raise ValueError(f"Duplicate menu item id: {item.id!r}")
            self.by_id[item.id] = item

    def __contains__(self, item_id: object) -> bool:
        return item_id in self.by_id

    def as_prompt_text(self) -> str:
        """The menu as it is presented to the model (stable, readable JSON)."""

        payload = [item.model_dump() for item in self.items]
        return json.dumps(payload, ensure_ascii=False, indent=2)


def load_menu(path: Path = MENU_FILE) -> Menu:
    """Read and validate jelovnik.json; raises on a malformed file."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Menu file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Menu file {path} is not valid JSON: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError(f"Menu file {path} must contain a JSON array.")
    return Menu([MenuItem.model_validate(entry) for entry in data])
