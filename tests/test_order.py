"""Tests for the order extraction service.

The Gemini boundary (app.gemini.extract_order) is mocked in the endpoint
tests, so the suite runs without an API key and without network access.
The Gemini-facing pieces (config building, strict validation, retry) are
tested directly with a fake client.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import gemini
from app.config import ConfigError, load_settings
from app.main import create_app, normalize_order
from app.menu import MenuItem, Menu, load_menu
from app.schemas import GeminiItem, GeminiOrder, GeminiUnavailable

APP_DIR = Path(__file__).resolve().parent.parent
MENU_FILE = APP_DIR / "jelovnik.json"

ENV_VARS = ("GEMINI_MODEL", "GEMINI_THINKING_LEVEL", "GEMINI_TEMPERATURE")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_menu() -> Menu:
    """The real menu shipped with the task."""

    return load_menu(MENU_FILE)


def fake_extract_order(result: GeminiOrder):
    """Builds a stand-in for app.gemini.extract_order returning `result`."""

    def _extract(order_text, menu, settings, system_prompt=None, **kwargs):
        return result

    return _extract


class FakeResponse:
    """Just enough of the SDK response for _extract_once: a .text attribute."""

    def __init__(self, text: str) -> None:
        self.text = text


def make_fake_client(script):
    """Stands in for genai.Client; returns scripted .text values or raises.

    The last script entry repeats once the script is exhausted.
    """

    class _Models:
        def __init__(self, outer) -> None:
            self._outer = outer

        def generate_content(self, **kwargs):
            self._outer.calls += 1
            index = min(self._outer.calls - 1, len(self._outer._script) - 1)
            step = self._outer._script[index]
            if isinstance(step, Exception):
                raise step
            return FakeResponse(step)

    class _Client:
        def __init__(self) -> None:
            self.models = _Models(self)
            self.calls = 0
            self._script = script

    return _Client()


@pytest.fixture()
def clean_env(monkeypatch):
    """Removes optional Gemini overrides, keeps tests order-independent."""

    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture()
def client(monkeypatch, clean_env):
    """A TestClient with a dummy API key.

    Entering the client context triggers the lifespan, which re-reads the
    environment — hence setenv *before* entering.
    """

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Endpoint behaviour
# ---------------------------------------------------------------------------


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "gemini-3.8-flash"
    assert body["thinking_level"] == "low"
    assert body["api_key_set"] is True


def test_order_happy_path(client, monkeypatch):
    monkeypatch.setattr(
        gemini,
        "extract_order",
        fake_extract_order(
            GeminiOrder(
                items=[
                    GeminiItem(id="margarita", quantity=2),
                    GeminiItem(id="coca_cola", quantity=1),
                ],
                unavailable=[],
            )
        ),
    )
    response = client.post("/order", json={"text": "dvije margarite i jednu colu"})
    assert response.status_code == 200
    assert response.json() == {
        "items": [
            {"id": "margarita", "quantity": 2},
            {"id": "coca_cola", "quantity": 1},
        ],
        "unavailable": [],
    }


def test_order_unavailable_passthrough(client, monkeypatch):
    """What is not on the menu comes back separately, untouched (task rule 4)."""

    monkeypatch.setattr(
        gemini,
        "extract_order",
        fake_extract_order(
            GeminiOrder(
                items=[GeminiItem(id="margarita", quantity=1)],
                unavailable=[GeminiUnavailable(text="hamburger", quantity=2)],
            )
        ),
    )
    response = client.post("/order", json={"text": "dva hamburgera i jednu margaritu"})
    assert response.status_code == 200
    assert response.json() == {
        "items": [{"id": "margarita", "quantity": 1}],
        "unavailable": [{"text": "hamburger", "quantity": 2}],
    }


def test_blank_text_is_rejected(client):
    assert client.post("/order", json={"text": ""}).status_code == 422
    assert client.post("/order", json={"text": "   "}).status_code == 422


def test_missing_text_field_is_rejected(client):
    assert client.post("/order", json={}).status_code == 422


def test_missing_api_key_answers_500(monkeypatch, clean_env):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    app = create_app()
    with TestClient(app) as test_client:
        response = test_client.post("/order", json={"text": "jednu colu"})
    assert response.status_code == 500
    assert "GEMINI_API_KEY" in response.json()["detail"]


def test_gemini_failure_answers_502(client, monkeypatch):
    def _raise(*args, **kwargs):
        raise gemini.GeminiError("Gemini could not produce a valid order")

    monkeypatch.setattr(gemini, "extract_order", _raise)
    response = client.post("/order", json={"text": "jednu colu"})
    assert response.status_code == 502
    assert "Gemini" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Post-processing (never trust the model)
# ---------------------------------------------------------------------------


def test_unknown_id_is_moved_to_unavailable():
    """A hallucinated id must not vanish and must not become a fake order."""

    menu = make_menu()
    result = GeminiOrder(
        items=[
            GeminiItem(id="hamburger", quantity=2),
            GeminiItem(id="margarita", quantity=1),
        ]
    )
    response = normalize_order(result, menu)
    assert [i.model_dump() for i in response.items] == [
        {"id": "margarita", "quantity": 1}
    ]
    assert [u.model_dump() for u in response.unavailable] == [
        {"text": "hamburger", "quantity": 2}
    ]


def test_duplicate_items_are_merged():
    menu = make_menu()
    result = GeminiOrder(
        items=[
            GeminiItem(id="margarita", quantity=1),
            GeminiItem(id="margarita", quantity=2),
            GeminiItem(id="pivo", quantity=1),
        ],
        unavailable=[
            GeminiUnavailable(text="hamburger", quantity=1),
            GeminiUnavailable(text="Hamburger", quantity=1),
        ],
    )
    response = normalize_order(result, menu)
    assert [i.model_dump() for i in response.items] == [
        {"id": "margarita", "quantity": 3},
        {"id": "pivo", "quantity": 1},
    ]
    assert [u.model_dump() for u in response.unavailable] == [
        {"text": "hamburger", "quantity": 2}
    ]


# ---------------------------------------------------------------------------
# Configuration (what we do and do not send to Gemini)
# ---------------------------------------------------------------------------


def test_build_config_sends_no_temperature_for_gemini_3(monkeypatch):
    """On Gemini 3.x the backend ignores temperature/top_p/top_k — we do not
    send them; determinism comes from thinking_level + response_schema."""

    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "low")
    monkeypatch.setenv("GEMINI_TEMPERATURE", "0.2")  # must NOT be sent for 3.x
    settings = load_settings()

    assert settings.temperature is None  # dropped at load time for 3.x
    config = gemini.build_config(settings, "system prompt")
    assert "temperature" not in config
    assert config["thinking_config"] == {"thinking_level": "low"}
    assert config["response_mime_type"] == "application/json"
    assert config["response_schema"] is GeminiOrder


def test_build_config_sends_temperature_for_legacy_model(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "low")
    monkeypatch.setenv("GEMINI_TEMPERATURE", "0")
    settings = load_settings()

    assert settings.temperature == 0.0
    config = gemini.build_config(settings, "system prompt")
    assert config["temperature"] == 0.0


def test_load_settings_rejects_bad_thinking_level(monkeypatch):
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "super")
    with pytest.raises(ConfigError):
        load_settings()


def test_load_settings_rejects_bad_temperature(monkeypatch):
    monkeypatch.setenv("GEMINI_TEMPERATURE", "not-a-number")
    with pytest.raises(ConfigError):
        load_settings()


# ---------------------------------------------------------------------------
# Gemini call with a fake client (no network)
# ---------------------------------------------------------------------------


def test_extract_order_parses_structured_response(clean_env):
    settings = load_settings()
    client = make_fake_client(
        ['{"items": [{"id": "margarita", "quantity": 2}], "unavailable": []}']
    )
    result = gemini.extract_order("dvije margarite", make_menu(), settings, client=client)
    assert [i.model_dump() for i in result.items] == [
        {"id": "margarita", "quantity": 2}
    ]
    assert result.unavailable == []
    assert client.calls == 1


def test_extract_order_retries_once_then_succeeds(clean_env):
    settings = load_settings()
    client = make_fake_client(
        [
            RuntimeError("transient API error"),
            '{"items": [{"id": "pivo", "quantity": 2}], "unavailable": []}',
        ]
    )
    result = gemini.extract_order("dva piva", make_menu(), settings, client=client)
    assert [i.model_dump() for i in result.items] == [{"id": "pivo", "quantity": 2}]
    assert client.calls == 2  # failed once, retried, succeeded


def test_extract_order_gives_up_after_two_attempts(clean_env):
    settings = load_settings()
    client = make_fake_client([RuntimeError("API down")])
    with pytest.raises(gemini.GeminiError):
        gemini.extract_order("jednu colu", make_menu(), settings, client=client)
    assert client.calls == 2  # initial try + exactly one retry


def test_extract_order_rejects_nonpositive_quantity(clean_env):
    settings = load_settings()
    client = make_fake_client(
        ['{"items": [{"id": "margarita", "quantity": 0}], "unavailable": []}']
    )
    with pytest.raises(gemini.GeminiError):
        gemini.extract_order("nula margarita", make_menu(), settings, client=client)
    assert client.calls == 2  # invalid answer retried once, then gave up


def test_extract_order_requires_api_key(clean_env):
    settings = load_settings()
    assert settings.api_key is None
    with pytest.raises(gemini.GeminiError) as excinfo:
        gemini.extract_order("jednu colu", make_menu(), settings)
    assert "GEMINI_API_KEY" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Menu loading
# ---------------------------------------------------------------------------


def test_load_real_menu():
    menu = make_menu()
    assert len(menu.items) == 10
    assert "margarita" in menu
    assert "pivo" in menu


def test_load_menu_rejects_duplicate_ids(tmp_path):
    data = (
        '[{"id": "x", "naziv": "X", "kategorija": "pizza", "cijena_eur": 1, "bez_mesa": true},'
        ' {"id": "x", "naziv": "X2", "kategorija": "pizza", "cijena_eur": 2, "bez_mesa": true}]'
    )
    path = tmp_path / "menu.json"
    path.write_text(data, encoding="utf-8")
    with pytest.raises(ValueError):
        load_menu(path)


def test_menu_prompt_text_is_valid_json():
    menu = Menu(
        [
            MenuItem(
                id="margarita",
                naziv="Pizza Margarita",
                kategorija="pizza",
                cijena_eur=9.0,
                bez_mesa=True,
            )
        ]
    )
    text = menu.as_prompt_text()
    assert '"id": "margarita"' in text
    assert "Pizza Margarita" in text
