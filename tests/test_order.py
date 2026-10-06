"""Tests for the order extraction service.

The Gemini boundary (app.gemini.extract_order) is mocked in the endpoint
tests, so the suite runs without an API key and without network access.
The Gemini-facing pieces (config building, strict validation, retry and
model fallback) are tested directly with a fake client.

Every test gets a deterministic environment via the autouse `isolated_env`
fixture: no local .env file, no variables leaking in or out of a test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv as real_load_dotenv
from fastapi.testclient import TestClient

from app import config as config_module
from app import gemini
from app.config import DEFAULT_FALLBACK_MODELS, ConfigError, Settings, load_settings
from app.main import create_app, normalize_order
from app.menu import MenuItem, Menu, load_menu
from app.schemas import GeminiItem, GeminiOrder, GeminiUnavailable

APP_DIR = Path(__file__).resolve().parent.parent
MENU_FILE = APP_DIR / "jelovnik.json"

ENV_VARS = (
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_THINKING_LEVEL",
    "GEMINI_TEMPERATURE",
    "GEMINI_FALLBACK_MODELS",
    "GEMINI_SIMULATE_FAILURE",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_menu() -> Menu:
    """The real menu shipped with the task."""

    return load_menu(MENU_FILE)


def make_settings(model: str = "gemini-3.8-flash", api_key: str | None = "test-key",
                  fallback_models: tuple[str, ...] = ()) -> Settings:
    """Explicit settings for gemini-call tests (no environment dependence)."""

    return Settings(
        model=model,
        thinking_level="low",
        temperature=None,
        api_key=api_key,
        fallback_models=fallback_models,
    )


def fake_extract_order(result: GeminiOrder):
    """Builds a stand-in for app.gemini.extract_order returning `result`."""

    def _extract(order_text, menu, settings, system_prompt=None, **kwargs):
        return result

    return _extract


class FakeResponse:
    """Just enough of the SDK response for _extract_once: a .text attribute."""

    def __init__(self, text: str) -> None:
        self.text = text


class FakeApiError(RuntimeError):
    """Mimics google.genai.errors.APIError: carries an HTTP status code."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def make_fake_client(scripts):
    """Stands in for genai.Client, scripted per model.

    scripts: {model: [step, ...]} where a step is either an exception to
    raise or the response .text to return. The last step for a model repeats;
    a call with an unexpected model raises KeyError.
    """

    class _Models:
        def __init__(self, outer) -> None:
            self._outer = outer

        def generate_content(self, **kwargs):
            model = kwargs["model"]
            script = self._outer._scripts[model]
            count = self._outer.model_calls.get(model, 0) + 1
            self._outer.model_calls[model] = count
            step = script[min(count - 1, len(script) - 1)]
            if isinstance(step, Exception):
                raise step
            return FakeResponse(step)

    class _Client:
        def __init__(self) -> None:
            self.models = _Models(self)
            self.model_calls: dict[str, int] = {}
            self._scripts = scripts

    return _Client()


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """Deterministic environment for every test.

    - the developer's local .env file is not loaded,
    - no Gemini variable leaks in or out of a test.

    Tests that explicitly exercise .env handling re-bind the real
    load_dotenv and point ENV_FILE at a temporary file.
    """

    monkeypatch.setattr(config_module, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(gemini, "RETRY_BACKOFF_S", 0)
    monkeypatch.setattr(gemini, "QUOTA_BACKOFF_S", 0)
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture()
def client(monkeypatch):
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
    assert body["fallback_models"] == list(DEFAULT_FALLBACK_MODELS)
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
        "suggestions": [],
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
        "suggestions": [],
    }


def test_order_meat_free_question_returns_suggestions(client, monkeypatch):
    """Optional task case: a question is not an order — matching menu ids
    are suggested, nothing is ordered."""

    monkeypatch.setattr(
        gemini,
        "extract_order",
        fake_extract_order(
            GeminiOrder(
                items=[],
                unavailable=[],
                suggestions=["margarita", "vegetariana",
                             "quattro_formaggi", "mijesana_salata"],
            )
        ),
    )
    response = client.post(
        "/order", json={"text": "imate li nešto bez mesa za nas dvoje?"}
    )
    assert response.status_code == 200
    assert response.json() == {
        "items": [],
        "unavailable": [],
        "suggestions": ["margarita", "vegetariana",
                        "quattro_formaggi", "mijesana_salata"],
    }


def test_order_change_of_mind_final_statement_wins(client, monkeypatch):
    """Optional task case: 'tri margarite i colu, ma ne, ipak dvije
    margarite' -> margarita x 2 (corrected) and the cola stands."""

    monkeypatch.setattr(
        gemini,
        "extract_order",
        fake_extract_order(
            GeminiOrder(
                items=[GeminiItem(id="margarita", quantity=2),
                       GeminiItem(id="coca_cola", quantity=1)],
            )
        ),
    )
    response = client.post(
        "/order",
        json={"text": "tri margarite i colu, ma ne, ipak dvije margarite"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "items": [{"id": "margarita", "quantity": 2},
                  {"id": "coca_cola", "quantity": 1}],
        "unavailable": [],
        "suggestions": [],
    }


def test_blank_text_is_rejected(client):
    assert client.post("/order", json={"text": ""}).status_code == 422
    assert client.post("/order", json={"text": "   "}).status_code == 422


def test_missing_text_field_is_rejected(client):
    assert client.post("/order", json={}).status_code == 422


def test_missing_api_key_answers_500(monkeypatch):
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


def test_auth_failure_answers_500(client, monkeypatch):
    """A rejected API key is our configuration problem, not an outage."""

    def _raise(*args, **kwargs):
        raise gemini.GeminiAuthError("Gemini rejected the API key")

    monkeypatch.setattr(gemini, "extract_order", _raise)
    response = client.post("/order", json={"text": "jednu colu"})
    assert response.status_code == 500
    assert "API key" in response.json()["detail"]


def test_quota_failure_answers_503_with_retry_after(client, monkeypatch):
    """Exhausted free-tier quota: 503 plus a Retry-After hint."""

    def _raise(*args, **kwargs):
        raise gemini.GeminiQuotaError("Free-tier rate limit reached (429)")

    monkeypatch.setattr(gemini, "extract_order", _raise)
    response = client.post("/order", json={"text": "jednu colu"})
    assert response.status_code == 503
    assert "429" in response.json()["detail"]
    assert response.headers["Retry-After"] == str(gemini.QUOTA_RETRY_AFTER_S)


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


def test_unknown_suggestions_dropped_and_duplicates_merged():
    """Never trust the model: only ids that exist on the menu survive."""

    menu = make_menu()
    result = GeminiOrder(
        suggestions=["margarita", "hamburger", "margarita", "vegetariana"]
    )
    response = normalize_order(result, menu)
    assert response.suggestions == ["margarita", "vegetariana"]
    assert response.items == []
    assert response.unavailable == []


def test_validate_result_rejects_empty_suggestion():
    result = GeminiOrder(suggestions=["  "])
    with pytest.raises(ValueError):
        gemini._validate_result(result)


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
# Fault injection (GEMINI_SIMULATE_FAILURE) — demoing failure modes live
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind, expected_status", [
    ("auth", 500),       # rejected key: our configuration problem
    ("quota", 503),      # free-tier quota exhausted: try later
    ("transient", 502),  # whole chain unreachable: upstream problem
])
def test_simulated_failure_answers_the_right_status(monkeypatch, kind, expected_status):
    """Full HTTP path, no mocks and no API key needed — deterministic demo."""

    monkeypatch.setenv("GEMINI_SIMULATE_FAILURE", kind)
    app = create_app()
    with TestClient(app) as test_client:
        response = test_client.post("/order", json={"text": "dvije margarite"})
    assert response.status_code == expected_status
    assert "Simulated failure" in response.json()["detail"]


def test_simulated_quota_advertises_retry_after(monkeypatch):
    monkeypatch.setenv("GEMINI_SIMULATE_FAILURE", "quota")
    app = create_app()
    with TestClient(app) as test_client:
        response = test_client.post("/order", json={"text": "jednu colu"})
    assert response.status_code == 503
    assert response.headers["Retry-After"] == str(gemini.QUOTA_RETRY_AFTER_S)


def test_simulated_failure_shown_in_health(monkeypatch):
    monkeypatch.setenv("GEMINI_SIMULATE_FAILURE", "transient")
    app = create_app()
    with TestClient(app) as test_client:
        body = test_client.get("/health").json()
    assert body["simulate_failure"] == "transient"


def test_simulated_failure_rejects_unknown_kind(monkeypatch):
    monkeypatch.setenv("GEMINI_SIMULATE_FAILURE", "everything-at-once")
    with pytest.raises(ConfigError):
        load_settings()


def test_default_fallback_chain_when_unset():
    """Google's newest models occasionally answer 503 'high demand'; the
    default is a small chain of other free-tier models."""

    settings = load_settings()
    assert settings.model == "gemini-3.8-flash"
    assert settings.fallback_models == DEFAULT_FALLBACK_MODELS


def test_fallback_can_be_disabled(monkeypatch):
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "")
    settings = load_settings()
    assert settings.fallback_models == ()


def test_fallback_never_repeats_primary(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite, gemini-3.7-flash")
    settings = load_settings()
    assert settings.fallback_models == ("gemini-3.7-flash",)


# ---------------------------------------------------------------------------
# .env handling (local development convenience)
# ---------------------------------------------------------------------------


def test_dotenv_file_is_loaded_when_present(monkeypatch, tmp_path):
    """A .env file at the project root populates the environment locally."""

    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_THINKING_LEVEL=medium\n", encoding="utf-8")
    monkeypatch.setattr(config_module, "ENV_FILE", env_file)
    monkeypatch.setattr(config_module, "load_dotenv", real_load_dotenv)

    settings = load_settings()
    assert settings.thinking_level == "medium"


def test_real_environment_wins_over_dotenv(monkeypatch, tmp_path):
    """Real environment variables always beat .env (override=False)."""

    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_THINKING_LEVEL=high\n", encoding="utf-8")
    monkeypatch.setattr(config_module, "ENV_FILE", env_file)
    monkeypatch.setattr(config_module, "load_dotenv", real_load_dotenv)
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "low")

    settings = load_settings()
    assert settings.thinking_level == "low"


# ---------------------------------------------------------------------------
# Gemini call with a fake client (no network): retry and fallback
# ---------------------------------------------------------------------------


OK_PIVO = '{"items": [{"id": "pivo", "quantity": 2}], "unavailable": []}'


def test_extract_order_parses_structured_response():
    settings = make_settings()
    client = make_fake_client(
        {"gemini-3.8-flash": ['{"items": [{"id": "margarita", "quantity": 2}], "unavailable": []}']}
    )
    result = gemini.extract_order("dvije margarite", make_menu(), settings, client=client)
    assert [i.model_dump() for i in result.items] == [
        {"id": "margarita", "quantity": 2}
    ]
    assert result.unavailable == []
    assert client.model_calls == {"gemini-3.8-flash": 1}


def test_extract_order_retries_once_then_succeeds():
    settings = make_settings()
    client = make_fake_client(
        {"gemini-3.8-flash": [RuntimeError("transient API error"), OK_PIVO]}
    )
    result = gemini.extract_order("dva piva", make_menu(), settings, client=client)
    assert [i.model_dump() for i in result.items] == [{"id": "pivo", "quantity": 2}]
    assert client.model_calls == {"gemini-3.8-flash": 2}  # failed once, retried


def test_extract_order_gives_up_after_two_attempts():
    settings = make_settings()
    client = make_fake_client({"gemini-3.8-flash": [RuntimeError("API down")]})
    with pytest.raises(gemini.GeminiError):
        gemini.extract_order("jednu colu", make_menu(), settings, client=client)
    assert client.model_calls == {"gemini-3.8-flash": 2}  # try + exactly one retry


def test_extract_order_rejects_nonpositive_quantity():
    settings = make_settings()
    client = make_fake_client(
        {"gemini-3.8-flash": ['{"items": [{"id": "margarita", "quantity": 0}], "unavailable": []}']}
    )
    with pytest.raises(gemini.GeminiError):
        gemini.extract_order("nula margarita", make_menu(), settings, client=client)
    assert client.model_calls == {"gemini-3.8-flash": 2}


def test_extract_order_falls_back_to_next_model():
    """A '503 high demand' primary is retried once, then the next free model
    in the chain takes over."""

    settings = make_settings(fallback_models=("gemini-3.5-flash-lite",))
    client = make_fake_client(
        {
            "gemini-3.8-flash": [RuntimeError("503 high demand")],
            "gemini-3.5-flash-lite": [OK_PIVO],
        }
    )
    result = gemini.extract_order("dva piva", make_menu(), settings, client=client)
    assert [i.model_dump() for i in result.items] == [{"id": "pivo", "quantity": 2}]
    assert client.model_calls == {
        "gemini-3.8-flash": 2,        # try + one retry
        "gemini-3.5-flash-lite": 1,   # fallback succeeds immediately
    }


def test_extract_order_reports_whole_chain_in_error():
    settings = make_settings(fallback_models=("gemini-3.5-flash-lite",))
    client = make_fake_client(
        {
            "gemini-3.8-flash": [RuntimeError("503 high demand")],
            "gemini-3.5-flash-lite": [RuntimeError("also down")],
        }
    )
    with pytest.raises(gemini.GeminiError) as excinfo:
        gemini.extract_order("jednu colu", make_menu(), settings, client=client)
    message = str(excinfo.value)
    assert "gemini-3.8-flash" in message
    assert "gemini-3.5-flash-lite" in message
    assert client.model_calls == {"gemini-3.8-flash": 2, "gemini-3.5-flash-lite": 2}


def test_auth_error_stops_immediately():
    """A rejected key (401/403) must not burn retries or fallback models."""

    settings = make_settings(fallback_models=("gemini-3.5-flash-lite",))
    client = make_fake_client(
        {
            "gemini-3.8-flash": [FakeApiError(403, "API key not valid")],
            "gemini-3.5-flash-lite": [OK_PIVO],  # must never be reached
        }
    )
    with pytest.raises(gemini.GeminiAuthError) as excinfo:
        gemini.extract_order("jednu colu", make_menu(), settings, client=client)
    assert "GEMINI_API_KEY" in str(excinfo.value)
    assert client.model_calls == {"gemini-3.8-flash": 1}  # no retry, no fallback


def test_quota_exhausted_on_all_models():
    """Every model answering 429 means the free-tier quota is gone: 503."""

    settings = make_settings(fallback_models=("gemini-3.5-flash-lite",))
    client = make_fake_client(
        {
            "gemini-3.8-flash": [FakeApiError(429, "rate limit")],
            "gemini-3.5-flash-lite": [FakeApiError(429, "rate limit")],
        }
    )
    with pytest.raises(gemini.GeminiQuotaError) as excinfo:
        gemini.extract_order("jednu colu", make_menu(), settings, client=client)
    assert excinfo.value.status_code == 503
    assert excinfo.value.retry_after_s == gemini.QUOTA_RETRY_AFTER_S
    assert client.model_calls == {"gemini-3.8-flash": 2, "gemini-3.5-flash-lite": 2}


def test_quota_on_primary_then_fallback_succeeds():
    """A 429 burst on the primary model is retried, then the chain continues."""

    settings = make_settings(fallback_models=("gemini-3.5-flash-lite",))
    client = make_fake_client(
        {
            "gemini-3.8-flash": [FakeApiError(429, "rate limit")],
            "gemini-3.5-flash-lite": [OK_PIVO],
        }
    )
    result = gemini.extract_order("dva piva", make_menu(), settings, client=client)
    assert [i.model_dump() for i in result.items] == [{"id": "pivo", "quantity": 2}]
    assert client.model_calls == {"gemini-3.8-flash": 2, "gemini-3.5-flash-lite": 1}


def test_extract_order_requires_api_key():
    settings = make_settings(api_key=None)
    with pytest.raises(gemini.GeminiAuthError) as excinfo:
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
