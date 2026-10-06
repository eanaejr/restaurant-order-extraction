"""Presentation script: shows the whole story in front of the examiners.

Usage (two terminals):
    # 1) the normal service (reads .env / GEMINI_API_KEY)
    uvicorn app.main:app --reload

    # 2) the demo
    python demo.py

What it shows, in order:
 1. The five sentences from the task, asserted (three mandatory + the two
    optional cases: meat-free suggestions and the changed mind)
 2. Protocol edge cases over HTTP (blank text -> 422, /health)
 3. Informational cases (a question about an item, politeness, gibberish)
    — printed so you can talk about them; model behaviour, so no hard
    asserts
 4. All three failure modes, deterministically — spins up temporary
    services with GEMINI_SIMULATE_FAILURE and shows the exact answers
    for a rejected key (500), exhausted free-tier quota (503 +
    Retry-After) and a fully unreachable Gemini (502)

Exit code 0 = every asserted check passed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import httpx

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")

MANDATORY_CASES = [
    {
        "text": "dvije margarite i jednu colu",
        "items": {"margarita": 2, "coca_cola": 1},
        "unavailable": {},
        "suggestions": set(),
    },
    {
        "text": "jednu kapričozu i dva piva molim",
        "items": {"capricciosa": 1, "pivo": 2},
        "unavailable": {},
        "suggestions": set(),
    },
    {
        "text": "dva hamburgera i jednu margaritu",
        "items": {"margarita": 1},
        "unavailable": {"hamburger": 2},
        "suggestions": set(),
    },
    {
        # Optional case: a question, not an order -> suggestions, empty order.
        "text": "imate li nešto bez mesa za nas dvoje?",
        "items": {},
        "unavailable": {},
        "suggestions": {"margarita", "vegetariana", "quattro_formaggi", "mijesana_salata"},
    },
    {
        # Optional case: the guest changes their mind mid-sentence.
        "text": "tri margarite i colu, ma ne, ipak dvije margarite",
        "items": {"margarita": 2, "coca_cola": 1},
        "unavailable": {},
        "suggestions": set(),
    },
]

# Model-dependent behaviour: printed, not asserted.
INFORMATIONAL_CASES = [
    "je li diavola jako ljuta?",               # a question about an item, not an order
    "ništa, hvala lijepa",                      # explicitly nothing
    "asdfgh qwerty 12345",                      # gibberish
    "dva piva i tri margarite, molim lijepo",   # politeness + number words
]

# Failure modes: kind -> expected HTTP status (+ Retry-After for quota).
FAILURE_MODES = [
    ("auth", 500),
    ("quota", 503),
    ("transient", 502),
]

passed, failed = 0, 0


def header(title: str) -> None:
    line = "=" * 64
    print(f"\n{line}\n {title}\n{line}")


def ok(message: str) -> None:
    global passed
    passed += 1
    print(f"  PASS  {message}")


def fail(message: str) -> None:
    global failed
    failed += 1
    print(f"  FAIL  {message}")


def show_mandatory(client: httpx.Client) -> None:
    header("1) The five sentences from the task (asserted)")
    for case in MANDATORY_CASES:
        response = client.post("/order", json={"text": case["text"]})
        body = response.json()
        got_items = {i["id"]: i["quantity"] for i in body.get("items", [])}
        got_unavailable = {u["text"]: u["quantity"] for u in body.get("unavailable", [])}
        got_suggestions = set(body.get("suggestions", []))
        if (response.status_code == 200
                and got_items == case["items"]
                and got_unavailable == case["unavailable"]
                and got_suggestions == case["suggestions"]):
            ok(f"{case['text']!r} -> items={got_items} "
               f"unavailable={got_unavailable} "
               f"suggestions={sorted(got_suggestions)}")
        else:
            fail(f"{case['text']!r}: HTTP {response.status_code}, got "
                 f"items={got_items} unavailable={got_unavailable} "
                 f"suggestions={sorted(got_suggestions)}, "
                 f"expected items={case['items']} "
                 f"unavailable={case['unavailable']} "
                 f"suggestions={sorted(case['suggestions'])}")


def show_protocol(client: httpx.Client) -> None:
    header("2) Protocol edge cases (asserted, no Gemini involved)")
    response = client.post("/order", json={"text": "   "})
    if response.status_code == 422:
        ok("blank/whitespace text -> 422")
    else:
        fail(f"blank text: expected 422, got {response.status_code} {response.text}")

    response = client.post("/order", json={})
    if response.status_code == 422:
        ok("missing 'text' field -> 422")
    else:
        fail(f"missing field: expected 422, got {response.status_code}")

    response = client.get("/health")
    body = response.json()
    if response.status_code == 200 and body.get("status") == "ok":
        ok(f"/health -> model={body.get('model')}, "
           f"fallbacks={body.get('fallback_models')}, api_key_set={body.get('api_key_set')}")
    else:
        fail(f"/health unexpected: {response.status_code} {response.text}")


def show_informational(client: httpx.Client) -> None:
    header("3) Unusual input (informational — model behaviour, not asserted)")
    for text in INFORMATIONAL_CASES:
        response = client.post("/order", json={"text": text})
        print(f"  {text!r}\n    -> HTTP {response.status_code}: {response.text}")


def spawn_failure_service(kind: str, port: int) -> subprocess.Popen:
    """A temporary service with GEMINI_SIMULATE_FAILURE=kind."""

    env = {**os.environ, "GEMINI_SIMULATE_FAILURE": kind}
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def show_failure_modes() -> None:
    header("4) Failure modes, deterministically (GEMINI_SIMULATE_FAILURE)")
    for index, (kind, expected_status) in enumerate(FAILURE_MODES):
        port = 8101 + index
        url = f"http://127.0.0.1:{port}"
        process = spawn_failure_service(kind, port)
        try:
            # Wait for the temporary service to come up.
            with httpx.Client(base_url=url, timeout=5.0) as probe:
                for _ in range(50):
                    try:
                        if probe.get("/health").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.2)
                else:
                    fail(f"{kind}: temporary service on port {port} did not start")
                    continue
                response = probe.post("/order", json={"text": "dvije margarite"})
            if response.status_code != expected_status:
                fail(f"{kind}: expected HTTP {expected_status}, "
                     f"got {response.status_code} {response.text}")
                continue
            detail = ""
            if response.status_code == 503:
                detail = f", Retry-After={response.headers.get('Retry-After')}"
                if not response.headers.get("Retry-After"):
                    fail("quota: expected a Retry-After header")
                    continue
            body = response.json()
            ok(f"{kind} -> HTTP {response.status_code}{detail}: "
               f"{body.get('detail', '')[:120]}")
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def main() -> int:
    print(f"Demo against the normal service at {BASE_URL}")
    try:
        with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
            client.get("/health")
    except httpx.HTTPError as exc:
        print(f"Cannot reach {BASE_URL} — start it first: "
              f"uvicorn app.main:app --reload ({exc})")
        return 1

    with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
        show_mandatory(client)
        show_protocol(client)
        show_informational(client)
    show_failure_modes()

    header(f"Result: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
