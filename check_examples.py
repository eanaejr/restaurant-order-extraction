"""Sends the example sentences from the task against a running service.

Usage:
    uvicorn app.main:app --reload        # in one terminal (needs GEMINI_API_KEY)
    python check_examples.py             # in another

BASE_URL can be overridden via the environment or the first argument.
All five sentences from the task ("Primjeri za provjeru") are asserted,
including the two optional cases: the meat-free question (suggestions)
and the guest changing their mind mid-sentence.
Exit code 0 = all checks passed, 1 = at least one failed.
"""

from __future__ import annotations

import os
import sys

import httpx

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")

# The five sentences from the task, with the expected results.
CASES = [
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


def run_cases(client: httpx.Client) -> bool:
    ok = True
    for case in CASES:
        try:
            response = client.post("/order", json={"text": case["text"]})
            body = response.json()
        except httpx.HTTPError as exc:
            print(f"FAIL  {case['text']!r}: request failed: {exc}")
            ok = False
            continue

        if response.status_code != 200:
            print(f"FAIL  {case['text']!r}: HTTP {response.status_code}: {body}")
            ok = False
            continue

        got_items = {item["id"]: item["quantity"] for item in body.get("items", [])}
        got_unavailable = {
            item["text"]: item["quantity"] for item in body.get("unavailable", [])
        }
        got_suggestions = set(body.get("suggestions", []))
        if (got_items == case["items"]
                and got_unavailable == case["unavailable"]
                and got_suggestions == case["suggestions"]):
            print(f"PASS  {case['text']!r}")
            print(f"      items={got_items} unavailable={got_unavailable} "
                  f"suggestions={sorted(got_suggestions)}")
        else:
            print(f"FAIL  {case['text']!r}")
            print(f"      expected items={case['items']} "
                  f"unavailable={case['unavailable']} "
                  f"suggestions={sorted(case['suggestions'])}")
            print(f"      got      items={got_items} unavailable={got_unavailable} "
                  f"suggestions={sorted(got_suggestions)}")
            ok = False
    return ok


def run_protocol_checks(client: httpx.Client) -> bool:
    """Checks that need no Gemini call: input validation and liveness."""

    ok = True

    response = client.post("/order", json={"text": "   "})
    if response.status_code == 422:
        print("PASS  protocol: blank text -> 422")
    else:
        print(f"FAIL  protocol: blank text -> expected 422, got {response.status_code}")
        ok = False

    response = client.get("/health")
    if response.status_code == 200 and response.json().get("status") == "ok":
        print("PASS  protocol: GET /health -> ok")
    else:
        print(f"FAIL  protocol: GET /health -> unexpected {response.status_code} {response.text}")
        ok = False
    return ok


def main() -> int:
    with httpx.Client(base_url=BASE_URL, timeout=90.0) as client:
        try:
            health = client.get("/health").json()
        except httpx.HTTPError as exc:
            print(f"Cannot reach {BASE_URL} — is the service running? ({exc})")
            return 1
        if not health.get("api_key_set"):
            print("Warning: the service reports api_key_set=false; "
                  "POST /order will return 500. Set GEMINI_API_KEY and restart it.")
        ok = run_cases(client) & run_protocol_checks(client)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
