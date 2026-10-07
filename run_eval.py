"""Runs the evaluation set (eval_set.json) against a running service
and prints the pass rate.

Usage:
    uvicorn app.main:app --reload    # terminal 1 (needs GEMINI_API_KEY)
    python run_eval.py               # terminal 2
    python run_eval.py --start=13    # resume from case 13

Sends every sentence from eval_set.json, compares the answers with the
expected results and prints the pass rate (a measured number instead of a
feeling). Requests are paced below the free-tier 5/min rate limit; on a
503 the script waits for Retry-After and retries that sentence once;
transport errors (timeouts, resets) count as failed cases instead of
aborting the run. The client is patient on purpose: the service cuts a
stalled Gemini call after 60s, but the whole retry/fallback chain can
legitimately take minutes.
Exit code 0 = every case passed.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
EVAL_FILE = Path(__file__).resolve().parent / "eval_set.json"
PACE_S = 13.0  # stays under the free-tier 5 requests/min limit
TIMEOUT_S = 400.0  # the retry/fallback chain can legitimately take minutes


def _start_from_args() -> int:
    for arg in sys.argv[1:]:
        if arg.startswith("--start="):
            value = arg.split("=", 1)[1]
            if value.isdigit():
                return max(1, int(value))
    return 1


def load_cases() -> list[dict]:
    cases = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise SystemExit(f"No evaluation cases found in {EVAL_FILE}")
    return cases


def _matches(body: dict, case: dict) -> bool:
    got_items = {item["id"]: item["quantity"] for item in body.get("items", [])}
    got_unavailable = {item["text"]: item["quantity"] for item in body.get("unavailable", [])}
    got_suggestions = set(body.get("suggestions", []))
    return (got_items == case["items"]
            and got_unavailable == case["unavailable"]
            and got_suggestions == set(case["suggestions"]))


def _post(client: httpx.Client, case: dict) -> httpx.Response:
    response = client.post("/order", json={"text": case["text"]})
    if response.status_code == 503:
        wait = int(response.headers.get("Retry-After", "60"))
        print(f"      503 — waiting {wait}s (rate limit), one retry")
        time.sleep(wait)
        response = client.post("/order", json={"text": case["text"]})
    return response


def _ask(client: httpx.Client, case: dict) -> tuple[bool, str]:
    try:
        response = _post(client, case)
    except httpx.HTTPError as exc:
        return False, f"request failed: {exc}"
    if response.status_code != 200:
        return False, f"HTTP {response.status_code}: {response.json().get('detail', '')}"
    if _matches(response.json(), case):
        return True, ""
    body = response.json()
    got_items = {i["id"]: i["quantity"] for i in body.get("items", [])}
    got_unavailable = {u["text"]: u["quantity"] for u in body.get("unavailable", [])}
    got_suggestions = sorted(set(body.get("suggestions", [])))
    return False, (
        f"expected items={case['items']} unavailable={case['unavailable']} "
        f"suggestions={sorted(case['suggestions'])}\n"
        f"      got      items={got_items} unavailable={got_unavailable} "
        f"suggestions={got_suggestions}"
    )


def main() -> int:
    start = _start_from_args()
    cases = load_cases()
    todo = cases[start - 1:]
    if not todo:
        print(f"--start={start} is beyond the last case ({len(cases)})")
        return 1
    print(f"Evaluating {len(todo)} of {len(cases)} sentences against {BASE_URL} "
          f"(paced ~{PACE_S:.0f}s apart — free-tier rate limits)")
    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT_S) as client:
        health = client.get("/health").json()
        print(f"model={health.get('model')}, fallbacks={health.get('fallback_models')}\n")
        passed = failed = 0
        for index, case in enumerate(todo, start):
            if index > start:
                time.sleep(PACE_S)
            ok, detail = _ask(client, case)
            if ok:
                passed += 1
                print(f"PASS  [{index}/{len(cases)}] {case['text']!r}")
            else:
                failed += 1
                print(f"FAIL  [{index}/{len(cases)}] {case['text']!r}")
                print(f"      {detail}")
    rate = passed / len(todo) * 100
    print(f"\nResult: {passed}/{len(todo)} passed ({rate:.1f}%)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
