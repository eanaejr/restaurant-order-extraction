"""Data-quality checks for eval_set.json (the sentence set run by run_eval.py)."""

from __future__ import annotations

import json
from pathlib import Path

from app.menu import load_menu

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = PROJECT_ROOT / "eval_set.json"
MENU_FILE = PROJECT_ROOT / "jelovnik.json"


def _load_cases() -> list[dict]:
    cases = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    assert isinstance(cases, list) and len(cases) >= 20
    return cases


def test_every_case_is_well_formed():
    texts = []
    for case in _load_cases():
        assert case["text"].strip(), case
        assert isinstance(case["items"], dict), case
        assert isinstance(case["unavailable"], dict), case
        assert isinstance(case["suggestions"], list), case
        for quantity in {**case["items"], **case["unavailable"]}.values():
            assert isinstance(quantity, int) and quantity >= 1, case
        assert not (set(case["items"]) & set(case["unavailable"])), case
        texts.append(case["text"])
    assert len(texts) == len(set(texts))


def test_expected_ids_and_suggestions_exist_on_the_menu():
    menu = load_menu(MENU_FILE)
    for case in _load_cases():
        for item_id in case["items"]:
            assert item_id in menu, (item_id, case)
        for suggestion in case["suggestions"]:
            assert suggestion in menu, (suggestion, case)
