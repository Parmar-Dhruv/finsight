"""
Unit tests for Task J-5 comparative evaluation helpers (eval/j5_comparative_eval.py).
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from eval.j5_comparative_eval import (
    compute_faithfulness_proxy,
    compute_numeric_match,
    compute_token_f1,
    extract_numeric_values,
    extract_query_filters,
)


def test_extract_query_filters_company_and_single_year():
    ticker, year = extract_query_filters("What was Apple's total net revenue for fiscal year 2024?")
    assert ticker == "AAPL"
    assert year == 2024


def test_extract_query_filters_temporal_comparison_two_years():
    ticker, year = extract_query_filters(
        "How did Microsoft's cloud revenue change in 2024 compared to 2023?"
    )
    assert ticker == "MSFT"
    assert year is None


def test_extract_numeric_values_ignores_years():
    nums = extract_numeric_values("Apple's FY2024 net sales were $391.0 billion, up 2.0%.")
    assert "391.0" in nums
    assert "2.0" in nums
    assert "2024" not in nums


def test_compute_numeric_match_handles_formatting():
    gold = "Apple's total net revenue for fiscal year 2024 was $391.0 billion."
    assert compute_numeric_match("Total net sales were $391 billion [SOURCE 1].", gold) is True
    assert compute_numeric_match("Total net sales were $100 billion [SOURCE 1].", gold) is False
    assert compute_numeric_match("No numbers here", "Qualitative risk factor answer.") is None


def test_compute_token_f1_strips_citation_tags():
    f1 = compute_token_f1(
        "Apple revenue was $391.0 billion [SOURCE 1].",
        "Apple revenue was $391.0 billion.",
    )
    assert f1 == 1.0


def test_compute_faithfulness_proxy():
    chunks = [{"text": "Total net sales were $391.0 billion in fiscal 2024."}]
    assert compute_faithfulness_proxy("Total net sales were $391.0 billion [SOURCE 1].", chunks, False) > 0.8
    assert compute_faithfulness_proxy("Anything", [], False) == 0.0
    assert compute_faithfulness_proxy("Refusal", chunks, True) == 1.0
