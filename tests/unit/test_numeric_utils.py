"""Tests normalize_number/extract_numbers/iter_numeric_claims -- pure regex
and arithmetic, no model or DB needed.
"""

from mmss.utils.text import extract_numbers, iter_numeric_claims, normalize_number


def test_normalize_plain_number_with_commas() -> None:
    assert normalize_number("50,623", None) == 50623.0


def test_normalize_accounting_negative_parens() -> None:
    assert normalize_number("(173)", None) == -173.0


def test_normalize_currency_symbol() -> None:
    assert normalize_number("$1,591", None) == 1591.0


def test_normalize_percent_is_not_scaled() -> None:
    assert normalize_number("24", "%") == 24.0


def test_normalize_billion_scale() -> None:
    assert normalize_number("50.6", "billion") == 50_600_000_000.0


def test_normalize_million_letter_suffix() -> None:
    assert normalize_number("5", "M") == 5_000_000.0


def test_extract_numbers_finds_all_claims_in_a_sentence() -> None:
    text = "Total revenues were $50,623 and $41,831 respectively"
    assert extract_numbers(text) == [50623.0, 41831.0]


def test_extract_numbers_applies_scale_suffix() -> None:
    text = "Revenue grew to $50.6 billion"
    assert extract_numbers(text) == [50_600_000_000.0]


def test_iter_numeric_claims_returns_matched_span_and_value() -> None:
    claims = iter_numeric_claims("Interest expense was (173) this quarter")
    assert claims == [("(173)", -173.0)]


def test_no_numbers_returns_empty_list() -> None:
    assert extract_numbers("This information is not available.") == []
