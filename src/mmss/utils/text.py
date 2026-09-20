"""Shared text helpers used across components (BM25 tokenization, numeric
normalization for the guardrail) so tokenization/number-parsing behavior is
defined once instead of per-component.

Known scope limitation for the numeric helpers below: a number is normalized
using only whatever scale suffix (M/B/K/%/word) appears directly next to it.
A financial table's own unit caption (e.g. "in millions, except per share
data") usually lives in a separate, earlier text chunk, not attached to each
number -- so a source row reading "50,623" (meaning $50,623 million) will not
match an answer phrased as "$50.6 billion", even though they're the same
quantity. This is fine for the common case, where the system prompt already
instructs the model to copy exact source figures rather than convert units,
but it's not a general unit-aware comparison.
"""

from __future__ import annotations

import re

_SCALE_MULTIPLIERS = {
    "k": 1_000,
    "thousand": 1_000,
    "m": 1_000_000,
    "million": 1_000_000,
    "b": 1_000_000_000,
    "billion": 1_000_000_000,
    "t": 1_000_000_000_000,
    "trillion": 1_000_000_000_000,
}

# Matches an optional accounting-style negative paren, optional currency
# symbol, the digits themselves (with commas/decimal), and an optional scale
# suffix -- e.g. "(173)", "$1,591", "50.6 billion", "24%".
_NUMERIC_CLAIM_RE = re.compile(
    r"""
    (?P<open_paren>\()?
    \s*[$€£]?\s*
    (?P<number>\d[\d,]*(?:\.\d+)?)
    \s*(?P<close_paren>\))?
    \s*(?P<scale>%|percent\b|thousand\b|million\b|billion\b|trillion\b|[kKmMbBtT]\b)?
    """,
    re.VERBOSE,
)


def simple_tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-only tokenizer for BM25 -- no external NLP dependency."""
    return re.findall(r"[a-z0-9]+", text.lower())


def normalize_number(raw_value: str, scale: str | None) -> float:
    """Expand a matched numeric literal + optional scale suffix (M/B/K/%) into
    a comparable float. `raw_value` may be parenthesized for accounting-style
    negatives (e.g. "(173)") and may carry a leading currency symbol.
    """
    cleaned = raw_value.strip()
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    if negative:
        cleaned = cleaned[1:-1]
    cleaned = cleaned.replace(",", "").lstrip("$€£").strip()
    value = float(cleaned)
    if negative:
        value = -value
    if scale and scale.lower() not in ("%", "percent"):
        value *= _SCALE_MULTIPLIERS.get(scale.lower(), 1)
    return value


def iter_numeric_claims(text: str) -> list[tuple[str, float]]:
    """Find every numeric claim in `text`, returned as (matched_span, value)
    pairs -- the span is needed by the guardrail's "strip" policy to replace
    specific ungrounded numbers in the generated answer text.
    """
    claims: list[tuple[str, float]] = []
    for match in _NUMERIC_CLAIM_RE.finditer(text):
        number = match.group("number")
        if number is None:
            continue
        has_parens = bool(match.group("open_paren") and match.group("close_paren"))
        raw_value = f"({number})" if has_parens else number
        try:
            value = normalize_number(raw_value, match.group("scale"))
        except ValueError:
            continue
        claims.append((match.group(0).strip(), value))
    return claims


def extract_numbers(text: str) -> list[float]:
    """Find every numeric claim in `text`, normalized via `normalize_number`."""
    return [value for _, value in iter_numeric_claims(text)]
