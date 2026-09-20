"""Tests QueryAnalyzer's intent classification and the derived signals
(use_pot, suggested_top_k, is_complex) -- pure regex, no LLM needed.
"""

from mmss.components.agentic.query_analyzer import QueryAnalyzer, QueryIntent


def test_factual_query() -> None:
    analysis = QueryAnalyzer().analyze("What was total revenue for the six months ended June 30, 2026?")

    assert analysis.intent == QueryIntent.FACTUAL
    assert analysis.use_pot is False
    assert analysis.suggested_top_k == 5
    assert analysis.is_complex is False


def test_numeric_query_triggers_pot() -> None:
    analysis = QueryAnalyzer().analyze("What was the year-over-year growth rate in automotive revenue?")

    assert analysis.intent == QueryIntent.NUMERIC
    assert analysis.use_pot is True
    assert analysis.is_complex is True


def test_comparative_query_widens_retrieval() -> None:
    # Deliberately avoids "margin"/"ratio"/etc -- those are NUMERIC_RE
    # keywords too, and NUMERIC is checked before COMPARATIVE, so a query
    # combining both would classify as NUMERIC, not COMPARATIVE.
    analysis = QueryAnalyzer().analyze("Compare automotive revenue to energy revenue")

    assert analysis.intent == QueryIntent.COMPARATIVE
    assert analysis.suggested_top_k == 10
    assert analysis.is_complex is True
    assert analysis.use_pot is False


def test_temporal_query_widens_retrieval_but_not_complex() -> None:
    analysis = QueryAnalyzer().analyze("How has revenue trended over the last few quarters?")

    assert analysis.intent == QueryIntent.TEMPORAL
    assert analysis.suggested_top_k == 10
    assert analysis.is_complex is False
    assert analysis.use_pot is False


def test_long_query_is_complex_even_if_factual() -> None:
    long_query = " ".join(["word"] * 31) + " what was revenue"
    analysis = QueryAnalyzer().analyze(long_query)

    assert analysis.is_complex is True


def test_numeric_pattern_takes_priority_over_comparative() -> None:
    # Contains both a comparative cue ("compare") and a numeric cue ("cagr")
    # -- numeric is checked first, since it's the one that decides use_pot.
    analysis = QueryAnalyzer().analyze("Compare and calculate the CAGR between 2020 and 2025")

    assert analysis.intent == QueryIntent.NUMERIC
    assert analysis.use_pot is True
