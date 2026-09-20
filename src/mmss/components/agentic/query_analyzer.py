"""Classifies a query's intent to drive retrieval breadth (top_k), generator
routing (simple vs. complex model), and whether to invoke the PoT executor.

Deliberately scoped down from the reference repo's QueryAnalyzer to only what
has a real consumer in this project: intent classification, suggested_top_k,
is_complex, and use_pot. NOT built at all (not just left unused): entity
extraction (Article/Section/company -- there's no metadata-filter consumer
for it, PGVectorStore.query()'s filters param is unimplemented), query
rewriting (same reason), prompt-injection detection (no adversarial users --
this is a personal, single-user tool), and should_skip_vision (we never call
a vision model at query time -- charts become plain text chunks once, at
ingest; there's no per-query vision step for it to skip).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class QueryIntent(str, Enum):
    FACTUAL = "factual"  # simple lookup, e.g. "what was total revenue?"
    NUMERIC = "numeric"  # needs arithmetic on retrieved numbers -- routes to PoT
    COMPARATIVE = "comparative"  # "compare X to Y" -- needs chunks from multiple places
    TEMPORAL = "temporal"  # "how has X trended" -- needs chunks across periods


@dataclass
class QueryAnalysis:
    intent: QueryIntent
    use_pot: bool
    suggested_top_k: int
    is_complex: bool


# Broadened beyond finance-only vocabulary (cagr/margin/yoy) to also catch
# research-paper phrasing (improvement/accuracy/outperform) once this project
# started being used for both domains -- same intent categories, wider net.
_NUMERIC_RE = re.compile(
    r"\b(calculat|cagr|compound annual|growth rate|yoy|qoq|year.over.year|"
    r"quarter.over.quarter|percent(age)?|ratio|margin|increase|decrease|"
    r"changed?|how much (more|less|better|worse|higher|lower)|average|mean|"
    r"improv(ed?|ement)|reduc(ed?|tion))\b",
    re.I,
)
_COMPARATIVE_RE = re.compile(
    r"\b(compared?|comparing|versus|vs\.?|difference between|relative to|"
    r"outperform\w*|baseline)\b",
    re.I,
)
_TEMPORAL_RE = re.compile(r"\b(trend|over the (last|past)|historical|since \d{4}|trajectory)\b", re.I)

_WIDE_RETRIEVAL_TOP_K = 10
_DEFAULT_TOP_K = 5
_LONG_QUERY_WORD_COUNT = 30


class QueryAnalyzer:
    """Pure regex classification -- no LLM call, no registry entry (there's
    only one way to do this in this project, not a local/API vendor choice).
    """

    def analyze(self, query: str) -> QueryAnalysis:
        if _NUMERIC_RE.search(query):
            intent = QueryIntent.NUMERIC
        elif _COMPARATIVE_RE.search(query):
            intent = QueryIntent.COMPARATIVE
        elif _TEMPORAL_RE.search(query):
            intent = QueryIntent.TEMPORAL
        else:
            intent = QueryIntent.FACTUAL

        use_pot = intent == QueryIntent.NUMERIC
        suggested_top_k = (
            _WIDE_RETRIEVAL_TOP_K
            if intent in (QueryIntent.COMPARATIVE, QueryIntent.TEMPORAL)
            else _DEFAULT_TOP_K
        )
        is_complex = (
            intent in (QueryIntent.COMPARATIVE, QueryIntent.NUMERIC)
            or len(query.split()) > _LONG_QUERY_WORD_COUNT
        )

        return QueryAnalysis(
            intent=intent,
            use_pot=use_pot,
            suggested_top_k=suggested_top_k,
            is_complex=is_complex,
        )
