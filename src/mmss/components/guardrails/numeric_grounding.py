"""Numeric-grounding guardrail: verifies every number in the answer traces back
to a source chunk.

Algorithm: regex-extract numeric tokens from the answer (ints/decimals/
percentages/currency/scale suffixes, via utils/text.py's iter_numeric_claims)
-> extract the same from the source chunks' text -> a claim is grounded if
some source number matches it within `numeric_tolerance` (relative
difference) -> grounding_ratio drives the configured policy (flag/reject/
strip) in pipeline/generate.py.

Deliberately not copied from the reference repo's FinancialGuardrails.
check_numeric_grounding(): that method accepts a `tolerance` parameter but
never reads it in the method body -- the check there is a plain substring
match with no percentage tolerance and no M/B/K unit normalization. Ours
actually uses numeric_tolerance via _within_tolerance below.
"""

from __future__ import annotations

from mmss.components.base import GenerationResult, ScoredChunk
from mmss.components.guardrails.base import Guardrail, GuardrailResult
from mmss.utils.text import extract_numbers, iter_numeric_claims


class NumericGroundingGuardrail(Guardrail):
    def __init__(self, numeric_tolerance: float = 0.01, min_grounding_ratio: float = 0.8) -> None:
        self._numeric_tolerance = numeric_tolerance
        self._min_grounding_ratio = min_grounding_ratio

    def check(
        self, answer: GenerationResult, source_chunks: list[ScoredChunk]
    ) -> GuardrailResult:
        answer_claims = iter_numeric_claims(answer.text)
        if not answer_claims:
            return GuardrailResult(passed=True, grounding_ratio=1.0, ungrounded_claims=[])

        context_values = [
            value for scored in source_chunks for value in extract_numbers(scored.chunk.text)
        ]

        ungrounded = [
            span
            for span, value in answer_claims
            if not any(self._within_tolerance(value, c) for c in context_values)
        ]

        grounding_ratio = 1 - (len(ungrounded) / len(answer_claims))
        passed = grounding_ratio >= self._min_grounding_ratio
        return GuardrailResult(
            passed=passed, grounding_ratio=grounding_ratio, ungrounded_claims=ungrounded
        )

    def _within_tolerance(self, a: float, b: float) -> bool:
        if a == 0 and b == 0:
            return True
        denom = max(abs(a), abs(b))
        if denom == 0:
            return True
        return abs(a - b) / denom <= self._numeric_tolerance
