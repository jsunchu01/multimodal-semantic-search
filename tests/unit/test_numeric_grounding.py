"""Tests NumericGroundingGuardrail.check() against synthetic answer/context
text -- no LLM or retrieval needed, just the grounding logic itself.
"""

from mmss.components.base import Chunk, GenerationResult, ScoredChunk
from mmss.components.guardrails.numeric_grounding import NumericGroundingGuardrail


def _scored(text: str) -> ScoredChunk:
    chunk = Chunk(
        id="c1", doc_id="doc1", text=text, chunk_type="table",
        page_start=1, page_end=1, source_parser="test",
    )
    return ScoredChunk(chunk=chunk, score=1.0, source="hybrid")


def test_grounded_answer_passes() -> None:
    guardrail = NumericGroundingGuardrail(numeric_tolerance=0.01, min_grounding_ratio=0.8)
    answer = GenerationResult(text="Total revenues were $50,623.")
    context = [_scored("Total revenues -- Six Months Ended June 30, 2026: $ 50,623")]

    result = guardrail.check(answer, context)

    assert result.passed
    assert result.grounding_ratio == 1.0
    assert result.ungrounded_claims == []


def test_ungrounded_number_fails() -> None:
    guardrail = NumericGroundingGuardrail(numeric_tolerance=0.01, min_grounding_ratio=0.8)
    answer = GenerationResult(text="Total revenues were $99,999.")
    context = [_scored("Total revenues -- Six Months Ended June 30, 2026: $ 50,623")]

    result = guardrail.check(answer, context)

    assert not result.passed
    assert result.grounding_ratio == 0.0
    assert "$99,999" in result.ungrounded_claims


def test_number_within_tolerance_still_passes() -> None:
    # relative difference between 50,623 and 50,600 is ~0.045%, well under 1%
    guardrail = NumericGroundingGuardrail(numeric_tolerance=0.01, min_grounding_ratio=0.8)
    answer = GenerationResult(text="Total revenues were approximately $50,600.")
    context = [_scored("Total revenues -- Six Months Ended June 30, 2026: $ 50,623")]

    result = guardrail.check(answer, context)

    assert result.passed


def test_number_outside_tolerance_fails() -> None:
    # relative difference between 50,623 and 45,000 is ~11%, over 1%
    guardrail = NumericGroundingGuardrail(numeric_tolerance=0.01, min_grounding_ratio=0.8)
    answer = GenerationResult(text="Total revenues were $45,000.")
    context = [_scored("Total revenues -- Six Months Ended June 30, 2026: $ 50,623")]

    result = guardrail.check(answer, context)

    assert not result.passed


def test_no_numbers_in_answer_trivially_passes() -> None:
    guardrail = NumericGroundingGuardrail()
    answer = GenerationResult(text="This information is not available in the provided documents.")

    result = guardrail.check(answer, [])

    assert result.passed
    assert result.grounding_ratio == 1.0


def test_partial_grounding_ratio() -> None:
    guardrail = NumericGroundingGuardrail(numeric_tolerance=0.01, min_grounding_ratio=0.8)
    answer = GenerationResult(text="Revenue was $50,623 and margin was 99%.")
    context = [_scored("Total revenues -- Six Months Ended June 30, 2026: $ 50,623")]

    result = guardrail.check(answer, context)

    assert result.grounding_ratio == 0.5
    assert not result.passed
    assert result.ungrounded_claims == ["99%"]
