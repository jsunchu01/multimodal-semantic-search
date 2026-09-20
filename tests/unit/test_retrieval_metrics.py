"""Tests recall_at_k and mean_reciprocal_rank -- pure set/rank arithmetic
over ScoredChunk lists, no DB or model needed."""

from mmss.components.base import Chunk, ScoredChunk
from mmss.components.evaluator.retrieval_metrics import mean_reciprocal_rank, recall_at_k


def _scored(id_: str) -> ScoredChunk:
    chunk = Chunk(
        id=id_, doc_id="doc1", text=f"text {id_}", chunk_type="text",
        page_start=1, page_end=1, source_parser="test",
    )
    return ScoredChunk(chunk=chunk, score=1.0, source="hybrid")


def test_recall_at_k_all_expected_found() -> None:
    retrieved = [_scored("a"), _scored("b"), _scored("c")]
    assert recall_at_k(retrieved, {"a", "c"}, k=3) == 1.0


def test_recall_at_k_partial_match() -> None:
    retrieved = [_scored("a"), _scored("b"), _scored("c")]
    assert recall_at_k(retrieved, {"a", "z"}, k=3) == 0.5


def test_recall_at_k_respects_k_cutoff() -> None:
    retrieved = [_scored("a"), _scored("b"), _scored("c")]
    # "c" is expected but ranked 3rd -- excluded when k=2.
    assert recall_at_k(retrieved, {"c"}, k=2) == 0.0


def test_recall_at_k_no_expected_ids_is_vacuously_satisfied() -> None:
    retrieved = [_scored("a")]
    assert recall_at_k(retrieved, set(), k=5) == 1.0


def test_mrr_reciprocal_of_first_matching_rank() -> None:
    retrieved = [_scored("x"), _scored("a"), _scored("b")]
    assert mean_reciprocal_rank(retrieved, {"a"}) == 0.5  # rank 2 -> 1/2


def test_mrr_uses_earliest_matching_rank_when_multiple_expected_present() -> None:
    retrieved = [_scored("x"), _scored("a"), _scored("b")]
    assert mean_reciprocal_rank(retrieved, {"a", "b"}) == 0.5  # first hit at rank 2


def test_mrr_no_match_is_zero() -> None:
    retrieved = [_scored("x"), _scored("y")]
    assert mean_reciprocal_rank(retrieved, {"z"}) == 0.0


def test_mrr_no_expected_ids_is_vacuously_satisfied() -> None:
    retrieved = [_scored("x")]
    assert mean_reciprocal_rank(retrieved, set()) == 1.0
