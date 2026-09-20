"""Tests LocalCrossEncoderReranker's sort/truncate logic and graceful-degrade
path, plus NoOpReranker -- no real model is loaded. CrossEncoder.predict is
stood in for via a fake model object assigned directly to `._model`, so these
stay fast, offline unit tests (same philosophy as the BM25/hybrid tests: no
DB, embedder, or real model needed, just the reranking logic itself).
"""

from mmss.components.base import Chunk, ScoredChunk
from mmss.components.reranker.local_cross_encoder import LocalCrossEncoderReranker
from mmss.components.reranker.noop_reranker import NoOpReranker


def _scored(id_: str) -> ScoredChunk:
    chunk = Chunk(
        id=id_, doc_id="doc1", text=f"text {id_}", chunk_type="text",
        page_start=1, page_end=1, source_parser="test",
    )
    return ScoredChunk(chunk=chunk, score=1.0, source="hybrid")


class _FakeModel:
    """Stands in for CrossEncoder -- returns predetermined scores, one per pair,
    in the same order the pairs were given."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def predict(self, pairs):
        assert len(pairs) == len(self._scores)
        return self._scores


def test_rerank_sorts_by_model_score_descending() -> None:
    candidates = [_scored("a"), _scored("b"), _scored("c")]
    reranker = LocalCrossEncoderReranker()
    reranker._model = _FakeModel([0.1, 0.9, 0.5])  # a=0.1, b=0.9, c=0.5

    results = reranker.rerank("query", candidates, top_k=3)

    assert [r.chunk.id for r in results] == ["b", "c", "a"]
    assert results[0].score == 0.9
    assert all(r.source == "rerank" for r in results)


def test_rerank_respects_top_k() -> None:
    candidates = [_scored("a"), _scored("b"), _scored("c")]
    reranker = LocalCrossEncoderReranker()
    reranker._model = _FakeModel([0.1, 0.9, 0.5])

    results = reranker.rerank("query", candidates, top_k=1)

    assert [r.chunk.id for r in results] == ["b"]


def test_rerank_empty_candidates_returns_empty() -> None:
    reranker = LocalCrossEncoderReranker()
    assert reranker.rerank("query", [], top_k=5) == []


def test_rerank_degrades_to_unranked_truncation_when_model_unavailable() -> None:
    candidates = [_scored("a"), _scored("b"), _scored("c")]
    reranker = LocalCrossEncoderReranker()
    reranker._get_model = lambda: None  # simulate sentence-transformers not installed

    results = reranker.rerank("query", candidates, top_k=2)

    assert [r.chunk.id for r in results] == ["a", "b"]  # input order preserved, just truncated
    assert results[0].source == "hybrid"  # untouched -- no reranking actually happened


def test_noop_reranker_truncates_without_reordering() -> None:
    candidates = [_scored("a"), _scored("b"), _scored("c")]
    reranker = NoOpReranker()

    results = reranker.rerank("query", candidates, top_k=2)

    assert [r.chunk.id for r in results] == ["a", "b"]
    assert results[0].source == "hybrid"
