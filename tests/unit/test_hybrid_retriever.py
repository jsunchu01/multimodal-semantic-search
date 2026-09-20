"""Tests HybridRetriever's RRF fusion against fake retrievers with known
rankings -- no DB, embedder, or BM25 corpus needed, just the fusion math.
"""

from mmss.components.base import Chunk, ScoredChunk
from mmss.components.retriever.base import Retriever
from mmss.components.retriever.hybrid import HybridRetriever


def _chunk(id_: str) -> Chunk:
    return Chunk(
        id=id_, doc_id="doc1", text=f"text {id_}", chunk_type="text",
        page_start=1, page_end=1, source_parser="test",
    )


class _FakeRetriever(Retriever):
    """Returns chunks in a fixed, predetermined order regardless of query."""

    def __init__(self, name: str, ordered_ids: list[str]) -> None:
        self._name = name
        self._ordered_ids = ordered_ids

    @property
    def name(self) -> str:
        return self._name

    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]:
        return [
            ScoredChunk(chunk=_chunk(cid), score=1.0, source=self._name)
            for cid in self._ordered_ids[:top_k]
        ]


def test_rrf_combines_rankings_from_both_sources() -> None:
    dense = _FakeRetriever("dense", ["a", "b", "c"])
    bm25 = _FakeRetriever("bm25", ["b", "a", "d"])

    hybrid = HybridRetriever([(dense, 10), (bm25, 10)], rrf_k=60)
    results = hybrid.retrieve("query", top_k=4)

    # a: 1/61 (dense rank1) + 1/62 (bm25 rank2) == b: 1/62 (dense rank2) + 1/61 (bm25 rank1)
    # -- exact tie. Stable sort keeps "a" first since it's inserted into the
    # scores dict before "b" (dense is processed first, and "a" ranks ahead
    # of "b" within dense). c/d tie the same way, "c" before "d".
    assert [r.chunk.id for r in results] == ["a", "b", "c", "d"]


def test_rrf_chunk_in_only_one_source_still_included() -> None:
    dense = _FakeRetriever("dense", ["x"])
    bm25 = _FakeRetriever("bm25", [])

    hybrid = HybridRetriever([(dense, 10), (bm25, 10)], rrf_k=60)
    results = hybrid.retrieve("query", top_k=5)

    assert len(results) == 1
    assert results[0].chunk.id == "x"


def test_rrf_respects_final_top_k() -> None:
    dense = _FakeRetriever("dense", ["a", "b", "c", "d", "e"])
    bm25 = _FakeRetriever("bm25", [])

    hybrid = HybridRetriever([(dense, 10), (bm25, 10)], rrf_k=60)
    results = hybrid.retrieve("query", top_k=2)

    assert [r.chunk.id for r in results] == ["a", "b"]
