"""Hybrid retriever: fuses multiple retrievers via Reciprocal Rank Fusion.

RRF score for chunk d: sum(1 / (k + rank_r(d))) over every retriever r that
returned d, default k=60. Fusion key is Chunk.id directly -- a stable id
already assigned at chunking time -- rather than reconstructing an identity
key from document+page+text-hash the way the reference repo does, since we
already carry a real id through the whole pipeline.

Each retriever is paired with its own fetch depth (top_k_dense/top_k_bm25
from RetrievalConfig) so the fusion pool is wider than the final top_k
actually requested -- that's what lets RRF meaningfully re-rank across
sources instead of just re-scoring whatever a single retriever already
picked as its own top result.
"""

from __future__ import annotations

from mmss.components.base import ScoredChunk
from mmss.components.retriever.base import Retriever


class HybridRetriever(Retriever):
    def __init__(self, retrievers: list[tuple[Retriever, int]], rrf_k: int = 60) -> None:
        """retrievers: (retriever, per_retriever_fetch_k) pairs."""
        self._retrievers = retrievers
        self._rrf_k = rrf_k

    @property
    def name(self) -> str:
        return "hybrid"

    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]:
        scores: dict[str, float] = {}
        chunk_by_id: dict[str, ScoredChunk] = {}

        for retriever, fetch_k in self._retrievers:
            ranked = retriever.retrieve(query, fetch_k)
            for rank, scored in enumerate(ranked, start=1):
                key = scored.chunk.id
                scores[key] = scores.get(key, 0.0) + 1.0 / (self._rrf_k + rank)
                if key not in chunk_by_id:
                    chunk_by_id[key] = scored

        ranked_ids = sorted(scores, key=lambda k: scores[k], reverse=True)
        return [
            ScoredChunk(chunk=chunk_by_id[key].chunk, score=scores[key], source="hybrid")
            for key in ranked_ids[:top_k]
        ]
