"""Local sentence-transformers CrossEncoder reranker (default: ms-marco-MiniLM-L-6-v2).

Verified against the reference repo's actual reranker/__init__.py: a real
cross-encoder jointly encodes [query, chunk_text] pairs through one model
(model.predict(pairs)) rather than comparing two separate embeddings -- that
query/document interaction is what lets it correct cases where dense/BM25/RRF
ranked something too low (e.g. table chunks whose HTML markup diluted their
BM25 score in M3).

Kept synchronous, unlike the reference repo's asyncio.to_thread wrap -- their
wrap only existed to keep an async app from blocking on a CPU/GPU-bound call;
we're sync end-to-end through M6, async is deferred to M7 where it actually
pays off (concurrent network calls, not local inference).

Graceful degrade on missing dependency: returns the input candidates unranked
(truncated to top_k, original score/source untouched) rather than crashing --
same fallback shape as FallbackParser in components/parser/__init__.py.
"""

from __future__ import annotations

from mmss.components.base import ScoredChunk
from mmss.components.reranker.base import Reranker
from mmss.registry import register
from mmss.utils.logging import get_logger

logger = get_logger(__name__)


@register("reranker", "local_cross_encoder")
class LocalCrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._model = None  # lazy-loaded

    @property
    def name(self) -> str:
        return f"local_cross_encoder:{self._model_name}"

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self._model_name)
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed -- reranker disabled, "
                    "returning candidates unranked (pip install sentence-transformers)"
                )
                return None
        return self._model

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []

        model = self._get_model()
        if model is None:
            return candidates[:top_k]

        pairs = [[query, c.chunk.text] for c in candidates]
        scores = model.predict(pairs)

        reranked = sorted(
            zip(scores, candidates, strict=True), key=lambda pair: pair[0], reverse=True
        )
        return [
            ScoredChunk(chunk=c.chunk, score=float(score), source="rerank")
            for score, c in reranked[:top_k]
        ]
