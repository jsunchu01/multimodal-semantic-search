"""Pass-through reranker: no reordering, just truncates to top_k.

Lets reranking be disabled via config (provider: noop), or used as the "before"
side of a before/after comparison the same way the CLI already compares dense
vs. hybrid -- run the same query with --strategy hybrid vs. --strategy reranked.
Score/source are left untouched since no actual reranking happened.
"""

from __future__ import annotations

from mmss.components.base import ScoredChunk
from mmss.components.reranker.base import Reranker
from mmss.registry import register


@register("reranker", "noop")
class NoOpReranker(Reranker):
    @property
    def name(self) -> str:
        return "noop"

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        return candidates[:top_k]
