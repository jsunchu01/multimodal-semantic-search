"""Generic API reranker wrapper (e.g. Cohere Rerank -- vendor picked later)."""

from __future__ import annotations

from mmss.components.base import ScoredChunk
from mmss.components.reranker.base import Reranker
from mmss.registry import register


@register("reranker", "api_reranker")
class APIReranker(Reranker):
    def __init__(self, vendor: str, model_name: str) -> None:
        self._vendor = vendor
        self._model_name = model_name

    @property
    def name(self) -> str:
        return f"api_reranker:{self._vendor}:{self._model_name}"

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        raise NotImplementedError("Implemented in M7 -- pluggable API backends")
