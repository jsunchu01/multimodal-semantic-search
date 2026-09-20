"""Reranker interface: reorders candidate chunks for a given query."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mmss.components.base import ScoredChunk


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]: ...

    @property
    @abstractmethod
    def name(self) -> str: ...
