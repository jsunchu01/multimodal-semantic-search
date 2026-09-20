"""Retriever interface: text query -> scored chunks."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mmss.components.base import ScoredChunk


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]: ...

    @property
    @abstractmethod
    def name(self) -> str: ...
