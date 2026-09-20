"""VectorStore interface: persist chunk embeddings, query by similarity."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mmss.components.base import Chunk, ScoredChunk


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    @abstractmethod
    def query(
        self, embedding: list[float], top_k: int, filters: dict | None = None
    ) -> list[ScoredChunk]: ...

    @abstractmethod
    def get_all(self, filters: dict | None = None) -> list[Chunk]:
        """Returns every indexed chunk -- used to build a fresh BM25 corpus
        each query, since our CLI is a new process per invocation rather
        than a long-lived server with a persisted keyword index."""
        ...

    @abstractmethod
    def persist(self) -> None: ...

    @abstractmethod
    def delete(self, doc_id: str) -> None:
        """Removes every chunk belonging to doc_id. No-op if doc_id isn't present."""
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...
