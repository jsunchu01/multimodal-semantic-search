"""Chroma-backed vector store: embedded, persists to disk, no external service needed."""

from __future__ import annotations

from mmss.components.base import Chunk, ScoredChunk
from mmss.components.vector_store.base import VectorStore
from mmss.registry import register


@register("vector_store", "chroma")
class ChromaVectorStore(VectorStore):
    def __init__(
        self,
        persist_directory: str = "data/index/chroma",
        collection_name: str = "mmss_chunks",
    ) -> None:
        self._persist_directory = persist_directory
        self._collection_name = collection_name

    @property
    def name(self) -> str:
        return f"chroma:{self._collection_name}"

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        raise NotImplementedError("Implemented in M2 -- local semantic search")

    def query(
        self, embedding: list[float], top_k: int, filters: dict | None = None
    ) -> list[ScoredChunk]:
        raise NotImplementedError("Implemented in M2 -- local semantic search")

    def get_all(self, filters: dict | None = None) -> list[Chunk]:
        raise NotImplementedError("Implemented in M2 -- local semantic search")

    def persist(self) -> None:
        raise NotImplementedError("Implemented in M2 -- local semantic search")

    def delete(self, doc_id: str) -> None:
        raise NotImplementedError("Implemented in M2 -- local semantic search")
