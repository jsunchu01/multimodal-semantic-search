"""FAISS-backed vector store: an alternative to pgvector, manual ID/metadata bookkeeping.

Not on the milestone plan yet -- registered so it's a config change away if
running a local Postgres ever becomes more overhead than it's worth.
"""

from __future__ import annotations

from mmss.components.base import Chunk, ScoredChunk
from mmss.components.vector_store.base import VectorStore
from mmss.registry import register


@register("vector_store", "faiss")
class FAISSVectorStore(VectorStore):
    def __init__(self, index_path: str = "data/index/faiss.index") -> None:
        self._index_path = index_path

    @property
    def name(self) -> str:
        return f"faiss:{self._index_path}"

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        raise NotImplementedError("Optional alternative to Chroma -- not yet scheduled on the milestone plan")

    def query(
        self, embedding: list[float], top_k: int, filters: dict | None = None
    ) -> list[ScoredChunk]:
        raise NotImplementedError("Optional alternative to pgvector -- not yet scheduled on the milestone plan")

    def get_all(self, filters: dict | None = None) -> list[Chunk]:
        raise NotImplementedError("Optional alternative to pgvector -- not yet scheduled on the milestone plan")

    def persist(self) -> None:
        raise NotImplementedError("Optional alternative to pgvector -- not yet scheduled on the milestone plan")

    def delete(self, doc_id: str) -> None:
        raise NotImplementedError("Optional alternative to pgvector -- not yet scheduled on the milestone plan")
