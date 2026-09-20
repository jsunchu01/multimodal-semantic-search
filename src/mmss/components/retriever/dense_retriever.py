"""Dense retriever: embeds the query, searches the vector store."""

from __future__ import annotations

from mmss.components.base import ScoredChunk
from mmss.components.embedder.base import Embedder
from mmss.components.retriever.base import Retriever
from mmss.components.vector_store.base import VectorStore


class DenseRetriever(Retriever):
    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    @property
    def name(self) -> str:
        return "dense"

    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]:
        query_embedding = self._embedder.embed_query(query)
        return self._vector_store.query(query_embedding, top_k=top_k)
