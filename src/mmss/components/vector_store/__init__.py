"""Vector store package: pluggable chunk-embedding storage/query backends."""

from __future__ import annotations

from functools import lru_cache

from mmss.components.vector_store import faiss_store, pgvector_store  # noqa: F401  (registers both)
from mmss.components.vector_store.base import VectorStore
from mmss.config import get_config
from mmss.registry import build


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    # See get_embedder()'s comment -- same process-lifetime caching, here
    # avoiding a fresh Postgres connection per call in a long-lived process.
    cfg = get_config().vector_store
    return build("vector_store", cfg.provider, **cfg.params)
