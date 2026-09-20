"""Reranker package: pluggable local/API cross-encoder reranking backends."""

from __future__ import annotations

from functools import lru_cache

from mmss.components.reranker import (  # noqa: F401  (registers all three)
    api_reranker,
    local_cross_encoder,
    noop_reranker,
)
from mmss.components.reranker.base import Reranker
from mmss.config import get_config
from mmss.registry import build


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    # See get_embedder()'s comment -- same process-lifetime caching, here
    # avoiding reloading the cross-encoder model from disk per call.
    cfg = get_config().reranker
    return build("reranker", cfg.provider, **cfg.params)
