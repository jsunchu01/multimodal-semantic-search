"""Embedder package: pluggable local/API text-embedding backends."""

from __future__ import annotations

from functools import lru_cache

from mmss.components.embedder import api_embedder, local_embedder  # noqa: F401  (registers both)
from mmss.components.embedder.base import Embedder
from mmss.config import get_config
from mmss.registry import build


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    # Cached for the life of the process: constructing a fresh instance per
    # call is free for the one-shot CLI (each invocation is its own process
    # anyway) but would reload the local embedding model from disk on every
    # single call in a long-lived process like the Streamlit UI (M10). Takes
    # no arguments (reads get_config() internally), so this is just "build
    # once, reuse after that" -- safe as long as config doesn't change
    # mid-process, which none of our entrypoints do today.
    cfg = get_config().embedder
    return build("embedder", cfg.provider, **cfg.params)
