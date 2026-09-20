"""Generator package: pluggable local/API answer-generation backends."""

from __future__ import annotations

from functools import lru_cache

from mmss.components.generator import api_llm, local_llm  # noqa: F401  (registers both)
from mmss.components.generator.base import Generator
from mmss.config import get_config
from mmss.registry import build


@lru_cache(maxsize=1)
def get_generator() -> Generator:
    # See get_embedder()'s comment -- same process-lifetime caching. Cheaper
    # to skip for API/Ollama generators (no in-process model weights either
    # way), but consistent with the other four getters rather than a special
    # case, and free of downside.
    cfg = get_config().generator
    return build("generator", cfg.provider, **cfg.params)
