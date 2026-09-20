"""Vision package: pluggable chart/figure extraction backends."""

from __future__ import annotations

from functools import lru_cache

from mmss.components.vision import api_vlm, local_vlm  # noqa: F401  (registers both)
from mmss.components.vision.base import VisionExtractor
from mmss.config import get_config
from mmss.registry import build


@lru_cache(maxsize=1)
def get_vision_extractor() -> VisionExtractor:
    # See get_embedder()'s comment -- same process-lifetime caching.
    cfg = get_config().vision
    return build("vision", cfg.provider, **cfg.params)
