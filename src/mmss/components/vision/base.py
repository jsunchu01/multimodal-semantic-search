"""VisionExtractor interface: pulls structured data out of a chart/figure image."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mmss.components.base import VisionExtractionResult


class VisionExtractor(ABC):
    @abstractmethod
    def extract(
        self, image_path: str, context_text: str | None = None
    ) -> VisionExtractionResult: ...

    @property
    @abstractmethod
    def name(self) -> str: ...
