"""Parser package: pluggable DocumentParser backends behind a primary/fallback wrapper."""

from __future__ import annotations

from mmss.components.base import RawElement
from mmss.components.parser import docling_parser, unstructured_parser  # noqa: F401  (registers both)
from mmss.components.parser.base import Chunker, DocumentParser
from mmss.components.parser.chunker import DefaultChunker
from mmss.config import get_config
from mmss.registry import build
from mmss.utils.logging import get_logger

logger = get_logger(__name__)


class FallbackParser(DocumentParser):
    """Tries `primary`, falls back to `fallback` on ANY failure (import or runtime).

    This is the one gap we deliberately fixed vs. the reference project: there,
    the Docling backend only caught ImportError, so a runtime parse failure
    propagated instead of degrading gracefully.
    """

    def __init__(self, primary: DocumentParser, fallback: DocumentParser) -> None:
        self._primary = primary
        self._fallback = fallback

    @property
    def name(self) -> str:
        return f"{self._primary.name}(fallback={self._fallback.name})"

    def parse(self, file_path: str) -> list[RawElement]:
        try:
            return self._primary.parse(file_path)
        except Exception:
            logger.warning(
                "Primary parser %s failed on %s, falling back to %s",
                self._primary.name,
                file_path,
                self._fallback.name,
            )
            return self._fallback.parse(file_path)


def get_parser() -> DocumentParser:
    cfg = get_config().parser
    primary = build("parser", cfg.primary)
    fallback = build("parser", cfg.fallback)
    return FallbackParser(primary=primary, fallback=fallback)


def get_chunker() -> Chunker:
    cfg = get_config().chunking
    return DefaultChunker(max_characters=cfg.max_characters)
