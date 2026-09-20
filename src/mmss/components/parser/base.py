"""Parser-package interfaces: DocumentParser turns a file into RawElements,
Chunker turns those into Chunks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from mmss.components.base import Chunk, RawElement


class DocumentParser(ABC):
    """Extracts text/tables/layout from a document, before chunking."""

    @abstractmethod
    def parse(self, file_path: str) -> list[RawElement]: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class Chunker(ABC):
    """Turns a parser's RawElements into retrieval-sized Chunks.

    Owns the layout-aware behavior that's shared across every parser backend:
    caption<->table/chart pairing, cross-page merges, and char-budget splitting
    -- kept here so chunk shape doesn't depend on which parser backend ran.
    """

    @abstractmethod
    def chunk(self, elements: list[RawElement], doc_id: str) -> list[Chunk]: ...
