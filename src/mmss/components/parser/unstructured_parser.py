"""Unstructured-backed parser: the fallback DoclingParser degrades to on
either an import failure (docling not installed) or a runtime parse failure.

Uses the general partition() dispatcher (not partition_pdf()) so multi-format
input works, not just PDF -- confirmed from partition()'s own source that it
auto-detects file type via libmagic and routes to the right partition_*
function. Two settings have to be right together to actually get table
structure inference, also confirmed from source: `strategy` must be
"hi_res" (the default "auto" resolves to "fast" for any file with a real
text layer, which silently skips table inference -- true for essentially
every born-digital financial filing), and "pdf" must be explicitly removed
from `skip_infer_table_types` (it's in the default skip list even when
pdf_infer_table_structure=True is passed).

NOT independently runtime-tested here -- expect a fix-up pass once this
runs against a real file.
"""

from __future__ import annotations

from mmss.components.base import RawElement
from mmss.components.parser.base import DocumentParser
from mmss.registry import register

_HEADING_TYPES = {"title"}
_TABLE_TYPES = {"table"}
_IMAGE_TYPES = {"image", "figure"}
_CAPTION_TYPES = {"figurecaption"}
_SKIP_TYPES = {"header", "footer"}


@register("parser", "unstructured")
class UnstructuredParser(DocumentParser):
    @property
    def name(self) -> str:
        return "unstructured"

    def parse(self, file_path: str) -> list[RawElement]:
        from unstructured.partition.auto import partition

        raw_elements = partition(
            filename=file_path,
            strategy="hi_res",
            pdf_infer_table_structure=True,
            skip_infer_table_types=[],
            extract_images_in_pdf=True,
        )

        elements: list[RawElement] = []
        for el in raw_elements:
            text = str(el).strip()
            if not text:
                continue
            el_type = type(el).__name__.lower()
            if el_type in _SKIP_TYPES:
                continue
            mapped_type = (
                "heading"
                if el_type in _HEADING_TYPES
                else "table"
                if el_type in _TABLE_TYPES
                else "image"
                if el_type in _IMAGE_TYPES
                else "caption"
                if el_type in _CAPTION_TYPES
                else "text"
            )
            page_number = getattr(el.metadata, "page_number", None) if hasattr(el, "metadata") else None
            elements.append(
                RawElement(
                    type=mapped_type,
                    text=text,
                    page_number=page_number,
                    source_parser=self.name,
                    # Unstructured's coordinates are an arbitrary polygon, not
                    # a simple box -- not approximating a bbox for this
                    # backend in M1 rather than doing it incorrectly.
                    bbox=None,
                )
            )
        return elements
