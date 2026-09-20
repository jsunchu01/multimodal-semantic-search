"""Docling-backed parser: own layout model + TableFormer for table structure
recovery (row/col grid, header cells, merged cells, borderless tables) --
the best of the parser options we compared for financial-statement tables.

Import of `docling` stays lazy inside parse() so this module loads fine even
before the `parsing` extra is installed; FallbackParser (see parser/__init__.py)
catches both ImportError and any runtime parse failure and degrades to
UnstructuredParser -- unlike the reference project, which only caught
ImportError and let runtime failures propagate unhandled.

Verified against Docling's current docs: DocumentConverter().convert(source)
.document is the real API -- NOT result.pages[].elements, which is what the
reference repo's (broken) code used. Element position comes from
item.prov[0].page_no / item.prov[0].bbox. Table content comes from
table.export_to_html(doc=doc) (no tabulate/pandas dependency needed, and
it's the same format the reference repo's own table_extractor.py expects
downstream).

NOT independently runtime-tested here (no PDF execution available) -- the
exact DocItemLabel enum member names, bbox field names, and coordinate
origin convention are taken from docs/search, not confirmed against a live
Docling install. Expect a fix-up pass once this runs against a real PDF.

M6 adds real chart image extraction: verified against Docling's own example
(docs/examples/export_figures.py) that this requires PdfPipelineOptions(
generate_picture_images=True) passed into DocumentConverter via
format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=...)} --
without it, picture_item.get_image(doc) has nothing to return. Each picture
is saved as a PNG under data/processed/images/ and its path recorded in
RawElement.metadata["image_path"] for the vision-extraction ingest stage to
pick up (chunker.py carries this metadata through into the chart Chunk).
"""

from __future__ import annotations

from pathlib import Path

from mmss.components.base import RawElement
from mmss.components.parser.base import DocumentParser
from mmss.registry import register
from mmss.utils.logging import get_logger

logger = get_logger(__name__)

_HEADING_LABELS = {"title", "section_header"}
_CAPTION_LABELS = {"caption"}
_SKIP_LABELS = {"page_header", "page_footer"}
_IMAGES_DIR = Path("data/processed/images")


def _bbox_to_dict(bbox: object) -> dict | None:
    """Best-effort conversion of Docling's BoundingBox to a plain dict.

    Field names vary across Docling versions (l/t/r/b vs x0/y0/x1/y1) --
    this defends against both rather than assuming one, and returns None
    rather than raising if neither shape matches.
    """
    if bbox is None:
        return None
    for names in (("l", "t", "r", "b"), ("x0", "y0", "x1", "y1")):
        if all(hasattr(bbox, n) for n in names):
            return {n: getattr(bbox, n) for n in names}
    return None


def _page_and_bbox(item: object) -> tuple[int | None, dict | None]:
    prov_list = getattr(item, "prov", None)
    if not prov_list:
        return None, None
    prov = prov_list[0]
    return getattr(prov, "page_no", None), _bbox_to_dict(getattr(prov, "bbox", None))


@register("parser", "docling")
class DoclingParser(DocumentParser):
    @property
    def name(self) -> str:
        return "docling"

    def parse(self, file_path: str) -> list[RawElement]:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_picture_images = True

        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        result = converter.convert(file_path)
        doc = result.document

        elements: list[RawElement] = []

        for text_item in doc.texts:
            text = (getattr(text_item, "text", "") or "").strip()
            if not text:
                continue
            label = str(getattr(text_item, "label", "text")).lower()
            if label in _SKIP_LABELS:
                continue
            elem_type = (
                "heading"
                if label in _HEADING_LABELS
                else "caption"
                if label in _CAPTION_LABELS
                else "text"
            )
            page_no, bbox = _page_and_bbox(text_item)
            elements.append(
                RawElement(
                    type=elem_type,
                    text=text,
                    page_number=page_no,
                    source_parser=self.name,
                    bbox=bbox,
                )
            )

        for table_item in doc.tables:
            try:
                html = table_item.export_to_html(doc=doc)
            except Exception:
                html = ""
            html = html.strip()
            if not html:
                continue
            page_no, bbox = _page_and_bbox(table_item)
            elements.append(
                RawElement(
                    type="table",
                    text=html,
                    page_number=page_no,
                    source_parser=self.name,
                    bbox=bbox,
                )
            )

        doc_stem = Path(file_path).stem
        for i, picture_item in enumerate(doc.pictures):
            page_no, bbox = _page_and_bbox(picture_item)
            metadata: dict = {}
            try:
                image = picture_item.get_image(doc)
                if image is not None:
                    _IMAGES_DIR.mkdir(parents=True, exist_ok=True)
                    image_path = _IMAGES_DIR / f"{doc_stem}_picture_{i:03d}.png"
                    image.save(image_path, "PNG")
                    metadata["image_path"] = str(image_path)
            except Exception:
                logger.warning(
                    "Could not export image for picture %d on page %s -- vision stage will skip it",
                    i,
                    page_no,
                )
            elements.append(
                RawElement(
                    type="image",
                    text="",
                    page_number=page_no,
                    source_parser=self.name,
                    bbox=bbox,
                    metadata=metadata,
                )
            )

        # doc.texts / doc.tables / doc.pictures are separate lists, not one
        # reading-order stream. This sorts by page only (a stable sort, so
        # within a page the original per-list order is preserved) -- it
        # does NOT precisely interleave text/table/image by vertical
        # position on the page. That mainly affects caption-pairing
        # precision on pages with multiple tables/figures; a bbox-based
        # interleave would need Docling's `coord_origin` field to get the
        # sort direction right, which we haven't verified, so we're not
        # guessing at it here.
        elements.sort(key=lambda e: e.page_number if e.page_number is not None else 0)
        return elements
