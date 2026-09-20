"""Flattens an HTML table (e.g. Docling's table.export_to_html()) into
natural-language sentences.

Motivated by a concrete finding: BM25 tokenizes raw markup tags ("table",
"tbody", "tr", "td", "th") as noise tokens that dilute a table chunk's score,
and the cross-encoder reranker (trained on natural-language passage pairs,
not HTML) actively demoted a table containing the exact answer to a query
below ten worse text chunks. Flattening at chunk-creation time fixes both,
plus dense embeddings, since every downstream consumer reads the same text.

Verified against BeautifulSoup's docs: BeautifulSoup(html, "html.parser") uses
the stdlib parser (no lxml needed), tag.get(attr) returns None if the
attribute is missing, tag.get_text(strip=True) gives clean cell text.

Header-row detection is a convention observed directly in real Docling output
from a Tesla 10-Q, not a general HTML-table spec: financial-statement tables
consistently use an empty corner cell above the row-label column in every
header row, and a non-empty first cell in every data row. Tables that don't
match this shape fall back to plain tag-stripped text -- still removing the
markup-token noise even when we can't produce full natural-language rows.
"""

from __future__ import annotations


def flatten_html_table(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    rows = soup.find_all("tr")
    if not rows:
        return soup.get_text(" ", strip=True)

    header_rows: list[list[tuple[str, int]]] = []  # per row: [(text, colspan), ...]
    data_rows: list[list[str]] = []

    for row in rows:
        cells = row.find_all(["td", "th"])
        if not cells:
            continue
        first_text = cells[0].get_text(strip=True)
        if first_text == "":
            header_rows.append(
                [(c.get_text(strip=True), int(c.get("colspan", 1))) for c in cells[1:]]
            )
        else:
            data_rows.append([c.get_text(strip=True) for c in cells])

    if not header_rows or not data_rows:
        return soup.get_text(" ", strip=True)

    expanded_headers: list[list[str]] = []
    for header_row in header_rows:
        expanded: list[str] = []
        for text, colspan in header_row:
            expanded.extend([text] * max(colspan, 1))
        expanded_headers.append(expanded)

    num_cols = max(len(h) for h in expanded_headers)
    column_labels = [
        " ".join(h[col] for h in expanded_headers if col < len(h) and h[col])
        for col in range(num_cols)
    ]

    lines: list[str] = []
    for row in data_rows:
        label, values = row[0], row[1:]
        if not any(v for v in values):
            lines.append(f"{label}:")
            continue
        parts = [
            f"{column_labels[i]}: {v}" if i < len(column_labels) and column_labels[i] else v
            for i, v in enumerate(values)
            if v
        ]
        lines.append(f"{label} -- {'; '.join(parts)}")

    return "\n".join(lines)
