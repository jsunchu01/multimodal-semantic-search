"""Tests flatten_html_table against real table shapes pulled from the Tesla
10-Q test fixture (two-level colspan header, single-row header, a section
label row with no values) plus the tag-stripped fallback for irregular
tables. No BM25/embedder/reranker involved -- just the HTML->text logic.
"""

from mmss.utils.table_flatten import flatten_html_table


def test_flattens_two_level_colspan_header() -> None:
    html = (
        "<table><tbody>"
        '<tr><td></td><th colspan="2">Three Months Ended June 30,</th>'
        '<th colspan="2">Six Months Ended June 30,</th></tr>'
        "<tr><td></td><th>2026</th><th>2025</th><th>2026</th><th>2025</th></tr>"
        "<tr><th>Total revenues</th><td>$ 28,236</td><td>$ 22,496</td>"
        "<td>$ 50,623</td><td>$ 41,831</td></tr>"
        "</tbody></table>"
    )

    result = flatten_html_table(html)

    assert result == (
        "Total revenues -- Three Months Ended June 30, 2026: $ 28,236; "
        "Three Months Ended June 30, 2025: $ 22,496; "
        "Six Months Ended June 30, 2026: $ 50,623; "
        "Six Months Ended June 30, 2025: $ 41,831"
    )


def test_flattens_single_row_header() -> None:
    html = (
        "<table><tbody>"
        "<tr><td></td><th>June 30, 2026</th><th>December 31, 2025</th></tr>"
        "<tr><th>United States</th><td>$ 42,213</td><td>$ 35,847</td></tr>"
        "</tbody></table>"
    )

    result = flatten_html_table(html)

    assert result == "United States -- June 30, 2026: $ 42,213; December 31, 2025: $ 35,847"


def test_section_label_row_with_no_values_has_no_trailing_dashes() -> None:
    html = (
        "<table><tbody>"
        "<tr><td></td><th>2026</th><th>2025</th></tr>"
        "<tr><th>Cash Flows from Operating Activities</th><td></td><td></td></tr>"
        "<tr><th>Net income</th><td>1,619</td><td>1,610</td></tr>"
        "</tbody></table>"
    )

    result = flatten_html_table(html)

    assert result == (
        "Cash Flows from Operating Activities:\n"
        "Net income -- 2026: 1,619; 2025: 1,610"
    )


def test_multiple_data_rows_produce_multiple_lines() -> None:
    html = (
        "<table><tbody>"
        "<tr><td></td><th>2026</th><th>2025</th></tr>"
        "<tr><th>Automotive</th><td>9,637</td><td>9,678</td></tr>"
        "<tr><th>Energy</th><td>1,200</td><td>1,100</td></tr>"
        "</tbody></table>"
    )

    result = flatten_html_table(html)

    assert result == "Automotive -- 2026: 9,637; 2025: 9,678\nEnergy -- 2026: 1,200; 2025: 1,100"


def test_falls_back_to_tag_stripped_text_when_no_header_row_detected() -> None:
    # Every row's first cell is non-empty -- no header row by our convention.
    html = "<table><tbody><tr><td>a</td><td>b</td></tr></tbody></table>"

    result = flatten_html_table(html)

    assert result == "a b"


def test_empty_table_returns_empty_string() -> None:
    assert flatten_html_table("<table><tbody></tbody></table>") == ""
