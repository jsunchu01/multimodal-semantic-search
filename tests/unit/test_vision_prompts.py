"""Tests parse_structured_response's JSON parsing + graceful degradation --
no VLM or image file needed, just the parsing/fallback logic.
"""

from mmss.components.vision.prompts import parse_structured_response


def test_parses_valid_json_response() -> None:
    raw = (
        '{"description": "A bar chart of quarterly revenue.", '
        '"chart_type": "bar", '
        '"data_points": [{"label": "Q1 2026", "value": "$21.5B"}, {"label": "Q2 2026", "value": "$28.2B"}]}'
    )

    result = parse_structured_response(raw)

    assert result.chart_type == "bar"
    assert result.data_points == [
        {"label": "Q1 2026", "value": "$21.5B"},
        {"label": "Q2 2026", "value": "$28.2B"},
    ]
    assert "A bar chart of quarterly revenue." in result.description
    assert "Q1 2026: $21.5B" in result.description
    assert "Q2 2026: $28.2B" in result.description
    assert result.raw_response == raw


def test_strips_markdown_code_fences() -> None:
    raw = '```json\n{"description": "A pie chart.", "chart_type": "pie", "data_points": []}\n```'

    result = parse_structured_response(raw)

    assert result.chart_type == "pie"
    assert result.description == "A pie chart."


def test_no_data_points_leaves_description_unchanged() -> None:
    raw = '{"description": "A line chart with no readable values.", "chart_type": "line", "data_points": []}'

    result = parse_structured_response(raw)

    assert result.description == "A line chart with no readable values."
    assert result.data_points == []


def test_malformed_json_degrades_to_description_only() -> None:
    raw = "This is not JSON at all, just a plain description of the chart."

    result = parse_structured_response(raw)

    assert result.description == raw
    assert result.chart_type is None
    assert result.data_points == []
    assert result.raw_response == raw


def test_json_with_missing_keys_degrades_gracefully() -> None:
    raw = '{"description": "Only a description, no other keys."}'

    result = parse_structured_response(raw)

    assert result.description == "Only a description, no other keys."
    assert result.chart_type is None
    assert result.data_points == []
