"""Shared prompts + structured-JSON parsing for chart/figure extraction, used
by both the local (Ollama) and API (OpenAI/Together) vision extractors.

Two modes, per the user's decision: description-only is the default (safer
-- a small local model is more likely to produce malformed JSON than GPT-4o),
structured JSON extraction (chart_type + data_points) is opt-in via
`structured=True` on either extractor. A JSON parse failure always degrades
to description-only rather than raising -- same resilience pattern as
FallbackParser (components/parser/__init__.py).

When data_points are present, they're also rendered into `description` as
plain "label: value" lines -- the same lesson M4 already proved for tables:
numbers have to be in the actual chunk text for BM25/dense retrieval and the
numeric-grounding guardrail to see them, not just tucked away in metadata.
"""

from __future__ import annotations

import json

from mmss.components.base import VisionExtractionResult
from mmss.utils.logging import get_logger

logger = get_logger(__name__)

DESCRIPTION_ONLY_PROMPT = (
    "You are analyzing a chart or figure from a document (e.g. a financial filing, research "
    "paper, or report). Describe it precisely and exhaustively: chart type, title and axis "
    "labels (with units), every data value visible, legend entries, key trends/peaks/troughs, "
    "and any footnotes or source attributions. Every number matters for accuracy -- do not "
    "summarize away specific values. Respond in plain prose only, no JSON or markdown."
)

STRUCTURED_JSON_PROMPT = (
    "You are analyzing a chart or figure from a document (e.g. a financial filing, research "
    "paper, or report). Extract it as a single JSON object with EXACTLY these keys and nothing "
    "else:\n\n"
    "{\n"
    '  "description": "a precise prose description: chart type, title, axis labels, trends, legend, footnotes",\n'
    '  "chart_type": "one of: bar, line, pie, area, scatter, table, other",\n'
    '  "data_points": [{"label": "<x-axis category or series name>", "value": "<exact value as shown, including units>"}]\n'
    "}\n\n"
    "Include EVERY data value visible in the chart in data_points -- every number matters for "
    "accuracy. Respond with ONLY the JSON object: no markdown code fences, no extra text."
)


def parse_structured_response(raw_text: str) -> VisionExtractionResult:
    """Best-effort JSON parse of a structured chart response; degrades to
    description-only (the raw text verbatim) on any failure.
    """
    try:
        text = raw_text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()

        data = json.loads(text)
        description = (data.get("description") or "").strip()
        chart_type = data.get("chart_type")
        data_points = data.get("data_points") or []

        lines = [
            f"{p.get('label', '?')}: {p.get('value', '?')}"
            for p in data_points
            if isinstance(p, dict)
        ]
        if lines:
            description = f"{description}\n\nData points:\n" + "\n".join(lines)

        return VisionExtractionResult(
            description=description or raw_text,
            chart_type=chart_type,
            data_points=data_points,
            raw_response=raw_text,
        )
    except (json.JSONDecodeError, AttributeError, TypeError):
        logger.warning("Structured chart JSON parse failed -- falling back to description-only")
        return VisionExtractionResult(description=raw_text, raw_response=raw_text)
