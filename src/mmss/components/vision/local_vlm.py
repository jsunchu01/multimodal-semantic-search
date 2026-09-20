"""Ollama-hosted local VLM (default: qwen2.5vl:3b) for chart/figure extraction.

Model name verified against Ollama's actual model library -- "qwen2-vl" (the
name in our earlier config draft) isn't a real tag; the real model is
"qwen2.5vl", available as qwen2.5vl:3b/7b/32b/72b.

Image input verified against Ollama's own docs: a message carries an
"images" list of base64-encoded strings -- NOT an OpenAI-style content array
with an image_url object. Uses the same official `ollama` client as
local_llm.py, not raw HTTP.
"""

from __future__ import annotations

import base64

from mmss.components.base import VisionExtractionResult
from mmss.components.vision.base import VisionExtractor
from mmss.components.vision.prompts import (
    DESCRIPTION_ONLY_PROMPT,
    STRUCTURED_JSON_PROMPT,
    parse_structured_response,
)
from mmss.registry import register


@register("vision", "local_vlm")
class LocalVLMExtractor(VisionExtractor):
    def __init__(
        self,
        model: str = "qwen2.5vl:3b",
        host: str = "http://localhost:11434",
        structured: bool = False,
    ) -> None:
        self._model = model
        self._host = host
        self._structured = structured
        self._client = None

    @property
    def name(self) -> str:
        return f"local_vlm:{self._model}"

    def _get_client(self):
        if self._client is None:
            from ollama import Client

            self._client = Client(host=self._host)
        return self._client

    def extract(
        self, image_path: str, context_text: str | None = None
    ) -> VisionExtractionResult:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        prompt = STRUCTURED_JSON_PROMPT if self._structured else DESCRIPTION_ONLY_PROMPT
        if context_text:
            prompt = f"{prompt}\n\nSurrounding document text for context:\n{context_text}"

        client = self._get_client()
        response = client.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt, "images": [image_b64]}],
        )
        raw_text = response.message.content

        if self._structured:
            return parse_structured_response(raw_text)
        return VisionExtractionResult(description=raw_text, raw_response=raw_text)
