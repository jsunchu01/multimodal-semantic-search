"""API vision extractor -- "openai" (GPT-4o vision) and "together" (Qwen2-VL
hosted on Together.ai, cheaper than OpenAI and able to run bigger open models
than can be run locally) both implemented; other vendors raise
NotImplementedError, same vendor-guard pattern as api_embedder.py/api_llm.py.

OpenAI: image passed as a content-array item {"type": "image_url",
"image_url": {"url": "data:image/png;base64,..."}}, same shape used in
api_llm.py's verified chat completions usage.

Together.ai: verified its request shape is OpenAI-compatible and it's meant
to be used via the official `openai` package pointed at a custom base_url --
no separate `together` SDK needed (confirmed from Together's own docs).
Verified the current base URL is https://api.together.ai/v1 -- the reference
repo hardcodes https://api.together.xyz/v1, which is stale. NOT verified:
whether Together's vision endpoint accepts a base64 data URI for the image
the same way OpenAI does -- docs only showed publicly-hosted HTTP URL
examples. This is a best-effort assumption (Together advertises OpenAI-
compatible request shapes generally) until a live call confirms it either
way; if it turns out data URIs aren't accepted, this needs revisiting.
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

_TOGETHER_BASE_URL = "https://api.together.ai/v1"


@register("vision", "api_vlm")
class APIVisionExtractor(VisionExtractor):
    def __init__(self, vendor: str, model_name: str, structured: bool = False) -> None:
        if vendor not in ("openai", "together"):
            raise NotImplementedError(
                f"API vision vendor '{vendor}' is not implemented -- only "
                "'openai' and 'together' are supported right now"
            )
        self._vendor = vendor
        self._model_name = model_name
        self._structured = structured
        self._client = None

    @property
    def name(self) -> str:
        return f"api_vlm:{self._vendor}:{self._model_name}"

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            if self._vendor == "openai":
                self._client = OpenAI()  # reads OPENAI_API_KEY from the environment
            else:
                import os

                self._client = OpenAI(
                    base_url=_TOGETHER_BASE_URL,
                    api_key=os.environ.get("TOGETHER_API_KEY"),
                )
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
        response = client.chat.completions.create(
            model=self._model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            max_completion_tokens=1500,
        )
        raw_text = response.choices[0].message.content

        if self._structured:
            return parse_structured_response(raw_text)
        return VisionExtractionResult(description=raw_text, raw_response=raw_text)
