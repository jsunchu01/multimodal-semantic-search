"""Generic API LLM wrapper -- "openai" implemented (same openai SDK and lazy-
client pattern already used in embedder/api_embedder.py); other vendors
raise NotImplementedError until picked, same vendor-guard as api_embedder.

Verified against OpenAI's current chat completions API: max_tokens is
accepted but deprecated in favor of max_completion_tokens, and max_tokens is
outright incompatible with o-series reasoning models -- using
max_completion_tokens here so this keeps working if the configured model
name is ever changed to an o-series model, not just gpt-4o-mini.
"""

from __future__ import annotations

from mmss.components.base import GenerationResult
from mmss.components.generator.base import Generator
from mmss.registry import register


@register("generator", "api_llm")
class APILLMGenerator(Generator):
    def __init__(self, vendor: str, model_name: str) -> None:
        if vendor != "openai":
            raise NotImplementedError(
                f"API generator vendor '{vendor}' is not implemented -- only "
                "'openai' is supported right now"
            )
        self._vendor = vendor
        self._model_name = model_name
        self._client = None

    @property
    def name(self) -> str:
        return f"api_llm:{self._vendor}:{self._model_name}"

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()  # reads OPENAI_API_KEY from the environment
        return self._client

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> GenerationResult:
        client = self._get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
        )
        text = response.choices[0].message.content
        return GenerationResult(text=text, raw_response=str(response))
