"""Ollama-hosted local LLM (e.g. llama3.1:8b) for answer generation.

Uses the official `ollama` Python client (already an existing [local] extra),
not raw HTTP -- verified its Client(host=...).chat(model=..., messages=...,
options=...) signature directly from ollama-python's source: `options` takes
a plain dict passed straight through to Ollama's REST API, where the max-
output-tokens knob is "num_predict" (confirmed from Ollama's own Modelfile
docs -- there is no "max_tokens" option). Response content is at
response.message.content.
"""

from __future__ import annotations

from mmss.components.base import GenerationResult
from mmss.components.generator.base import Generator
from mmss.registry import register


@register("generator", "local_llm")
class LocalLLMGenerator(Generator):
    def __init__(self, model: str = "llama3.1:8b", host: str = "http://localhost:11434") -> None:
        self._model = model
        self._host = host
        self._client = None

    @property
    def name(self) -> str:
        return f"local_llm:{self._model}"

    def _get_client(self):
        if self._client is None:
            from ollama import Client

            self._client = Client(host=self._host)
        return self._client

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> GenerationResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        client = self._get_client()
        response = client.chat(
            model=self._model,
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens},
        )

        return GenerationResult(text=response.message.content, raw_response=str(response))
