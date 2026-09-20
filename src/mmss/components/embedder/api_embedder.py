"""Generic API embedder wrapper. Currently implements vendor="openai" only --
Voyage/Cohere are intentionally not implemented (raise a clear error) since
they're not being used for this project. Requires OPENAI_API_KEY in the
environment (loaded from .env via config.py's load_dotenv() call) -- the
openai SDK reads it automatically, we never touch the key value directly.

Verified against OpenAI's current embeddings API: client.embeddings.create(
model=..., input=texts) returns response.data, a list of objects each with
an .embedding (list of floats) and .index -- sorting by .index before
extracting vectors guards against the API returning results out of input
order, since the index field exists specifically for that reordering case.

M7: batches of 100 texts now run concurrently (bounded thread pool) instead
of sequentially -- a real wall-clock win, since each batch request is pure
network wait. No retry/backoff logic needed on top of this: the openai SDK
already retries 429/5xx/408/409/connection errors automatically (2 retries,
exponential backoff by default, verified from its README) -- that's a
benefit of using the official SDK instead of hand-rolled HTTP.
"""

from __future__ import annotations

from mmss.components.embedder.base import Embedder
from mmss.registry import register
from mmss.utils.concurrency import run_concurrently

_OPENAI_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

_BATCH_SIZE = 100
_MAX_CONCURRENT_REQUESTS = 5


@register("embedder", "api_embedder")
class APIEmbedder(Embedder):
    def __init__(self, vendor: str, model_name: str) -> None:
        if vendor != "openai":
            raise NotImplementedError(
                f"API embedder vendor '{vendor}' is not implemented -- only "
                "'openai' is supported right now"
            )
        self._vendor = vendor
        self._model_name = model_name
        self._client = None

    @property
    def name(self) -> str:
        return f"api_embedder:{self._vendor}:{self._model_name}"

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()  # reads OPENAI_API_KEY from the environment
        return self._client

    @property
    def dimension(self) -> int:
        if self._model_name in _OPENAI_DIMENSIONS:
            return _OPENAI_DIMENSIONS[self._model_name]
        # Unknown model name -- ask the API directly rather than guessing.
        return len(self.embed_documents(["dimension probe"])[0])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        batches = [texts[start : start + _BATCH_SIZE] for start in range(0, len(texts), _BATCH_SIZE)]

        def _embed_batch(batch: list[str]) -> list[list[float]]:
            response = client.embeddings.create(model=self._model_name, input=batch)
            ordered = sorted(response.data, key=lambda d: d.index)
            return [d.embedding for d in ordered]

        batch_results = run_concurrently(_embed_batch, batches, max_workers=_MAX_CONCURRENT_REQUESTS)

        vectors: list[list[float]] = []
        for batch_result in batch_results:
            vectors.extend(batch_result)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
