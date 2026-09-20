"""Local sentence-transformers embedder (default: BAAI/bge-small-en-v1.5).

Verified against sentence-transformers' current docs: normalize_embeddings=True
on encode() is required for cosine similarity to behave correctly (L2-normalizes
each vector), and model.get_sentence_embedding_dimension() gives the real
output dimension dynamically -- unlike the reference repo, which hardcodes
384 for bge-small regardless of what model_name is actually configured.
"""

from __future__ import annotations

from mmss.components.embedder.base import Embedder
from mmss.registry import register


@register("embedder", "local_bge")
class LocalBGEEmbedder(Embedder):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu") -> None:
        self._model_name = model_name
        self._device = device
        self._model = None

    @property
    def name(self) -> str:
        return f"local_bge:{self._model_name}"

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model

    @property
    def dimension(self) -> int:
        return self._get_model().get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        vectors = model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
