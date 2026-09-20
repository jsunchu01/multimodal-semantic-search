"""Confirms every stub provider is wired into the registry, even before M1+ fill in real logic."""

import pytest

from mmss.components import (  # noqa: F401  (imports trigger registration)
    embedder,
    generator,
    parser,
    reranker,
    vector_store,
    vision,
)
from mmss.registry import available, build


def test_parser_providers_registered() -> None:
    assert set(available("parser")) >= {"docling", "unstructured"}


def test_embedder_providers_registered() -> None:
    assert set(available("embedder")) >= {"local_bge", "api_embedder"}


def test_vision_providers_registered() -> None:
    assert set(available("vision")) >= {"local_vlm", "api_vlm"}


def test_reranker_providers_registered() -> None:
    assert set(available("reranker")) >= {"local_cross_encoder", "api_reranker", "noop"}


def test_generator_providers_registered() -> None:
    assert set(available("generator")) >= {"local_llm", "api_llm"}


def test_vector_store_providers_registered() -> None:
    assert set(available("vector_store")) >= {"pgvector", "faiss"}


def test_api_embedder_rejects_unimplemented_vendor() -> None:
    with pytest.raises(NotImplementedError):
        build("embedder", "api_embedder", vendor="cohere", model_name="embed-english-v3.0")


def test_api_llm_rejects_unimplemented_vendor() -> None:
    with pytest.raises(NotImplementedError):
        build("generator", "api_llm", vendor="anthropic", model_name="claude-3-5-sonnet")


def test_api_vlm_rejects_unimplemented_vendor() -> None:
    with pytest.raises(NotImplementedError):
        build("vision", "api_vlm", vendor="gemini", model_name="gemini-2.5-flash")


def test_api_vlm_accepts_together_vendor() -> None:
    extractor = build("vision", "api_vlm", vendor="together", model_name="Qwen/Qwen2-VL-72B-Instruct")
    assert extractor.name == "api_vlm:together:Qwen/Qwen2-VL-72B-Instruct"
