"""Tests APIEmbedder's concurrent-batching logic -- a fake client stands in
for openai.OpenAI() (assigned directly to ._client, bypassing the real lazy
init, same pattern as the reranker's degrade-path tests), so this stays an
offline unit test with no real API key or network call needed.
"""

from mmss.components.embedder.api_embedder import APIEmbedder


class _FakeItem:
    def __init__(self, index: int, embedding: list[float]) -> None:
        self.index = index
        self.embedding = embedding


class _FakeResponse:
    def __init__(self, data: list[_FakeItem]) -> None:
        self.data = data


class _FakeEmbeddingsAPI:
    def __init__(self) -> None:
        self.batches_seen: list[list[str]] = []

    def create(self, model: str, input: list[str]) -> _FakeResponse:
        self.batches_seen.append(list(input))
        # Embedding value == the numeric suffix in "chunk {i}" -- lets tests
        # verify the final vector order matches the original text order,
        # independent of which thread/batch actually processed which text.
        data = [_FakeItem(i, [float(text.rsplit(" ", 1)[-1])]) for i, text in enumerate(input)]
        return _FakeResponse(data)


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddingsAPI()


def test_embed_documents_splits_into_100_item_batches() -> None:
    embedder = APIEmbedder(vendor="openai", model_name="text-embedding-3-small")
    fake_client = _FakeOpenAIClient()
    embedder._client = fake_client

    texts = [f"chunk {i}" for i in range(250)]
    embedder.embed_documents(texts)

    batch_lengths = sorted((len(b) for b in fake_client.embeddings.batches_seen), reverse=True)
    assert batch_lengths == [100, 100, 50]


def test_embed_documents_preserves_original_order_across_batches() -> None:
    embedder = APIEmbedder(vendor="openai", model_name="text-embedding-3-small")
    fake_client = _FakeOpenAIClient()
    embedder._client = fake_client

    texts = [f"chunk {i}" for i in range(250)]
    vectors = embedder.embed_documents(texts)

    assert [v[0] for v in vectors] == [float(i) for i in range(250)]


def test_embed_documents_single_batch() -> None:
    embedder = APIEmbedder(vendor="openai", model_name="text-embedding-3-small")
    fake_client = _FakeOpenAIClient()
    embedder._client = fake_client

    vectors = embedder.embed_documents(["chunk 0", "chunk 1"])

    assert vectors == [[0.0], [1.0]]
    assert len(fake_client.embeddings.batches_seen) == 1


def test_embed_documents_empty_input() -> None:
    embedder = APIEmbedder(vendor="openai", model_name="text-embedding-3-small")
    embedder._client = _FakeOpenAIClient()

    assert embedder.embed_documents([]) == []
