from mmss.config import get_config, reset_config


def test_default_config_loads() -> None:
    reset_config()
    cfg = get_config()
    assert cfg.parser.primary == "docling"
    assert cfg.parser.fallback == "unstructured"
    assert cfg.embedder.provider == "local_bge"
    assert cfg.vector_store.provider == "pgvector"
    assert cfg.guardrail.on_violation == "flag"
    assert cfg.chunking.max_characters == 2000
