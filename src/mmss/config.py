"""Central settings: a YAML file is the base, env vars / .env overlay on top
for secrets and quick ad-hoc swaps.

Precedence, highest first: init kwargs > env vars > .env file > YAML file > field defaults.
Pick which YAML file to load via the MMSS_CONFIG_FILE env var (defaults to
config/default.yaml); flip a single slot ad hoc via e.g. MMSS_GENERATOR__PROVIDER=api_llm.
"""

from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# pydantic-settings' own env_file=".env" handling (below) only populates
# fields defined on Settings -- it does NOT export .env into os.environ.
# Vendor SDKs (e.g. `openai`) read their API keys straight from os.environ,
# so without this call a key sitting in .env would never actually reach
# them.
load_dotenv()


class ProviderConfig(BaseModel):
    provider: str
    params: dict = {}


class ParserConfig(BaseModel):
    primary: Literal["docling", "unstructured"] = "docling"
    fallback: Literal["unstructured"] = "unstructured"


class ChunkingConfig(BaseModel):
    max_characters: int = 2000


class RetrievalConfig(BaseModel):
    top_k_dense: int = 20
    top_k_bm25: int = 20
    rrf_k: int = 60
    final_top_k: int = 10
    # How much wider than final_top_k the hybrid fetch goes before handing
    # candidates to the reranker. Verified from the reference repo's actual
    # retriever/__init__.py: `candidates = fused[: top_k_final * 3]` -- we
    # apply the same 3x multiplier, but to whatever top_k the caller asks
    # for (not a fixed config constant), so `--top-k` stays meaningful at
    # every strategy.
    rerank_candidate_multiplier: int = 3


class GuardrailConfig(BaseModel):
    numeric_tolerance: float = 0.01
    on_violation: Literal["flag", "reject", "strip"] = "flag"
    min_grounding_ratio: float = 0.8


class PoTConfig(BaseModel):
    enabled: bool = True
    timeout_seconds: float = 5.0


class EvalConfig(BaseModel):
    judge_model: str = "gpt-4o-mini"
    regression_threshold: float = 0.05
    history_path: str = "evals/history.json"


class PIIConfig(BaseModel):
    enabled: bool = True
    entities: list[str] = [
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "US_SSN",
        "CREDIT_CARD",
        "IBAN_CODE",
        "US_BANK_NUMBER",
    ]
    enable_financial_patterns: bool = True


class Settings(BaseSettings):
    parser: ParserConfig = ParserConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    embedder: ProviderConfig
    vision: ProviderConfig
    generator: ProviderConfig
    # Optional escalation target for complex queries (QueryAnalyzer.is_complex)
    # -- e.g. local Ollama by default, api_llm/gpt-4o-mini for hard questions.
    # None (the default, and what default.yaml/local.yaml leave it as) means
    # complexity routing has no effect: every query uses `generator`.
    generator_complex: ProviderConfig | None = None
    reranker: ProviderConfig
    vector_store: ProviderConfig
    retrieval: RetrievalConfig = RetrievalConfig()
    guardrail: GuardrailConfig = GuardrailConfig()
    pot: PoTConfig = PoTConfig()
    pii: PIIConfig = PIIConfig()
    eval: EvalConfig = EvalConfig()

    model_config = SettingsConfigDict(
        env_prefix="MMSS_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_file = os.environ.get("MMSS_CONFIG_FILE", "config/default.yaml")
        yaml_source = YamlConfigSettingsSource(settings_cls, yaml_file=yaml_file)
        return (init_settings, env_settings, dotenv_settings, yaml_source)


_settings: Settings | None = None


def get_config() -> Settings:
    """Return the process-wide Settings singleton, loading it on first use."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_config() -> None:
    """Clear the cached Settings singleton (mainly for tests / switching config files)."""
    global _settings
    _settings = None
