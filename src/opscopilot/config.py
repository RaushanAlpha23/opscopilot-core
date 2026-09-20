"""Typed settings, read from the environment or passed explicitly.

Two changes from the previous SDK, both about making the package usable as a
library rather than only as an app:

1. Nothing is required at import time. The old app-level `Settings` raised a
   validation error at process start if MISTRAL_API_KEY was absent, which is
   correct for a server but wrong for a library — importing opscopilot must
   never explode. Keys are validated lazily, at the point a provider is
   actually constructed.
2. Every field can be supplied three ways: constructor argument, environment
   variable (prefixed `OPSCOPILOT_`), or `.env` file — in that precedence.
   The prefix prevents collisions with a host application's own config.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPSCOPILOT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Models ---
    # Provider specs are "<provider>:<model>", e.g. "mistral:mistral-small-latest"
    # or "openai:gpt-4o". Swapping provider is a config change, not a code change.
    llm: str = "mistral:mistral-small-latest"
    vision_llm: str | None = None  # defaults to `llm` when unset
    embeddings: str = "hash:384"  # offline default; see embeddings/registry.py

    mistral_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""

    # --- Vector store ---
    # "memory://" needs no infrastructure at all, which is what makes the
    # quickstart in the README runnable without docker-compose.
    vector_store: str = "memory://"
    collection_code: str = "opscopilot_code"
    collection_schema: str = "opscopilot_schema"

    # --- State store ---
    state_store: str = "memory://"
    state_ttl_seconds: int = 60 * 60 * 24

    # --- Ticketing ---
    github_token: str = ""
    github_repo: str = ""
    github_labels: list[str] = Field(default_factory=lambda: ["ai-triaged", "bug", "needs-review"])

    # --- Pipeline behaviour ---
    confidence_threshold: float = 0.6
    max_evidence_rounds: int = 2
    code_top_k: int = 5
    schema_top_k: int = 3
    max_chunk_chars: int = 1200
    request_timeout_seconds: int = 60
    max_retries: int = 2  # re-prompts when the model returns malformed JSON
    llm_max_attempts: int = 6  # HTTP attempts per LLM call on 429/5xx, with backoff
    min_seconds_between_calls: float = 0.0  # client-side pacing; raise on free tiers

    @property
    def effective_vision_llm(self) -> str:
        return self.vision_llm or self.llm


def load_settings(**overrides: object) -> Settings:
    """Build Settings from env/.env, with explicit keyword overrides winning."""
    clean = {k: v for k, v in overrides.items() if v is not None}
    return Settings(**clean)  # type: ignore[arg-type]
