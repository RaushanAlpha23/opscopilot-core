"""Resolve an embedding spec string into an Embedder."""

from __future__ import annotations

from ..config import Settings
from ..exceptions import ConfigurationError
from .base import Embedder

_KNOWN = ("hash", "st", "openai")


def build_embedder(spec: str, settings: Settings) -> Embedder:
    """Specs: 'hash:384', 'st:sentence-transformers/all-MiniLM-L6-v2',
    'openai:text-embedding-3-small'."""
    if ":" not in spec:
        raise ConfigurationError(
            f"Invalid embeddings spec '{spec}'. Expected '<provider>:<model>', "
            f"e.g. 'st:sentence-transformers/all-MiniLM-L6-v2'. Known: {', '.join(_KNOWN)}."
        )
    provider, _, model = spec.partition(":")
    provider = provider.strip().lower()

    if provider == "hash":
        from .hashing import HashingEmbedder

        try:
            dimension = int(model) if model else 384
        except ValueError:
            raise ConfigurationError(
                f"The 'hash' embedder expects a dimension, got '{model}'. Use e.g. 'hash:384'."
            ) from None
        return HashingEmbedder(dimension)

    if provider == "st":
        from .sentence_transformers import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(model or "sentence-transformers/all-MiniLM-L6-v2")

    if provider == "openai":
        from .openai import OpenAIEmbedder

        return OpenAIEmbedder(model or "text-embedding-3-small", api_key=settings.openai_api_key)

    raise ConfigurationError(
        f"Unknown embeddings provider '{provider}'. Known: {', '.join(_KNOWN)}."
    )
