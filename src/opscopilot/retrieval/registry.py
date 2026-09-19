"""Resolve a vector-store URL into a VectorStore."""

from __future__ import annotations

from ..config import Settings
from ..exceptions import ConfigurationError
from .base import VectorStore


def build_vector_store(url: str, settings: Settings) -> VectorStore:
    """Supported: 'memory://', 'memory:///path/to/index.json', 'http(s)://host:6333'."""
    if url.startswith("memory://"):
        from .memory import InMemoryVectorStore

        path = url[len("memory://") :] or None
        return InMemoryVectorStore(path)

    if url.startswith(("http://", "https://")):
        from .qdrant import QdrantVectorStore

        return QdrantVectorStore(url)

    raise ConfigurationError(
        f"Unrecognised vector store URL '{url}'. Use 'memory://' or a Qdrant "
        "HTTP URL such as 'http://localhost:6333'."
    )
