"""Turn a repository into an embedded, searchable index."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..embeddings.base import Embedder
from ..models import CodeChunk, RetrievedChunk
from .base import VectorStore
from .chunking import chunk_repository, chunk_source

EMBED_BATCH = 64


class Indexer:
    """Owns the embed-and-store half of retrieval.

    Kept separate from the engine so it can be used on its own — indexing is
    a batch job that usually runs on a different schedule (CI, a cron) than
    incident handling.
    """

    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def index_chunks(
        self,
        collection: str,
        chunks: list[CodeChunk],
        progress: Callable[[int, int], None] | None = None,
    ) -> int:
        if not chunks:
            return 0
        self.store.ensure_collection(collection, self.embedder.dimension)

        total = 0
        for i in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[i : i + EMBED_BATCH]
            vectors = self.embedder.embed([c.text for c in batch])
            total += self.store.upsert(collection, batch, vectors)
            if progress:
                progress(min(i + EMBED_BATCH, len(chunks)), len(chunks))
        return total

    def index_repository(
        self,
        root: str | Path,
        collection: str,
        extensions: set[str] | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> int:
        return self.index_chunks(collection, chunk_repository(root, extensions), progress)

    def index_text(self, collection: str, source: str, file_path: str) -> int:
        """Index a single in-memory source string — useful for schema DDL."""
        return self.index_chunks(collection, chunk_source(source, file_path))

    def search(self, collection: str, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        return self.store.search(collection, self.embedder.embed_one(query), top_k=top_k)
