"""Embedder protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    name: str
    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch. Always batched — the previous SDK embedded one chunk
        per API call inside the indexing loop, which made indexing a mid-sized
        repo both slow and expensive."""
        ...

    def embed_one(self, text: str) -> list[float]:
        ...
