"""Vector store protocol."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..models import CodeChunk, RetrievedChunk


@runtime_checkable
class VectorStore(Protocol):
    def ensure_collection(self, name: str, dimension: int) -> None:
        """Create the collection if absent. Must be idempotent."""
        ...

    def upsert(self, name: str, chunks: list[CodeChunk], vectors: list[list[float]]) -> int:
        ...

    def search(
        self,
        name: str,
        vector: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        ...

    def delete_collection(self, name: str) -> None:
        ...

    def count(self, name: str) -> int:
        ...
