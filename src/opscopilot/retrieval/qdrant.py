"""Qdrant-backed vector store."""

from __future__ import annotations

from typing import Any

from ..exceptions import MissingDependencyError, RetrievalError
from ..models import CodeChunk, RetrievedChunk

UPSERT_BATCH = 256


class QdrantVectorStore:
    name = "qdrant"

    def __init__(self, url: str = "http://localhost:6333", api_key: str | None = None) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover
            raise MissingDependencyError("qdrant-client", "qdrant") from exc
        self._client = QdrantClient(url=url, api_key=api_key)

    def ensure_collection(self, name: str, dimension: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        existing = {c.name for c in self._client.get_collections().collections}
        if name in existing:
            return
        self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )

    def upsert(self, name: str, chunks: list[CodeChunk], vectors: list[list[float]]) -> int:
        from qdrant_client.models import PointStruct

        if len(chunks) != len(vectors):
            raise RetrievalError("chunks and vectors must be the same length")
        if not chunks:
            return 0

        self.ensure_collection(name, len(vectors[0]))
        points = [
            PointStruct(id=chunk.id, vector=vector, payload=chunk.to_dict())
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        # Batched: the old SDK accumulated every point for an entire repo in one
        # list and sent a single upsert, which fails on large repos.
        for i in range(0, len(points), UPSERT_BATCH):
            self._client.upsert(collection_name=name, points=points[i : i + UPSERT_BATCH])
        return len(points)

    def search(
        self,
        name: str,
        vector: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        query_filter = None
        if where:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            query_filter = Filter(
                must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in where.items()]
            )

        # `query_points`, not the deprecated `search()` the old SDK called —
        # `search()` emits a DeprecationWarning on qdrant-client >= 1.9 and is
        # slated for removal.
        response = self._client.query_points(
            collection_name=name, query=vector, limit=top_k, query_filter=query_filter
        )
        return [
            RetrievedChunk(chunk=CodeChunk.from_dict(point.payload or {}), score=point.score)
            for point in response.points
        ]

    def delete_collection(self, name: str) -> None:
        self._client.delete_collection(collection_name=name)

    def count(self, name: str) -> int:
        return int(self._client.count(collection_name=name).count)
