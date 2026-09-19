"""In-process vector store.

Why this ships in the package: it removes infrastructure from the critical
path. A first-time user can index a repo and run a diagnosis without docker,
and the test suite exercises the real retrieval code path rather than a mock.
Persistence is optional — `save()`/`load()` write plain JSON, which is enough
for a CLI session or a small single-process deployment.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..exceptions import RetrievalError
from ..models import CodeChunk, RetrievedChunk


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise RetrievalError(
            f"Vector dimension mismatch: {len(a)} vs {len(b)}. This usually means the "
            "collection was indexed with a different embedding model than the one "
            "currently configured. Re-index, or point at a fresh collection."
        )
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class InMemoryVectorStore:
    name = "memory"

    def __init__(self, path: str | Path | None = None) -> None:
        self._collections: dict[str, dict[str, tuple[CodeChunk, list[float]]]] = {}
        self._dimensions: dict[str, int] = {}
        self._path = Path(path) if path else None
        if self._path and self._path.exists():
            self.load(self._path)

    def ensure_collection(self, name: str, dimension: int) -> None:
        self._collections.setdefault(name, {})
        self._dimensions.setdefault(name, dimension)

    def upsert(self, name: str, chunks: list[CodeChunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            raise RetrievalError("chunks and vectors must be the same length")
        if not chunks:
            return 0
        self.ensure_collection(name, len(vectors[0]))
        collection = self._collections[name]
        for chunk, vector in zip(chunks, vectors, strict=True):
            collection[chunk.id] = (chunk, vector)  # id-keyed: re-indexing updates in place
        if self._path:
            self.save(self._path)
        return len(chunks)

    def search(
        self,
        name: str,
        vector: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        collection = self._collections.get(name, {})
        hits: list[RetrievedChunk] = []
        for chunk, stored in collection.values():
            if where and any(getattr(chunk, k, None) != v for k, v in where.items()):
                continue
            hits.append(RetrievedChunk(chunk=chunk, score=cosine(vector, stored)))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def delete_collection(self, name: str) -> None:
        self._collections.pop(name, None)
        self._dimensions.pop(name, None)
        if self._path:
            self.save(self._path)

    def count(self, name: str) -> int:
        return len(self._collections.get(name, {}))

    # --- optional persistence ---

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dimensions": self._dimensions,
            "collections": {
                name: [
                    {"chunk": chunk.to_dict(), "vector": vector}
                    for chunk, vector in entries.values()
                ]
                for name, entries in self._collections.items()
            },
        }
        target.write_text(json.dumps(payload), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self._dimensions = payload.get("dimensions", {})
        self._collections = {
            name: {
                entry["chunk"]["id"]: (CodeChunk.from_dict(entry["chunk"]), entry["vector"])
                for entry in entries
            }
            for name, entries in payload.get("collections", {}).items()
        }
