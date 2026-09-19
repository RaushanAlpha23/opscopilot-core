"""Local embeddings via sentence-transformers (no API calls, no per-token cost)."""

from __future__ import annotations

from ..exceptions import MissingDependencyError

_KNOWN_DIMENSIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
}


class SentenceTransformerEmbedder:
    name = "st"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise MissingDependencyError("sentence-transformers", "local-embeddings") from exc

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        # Ask the model rather than trusting a hardcoded constant — the old code
        # hardcoded 384, which silently breaks the Qdrant collection the moment
        # anyone swaps in a 768-dim model.
        self.dimension = int(
            self._model.get_sentence_embedding_dimension()
            or _KNOWN_DIMENSIONS.get(model_name, 384)
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
