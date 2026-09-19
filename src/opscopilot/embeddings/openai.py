"""OpenAI embedding models."""

from __future__ import annotations

from ..exceptions import ConfigurationError, MissingDependencyError

_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
_BATCH = 128


class OpenAIEmbedder:
    name = "openai"

    def __init__(self, model: str = "text-embedding-3-small", api_key: str = "") -> None:
        if not api_key:
            raise ConfigurationError(
                "An OpenAI API key is required for OpenAI embeddings. "
                "Set OPSCOPILOT_OPENAI_API_KEY."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise MissingDependencyError("openai", "openai") from exc

        self.model = model
        self.dimension = _DIMENSIONS.get(model, 1536)
        self._client = OpenAI(api_key=api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        # Chunked: the embeddings endpoint caps inputs per request, and sending
        # an entire repo's chunks in one call would fail on a large index.
        for i in range(0, len(texts), _BATCH):
            batch = texts[i : i + _BATCH]
            response = self._client.embeddings.create(input=batch, model=self.model)
            out.extend(item.embedding for item in response.data)
        return out

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
