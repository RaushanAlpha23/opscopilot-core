"""A dependency-free, deterministic embedder.

This exists so that `pip install opscopilot-core` (no extras) yields a package
you can actually run: index a repo, retrieve chunks, exercise the pipeline in
tests and CI, all offline.

It is a hashed bag-of-tokens projection — real lexical signal, no semantic
signal. It will match `get_total_price` to a query mentioning
`get_total_price`, and it will not match "price is wrong" to
`calculate_subtotal`. Good enough for tests and keyword-ish lookups; use
`sentence-transformers` or an API embedder for production retrieval.
"""

from __future__ import annotations

import hashlib
import math
import re

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


_CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|[0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase tokens, plus sub-tokens split on snake_case and camelCase.

    Splitting happens on the ORIGINAL casing and lowercasing only afterwards —
    lowercasing first would erase the capital letters the camelCase split
    depends on, so `getTotalPrice` would never match a query saying
    "total price". That is most of the lexical recall on identifier-heavy code.
    """
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        tokens.append(raw.lower())
        parts = [p for p in raw.split("_") if p]
        if len(parts) > 1:
            tokens.extend(p.lower() for p in parts)
        for part in parts:
            camel = _CAMEL_RE.findall(part)
            if len(camel) > 1:
                tokens.extend(c.lower() for c in camel)
    return tokens


class HashingEmbedder:
    name = "hash"

    def __init__(self, dimension: int = 384) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self.dimension = dimension

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dimension

    def embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = _tokenize(text)
        if not tokens:
            return vector
        for token in tokens:
            vector[self._bucket(token)] += 1.0
        # Sublinear scaling then L2 normalisation, so cosine similarity is
        # not dominated by chunk length.
        vector = [math.log1p(v) for v in vector]
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]
