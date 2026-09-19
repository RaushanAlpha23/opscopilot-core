from __future__ import annotations

import pytest

from opscopilot.embeddings import HashingEmbedder
from opscopilot.exceptions import RetrievalError
from opscopilot.models import CodeChunk
from opscopilot.retrieval import Indexer, InMemoryVectorStore


@pytest.fixture
def indexer():
    return Indexer(InMemoryVectorStore(), HashingEmbedder(256))


def test_relevant_chunk_outranks_irrelevant_one(indexer, repo):
    indexer.index_repository(repo, "code")
    hits = indexer.search("code", "total price is wrong", top_k=2)
    assert hits[0].chunk.file_path == "cart.py"
    assert hits[0].score > hits[1].score


def test_reindexing_updates_in_place(indexer, repo):
    indexer.index_repository(repo, "code")
    first = indexer.store.count("code")
    indexer.index_repository(repo, "code")
    assert indexer.store.count("code") == first


def test_empty_query_returns_nothing_without_calling_the_store(indexer):
    assert indexer.search("code", "   ") == []


def test_dimension_mismatch_raises_a_clear_error():
    store = InMemoryVectorStore()
    store.upsert("c", [CodeChunk("1", "x", "a.py")], [[0.1] * 8])
    with pytest.raises(RetrievalError, match="dimension mismatch"):
        store.search("c", [0.1] * 16)


def test_memory_store_persists_to_disk(tmp_path, repo):
    path = tmp_path / "index.json"
    first = Indexer(InMemoryVectorStore(path), HashingEmbedder(256))
    first.index_repository(repo, "code")

    reopened = Indexer(InMemoryVectorStore(path), HashingEmbedder(256))
    assert reopened.store.count("code") == first.store.count("code")
    assert reopened.search("code", "total price", top_k=1)[0].chunk.file_path == "cart.py"


def test_camel_and_snake_case_identifiers_are_split_for_matching():
    embedder = HashingEmbedder(256)
    from opscopilot.retrieval.memory import cosine

    assert cosine(embedder.embed_one("getTotalPrice"), embedder.embed_one("total price")) > 0
