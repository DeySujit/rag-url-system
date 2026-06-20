"""Unit tests for dedup, vector-store factory, records and RAG plumbing.

External services (OpenAI, Postgres, vector DBs) are replaced with fakes.
"""
from __future__ import annotations

import pytest

from app.core.prompt import build_user_prompt
from app.core.vectorstore import (
    MilvusStore,
    PgVectorStore,
    PineconeStore,
    QdrantStore,
    WeaviateStore,
    get_vector_store,
)
from app.models.schemas import (
    Chunk,
    ChunkMetadata,
    RetrievedChunk,
    VectorRecord,
)
from app.services.rag_service import RAGService


# --------------------------------------------------------------------------- #
# Vector store factory
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "backend,cls",
    [
        ("pgvector", PgVectorStore),
        ("pinecone", PineconeStore),
        ("qdrant", QdrantStore),
        ("weaviate", WeaviateStore),
        ("milvus", MilvusStore),
    ],
)
def test_factory_returns_correct_backend(backend, cls):
    store = get_vector_store(backend=backend, dimension=2048)
    assert isinstance(store, cls)
    assert store.dimension == 2048


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError):
        get_vector_store(backend="does-not-exist")


# --------------------------------------------------------------------------- #
# Content hashing / dedup identity
# --------------------------------------------------------------------------- #
def test_content_hash_is_stable_and_scoped():
    h1 = Chunk.compute_hash("https://a.com", "hello")
    h2 = Chunk.compute_hash("https://a.com", "hello")
    h3 = Chunk.compute_hash("https://b.com", "hello")
    assert h1 == h2          # deterministic
    assert h1 != h3          # scoped by source_url
    # whitespace-insensitive
    assert Chunk.compute_hash("https://a.com", " hello ") == h1


def test_vector_record_shape():
    rec = VectorRecord(id="abc", embedding=[0.1, 0.2], text="t", metadata={"k": "v"})
    assert rec.id == "abc"
    assert rec.embedding == [0.1, 0.2]
    assert rec.metadata["k"] == "v"


def test_chunk_metadata_payload_is_json_safe():
    meta = ChunkMetadata(chunk_id="c1", source_url="https://a.com", token_count=10)
    payload = meta.as_payload()
    assert isinstance(payload["crawl_timestamp"], str)
    assert payload["chunk_id"] == "c1"


# --------------------------------------------------------------------------- #
# Prompt building
# --------------------------------------------------------------------------- #
def test_prompt_includes_context_and_citation_markers():
    sources = [
        RetrievedChunk(id="1", text="Paris is the capital of France.",
                       score=0.9, metadata={"source_url": "https://geo.com"}),
    ]
    prompt = build_user_prompt("What is the capital of France?", sources)
    assert "Paris is the capital" in prompt
    assert "[1]" in prompt
    assert "https://geo.com" in prompt


# --------------------------------------------------------------------------- #
# RAGService with fakes (no network)
# --------------------------------------------------------------------------- #
class _FakeEmbedder:
    async def embed_query(self, text: str):
        return [0.0] * 8


class _FakeStore:
    async def ensure_collection(self):
        return None

    async def query(self, vector, top_k=5, flt=None):
        return [
            RetrievedChunk(id="1", text="Answer lives here.", score=0.8,
                           metadata={"source_url": "https://x.com"})
        ]

    async def close(self):
        return None


class _FakeLLM:
    async def complete(self, system, user, temperature=0.1):
        return "The answer is here [1]."


@pytest.mark.asyncio
async def test_rag_service_end_to_end_with_fakes():
    svc = RAGService(embedder=_FakeEmbedder(), store=_FakeStore(), llm=_FakeLLM())
    resp = await svc.query("question?", top_k=3)
    assert resp.answer == "The answer is here [1]."
    assert len(resp.sources) == 1
    assert resp.sources[0].metadata["source_url"] == "https://x.com"


@pytest.mark.asyncio
async def test_rag_service_handles_no_results():
    class _Empty(_FakeStore):
        async def query(self, vector, top_k=5, flt=None):
            return []

    svc = RAGService(embedder=_FakeEmbedder(), store=_Empty(), llm=_FakeLLM())
    resp = await svc.query("nothing indexed?", top_k=3)
    assert resp.sources == []
    assert "No indexed content" in resp.answer
