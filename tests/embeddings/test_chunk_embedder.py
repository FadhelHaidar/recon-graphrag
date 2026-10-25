"""Tests for chunk embedding — the vectors that enable relevance-ranked sources."""

import asyncio
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from recon_graphrag.embeddings import ChunkEmbedder
from recon_graphrag.embeddings.chunks import _normalize
from recon_graphrag.graphdb.base import GraphStore


class FakeChunkStore:
    def __init__(self, chunks=None):
        self._chunks = list(chunks or [])
        self.graph_names = []
        self.upserts = []

    def get_unembedded_chunks(self, graph_name, limit=500, model_fingerprint=None):
        self.graph_names.append(graph_name)
        batch, self._chunks = self._chunks[:limit], self._chunks[limit:]
        return batch

    def upsert_vectors(self, ids, property_name, vectors, **kwargs):
        self.upserts.append((ids, property_name, vectors, kwargs))


class StuckChunkStore:
    """Never clears unembedded chunks — models persistent embed failure."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.fetches = 0
        self.upserts = []

    def get_unembedded_chunks(self, graph_name, limit=500, model_fingerprint=None):
        self.fetches += 1
        return list(self._chunks[:limit])

    def upsert_vectors(self, ids, property_name, vectors, **kwargs):
        self.upserts.append((ids, property_name, vectors, kwargs))


@pytest.fixture
def fake_embedder():
    embedder = MagicMock()
    embedder.async_embed_query = AsyncMock(return_value=[3.0, 4.0])
    return embedder


def test_normalize_scales_to_unit_length():
    assert _normalize([3.0, 4.0]) == [0.6, 0.8]


def test_normalize_leaves_zero_vector_alone():
    # A dot product with a zero vector is 0.0 (lowest rank) rather than a crash.
    assert _normalize([0.0, 0.0]) == [0.0, 0.0]


@pytest.mark.asyncio
async def test_embed_chunks_stores_normalized_vectors(fake_embedder):
    store = FakeChunkStore(chunks=[{"id": "c1", "chunk_id": "doc:0:0", "text": "alpha"}])

    await ChunkEmbedder(cast(GraphStore, store), fake_embedder).embed_chunks(batch_size=500)

    assert len(store.upserts) == 1
    ids, property_name, vectors, kwargs = store.upserts[0]
    assert ids == ["c1"]
    assert property_name == "embedding"
    assert vectors == [[0.6, 0.8]]
    # Chunk text is immutable, so no content fingerprint is needed to detect drift.
    assert "content_fingerprints" not in kwargs
    assert kwargs["model_fingerprint"] is not None
    assert set(store.graph_names) == {"entity-graph"}


@pytest.mark.asyncio
async def test_embed_chunks_keeps_successes_when_one_chunk_fails(fake_embedder):
    async def embed(text):
        # Fail by content, not by call order: chunks are embedded concurrently,
        # so a side_effect list would make this test flaky.
        if text == "alpha":
            raise RuntimeError("embedder down")
        return [0.0, 5.0]

    fake_embedder.async_embed_query = AsyncMock(side_effect=embed)
    store = FakeChunkStore(
        chunks=[
            {"id": "c1", "chunk_id": "doc:0:0", "text": "alpha"},
            {"id": "c2", "chunk_id": "doc:0:1", "text": "beta"},
        ]
    )

    await ChunkEmbedder(cast(GraphStore, store), fake_embedder).embed_chunks(batch_size=500)

    ids, _, vectors, _ = store.upserts[0]
    assert ids == ["c2"]
    assert vectors == [[0.0, 1.0]]


@pytest.mark.asyncio
async def test_embed_chunks_drains_in_batches(fake_embedder):
    store = FakeChunkStore(
        chunks=[{"id": f"c{i}", "chunk_id": f"doc:0:{i}", "text": f"t{i}"} for i in range(3)]
    )

    await ChunkEmbedder(cast(GraphStore, store), fake_embedder).embed_chunks(batch_size=2)

    assert [upsert[0] for upsert in store.upserts] == [["c0", "c1"], ["c2"]]


@pytest.mark.asyncio
async def test_embed_chunks_terminates_when_batch_never_clears(fake_embedder):
    store = StuckChunkStore(chunks=[{"id": "c1", "chunk_id": "doc:0:0", "text": "alpha"}])
    fake_embedder.async_embed_query = AsyncMock(side_effect=RuntimeError("embedder down"))

    # Without the no-progress guard this loops forever; bound it so a regression
    # fails loudly instead of hanging.
    await asyncio.wait_for(
        ChunkEmbedder(cast(GraphStore, store), fake_embedder).embed_chunks(batch_size=500),
        timeout=5,
    )

    assert store.upserts == []
    assert store.fetches == 2  # first attempt, then the repeat that trips the guard
