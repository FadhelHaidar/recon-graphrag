"""Tests for CommunityEmbedder."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from recon_graphrag.communities.embeddings import CommunityEmbedder


class FakeGraphStore:
    def __init__(self, communities=None, entities=None):
        self.queries = []
        self._communities = communities or []
        self._entities = entities or []
        self.upserted = []

    def execute_query(self, query, parameters=None):
        self.queries.append(query.strip())
        return []

    def get_unembedded_communities(self, graph_name, level):
        return self._communities

    def get_unembedded_entities(self, limit=500):
        batch = self._entities[:limit]
        self._entities = self._entities[limit:]
        return batch

    def upsert_vectors(self, ids, property_name, vectors):
        self.upserted.append((ids, property_name, vectors))


@pytest.fixture
def fake_embedder():
    embedder = MagicMock()
    embedder.async_embed_query = AsyncMock(return_value=[0.1] * 128)
    return embedder


@pytest.mark.asyncio
async def test_embed_entities_accepts_list_description(fake_embedder):
    store = FakeGraphStore(entities=[
        {
            "id": "e1",
            "labels": ["Movie"],
            "name": "Inception",
            "description": ["dream heist", "memory architecture"],
        },
    ])
    embedder = CommunityEmbedder(store, fake_embedder)

    await embedder.embed_entities(batch_size=500)

    fake_embedder.async_embed_query.assert_awaited_once_with(
        "Movie: Inception - dream heist, memory architecture"
    )
    assert len(store.upserted) == 1
