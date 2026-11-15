"""A failed batch must not block later pending entities."""

import asyncio
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from recon_graphrag.embeddings import EntityEmbedder
from recon_graphrag.graphdb.base import GraphStore


class PendingStore:
    def __init__(self):
        self.pending = [{"id": str(i), "name": f"node{i}", "description": "fact"} for i in range(4)]
        self.fetches = []
        self.upserted = []

    def get_unembedded_entities(self, graph_name, limit=500, model_fingerprint=None, *, exclude_ids=None):
        self.fetches.append(set(exclude_ids or []))
        return [item for item in self.pending if item["id"] not in (exclude_ids or [])][:limit]

    def upsert_vectors(self, ids, property_name, vectors, **kwargs):
        self.upserted.extend(ids)
        self.pending = [item for item in self.pending if item["id"] not in ids]


@pytest.mark.asyncio
async def test_failed_first_batch_does_not_starve_later_entities_and_next_pass_retries():
    store = PendingStore()
    llm = MagicMock()

    async def embed(text):
        if "node0" in text or "node1" in text:
            raise RuntimeError("failed provider")
        return [0.1]

    llm.async_embed_query = AsyncMock(side_effect=embed)
    worker = EntityEmbedder(cast(GraphStore, store), llm, graph_name="quality")
    await asyncio.wait_for(worker.embed_entities(batch_size=2), timeout=5)
    assert store.upserted == ["2", "3"]
    assert llm.async_embed_query.await_count == 4
    assert store.fetches[-1] == {"0", "1", "2", "3"}
    assert [item["id"] for item in store.pending] == ["0", "1"]
    llm.async_embed_query = AsyncMock(return_value=[0.1])
    await worker.embed_entities(batch_size=2)
    assert not store.pending
    assert llm.async_embed_query.await_count == 2


@pytest.mark.asyncio
async def test_all_failed_attempts_do_not_log_all_entities_already_embedded(caplog):
    store = PendingStore()
    llm = MagicMock(async_embed_query=AsyncMock(side_effect=RuntimeError("private")))
    with caplog.at_level("INFO"):
        await EntityEmbedder(cast(GraphStore, store), llm).embed_entities(batch_size=2)
    assert llm.async_embed_query.await_count == 4
    assert "all entities already have embeddings" not in caplog.text
    assert "4" in caplog.text
