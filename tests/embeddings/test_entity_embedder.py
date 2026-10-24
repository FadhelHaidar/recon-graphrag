"""Tests for entity embedding."""

from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from recon_graphrag.embeddings import EntityEmbedder
from recon_graphrag.graphdb.base import GraphStore


class FakeGraphStore:
    def __init__(self, communities=None, entities=None):
        self.queries = []
        self._communities = communities or []
        self._entities = entities or []
        self.upserted = []
        self.graph_names = []

    def execute_query(self, query, parameters=None):
        self.queries.append(query.strip())
        return []

    def get_unembedded_entities(self, graph_name, limit=500, model_fingerprint=None):
        self.graph_names.append(graph_name)
        batch = self._entities[:limit]
        self._entities = self._entities[limit:]
        return batch

    def upsert_vectors(self, ids, property_name, vectors, **kwargs):
        self.upserted.append((ids, property_name, vectors))


@pytest.fixture
def fake_embedder():
    embedder = MagicMock()
    embedder.async_embed_query = AsyncMock(return_value=[0.1] * 128)
    return embedder


class StuckGraphStore:
    """Never clears unembedded entities — models persistent embed failure."""

    def __init__(self, entities):
        self._entities = entities
        self.fetches = 0
        self.upserted = []

    def get_unembedded_entities(self, graph_name, limit=500, model_fingerprint=None):
        self.fetches += 1
        return list(self._entities[:limit])

    def upsert_vectors(self, ids, property_name, vectors, **kwargs):
        self.upserted.append((ids, property_name, vectors))


@pytest.mark.asyncio
async def test_embed_entities_terminates_when_batch_never_clears():
    import asyncio

    store = StuckGraphStore(entities=[{"id": "e1", "name": "Broken", "labels": ["Movie"]}])
    embedder = MagicMock()
    embedder.async_embed_query = AsyncMock(side_effect=RuntimeError("embedder down"))

    # Without the no-progress guard this loops forever; bound it so a regression
    # fails loudly instead of hanging.
    await asyncio.wait_for(
        EntityEmbedder(cast(GraphStore, store), embedder).embed_entities(batch_size=500),
        timeout=5,
    )

    assert store.upserted == []
    assert store.fetches == 2  # first attempt, then the repeat that trips the guard


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
    embedder = EntityEmbedder(cast(GraphStore, store), fake_embedder)

    await embedder.embed_entities(batch_size=500)

    fake_embedder.async_embed_query.assert_awaited_once_with(
        "Movie: Inception - dream heist, memory architecture"
    )
    assert len(store.upserted) == 1


@pytest.mark.asyncio
async def test_embedders_for_different_graphs_do_not_leak(fake_embedder):
    store_a = FakeGraphStore(entities=[
        {"id": "a1", "labels": ["Movie"], "name": "Alpha", "description": "alpha plot"},
    ])
    store_b = FakeGraphStore(entities=[
        {"id": "b1", "labels": ["Movie"], "name": "Beta", "description": "beta plot"},
    ])

    await EntityEmbedder(cast(GraphStore, store_a), fake_embedder, graph_name="graph-a").embed_entities(batch_size=500)
    await EntityEmbedder(cast(GraphStore, store_b), fake_embedder, graph_name="graph-b").embed_entities(batch_size=500)

    # Each embedder fetched only under its own graph (loop re-checks once for stray
    # rows even after a full drain) and embedded only its rows.
    assert set(store_a.graph_names) == {"graph-a"}
    assert set(store_b.graph_names) == {"graph-b"}
    assert store_a.upserted[0][0] == ["a1"]
    assert store_b.upserted[0][0] == ["b1"]
    texts = [call.args[0] for call in fake_embedder.async_embed_query.await_args_list]
    assert texts == ["Movie: Alpha - alpha plot", "Movie: Beta - beta plot"]


class StalenessFakeGraphStore:
    """Models the store's staleness predicate: missing, content, or model mismatch."""

    def __init__(self, entities):
        self.entities = entities
        self.upserted = []

    def get_unembedded_entities(self, graph_name, limit=500, model_fingerprint=None):
        stale = [
            dict(e)
            for e in self.entities
            if e.get("embedding") is None
            or e.get("embedding_content_fingerprint", "")
            != e.get("embedding_input_fingerprint", "")
            or (
                model_fingerprint is not None
                and e.get("embedding_model_fingerprint", "") != model_fingerprint
            )
        ]
        return stale[:limit]

    def upsert_vectors(self, ids, property_name, vectors, **kwargs):
        fingerprints = kwargs.get("content_fingerprints") or [None] * len(ids)
        model_fingerprint = kwargs.get("model_fingerprint")
        for entity_id, fingerprint in zip(ids, fingerprints, strict=True):
            for entity in self.entities:
                if entity["id"] != entity_id:
                    continue
                entity["embedding"] = [0.1] * 128
                entity["embedding_input_fingerprint"] = fingerprint
                entity["embedding_content_fingerprint"] = fingerprint
                if model_fingerprint is not None:
                    entity["embedding_model_fingerprint"] = model_fingerprint
        self.upserted.append((list(ids), list(fingerprints), model_fingerprint))


def _stored_entity(entity_id, name, description, **extra):
    from recon_graphrag.entity_text import entity_embedding_fingerprint

    fingerprint = entity_embedding_fingerprint(
        labels=["Movie"], name=name, description=description
    )
    return {
        "id": entity_id,
        "labels": ["Movie"],
        "name": name,
        "description": description,
        "embedding": [0.0] * 128,
        "embedding_content_fingerprint": fingerprint,
        "embedding_input_fingerprint": fingerprint,
        **extra,
    }


@pytest.mark.asyncio
async def test_changed_description_is_reembedded(fake_embedder):
    entity = _stored_entity("e1", "Inception", "an old summary")
    store = StalenessFakeGraphStore([entity])
    embedder = EntityEmbedder(cast(GraphStore, store), fake_embedder)
    entity["embedding_model_fingerprint"] = embedder.model_fingerprint

    await embedder.embed_entities()
    assert store.upserted == []  # fresh vector: nothing to do

    # The writer re-ingested the document and recorded the new expected text.
    from recon_graphrag.entity_text import entity_embedding_fingerprint

    entity["description"] = "a fresh summary"
    entity["embedding_input_fingerprint"] = entity_embedding_fingerprint(
        labels=["Movie"], name="Inception", description="a fresh summary"
    )

    await embedder.embed_entities()

    assert [call[0] for call in store.upserted] == [["e1"]]
    fake_embedder.async_embed_query.assert_awaited_once_with(
        "Movie: Inception - a fresh summary"
    )
    assert entity["embedding_input_fingerprint"] == entity["embedding_content_fingerprint"]


@pytest.mark.asyncio
async def test_model_change_reembeds_unchanged_text(fake_embedder):
    entity = _stored_entity("e1", "Inception", "plot", embedding_model_fingerprint="old")
    store = StalenessFakeGraphStore([entity])
    embedder = EntityEmbedder(cast(GraphStore, store), fake_embedder)

    await embedder.embed_entities()

    assert [call[0] for call in store.upserted] == [["e1"]]
    assert entity["embedding_model_fingerprint"] != "old"
    assert embedder.model_fingerprint == store.upserted[0][2]
