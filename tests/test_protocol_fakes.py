"""Scenario guards satisfy production interfaces and fail on unexpected calls."""

import pytest

from recon_graphrag.graphdb.base import GraphStore
from tests.protocol_fakes import AsyncOnlyEmbedder, AsyncOnlyLLM, UnsupportedGraphStore


def test_unsupported_graph_store_satisfies_protocol_and_rejects_queries():
    store: GraphStore = UnsupportedGraphStore()
    assert isinstance(store, GraphStore)
    with pytest.raises(AssertionError, match="not supported"):
        store.execute_query("RETURN 1")


@pytest.mark.asyncio
async def test_unsupported_graph_store_rejects_async_resolution():
    store: GraphStore = UnsupportedGraphStore()
    with pytest.raises(AssertionError, match="not supported"):
        await store.resolve_entities_exact()


def test_async_only_llm_rejects_synchronous_generation():
    with pytest.raises(AssertionError, match="Synchronous generation"):
        AsyncOnlyLLM().invoke("prompt")


def test_async_only_embedder_rejects_synchronous_embedding():
    with pytest.raises(AssertionError, match="Synchronous embedding"):
        AsyncOnlyEmbedder().embed_query("text")
