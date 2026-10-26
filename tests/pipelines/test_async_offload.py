"""Sync database boundaries in real async components must leave the loop free."""

import asyncio
import threading
import time
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from recon_graphrag.embeddings.entities import EntityEmbedder
from recon_graphrag.extraction.description_summarizer import DescriptionSummarizer
from recon_graphrag.extraction.structured import ColumnEntity, RowMapping
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.graphdb.neo4j.entity_resolution import _Neo4jEntityResolver
from recon_graphrag.llm.base import LLMResponse
from recon_graphrag.pipelines.graphrag_pipeline import GraphBuilderPipeline
from tests.graphdb.neo4j.test_neo4j_entity_resolution import FakeGraphStore as ResolutionStore
from tests.pipelines.test_build_from_rows import ROWS
from tests.pipelines.test_graph_builder import FakeGraphStore


class BlockingCalls:
    def __init__(self):
        self.call_threads = []
        self.loop_thread = threading.get_ident()

    def block(self, label):
        self.call_threads.append((label, threading.get_ident()))
        time.sleep(0.2)

    def assert_offloaded(self, *labels):
        assert set(labels) <= {label for label, _ in self.call_threads}
        assert all(thread != self.loop_thread for _, thread in self.call_threads)


async def with_heartbeat(operation):
    # Start and synchronize the heartbeat before starting any blocking work.
    started = asyncio.Event()
    gaps = []

    async def heartbeat():
        previous = time.monotonic()
        started.set()
        while True:
            await asyncio.sleep(0.005)
            now = time.monotonic()
            gaps.append(now - previous)
            previous = now

    task = asyncio.create_task(heartbeat())
    await started.wait()
    try:
        result = await operation
        await asyncio.sleep(0.01)
        assert gaps
        assert max(gaps) < 0.15, f"event loop stalled for {max(gaps):.3f}s"
        return result
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


class BlockingWriteStore(FakeGraphStore, BlockingCalls):
    def __init__(self):
        FakeGraphStore.__init__(self)
        BlockingCalls.__init__(self)

    def backfill_descriptions(self):
        self.block("backfill")

    def validate_graph_build(self):
        self.block("validate")
        return {"entity_count": 1, "chunk_count": 1, "evidence_link_count": 1, "entity_relationship_count": 0}


class SlowWriter:
    def __init__(self, store):
        self.store = store

    def write_graph_document(self, graph_document):
        self.store.block("write")
        return {"documents": 1, "chunks": 1, "entities": 1, "relationships": 0, "evidence_links": 1, "claims": 0}


@pytest.mark.asyncio
async def test_pipeline_write_backfill_and_finalization_are_offloaded():
    store = BlockingWriteStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store, llm=MagicMock(), embedder=MagicMock(),
        graph_writer=SlowWriter(store), perform_entity_resolution=False,
        embed_entities=False, summarize_descriptions=False,
    )
    mapping = RowMapping(entities=[
        ColumnEntity(label="Company", identity_column="supplier_name"),
        ColumnEntity(label="Product", identity_column="product_name"),
    ])
    await with_heartbeat(pipeline.build_from_rows([ROWS[0]], mapping))
    store.assert_offloaded("write", "backfill", "validate")


class LocalProviders:
    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        return LLMResponse(content="A concise summary.")

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        return self.invoke(prompt, **kwargs)

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        return [0.1, 0.2]

    async def async_embed_query(self, text: str, **kwargs: Any) -> list[float]:
        return self.embed_query(text, **kwargs)


class SummaryStore(BlockingCalls):
    def __init__(self):
        super().__init__()
        self.saved = {}

    def fetch(self, kind):
        self.block(f"{kind}-read")
        if kind in self.saved:
            return []
        return [{"id": "1", "name": "Alice", "type": "KNOWS", "source_id": "a", "target_id": "b", "descriptions": ["Alice knows Bob."]}]

    def persist(self, kind, rows):
        self.block(f"{kind}-persist")
        self.saved[kind] = rows

    def get_entities_needing_summary(self, graph_name, limit):
        return self.fetch("entity")

    def get_relationships_needing_summary(self, graph_name, limit):
        return self.fetch("relationship")

    def persist_entity_summaries(self, graph_name, rows):
        self.persist("entity", rows)

    def persist_relationship_summaries(self, graph_name, rows):
        self.persist("relationship", rows)


@pytest.mark.asyncio
async def test_real_summarizer_reads_and_persists_off_loop():
    store = SummaryStore()
    summarizer = DescriptionSummarizer(LocalProviders(), cast(GraphStore, store), "g")
    assert await with_heartbeat(summarizer.summarize_entities()) == {"summarized": 1, "batches": 1}
    assert await with_heartbeat(summarizer.summarize_relationships()) == {"summarized": 1, "batches": 1}
    store.assert_offloaded("entity-read", "entity-persist", "relationship-read", "relationship-persist")
    assert all(rows[0]["description"] == "A concise summary." for rows in store.saved.values())


class EmbeddingStore(BlockingCalls):
    def __init__(self):
        super().__init__()
        self.saved = None

    def get_unembedded_entities(self, **kwargs):
        self.block("embedding-read")
        return [] if self.saved else [{"id": "1", "labels": ["Person"], "name": "Alice", "description": "Engineer"}]

    def upsert_vectors(self, ids, prop, vectors, **kwargs):
        self.block("embedding-upsert")
        self.saved = (ids, prop, vectors, kwargs)


@pytest.mark.asyncio
async def test_real_embedder_reads_and_upserts_off_loop():
    store = EmbeddingStore()
    await with_heartbeat(EntityEmbedder(cast(GraphStore, store), LocalProviders()).embed_entities())
    store.assert_offloaded("embedding-read", "embedding-upsert")
    assert store.saved is not None
    ids, prop, vectors, metadata = store.saved
    assert (ids, prop, vectors) == (["1"], "embedding", [[0.1, 0.2]])
    assert len(metadata["content_fingerprints"]) == 1
    assert metadata["model_fingerprint"]


class BlockingResolutionStore(ResolutionStore, BlockingCalls):
    def __init__(self):
        rows = [
            {"node_id": node_id, "entity_id": node_id, "graph_name": "g", "resolve_value": "Alice", "labels": ["Person"], "properties": {"name": "Alice"}}
            for node_id in ["1", "2"]
        ]
        ResolutionStore.__init__(self, rows=rows)
        BlockingCalls.__init__(self)

    def execute_query(self, query, parameters=None):
        if "apoc.version" in query:
            label = "preflight"
        elif "elementId(e) AS node_id" in query:
            label = "load"
        elif "apoc.refactor.mergeNodes" in query:
            label = "merge"
        else:
            label = "merge-support"
        self.block(label)
        return super().execute_query(query, parameters)


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["exact", "normalized", "fuzzy", "hybrid"])
@pytest.mark.parametrize("dry_run", [False, True])
async def test_real_resolver_preflight_load_and_merge_off_loop(strategy, dry_run):
    store = BlockingResolutionStore()
    result = await with_heartbeat(_Neo4jEntityResolver(store).resolve(graph_name="g", strategy=strategy, dry_run=dry_run))
    assert result["merged_groups"] == 1
    if dry_run:
        store.assert_offloaded("load")
        assert {label for label, _ in store.call_threads} == {"load"}
        assert result["merged_nodes"] == 0
    else:
        store.assert_offloaded("preflight", "load", "merge")
        assert result["merged_nodes"] > 0
