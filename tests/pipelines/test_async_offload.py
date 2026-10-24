"""F8: sync database work must run via to_thread, not block the event loop."""

import asyncio
import threading
import time
from unittest.mock import MagicMock

import pytest

from recon_graphrag.extraction.structured import ColumnEntity, RowMapping
from recon_graphrag.pipelines.graphrag_pipeline import GraphBuilderPipeline
from tests.pipelines.test_build_from_rows import ROWS
from tests.pipelines.test_graph_builder import FakeGraphStore


class BlockingWriteStore(FakeGraphStore):
    """Store that records the worker thread and blocks long enough to starve the loop."""

    def __init__(self, delay: float = 0.2):
        super().__init__()
        self.delay = delay
        self.call_threads = []
        self.runs = 0

    def _block_and_note(self, label: str):
        self.call_threads.append((label, threading.get_ident()))
        self.runs += 1
        time.sleep(self.delay)

    def validate_graph_build(self):
        self._block_and_note("validate")
        return {
            "entity_count": 1,
            "chunk_count": 1,
            "evidence_link_count": 1,
            "entity_relationship_count": 0,
        }


class _SlowWriter:
    def __init__(self, store: BlockingWriteStore):
        self.store = store

    def write_graph_document(self, graph_document):
        self.store._block_and_note("write")
        return {
            "documents": 1,
            "chunks": 1,
            "entities": 1,
            "relationships": 0,
            "evidence_links": 1,
            "claims": 0,
        }


def _heartbeat():
    async def heartbeat():
        ticks = 0
        deadline = time.monotonic() + 0.15
        while time.monotonic() < deadline:
            await asyncio.sleep(0.01)
            ticks += 1
        return ticks

    return heartbeat()


def _mapping():
    return RowMapping(
        entities=[
            ColumnEntity(label="Company", identity_column="supplier_name"),
            ColumnEntity(label="Product", identity_column="product_name"),
        ],
    )


@pytest.mark.asyncio
async def test_write_does_not_block_event_loop():
    store = BlockingWriteStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=MagicMock(),
        embedder=MagicMock(),
        graph_writer=_SlowWriter(store),
        perform_entity_resolution=False,
        embed_entities=False,
        summarize_descriptions=False,
    )
    loop_thread = threading.get_ident()

    # The store sleeps 0.2s; a healthy loop finishes several 0.01s heartbeat
    # ticks while that sync DB work happens in a worker thread.
    _, ticks = await asyncio.gather(
        pipeline.build_from_rows([ROWS[0]], _mapping(), finalize=False),
        _heartbeat(),
    )

    assert ticks >= 5
    label, worker_thread = store.call_threads[0]
    assert label == "write"
    assert worker_thread != loop_thread


@pytest.mark.asyncio
async def test_finalize_validate_does_not_block_event_loop():
    store = BlockingWriteStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=MagicMock(),
        embedder=MagicMock(),
        graph_writer=MagicMock(return_value={"documents": 1, "chunks": 1}),
        perform_entity_resolution=False,
        embed_entities=False,
        summarize_descriptions=False,
    )
    loop_thread = threading.get_ident()

    _, ticks = await asyncio.gather(
        pipeline.build_from_rows([ROWS[0]], _mapping()),
        _heartbeat(),
    )

    assert ticks >= 5
    validate_thread = dict(store.call_threads).get("validate")
    assert validate_thread is not None and validate_thread != loop_thread
