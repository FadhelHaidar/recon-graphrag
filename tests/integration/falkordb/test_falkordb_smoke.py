"""Integration smoke test for the FalkorDB backend.

Requires a running FalkorDB instance and opt-in via the environment.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

RUN_FLAG = "RUN_FALKORDB_INTEGRATION_TESTS"
REQUIRED_ENV = ["FALKORDB_HOST", "FALKORDB_PORT"]


def _falkordb_env_or_skip() -> None:
    if os.getenv(RUN_FLAG, "").lower() not in {"1", "true", "yes"}:
        pytest.skip(f"Set {RUN_FLAG}=1 to run FalkorDB integration tests.")
    missing = [name for name in REQUIRED_ENV if not os.getenv(name)]
    if missing:
        pytest.fail(f"Missing required FalkorDB env vars: {', '.join(missing)}")


@pytest.fixture
def falkordb_store():
    _falkordb_env_or_skip()
    from falkordb import FalkorDB

    from recon_graphrag.graphdb.falkordb.store import FalkorDBGraphStore

    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "6379"))
    client = FalkorDB(host=host, port=port)
    # Use a unique graph name to avoid collisions.
    graph_name = f"test-graph-{os.getpid()}"
    store = FalkorDBGraphStore(client, graph_name=graph_name)

    # Clean slate
    store.execute_query("MATCH (n) DETACH DELETE n")

    yield store

    # Teardown
    store.execute_query("MATCH (n) DETACH DELETE n")


@pytest.mark.integration
async def test_write_and_count(falkordb_store):
    from recon_graphrag.extraction.types import (
        ChunkRecord,
        DocumentRecord,
        EntityRecord,
        EvidenceLink,
        GraphDocument,
    )

    document = DocumentRecord(
        id="doc-1",
        text_hash="hash",
        graph_name=falkordb_store.graph_name,
        metadata={},
    )
    chunks = [
        ChunkRecord(
            id="chunk-1",
            document_id="doc-1",
            text="Alice works at Acme.",
            index=0,
            graph_name=falkordb_store.graph_name,
            metadata={},
        )
    ]
    entities = [
        EntityRecord(
            id="ent-1",
            type="Person",
            graph_name=falkordb_store.graph_name,
            properties={"name": "Alice"},
        )
    ]
    graph_doc = GraphDocument(
        document=document,
        chunks=chunks,
        entities=entities,
        relationships=[],
        evidence_links=[
            EvidenceLink(
                chunk_id="chunk-1",
                entity_id="ent-1",
                graph_name=falkordb_store.graph_name,
            )
        ],
    )

    falkordb_store.write_graph_document(graph_doc)
    assert falkordb_store.get_entity_count() == 1
    assert falkordb_store.get_chunk_count() == 1
    assert falkordb_store.get_evidence_link_count() == 1
