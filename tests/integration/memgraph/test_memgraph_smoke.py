"""Integration smoke test for the Memgraph backend.

Requires a running Memgraph instance with MAGE and opt-in via the environment.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

RUN_FLAG = "RUN_MEMGRAPH_INTEGRATION_TESTS"
REQUIRED_ENV = ["MEMGRAPH_URL"]


def _memgraph_env_or_skip() -> None:
    if os.getenv(RUN_FLAG, "").lower() not in {"1", "true", "yes"}:
        pytest.skip(f"Set {RUN_FLAG}=1 to run Memgraph integration tests.")
    missing = [name for name in REQUIRED_ENV if not os.getenv(name)]
    if missing:
        pytest.fail(f"Missing required Memgraph env vars: {', '.join(missing)}")


@pytest.fixture
def memgraph_store():
    _memgraph_env_or_skip()

    from recon_graphrag.graphdb.memgraph.store import MemgraphGraphStore

    url = os.getenv("MEMGRAPH_URL", "bolt://localhost:7687")
    user = os.getenv("MEMGRAPH_USERNAME", "")
    password = os.getenv("MEMGRAPH_PASSWORD", "")
    database = os.getenv("MEMGRAPH_DATABASE", "")
    if user and password:
        driver = GraphDatabase.driver(url, auth=(user, password))
    else:
        driver = GraphDatabase.driver(url)
    store = MemgraphGraphStore(driver, database=database or None)

    # Clean slate
    store.execute_query("MATCH (n) DETACH DELETE n")

    yield store

    # Teardown
    store.execute_query("MATCH (n) DETACH DELETE n")


@pytest.mark.integration
async def test_write_and_count(memgraph_store):
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
        graph_name="entity-graph",
        metadata={},
    )
    chunks = [
        ChunkRecord(
            id="chunk-1",
            document_id="doc-1",
            text="Alice works at Acme.",
            index=0,
            graph_name="entity-graph",
            metadata={},
        )
    ]
    entities = [
        EntityRecord(
            id="ent-1",
            type="Person",
            graph_name="entity-graph",
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
                graph_name="entity-graph",
            )
        ],
    )

    memgraph_store.write_graph_document(graph_doc)
    assert memgraph_store.get_entity_count() == 1
    assert memgraph_store.get_chunk_count() == 1
    assert memgraph_store.get_evidence_link_count() == 1
