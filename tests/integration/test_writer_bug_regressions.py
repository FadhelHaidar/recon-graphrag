"""Bug-fix persistence regressions: real databases, no AI/provider calls.

Each test states the defective behavior it pins. Run with::

    RUN_DATABASE_INTEGRATION_TESTS=1 uv run pytest \
        tests/integration/test_writer_bug_regressions.py -m database
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from uuid import uuid4

import pytest

from recon_graphrag.embeddings.entities import EntityEmbedder
from recon_graphrag.entity_text import text_fingerprint
from recon_graphrag.extraction.types import GraphDocument
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.pipelines.writer import CypherGraphWriter
from tests.integration.factories import get_memgraph_store, get_neo4j_store
from tests.integration.support import cleanup_graph, require_integration_env
from tests.pipelines.test_writer_replacement import build_document

pytestmark = [pytest.mark.integration, pytest.mark.database]

# The disposable databases hold 768-dimension vector indexes, so index setup
# must match them. The default 1536 would fail on a compatible existing index.
INDEX_DIM = 768


@pytest.fixture(scope="module", params=["neo4j", "memgraph"])
def backend(request):
    prefix = request.param.upper()
    require_integration_env(
        "RUN_DATABASE_INTEGRATION_TESTS",
        [f"{prefix}_URL"],
        "writer bug-fix persistence regressions",
        fail_on_missing=True,
    )
    store = get_neo4j_store() if request.param == "neo4j" else get_memgraph_store()
    store.create_indexes(embedding_dim=INDEX_DIM)
    try:
        yield store
    finally:
        driver = getattr(store, "driver", None)
        if driver is not None:
            driver.close()


@pytest.fixture
def case(backend):
    graph = f"bug-fix-{uuid4().hex}"
    try:
        yield backend, graph
    finally:
        cleanup_graph(backend, graph)


def claim_graph(store: GraphStore, graph: str, claim_id: str) -> dict:
    """Return the claim node and its source and subject targets."""
    rows = store.execute_query(
        """
        MATCH (c:Claim {id: $claim_id, graph_name: $graph_name})
        OPTIONAL MATCH (c)-[:SOURCED_FROM]->(chunk:Chunk)
        OPTIONAL MATCH (c)-[:SUBJECT_OF]->(entity:__Entity__)
        RETURN c.description AS description,
               collect(DISTINCT chunk.id) AS source_chunks,
               collect(DISTINCT entity.id) AS subjects
        """,
        {"claim_id": claim_id, "graph_name": graph},
    )
    return rows[0] if rows else {}


def claim_count(store: GraphStore, graph: str) -> int:
    rows = store.execute_query(
        "MATCH (c:Claim {graph_name: $graph_name}) RETURN count(c) AS count",
        {"graph_name": graph},
    )
    return int(rows[0]["count"]) if rows else 0


# --- B03: a retained claim survives its source chunk replacement ---


def test_retained_claim_moves_to_the_replacement_chunk(case) -> None:
    store, graph = case
    writer = CypherGraphWriter(store)
    writer.write_graph_document(
        build_document(
            chunks=["doc:1:c0", "doc:1:c1"],
            entities=[("e1", "Alice", "Alice leads Acme.")],
            evidence=[("doc:1:c0", "e1"), ("doc:1:c1", "e1")],
            claims=[("claim:1", "e1", "doc:1:c0")],
            graph_name=graph,
        )
    )

    receipt = writer.write_graph_document(
        build_document(
            chunks=["doc:1:c1"],
            entities=[("e1", "Alice", "Alice leads Acme.")],
            evidence=[("doc:1:c1", "e1")],
            claims=[("claim:1", "e1", "doc:1:c1")],
            graph_name=graph,
        )
    )

    assert receipt["claims"] == 1
    state = claim_graph(store, graph, "claim:1")
    assert state["source_chunks"] == ["doc:1:c1"]
    assert state["subjects"] == ["e1"]
    chunks = store.execute_query(
        "MATCH (c:Chunk {graph_name: $graph_name}) RETURN collect(c.id) AS ids",
        {"graph_name": graph},
    )
    assert sorted(chunks[0]["ids"]) == ["doc:1:c1"]


def test_omitted_claim_is_deleted_and_retained_claim_kept(case) -> None:
    store, graph = case
    writer = CypherGraphWriter(store)
    writer.write_graph_document(
        build_document(
            chunks=["doc:1:c0"],
            entities=[("e1", "Alice", "Alice leads Acme.")],
            claims=[
                ("claim:keep", "e1", "doc:1:c0"),
                ("claim:drop", "e1", "doc:1:c0"),
            ],
            graph_name=graph,
        )
    )
    assert claim_count(store, graph) == 2

    writer.write_graph_document(
        build_document(
            chunks=["doc:1:c0"],
            entities=[("e1", "Alice", "Alice leads Acme.")],
            claims=[("claim:keep", "e1", "doc:1:c0")],
            graph_name=graph,
        )
    )

    assert claim_count(store, graph) == 1
    assert claim_graph(store, graph, "claim:keep")["source_chunks"] == ["doc:1:c0"]
    assert claim_graph(store, graph, "claim:drop") == {}


def test_claim_replacement_leaves_other_graph_claims_untouched(case) -> None:
    store, graph = case
    writer = CypherGraphWriter(store)
    other = f"{graph}-other"
    try:
        writer.write_graph_document(
            build_document(
                chunks=["doc:1:c0"],
                entities=[("e1", "Alice", "Alice leads Acme.")],
                claims=[("claim:1", "e1", "doc:1:c0")],
                graph_name=other,
            )
        )
        writer.write_graph_document(
            build_document(
                chunks=["doc:1:c0"],
                entities=[("e1", "Alice", "Alice leads Acme.")],
                claims=[("claim:1", "e1", "doc:1:c0")],
                graph_name=graph,
            )
        )

        writer.write_graph_document(
            build_document(
                chunks=["doc:1:c0"],
                entities=[("e1", "Alice", "Alice leads Acme.")],
                claims=[],
                graph_name=graph,
            )
        )

        assert claim_count(store, graph) == 0
        assert claim_graph(store, other, "claim:1")["source_chunks"] == ["doc:1:c0"]
    finally:
        cleanup_graph(store, other)


# --- B06: references must resolve against the replacement's chunks ---


class AutoCommitStore:
    """A store without the transaction hook, exercising the auto-commit path."""

    _transaction = None

    def __init__(self, store: GraphStore):
        self._store = store

    def execute_query(self, query: str, parameters: dict | None = None):
        return self._store.execute_query(query, parameters)


def document_snapshot(store: GraphStore, graph: str) -> dict:
    rows = store.execute_query(
        """
        MATCH (d:Document {graph_name: $graph_name})
        OPTIONAL MATCH (c:Chunk {graph_name: $graph_name})
        OPTIONAL MATCH (e:__Entity__ {graph_name: $graph_name})
        OPTIONAL MATCH (cl:Claim {graph_name: $graph_name})
        RETURN collect(DISTINCT d.id) AS documents,
               collect(DISTINCT c.id) AS chunks,
               collect(DISTINCT e.id) AS entities,
               collect(DISTINCT cl.id) AS claims
        """,
        {"graph_name": graph},
    )[0]
    return {key: sorted(value) for key, value in rows.items()}


@pytest.mark.parametrize("auto_commit", [False, True], ids=["transaction", "auto-commit"])
def test_removed_chunk_reference_is_rejected_before_mutation(case, auto_commit) -> None:
    store, graph = case
    writer = CypherGraphWriter(store)
    writer.write_graph_document(
        build_document(
            chunks=["doc:1:c0", "doc:1:c1"],
            entities=[("e1", "Alice", "Alice leads Acme.")],
            evidence=[("doc:1:c0", "e1"), ("doc:1:c1", "e1")],
            graph_name=graph,
        )
    )
    before = document_snapshot(store, graph)

    target: GraphStore = (
        AutoCommitStore(store) if auto_commit else store  # type: ignore[assignment]
    )
    with pytest.raises(ValueError, match="doc:1:c0"):
        CypherGraphWriter(target).write_graph_document(
            build_document(
                chunks=["doc:1:c1"],
                entities=[("e1", "Alice", "Alice leads Acme.")],
                evidence=[("doc:1:c1", "e1")],
                claims=[("claim:1", "e1", "doc:1:c0")],
                graph_name=graph,
            )
        )

    assert document_snapshot(store, graph) == before
    assert claim_count(store, graph) == 0


def test_evidence_reference_to_removed_chunk_is_rejected(case) -> None:
    store, graph = case
    writer = CypherGraphWriter(store)
    writer.write_graph_document(
        build_document(
            chunks=["doc:1:c0", "doc:1:c1"],
            entities=[("e1", "Alice", "Alice leads Acme.")],
            evidence=[("doc:1:c0", "e1"), ("doc:1:c1", "e1")],
            graph_name=graph,
        )
    )
    before = document_snapshot(store, graph)

    with pytest.raises(ValueError, match="doc:1:c0"):
        writer.write_graph_document(
            build_document(
                chunks=["doc:1:c1"],
                entities=[("e1", "Alice", "Alice leads Acme.")],
                evidence=[("doc:1:c0", "e1")],
                graph_name=graph,
            )
        )

    assert document_snapshot(store, graph) == before


def test_shared_entity_reference_is_accepted_on_backends(case) -> None:
    store, graph = case
    writer = CypherGraphWriter(store)
    writer.write_graph_document(
        build_document(
            document_id="doc:other",
            chunks=["other:c0"],
            entities=[("e1", "Alice", "Alice leads Acme.")],
            graph_name=graph,
        )
    )

    receipt = writer.write_graph_document(
        build_document(
            chunks=["doc:1:c0"],
            entities=[],
            evidence=[("doc:1:c0", "e1")],
            claims=[("claim:1", "e1", "doc:1:c0")],
            graph_name=graph,
        )
    )

    assert receipt["claims"] == 1
    assert claim_graph(store, graph, "claim:1")["source_chunks"] == ["doc:1:c0"]


# --- B05: relationship identity must be validated before mutation ---


def relationship_rows(store: GraphStore, graph: str) -> list[dict]:
    return store.execute_query(
        """
        MATCH (s:__Entity__ {graph_name: $graph_name})-[r]->(t:__Entity__ {graph_name: $graph_name})
        WHERE r.graph_name = $graph_name
        RETURN r.id AS id, s.id AS source_id, t.id AS target_id, type(r) AS type
        ORDER BY type, source_id, target_id
        """,
        {"graph_name": graph},
    )


def seed_relationship_document(store: GraphStore, graph: str) -> None:
    CypherGraphWriter(store).write_graph_document(
        build_document(
            chunks=["doc:1:c0"],
            entities=[("e1", "Alice", None), ("e2", "Acme", None)],
            relationships=[("WORKS_AT", "e1", "e2", None)],
            graph_name=graph,
        )
    )


def test_missing_stored_relationship_id_fails_before_mutation(case) -> None:
    store, graph = case
    seed_relationship_document(store, graph)
    # Imported data can omit the domain ID property.
    store.execute_query(
        """
        MATCH (:__Entity__ {graph_name: $graph_name})-[r]->(:__Entity__ {graph_name: $graph_name})
        WHERE r.graph_name = $graph_name
        SET r.id = null
        """,
        {"graph_name": graph},
    )
    before = document_snapshot(store, graph)

    with pytest.raises(ValueError) as error:
        CypherGraphWriter(store).write_graph_document(
            build_document(
                chunks=["doc:1:c0"],
                entities=[("e1", "Alice", None), ("e2", "Acme", None)],
                relationships=[],
                graph_name=graph,
            )
        )

    message = str(error.value)
    assert "unusable id" in message
    assert "stored relationship" in message
    assert "WORKS_AT" in message
    assert graph in message
    assert len(relationship_rows(store, graph)) == 1
    assert document_snapshot(store, graph) == before


def test_empty_incoming_relationship_id_is_rejected(case) -> None:
    store, graph = case
    before = document_snapshot(store, graph)
    document = build_document(
        chunks=["doc:1:c0"],
        entities=[("e1", "Alice", None), ("e2", "Acme", None)],
        relationships=[("WORKS_AT", "e1", "e2", None)],
        graph_name=graph,
    )
    document.relationships[0].id = ""

    with pytest.raises(ValueError, match="submitted relationship"):
        CypherGraphWriter(store).write_graph_document(document)

    assert document_snapshot(store, graph) == before
    assert relationship_rows(store, graph) == []


def test_custom_string_relationship_id_round_trips(case) -> None:
    store, graph = case
    writer = CypherGraphWriter(store)
    document = build_document(
        chunks=["doc:1:c0"],
        entities=[("e1", "Alice", None), ("e2", "Acme", None)],
        relationships=[("WORKS_AT", "e1", "e2", None)],
        graph_name=graph,
    )
    document.relationships[0].id = "custom:works-at:alice"
    writer.write_graph_document(document)

    replacement = build_document(
        chunks=["doc:1:c0"],
        entities=[("e1", "Alice", None), ("e2", "Acme", None)],
        relationships=[("WORKS_AT", "e1", "e2", None)],
        graph_name=graph,
    )
    replacement.relationships[0].id = "custom:works-at:alice"
    receipt = writer.write_graph_document(replacement)

    assert receipt["relationships"] == 1
    assert [row["id"] for row in relationship_rows(store, graph)] == [
        "custom:works-at:alice"
    ]


# --- B01: child records must stay inside the document graph ---


def graph_node_count(store: GraphStore, graph: str) -> int:
    rows = store.execute_query(
        "MATCH (n {graph_name: $graph_name}) RETURN count(n) AS count",
        {"graph_name": graph},
    )
    return int(rows[0]["count"]) if rows else 0


def test_foreign_graph_entity_is_rejected_before_mutation(case) -> None:
    store, graph = case
    other = f"{graph}-other"
    try:
        document = build_document(
            chunks=["doc:1:c0"],
            entities=[("e1", "Alice", None)],
            graph_name=graph,
        )
        document.entities[0].graph_name = other

        with pytest.raises(ValueError, match="EntityRecord"):
            CypherGraphWriter(store).write_graph_document(document)

        assert graph_node_count(store, graph) == 0
        assert graph_node_count(store, other) == 0
    finally:
        cleanup_graph(store, other)


def test_foreign_graph_preserves_existing_graph(case) -> None:
    store, graph = case
    other = f"{graph}-other"
    try:
        CypherGraphWriter(store).write_graph_document(
            build_document(
                chunks=["doc:1:c0"],
                entities=[("e1", "Alice", None)],
                graph_name=other,
            )
        )
        before = graph_node_count(store, other)

        document = build_document(
            chunks=["doc:1:c0"],
            entities=[("e2", "Bob", None)],
            graph_name=graph,
        )
        document.entities[0].graph_name = other

        with pytest.raises(ValueError, match="EntityRecord"):
            CypherGraphWriter(store).write_graph_document(document)

        assert graph_node_count(store, other) == before
        assert graph_node_count(store, graph) == 0
    finally:
        cleanup_graph(store, other)


# --- B04: entity identity is (graph_name, id); labels accumulate ---


def entity_rows(store: GraphStore, graph: str) -> list[dict]:
    return list(
        store.execute_query(
            """
            MATCH (e:__Entity__ {graph_name: $graph_name})
            RETURN e.id AS id, labels(e) AS labels, e.type AS type,
                   e.name AS name, e.descriptions AS descriptions,
                   id(e) AS node_id
            ORDER BY e.id
            """,
            {"graph_name": graph},
        )
    )


def test_entity_write_reuses_untyped_existing_identity(case) -> None:
    store, graph = case
    seeded = store.execute_query(
        "CREATE (e:__Entity__ {id: 'e1', graph_name: $graph_name, name: 'Alice'}) "
        "RETURN id(e) AS node_id",
        {"graph_name": graph},
    )[0]["node_id"]

    receipt = CypherGraphWriter(store).write_graph_document(
        build_document(
            chunks=["doc:1:c0"],
            entities=[("e1", "Alice", None)],
            entity_type="Person",
            graph_name=graph,
        )
    )

    (row,) = entity_rows(store, graph)
    assert receipt["entities"] == 1
    assert row["id"] == "e1"
    assert "Person" in row["labels"]
    assert row["node_id"] == seeded


def test_entity_write_adds_label_without_duplicate_identity(case) -> None:
    store, graph = case
    writer = CypherGraphWriter(store)
    writer.write_graph_document(
        build_document(
            chunks=["doc:1:c0"],
            entities=[("e1", "Alice", "Alice leads Acme.")],
            entity_type="Person",
            graph_name=graph,
        )
    )

    writer.write_graph_document(
        build_document(
            document_id="doc:2",
            chunks=["doc:2:c0"],
            entities=[("e1", "Alice", "Alice leads Acme.")],
            entity_type="Organization",
            graph_name=graph,
        )
    )

    (row,) = entity_rows(store, graph)
    assert set(row["labels"]) == {"__Entity__", "Person", "Organization"}
    # The stored type field keeps the last written group convention.
    assert row["type"] == "Organization"
    assert row["descriptions"]


def test_entity_label_change_preserves_edges_and_provenance(case) -> None:
    store, graph = case
    writer = CypherGraphWriter(store)
    document = build_document(
        chunks=["doc:1:c0"],
        entities=[("e1", "Alice", "Alice leads Acme."), ("e2", "Acme", None)],
        relationships=[("WORKS_AT", "e1", "e2", None)],
        evidence=[("doc:1:c0", "e1")],
        entity_type="Person",
        graph_name=graph,
    )
    writer.write_graph_document(document)
    before = relationship_rows(store, graph)

    document.entities[0].type = "Manager"
    writer.write_graph_document(document)

    (alice,) = [row for row in entity_rows(store, graph) if row["id"] == "e1"]
    assert set(alice["labels"]) == {"__Entity__", "Person", "Manager"}
    assert alice["descriptions"] == ["Alice leads Acme."]
    assert relationship_rows(store, graph) == before
    evidence = store.execute_query(
        "MATCH (c:Chunk {graph_name: $graph_name})-[:FROM_CHUNK]->"
        "(e:__Entity__ {id: 'e1'}) RETURN count(*) AS count",
        {"graph_name": graph},
    )[0]["count"]
    assert evidence == 1


def test_entity_write_rejects_ambiguous_existing_identity(case) -> None:
    store, graph = case
    store.execute_query(
        "CREATE (a:__Entity__ {id: 'e1', graph_name: $graph_name, uid: 'u1'}), "
        "(b:__Entity__ {id: 'e1', graph_name: $graph_name, uid: 'u2'})",
        {"graph_name": graph},
    )

    with pytest.raises(ValueError) as error:
        CypherGraphWriter(store).write_graph_document(
            build_document(
                chunks=["doc:1:c0"],
                entities=[("e1", "Alice", None)],
                entity_type="Person",
                graph_name=graph,
            )
        )

    message = str(error.value)
    assert "ambiguous" in message
    assert "'e1' (2 nodes)" in message
    assert len(entity_rows(store, graph)) == 2
    chunks = store.execute_query(
        "MATCH (c:Chunk {graph_name: $graph_name}) RETURN count(c) AS count",
        {"graph_name": graph},
    )[0]["count"]
    assert chunks == 0


def test_entity_write_with_escaped_label_reuses_identity(case) -> None:
    store, graph = case
    writer = CypherGraphWriter(store)
    writer.write_graph_document(
        build_document(
            chunks=["doc:1:c0"],
            entities=[("e1", "Alice", None)],
            entity_type="Person",
            graph_name=graph,
        )
    )

    writer.write_graph_document(
        build_document(
            document_id="doc:2",
            chunks=["doc:2:c0"],
            entities=[("e1", "Alice", None)],
            entity_type="Data Source",
            graph_name=graph,
        )
    )

    (row,) = entity_rows(store, graph)
    assert set(row["labels"]) == {"__Entity__", "Person", "Data Source"}


def test_concurrent_identity_writes_keep_one_node(case) -> None:
    import threading

    store, graph = case
    barrier = threading.Barrier(2)
    outcomes: list[bool] = []

    def write(document_id: str) -> None:
        barrier.wait(timeout=10)
        try:
            CypherGraphWriter(store).write_graph_document(
                build_document(
                    document_id=document_id,
                    chunks=[f"{document_id}:c0"],
                    entities=[("e1", "Alice", f"text from {document_id}")],
                    entity_type="Person",
                    graph_name=graph,
                )
            )
            outcomes.append(True)
        except Exception:  # noqa: BLE001 - conflict surface differs per backend
            outcomes.append(False)

    threads = [threading.Thread(target=write, args=(f"doc:{i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert len(outcomes) == 2
    assert any(outcomes)
    assert len(entity_rows(store, graph)) == 1


def test_relationship_replacement_preserves_other_document_contributions(case):
    store, graph = case
    writer = CypherGraphWriter(store)
    writer.write_graph_document(
        build_document(
            document_id="doc:a",
            graph_name=graph,
            chunks=["a0"],
            entities=[("e1", "Alice", None), ("e2", "Bob", None)],
            relationships=[("RELATES", "e1", "e2", "from A")],
        )
    )
    writer.write_graph_document(
        build_document(
            document_id="doc:b",
            graph_name=graph,
            chunks=["b0"],
            entities=[("e1", "Alice", None), ("e2", "Bob", None)],
            relationships=[("RELATES", "e1", "e2", "from B")],
        )
    )
    edges = relationship_edges(store, graph)
    assert len(edges) == 1
    stored_id = edges[0]["id"]
    assert set(edges[0]["source_chunk_ids"]) == {"a0", "b0"}

    writer.write_graph_document(
        build_document(
            document_id="doc:a",
            graph_name=graph,
            chunks=["a1"],
            entities=[("e1", "Alice", None), ("e2", "Bob", None)],
        )
    )

    edges = relationship_edges(store, graph)
    assert len(edges) == 1
    assert edges[0]["id"] == stored_id
    assert edges[0]["rel_graph_name"] == graph
    assert set(edges[0]["source_chunk_ids"]) == {"b0"}


def test_omitted_relationship_is_found_by_previous_chunks(case) -> None:
    store, graph = case
    writer = CypherGraphWriter(store)
    writer.write_graph_document(local_document(graph))
    assert len(relationship_edges(store, graph)) == 1

    writer.write_graph_document(
        build_document(document_id="doc:1", graph_name=graph, chunks=["c0"])
    )

    assert relationship_edges(store, graph) == []


def test_relationship_self_loop_and_reverse_direction_are_distinct(case) -> None:
    store, graph = case
    CypherGraphWriter(store).write_graph_document(
        build_document(
            document_id="doc:1",
            graph_name=graph,
            chunks=["c0"],
            entities=[("e1", "Alice", None), ("e2", "Bob", None)],
            relationships=[
                ("RELATES", "e1", "e1", "loop"),
                ("RELATES", "e1", "e2", "forward"),
                ("RELATES", "e2", "e1", "backward"),
            ],
        )
    )

    rows = relationship_edges(store, graph)
    assert sorted((row["source_id"], row["target_id"]) for row in rows) == [
        ("e1", "e1"),
        ("e1", "e2"),
        ("e2", "e1"),
    ]
    assert len({row["id"] for row in rows}) == 3
    assert {row["rel_graph_name"] for row in rows} == {graph}


@pytest.mark.asyncio
async def test_resolve_then_replace_preserves_logical_edge_identity(case) -> None:
    store, graph = case
    seed_entities(store, graph, "e1", "dup", "e2")
    store.execute_query(
        """
        MATCH (e:__Entity__ {graph_name: $graph_name})
        SET e.name = CASE e.id WHEN 'e2' THEN 'Bob' ELSE 'Alice' END
        """,
        {"graph_name": graph},
    )
    seed_relationship(
        store,
        graph,
        rel_id="rel:1",
        rel_graph_name=graph,
        marker="doc-a",
        source_id="e1",
        chunk_ids=("doc:1:c0",),
    )
    seed_relationship(
        store,
        graph,
        rel_id="rel:2",
        rel_graph_name=graph,
        marker="doc-b",
        source_id="dup",
        chunk_ids=("doc:2:c0",),
    )

    result = await store.resolve_entities_exact(graph_name=graph, dry_run=False)
    assert result["merged_groups"] == 1

    keys = [
        (row["source_id"], row["target_id"], row["type"])
        for row in relationship_edges(store, graph)
    ]
    assert len(keys) == len(set(keys)), keys

    writer = CypherGraphWriter(store)
    writer.write_graph_document(
        build_document(
            document_id="doc:1",
            graph_name=graph,
            chunks=["doc:1:c0"],
            entities=[("e1", "Alice", None), ("e2", "Bob", None)],
            relationships=[("RELATES", "e1", "e2", "from A")],
        )
    )

    rows = relationship_edges(store, graph)
    assert len(rows) == 1
    assert rows[0]["rel_graph_name"] == graph


def test_concurrent_logical_edge_creation_does_not_duplicate(case) -> None:
    import threading

    store, graph = case
    seed_entities(store, graph, "e1", "e2")
    barrier = threading.Barrier(2)
    outcomes: list[bool] = []

    def write(document_id: str) -> None:
        barrier.wait(timeout=10)
        try:
            CypherGraphWriter(store).write_graph_document(
                build_document(
                    document_id=document_id,
                    graph_name=graph,
                    chunks=[f"{document_id}:c0"],
                    entities=[("e1", "Alice", None), ("e2", "Bob", None)],
                    relationships=[("RELATES", "e1", "e2", None)],
                )
            )
            outcomes.append(True)
        except Exception:  # noqa: BLE001 - conflict surface differs per backend
            outcomes.append(False)

    threads = [
        threading.Thread(target=write, args=(f"doc:{index}",)) for index in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert len(outcomes) == 2
    assert any(outcomes)
    rows = relationship_edges(store, graph)
    # A duplicate created by concurrent merges must stay detectable, never be
    # silently combined: the next write to the same key raises.
    if len(rows) > 1:
        with pytest.raises(ValueError, match="ambiguous"):
            CypherGraphWriter(store).write_graph_document(
                build_document(
                    document_id="doc:3",
                    graph_name=graph,
                    chunks=["doc:3:c0"],
                    entities=[("e1", "Alice", None), ("e2", "Bob", None)],
                    relationships=[("RELATES", "e1", "e2", None)],
                )
            )


def test_concurrent_relationship_contributions_keep_one_edge(case) -> None:
    import threading

    store, graph = case
    seed_entities(store, graph, "e1", "e2")
    seed_relationship(
        store,
        graph,
        rel_id="existing",
        rel_graph_name=graph,
        marker="seed",
        chunk_ids=("seed-chunk",),
    )
    barrier = threading.Barrier(2)
    outcomes: list[bool] = []

    def write(document_id: str) -> None:
        barrier.wait(timeout=10)
        try:
            CypherGraphWriter(store).write_graph_document(
                build_document(
                    document_id=document_id,
                    graph_name=graph,
                    chunks=[f"{document_id}:c0"],
                    entities=[("e1", "Alice", None), ("e2", "Bob", None)],
                    relationships=[("RELATES", "e1", "e2", None)],
                )
            )
            outcomes.append(True)
        except Exception:  # noqa: BLE001 - conflict surface differs per backend
            outcomes.append(False)

    threads = [
        threading.Thread(target=write, args=(f"doc:{index}",)) for index in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert len(outcomes) == 2
    assert any(outcomes)
    rows = relationship_edges(store, graph)
    assert len(rows) == 1
    assert rows[0]["id"] == "existing"
    assert rows[0]["rel_graph_name"] == graph
    chunks = rows[0]["source_chunk_ids"]
    # No contribution is invented or duplicated. Neo4j read-committed writes
    # can still drop one concurrent contribution; Memgraph retries the conflict
    # and keeps both. That lost-update window predates this fix (measured on the
    # baseline writer) and needs per-key serialization to close.
    assert chunks == list(dict.fromkeys(chunks))
    assert set(chunks) <= {"seed-chunk", "doc:0:c0", "doc:1:c0"}
    assert "seed-chunk" in chunks
    assert any(chunk.startswith("doc:") for chunk in chunks)


# ---------------------------------------------------------------------------
# B07: a replaced chunk keeps its ID but changes text, so its vector goes stale
# ---------------------------------------------------------------------------

VECTOR = [1.0] + [0.0] * (INDEX_DIM - 1)
MODEL = "model-v1"


def chunk_state(store: GraphStore, graph: str, chunk_id: str) -> dict:
    rows = store.execute_query(
        """
        MATCH (c:Chunk {id: $chunk_id, graph_name: $graph_name})
        RETURN c.text AS text,
               c.embedding AS embedding,
               c.embedding_input_fingerprint AS input_fingerprint,
               c.embedding_content_fingerprint AS content_fingerprint,
               c.embedding_model_fingerprint AS model_fingerprint
        """,
        {"chunk_id": chunk_id, "graph_name": graph},
    )
    assert rows, f"chunk {chunk_id} not written"
    return rows[0]


def chunk_node_id(store: GraphStore, graph: str, chunk_id: str) -> str:
    """Return the backend-internal chunk ID that ``upsert_vectors`` expects."""
    id_function = "id" if "memgraph" in type(store).__module__ else "elementId"
    rows = store.execute_query(
        f"""
        MATCH (c:Chunk {{id: $chunk_id, graph_name: $graph_name}})
        RETURN {id_function}(c) AS node_id
        """,
        {"chunk_id": chunk_id, "graph_name": graph},
    )
    return rows[0]["node_id"]


def chunk_document(graph: str, *, text: str, chunk_id: str = "doc:1:c0") -> GraphDocument:
    document = build_document(chunks=[chunk_id], graph_name=graph)
    document.chunks[0].text = text
    return document


def unembedded_chunk_ids(
    store: GraphStore, graph: str, model_fingerprint: str | None = MODEL
) -> set[str]:
    rows = store.get_unembedded_chunks(
        graph_name=graph, limit=100, model_fingerprint=model_fingerprint
    )
    return {row["chunk_id"] for row in rows}


def certify_chunk(
    store: GraphStore, graph: str, chunk_id: str, text: str, model_fingerprint: str = MODEL
) -> None:
    store.upsert_vectors(
        [chunk_node_id(store, graph, chunk_id)],
        "embedding",
        [VECTOR],
        content_fingerprints=[text_fingerprint(text)],
        model_fingerprint=model_fingerprint,
    )


def test_changed_chunk_text_clears_vector_and_certification(case) -> None:
    store, graph = case
    writer = CypherGraphWriter(store)
    writer.write_graph_document(chunk_document(graph, text="alpha"))
    certify_chunk(store, graph, "doc:1:c0", "alpha")
    assert chunk_state(store, graph, "doc:1:c0")["embedding"] is not None
    assert unembedded_chunk_ids(store, graph) == set()

    writer.write_graph_document(chunk_document(graph, text="beta"))

    state = chunk_state(store, graph, "doc:1:c0")
    assert state["text"] == "beta"
    assert state["embedding"] is None
    assert state["content_fingerprint"] is None
    assert state["model_fingerprint"] is None
    assert state["input_fingerprint"] == text_fingerprint("beta")
    assert unembedded_chunk_ids(store, graph) == {"doc:1:c0"}


def test_unchanged_chunk_text_keeps_the_certified_vector(case) -> None:
    store, graph = case
    writer = CypherGraphWriter(store)
    writer.write_graph_document(chunk_document(graph, text="alpha"))
    certify_chunk(store, graph, "doc:1:c0", "alpha")

    writer.write_graph_document(chunk_document(graph, text="alpha"))

    state = chunk_state(store, graph, "doc:1:c0")
    assert state["embedding"] is not None
    assert state["content_fingerprint"] == text_fingerprint("alpha")
    assert state["model_fingerprint"] == MODEL
    assert unembedded_chunk_ids(store, graph) == set()


def test_legacy_uncertified_chunk_is_refreshed_once(case) -> None:
    store, graph = case
    writer = CypherGraphWriter(store)
    writer.write_graph_document(chunk_document(graph, text="alpha"))
    # Simulate a chunk written before fingerprints existed: vector, no proof.
    store.execute_query(
        """
        MATCH (c:Chunk {id: $chunk_id, graph_name: $graph_name})
        SET c.embedding = $vector
        """,
        {"chunk_id": "doc:1:c0", "graph_name": graph, "vector": VECTOR},
    )
    assert unembedded_chunk_ids(store, graph) == {"doc:1:c0"}

    certify_chunk(store, graph, "doc:1:c0", "alpha")

    assert unembedded_chunk_ids(store, graph) == set()
    assert chunk_state(store, graph, "doc:1:c0")["content_fingerprint"] == text_fingerprint(
        "alpha"
    )


def test_chunk_model_change_requires_refresh(case) -> None:
    store, graph = case
    writer = CypherGraphWriter(store)
    writer.write_graph_document(chunk_document(graph, text="alpha"))
    certify_chunk(store, graph, "doc:1:c0", "alpha")

    assert unembedded_chunk_ids(store, graph, model_fingerprint="model-v2") == {"doc:1:c0"}
    assert unembedded_chunk_ids(store, graph, model_fingerprint=MODEL) == set()


def test_stale_chunk_vector_cannot_overwrite_new_text(case) -> None:
    store, graph = case
    writer = CypherGraphWriter(store)
    writer.write_graph_document(chunk_document(graph, text="alpha"))
    node_id = chunk_node_id(store, graph, "doc:1:c0")
    writer.write_graph_document(chunk_document(graph, text="beta"))

    # The embedder read "alpha" before the replacement landed; its result must
    # not certify the new text.
    store.upsert_vectors(
        [node_id],
        "embedding",
        [VECTOR],
        content_fingerprints=[text_fingerprint("alpha")],
        model_fingerprint=MODEL,
    )

    state = chunk_state(store, graph, "doc:1:c0")
    assert state["embedding"] is None
    assert state["content_fingerprint"] is None
    assert unembedded_chunk_ids(store, graph) == {"doc:1:c0"}


def test_chunk_replacement_embeds_only_the_changed_chunk(case) -> None:
    store, graph = case
    writer = CypherGraphWriter(store)
    writer.write_graph_document(
        build_document(chunks=["doc:1:c0", "doc:1:c1"], graph_name=graph)
    )
    certify_chunk(store, graph, "doc:1:c0", "text")
    certify_chunk(store, graph, "doc:1:c1", "text")
    assert unembedded_chunk_ids(store, graph) == set()

    replacement = build_document(chunks=["doc:1:c0", "doc:1:c1"], graph_name=graph)
    replacement.chunks[1].text = "changed"
    writer.write_graph_document(replacement)

    assert chunk_state(store, graph, "doc:1:c0")["embedding"] is not None
    assert chunk_state(store, graph, "doc:1:c1")["embedding"] is None
    assert unembedded_chunk_ids(store, graph) == {"doc:1:c1"}


# --- B02: logical relationship identity and graph ownership ---


def seed_entities(store: GraphStore, graph: str, *ids: str) -> None:
    store.execute_query(
        """
        UNWIND $ids AS id
        CREATE (:__Entity__:Person {id: id, graph_name: $graph_name})
        """,
        {"ids": list(ids), "graph_name": graph},
    )


def seed_relationship(
    store: GraphStore,
    graph: str,
    *,
    rel_id: str | None,
    rel_graph_name: str | None,
    marker: str,
    source_id: str = "e1",
    target_id: str = "e2",
    rel_type: str = "RELATES",
    chunk_ids: tuple[str, ...] = ("seed-chunk",),
) -> None:
    """Create one edge. ``None`` graph_name leaves the property unset."""
    store.execute_query(
        f"""
        MATCH (s:__Entity__ {{id: $source_id, graph_name: $graph_name}})
        MATCH (t:__Entity__ {{id: $target_id, graph_name: $graph_name}})
        CREATE (s)-[:{rel_type} {{
            graph_name: $rel_graph_name,
            id: $rel_id,
            marker: $marker,
            source_chunk_ids: $chunk_ids
        }}]->(t)
        """,
        {
            "graph_name": graph,
            "rel_graph_name": rel_graph_name,
            "rel_id": rel_id,
            "marker": marker,
            "source_id": source_id,
            "target_id": target_id,
            "chunk_ids": list(chunk_ids),
        },
    )


def relationship_edges(store: GraphStore, graph: str) -> list[dict]:
    """Every edge between endpoints of this graph, whatever its own scope."""
    return store.execute_query(
        """
        MATCH (s:__Entity__ {graph_name: $graph_name})-[r]->(t:__Entity__ {graph_name: $graph_name})
        RETURN s.id AS source_id, t.id AS target_id, type(r) AS type,
               r.id AS id, r.graph_name AS rel_graph_name, r.marker AS marker,
               r.source_chunk_ids AS source_chunk_ids
        ORDER BY source_id, target_id
        """,
        {"graph_name": graph},
    )


def local_document(graph: str) -> GraphDocument:
    return build_document(
        chunks=["c0"],
        entities=[("e1", "Alice", None), ("e2", "Bob", None)],
        relationships=[("RELATES", "e1", "e2", None)],
        graph_name=graph,
    )


def test_foreign_relationship_is_not_adopted_by_local_write(case):
    store, graph = case
    seed_entities(store, graph, "e1", "e2")
    seed_relationship(
        store, graph, rel_id="foreign-rel", rel_graph_name="other-graph", marker="foreign"
    )

    CypherGraphWriter(store).write_graph_document(local_document(graph))

    edges = relationship_edges(store, graph)
    foreign = [edge for edge in edges if edge["marker"] == "foreign"]
    assert len(foreign) == 1
    assert foreign[0]["id"] == "foreign-rel"
    assert foreign[0]["rel_graph_name"] == "other-graph"
    local = [edge for edge in edges if edge["rel_graph_name"] == graph]
    assert len(local) == 1


def test_unscoped_relationship_is_not_adopted_by_local_write(case):
    store, graph = case
    seed_entities(store, graph, "e1", "e2")
    seed_relationship(
        store, graph, rel_id="unscoped-rel", rel_graph_name=None, marker="unscoped"
    )

    CypherGraphWriter(store).write_graph_document(local_document(graph))

    edges = relationship_edges(store, graph)
    unscoped = [edge for edge in edges if edge["marker"] == "unscoped"]
    assert len(unscoped) == 1
    assert unscoped[0]["rel_graph_name"] is None
    local = [edge for edge in edges if edge["rel_graph_name"] == graph]
    assert len(local) == 1


def test_duplicate_logical_relationships_fail_before_mutation(case):
    store, graph = case
    seed_entities(store, graph, "e1", "e2")
    seed_relationship(store, graph, rel_id="first", rel_graph_name=graph, marker="first")
    seed_relationship(store, graph, rel_id="second", rel_graph_name=graph, marker="second")

    with pytest.raises(ValueError, match="ambiguous"):
        CypherGraphWriter(store).write_graph_document(local_document(graph))

    edges = relationship_edges(store, graph)
    assert {edge["id"] for edge in edges} == {"first", "second"}
    assert {edge["marker"] for edge in edges} == {"first", "second"}
    assert not store.execute_query(
        "MATCH (c:Chunk {graph_name: $graph_name}) RETURN count(c) AS count",
        {"graph_name": graph},
    )[0]["count"]


def test_existing_logical_relationship_keeps_stored_id(case):
    store, graph = case
    seed_entities(store, graph, "e1", "e2")
    seed_relationship(store, graph, rel_id="existing", rel_graph_name=graph, marker="seeded")

    CypherGraphWriter(store).write_graph_document(local_document(graph))

    edges = relationship_edges(store, graph)
    local = [edge for edge in edges if edge["rel_graph_name"] == graph]
    assert len(local) == 1
    assert local[0]["id"] == "existing"


def test_relationship_id_collision_is_rejected(case):
    store, graph = case
    seed_entities(store, graph, "e1", "e2", "e3")
    # A different logical key already holds the ID the incoming write would use.
    seed_relationship(
        store, graph, rel_id="e1:RELATES:e2", rel_graph_name=graph, marker="holder",
        target_id="e3",
    )

    with pytest.raises(ValueError, match="ambiguous"):
        CypherGraphWriter(store).write_graph_document(local_document(graph))

    edges = relationship_edges(store, graph)
    assert len(edges) == 1
    assert edges[0]["marker"] == "holder"


def test_shared_stored_id_rejects_replacement_before_mutation(case) -> None:
    """A stored ID shared by two logical keys must stop the write first."""
    store, graph = case
    CypherGraphWriter(store).write_graph_document(
        build_document(
            chunks=["doc:1:c0"],
            entities=[("e1", "Alice", None), ("e2", "Acme", None),
                      ("e3", "Beta", None)],
            relationships=[
                ("RELATES", "e1", "e2", None),
                ("RELATES", "e1", "e3", None),
            ],
            graph_name=graph,
        )
    )
    store.execute_query(
        """
        MATCH (:__Entity__ {graph_name: $graph_name})-[r]->(:__Entity__ {graph_name: $graph_name})
        WHERE r.graph_name = $graph_name
        SET r.id = 'shared-id'
        """,
        {"graph_name": graph},
    )
    before = relationship_edges(store, graph)

    replacement = build_document(
        chunks=["doc:1:c0"],
        entities=[("e1", "Alice", None), ("e2", "Acme", None),
                  ("e3", "Beta", None)],
        relationships=[("RELATES", "e1", "e2", None)],
        graph_name=graph,
    )
    replacement = replace(
        replacement,
        relationships=[replace(replacement.relationships[0], id="new-id")],
    )

    with pytest.raises(ValueError, match="ambiguous"):
        CypherGraphWriter(store).write_graph_document(replacement)

    assert relationship_edges(store, graph) == before


def test_omitted_key_duplicate_is_rejected_before_mutation(case) -> None:
    """Two owned edges for one key fail even when the replacement omits it."""
    store, graph = case
    CypherGraphWriter(store).write_graph_document(
        build_document(
            chunks=["doc:1:c0"],
            entities=[("e1", "Alice", None), ("e2", "Acme", None)],
            relationships=[("RELATES", "e1", "e2", None)],
            graph_name=graph,
        )
    )
    seed_relationship(
        store, graph, rel_id="parallel-id", rel_graph_name=graph,
        marker="parallel", chunk_ids=("doc:1:c0",),
    )
    before = relationship_edges(store, graph)
    assert len(before) == 2

    replacement = build_document(
        chunks=["doc:1:c0"],
        entities=[("e1", "Alice", None), ("e2", "Acme", None)],
        relationships=[],
        graph_name=graph,
    )

    with pytest.raises(ValueError, match="ambiguous"):
        CypherGraphWriter(store).write_graph_document(replacement)

    assert relationship_edges(store, graph) == before


@pytest.mark.parametrize("auto_commit", [False, True])
def test_omitted_key_duplicate_is_rejected_on_both_paths(case, auto_commit) -> None:
    """The duplicate preflight runs before any transaction or auto-commit write."""
    store, graph = case
    CypherGraphWriter(store).write_graph_document(
        build_document(
            chunks=["doc:1:c0"],
            entities=[("e1", "Alice", None), ("e2", "Acme", None)],
            relationships=[("RELATES", "e1", "e2", None)],
            graph_name=graph,
        )
    )
    seed_relationship(
        store, graph, rel_id="parallel-id", rel_graph_name=graph,
        marker="parallel", chunk_ids=("doc:1:c0",),
    )
    before = relationship_edges(store, graph)
    target: GraphStore = (
        AutoCommitStore(store) if auto_commit else store  # type: ignore[assignment]
    )

    with pytest.raises(ValueError, match="ambiguous"):
        CypherGraphWriter(target).write_graph_document(
            build_document(
                chunks=["doc:1:c0"],
                entities=[("e1", "Alice", None), ("e2", "Acme", None)],
                relationships=[],
                graph_name=graph,
            )
        )

    assert relationship_edges(store, graph) == before


# --- review F2: a changed domain label must keep embedding certification ---


class _FixedEmbedder:
    """Deterministic embedder; only the text decides nothing, the vector is fixed."""

    model = "fixed-model"

    def embed_query(self, text: str) -> list[float]:
        return list(VECTOR)

    async def async_embed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


def _embed_entities(store: GraphStore, graph: str) -> None:
    embedder = EntityEmbedder(store, _FixedEmbedder(), graph_name=graph)  # type: ignore[arg-type]
    asyncio.run(embedder.embed_entities())


def _uncertified_entity_ids(store: GraphStore, graph: str) -> list[str]:
    return [
        row["id"]
        for row in store.execute_query(
            """
            MATCH (e:__Entity__ {graph_name: $graph_name})
            WHERE e.embedding IS NULL
               OR e.embedding_content_fingerprint IS NULL
               OR e.embedding_input_fingerprint IS NULL
               OR e.embedding_content_fingerprint <> e.embedding_input_fingerprint
            RETURN e.id AS id
            ORDER BY id
            """,
            {"graph_name": graph},
        )
    ]


def test_changed_entity_label_keeps_embedding_certification(case) -> None:
    """Reusing an identity under a new domain type must still embed once."""
    store, graph = case
    writer = CypherGraphWriter(store)
    writer.write_graph_document(
        build_document(
            chunks=["doc:1:c0"],
            entities=[("e1", "Alice", "Alice leads Acme.")],
            graph_name=graph,
        )
    )
    _embed_entities(store, graph)
    assert _uncertified_entity_ids(store, graph) == []

    writer.write_graph_document(
        build_document(
            chunks=["doc:2:c0"],
            entities=[("e1", "Alice", "Alice leads Acme.")],
            document_id="doc:2",
            graph_name=graph,
            entity_type="Organization",
        )
    )

    _embed_entities(store, graph)

    assert _uncertified_entity_ids(store, graph) == []
    rows = store.execute_query(
        """
        MATCH (e:__Entity__ {id: 'e1', graph_name: $graph_name})
        RETURN labels(e) AS labels, e.type AS type, e.embedding IS NOT NULL AS embedded
        """,
        {"graph_name": graph},
    )
    assert len(rows) == 1
    assert rows[0]["embedded"] is True
    assert sorted(rows[0]["labels"]) == ["Organization", "Person", "__Entity__"]


def test_shared_id_across_relationship_types_fails_before_mutation(case):
    """An id owned by another type is still reachable by rebuild and cleanup."""
    store, graph = case
    seed_entities(store, graph, "a", "b", "c", "d")
    seed_relationship(
        store, graph, rel_id="stored", rel_graph_name=graph, marker="t1",
        source_id="a", target_id="b", rel_type="T1",
    )
    seed_relationship(
        store, graph, rel_id="stored", rel_graph_name=graph, marker="t2",
        source_id="c", target_id="d", rel_type="T2",
    )
    before = relationship_edges(store, graph)

    document = build_document(
        chunks=["c0"],
        entities=[("a", "Alice", None), ("b", "Bob", None)],
        relationships=[("T1", "a", "b", None)],
        graph_name=graph,
    )

    with pytest.raises(ValueError, match="ambiguous"):
        CypherGraphWriter(store).write_graph_document(document)

    assert relationship_edges(store, graph) == before


def test_two_physical_edges_sharing_id_and_key_are_rejected(case):
    """Two stored edges with one id and one key leave no safe MERGE target."""
    store, graph = case
    seed_entities(store, graph, "e1", "e2")
    seed_relationship(
        store, graph, rel_id="shared", rel_graph_name=graph, marker="first"
    )
    seed_relationship(
        store, graph, rel_id="shared", rel_graph_name=graph, marker="second"
    )
    before = relationship_edges(store, graph)

    with pytest.raises(ValueError, match="ambiguous"):
        CypherGraphWriter(store).write_graph_document(local_document(graph))

    assert relationship_edges(store, graph) == before


def test_owned_duplicate_with_shared_id_is_rejected_before_mutation(case):
    """Omitted keys still count: both physical rows arrive through ownership."""
    store, graph = case
    seed_entities(store, graph, "e1", "e2")
    CypherGraphWriter(store).write_graph_document(local_document(graph))
    store.execute_query(
        """
        MATCH (s:__Entity__ {id: 'e1', graph_name: $graph_name})
        MATCH (t:__Entity__ {id: 'e2', graph_name: $graph_name})
        MATCH (s)-[r:RELATES]->(t)
        CREATE (s)-[:RELATES {
            graph_name: r.graph_name,
            id: r.id,
            source_chunk_ids: r.source_chunk_ids
        }]->(t)
        """,
        {"graph_name": graph},
    )
    before = relationship_edges(store, graph)

    replacement = build_document(
        chunks=["c1"],
        entities=[("e1", "Alice", None), ("e2", "Bob", None)],
        graph_name=graph,
    )

    with pytest.raises(ValueError, match="ambiguous"):
        CypherGraphWriter(store).write_graph_document(replacement)

    assert relationship_edges(store, graph) == before


def test_unscoped_parallel_edge_does_not_block_owned_write(case):
    """An edge the writer does not own must not stop a valid key write."""
    store, graph = case
    seed_entities(store, graph, "e1", "e2")
    CypherGraphWriter(store).write_graph_document(local_document(graph))
    seed_relationship(
        store, graph, rel_id="legacy", rel_graph_name=None, marker="unscoped"
    )

    replacement = build_document(
        chunks=["c1"],
        entities=[("e1", "Alice", None), ("e2", "Bob", None)],
        relationships=[("RELATES", "e1", "e2", None)],
        graph_name=graph,
    )
    CypherGraphWriter(store).write_graph_document(replacement)

    edges = relationship_edges(store, graph)
    owned = [edge for edge in edges if edge["rel_graph_name"] == graph]
    unscoped = [edge for edge in edges if edge["marker"] == "unscoped"]
    assert len(owned) == 1
    assert owned[0]["source_chunk_ids"] == ["c1"]
    assert len(unscoped) == 1
    assert unscoped[0]["rel_graph_name"] is None


def test_reference_only_endpoints_are_checked_for_ambiguous_entities(case):
    """A relationship alone can address an ambiguous entity identity."""
    store, graph = case
    store.execute_query(
        """
        CREATE (:__Entity__:Person {id: 'a', graph_name: $graph_name, uid: 'a-1'}),
               (:__Entity__:Person {id: 'a', graph_name: $graph_name, uid: 'a-2'}),
               (:__Entity__:Person {id: 'b', graph_name: $graph_name, uid: 'b-1'})
        """,
        {"graph_name": graph},
    )

    document = build_document(
        chunks=["c0"],
        entities=[],
        relationships=[("RELATES", "a", "b", None)],
        graph_name=graph,
    )

    with pytest.raises(ValueError, match="ambiguous"):
        CypherGraphWriter(store).write_graph_document(document)

    assert relationship_edges(store, graph) == []


def test_omitted_key_duplicate_owned_by_another_document_is_rejected(case):
    """The id read sees a same-id edge that another document owns."""
    store, graph = case
    seed_entities(store, graph, "e1", "e2")
    CypherGraphWriter(store).write_graph_document(local_document(graph))
    store.execute_query(
        """
        MATCH (s:__Entity__ {id: 'e1', graph_name: $graph_name})
        MATCH (t:__Entity__ {id: 'e2', graph_name: $graph_name})
        MATCH (s)-[r:RELATES]->(t)
        CREATE (s)-[:RELATES {
            graph_name: r.graph_name,
            id: r.id,
            source_chunk_ids: ['doc:2:c0']
        }]->(t)
        """,
        {"graph_name": graph},
    )
    before = relationship_edges(store, graph)

    replacement = build_document(
        chunks=["c1"],
        entities=[("e1", "Alice", None), ("e2", "Bob", None)],
        graph_name=graph,
    )

    with pytest.raises(ValueError, match="ambiguous"):
        CypherGraphWriter(store).write_graph_document(replacement)

    assert relationship_edges(store, graph) == before


# --- Cross-component plumbing follow-up ---


def test_retained_claim_replaces_links_when_old_endpoints_survive(case):
    store, graph = case
    writer = CypherGraphWriter(store)
    writer.write_graph_document(build_document(
        chunks=["doc:1:c0", "doc:1:c1"],
        entities=[("e1", "Alice", None), ("e2", "Bob", None)], graph_name=graph,
        claims=[("claim:move", "e1", "doc:1:c0")]))
    replacement = build_document(
        chunks=["doc:1:c0", "doc:1:c1"],
        entities=[("e1", "Alice", None), ("e2", "Bob", None)], graph_name=graph,
        claims=[("claim:move", "e2", "doc:1:c1")])
    for _ in range(2):
        assert writer.write_graph_document(replacement)["claims"] == 1
        state = claim_graph(store, graph, "claim:move")
        assert state["subjects"] == ["e2"]
        assert state["source_chunks"] == ["doc:1:c1"]


@pytest.mark.parametrize("edge_graph,cross_graph", [
    (None, False), (None, True), ("foreign", False), ("foreign", True), ("owned", True),
])
def test_resolution_rejects_unsafe_incident_edges_without_mutation(case, edge_graph, cross_graph):
    store, graph = case
    if edge_graph == "owned":
        edge_graph = graph
    store.execute_query(
        """
        CREATE (a:__Entity__:Person {id: 'a', graph_name: $graph_name, name: 'Alice'}),
               (b:__Entity__:Person {id: 'b', graph_name: $graph_name, name: 'Alice'}),
               (t:__Entity__:Person {id: 'target', graph_name: $target_graph})
        CREATE (b)-[:KNOWS {graph_name: $edge_graph, marker: 'unchanged'}]->(t)
        """, {"graph_name": graph, "target_graph": graph + '-foreign' if cross_graph else graph,
               "edge_graph": edge_graph})
    try:
        before = store.execute_query(
            "MATCH (e:__Entity__ {graph_name: $graph_name}) RETURN e.id AS id, properties(e) AS props ORDER BY id",
            {"graph_name": graph})
        with pytest.raises(ValueError, match="foreign or unscoped"):
            asyncio.run(store.resolve_entities_exact(graph_name=graph))
        assert before == store.execute_query(
            "MATCH (e:__Entity__ {graph_name: $graph_name}) RETURN e.id AS id, properties(e) AS props ORDER BY id",
            {"graph_name": graph})
        rows = store.execute_query(
            "MATCH (b:__Entity__ {id: 'b', graph_name: $graph_name})-[r:KNOWS]->(t) RETURN properties(r) AS props",
            {"graph_name": graph})
        assert rows[0]["props"].get("graph_name") == edge_graph
        assert rows[0]["props"]["marker"] == "unchanged"
    finally:
        cleanup_graph(store, graph + '-foreign')


def test_resolution_preserves_legacy_structural_edges(case):
    store, graph = case
    CypherGraphWriter(store).write_graph_document(build_document(
        chunks=["doc:1:c0"], entities=[("a", "Alice", None), ("b", "Alice", None)],
        claims=[("claim:1", "b", "doc:1:c0")], graph_name=graph))
    result = asyncio.run(store.resolve_entities_exact(graph_name=graph))
    assert result["merged_nodes"] == 2
    assert len(claim_graph(store, graph, "claim:1")["subjects"]) == 1
    assert store.execute_query(
        "MATCH (ch:Chunk {graph_name: $graph_name})-[:FROM_CHUNK]->(e:__Entity__) RETURN count(e) AS count",
        {"graph_name": graph})[0]["count"] == 1


def test_legacy_vector_rejects_concurrent_type_change(case):
    from recon_graphrag.entity_text import entity_embedding_fingerprint

    store, graph = case
    id_func = "elementId" if type(store).__name__ == "Neo4jGraphStore" else "id"
    rows = store.execute_query(
        f"""CREATE (e:__Entity__:Person:Organization {{id: 'a', graph_name: $graph_name,
            type: 'Person', name: 'Alice', description: 'same'}}) RETURN {id_func}(e) AS id""",
        {"graph_name": graph})
    node_id = rows[0]["id"]
    execute = store.execute_query

    def interleaved(query, parameters=None):
        result = execute(query, parameters)
        if "n.type AS type" in query and "embedding_input_fingerprint IS NULL" in query:
            execute("MATCH (e:__Entity__ {graph_name: $graph_name}) SET e.type = 'Organization'", {"graph_name": graph})
        return result

    store.execute_query = interleaved
    try:
        fingerprint = entity_embedding_fingerprint(labels=["Person"], name="Alice", description="same", entity_type="Person")
        store.upsert_vectors([node_id], "embedding", [[0.1] * INDEX_DIM], content_fingerprints=[fingerprint])
    finally:
        store.execute_query = execute
    assert execute("MATCH (e:__Entity__ {graph_name: $graph_name}) RETURN e.embedding AS vector", {"graph_name": graph})[0]["vector"] is None


@pytest.mark.parametrize("mode", ["local", "drift"])
def test_retrieval_uses_recorded_type_and_only_owned_edges(case, mode):
    store, graph = case
    id_func = "elementId" if type(store).__name__ == "Neo4jGraphStore" else "id"
    rows = store.execute_query(
        f"""
        CREATE (a:__Entity__:Person:Organization {{id: 'a', name: 'Acme', type: 'Organization', graph_name: $graph_name}}),
               (b:__Entity__:Person {{id: 'b', name: 'Bob', type: 'Person', graph_name: $graph_name}})
        CREATE (a)-[:OWNED {{graph_name: $graph_name}}]->(b),
               (a)-[:UNSCOPED]->(b),
               (b)-[:FOREIGN {{graph_name: 'foreign'}}]->(a)
        RETURN {id_func}(a) AS id
        """, {"graph_name": graph})
    context = store.fetch_entity_context([{"id": rows[0]["id"], "score": 1.0}], mode=mode, graph_name=graph)
    assert context
    for row in context:
        assert row["title"] == "Acme (Organization)"
        for rel in row["relationship_records"]:
            assert rel["rel"] == "OWNED"
            assert rel["entity"] == "Organization: Acme"
    assert any(row["relationship_records"] for row in context)


def test_community_retrieval_filters_relationship_scope(case):
    store, graph = case
    store.execute_query(
        """
        CREATE (c:Community {id: 'community', graph_name: $graph_name, level: 0}),
               (a:__Entity__ {id: 'a', graph_name: $graph_name}),
               (b:__Entity__ {id: 'b', graph_name: $graph_name})
        CREATE (a)-[:IN_COMMUNITY]->(c), (b)-[:IN_COMMUNITY]->(c),
               (a)-[:OWNED {graph_name: $graph_name}]->(b),
               (a)-[:UNSCOPED]->(b), (a)-[:FOREIGN {graph_name: 'foreign'}]->(b)
        """, {"graph_name": graph})
    members = store.fetch_community_member_graph(graph, 'community', 0)
    assert members[0]["edge_count"] == 1
    assert [edge["type"] for edge in members[0]["edges"]] == ["OWNED"]
    context = store.get_community_ranked_context(graph, 'community', 0)
    assert [row["rel_type"] for row in context if row["rel_type"]] == ["OWNED"]


@pytest.mark.parametrize("edge_graph", [None, "foreign"])
@pytest.mark.parametrize("incoming", [False, True])
def test_resolution_rejects_foreign_collision_at_canonical_endpoint(case, edge_graph, incoming):
    store, graph = case
    unsafe_edge = "(t)-[:KNOWS {graph_name: $edge_graph, marker: 'unsafe'}]->(a)" if incoming else "(a)-[:KNOWS {graph_name: $edge_graph, marker: 'unsafe'}]->(t)"
    owned_edge = "(t)-[:KNOWS {graph_name: $graph_name, marker: 'owned'}]->(b)" if incoming else "(b)-[:KNOWS {graph_name: $graph_name, marker: 'owned'}]->(t)"
    store.execute_query(
        f"""
        CREATE (a:__Entity__:Person {{id: 'a', name: 'Alice', graph_name: $graph_name}}),
               (b:__Entity__:Person {{id: 'b', name: 'Alice', graph_name: $graph_name}}),
               (t:__Entity__:Person {{id: 't', name: 'Target', graph_name: $graph_name}})
        CREATE {unsafe_edge}, {owned_edge}
        """, {"graph_name": graph, "edge_graph": edge_graph})
    query = """MATCH (s:__Entity__ {graph_name: $graph_name})-[r]->(t)
               RETURN s.id AS source, t.id AS target, properties(r) AS props ORDER BY source, target"""
    before = store.execute_query(query, {"graph_name": graph})
    with pytest.raises(ValueError, match="foreign or unscoped"):
        asyncio.run(store.resolve_entities_exact(graph_name=graph))
    assert store.execute_query(query, {"graph_name": graph}) == before


def test_claim_link_replacement_preserves_foreign_links(case):
    store, graph = case
    writer = CypherGraphWriter(store)
    original = build_document(
        chunks=["doc:1:c0", "doc:1:c1"], entities=[("e1", "Alice", None), ("e2", "Bob", None)],
        claims=[("claim:1", "e1", "doc:1:c0")], graph_name=graph)
    writer.write_graph_document(original)
    store.execute_query(
        """MATCH (c:Claim {id: 'claim:1', graph_name: $graph_name}),
                 (e:__Entity__ {id: 'e1', graph_name: $graph_name})
           CREATE (c)-[:SUBJECT_OF {graph_name: 'foreign', marker: 'preserve'}]->(e)
        """, {"graph_name": graph})
    foreign_graph = graph + '-foreign'
    store.execute_query(
        """MATCH (c:Claim {id: 'claim:1', graph_name: $graph_name})
           CREATE (foreign:Chunk {id: 'foreign', graph_name: $foreign_graph}),
                  (c)-[:SOURCED_FROM {marker: 'preserve'}]->(foreign)
        """, {"graph_name": graph, "foreign_graph": foreign_graph})
    try:
        writer.write_graph_document(build_document(
            chunks=["doc:1:c0", "doc:1:c1"], entities=[("e1", "Alice", None), ("e2", "Bob", None)],
            claims=[("claim:1", "e2", "doc:1:c1")], graph_name=graph))
        rows = store.execute_query(
            """MATCH (c:Claim {id: 'claim:1', graph_name: $graph_name})-[r]->(endpoint)
               RETURN type(r) AS type, r.marker AS marker, endpoint.id AS endpoint_id,
                      r.graph_name AS relationship_graph, endpoint.graph_name AS endpoint_graph""",
            {"graph_name": graph})
        preserved = [row for row in rows if row["marker"] == 'preserve']
        assert len(preserved) == 2
        assert any(row["endpoint_id"] == 'e1' and row["relationship_graph"] == 'foreign' for row in preserved)
        assert any(row["endpoint_id"] == 'foreign' and row["endpoint_graph"] == foreign_graph for row in preserved)
        assert {(row["type"], row["endpoint_id"]) for row in rows if row["marker"] is None} == {
            ('SUBJECT_OF', 'e2'), ('SOURCED_FROM', 'doc:1:c1')}
    finally:
        cleanup_graph(store, foreign_graph)


def test_claim_replacement_does_not_adopt_foreign_links_at_current_endpoints(case):
    store, graph = case
    writer = CypherGraphWriter(store)
    document = build_document(
        chunks=["doc:1:c0"], entities=[("e1", "Alice", None)],
        claims=[("claim:1", "e1", "doc:1:c0")], graph_name=graph,
    )
    writer.write_graph_document(document)
    store.execute_query(
        """MATCH (c:Claim {id: 'claim:1', graph_name: $graph_name}),
                 (e:__Entity__ {id: 'e1', graph_name: $graph_name}),
                 (ch:Chunk {id: 'doc:1:c0', graph_name: $graph_name})
           UNWIND [1, 2] AS occurrence
           CREATE (c)-[:SUBJECT_OF {graph_name: 'foreign', marker: occurrence}]->(e),
                  (c)-[:SOURCED_FROM {graph_name: 'foreign', marker: occurrence}]->(ch)
        """, {"graph_name": graph},
    )
    for _ in range(2):
        assert writer.write_graph_document(document)["claims"] == 1
        rows = store.execute_query(
            """MATCH (c:Claim {id: 'claim:1', graph_name: $graph_name})-[r]->(endpoint)
               RETURN type(r) AS type, r.graph_name AS owner, r.marker AS marker,
                      endpoint.id AS endpoint ORDER BY type, owner, marker""",
            {"graph_name": graph},
        )
        assert len(rows) == 6
        assert len([row for row in rows if row["owner"] == graph]) == 2
        assert sorted(row["marker"] for row in rows if row["owner"] == "foreign") == [1, 1, 2, 2]


def test_identical_claim_inputs_preserve_receipt_multiplicity(case):
    store, graph = case
    document = build_document(
        chunks=["doc:1:c0"], entities=[("e1", "Alice", None)],
        claims=[("claim:1", "e1", "doc:1:c0")], graph_name=graph,
    )
    document.claims.append(document.claims[0])
    writer = CypherGraphWriter(store)
    for _ in range(2):
        assert writer.write_graph_document(document)["claims"] == 2
        rows = store.execute_query(
            """MATCH (c:Claim {id: 'claim:1', graph_name: $graph_name})-[r]->()
               RETURN type(r) AS type, r.graph_name AS owner ORDER BY type""",
            {"graph_name": graph},
        )
        assert rows == [
            {"type": "SOURCED_FROM", "owner": graph},
            {"type": "SUBJECT_OF", "owner": graph},
        ]
