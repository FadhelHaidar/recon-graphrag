"""Opt-in persistence regressions: real databases, no AI/provider calls."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from neo4j.exceptions import Neo4jError

from recon_graphrag.embeddings.base import BaseEmbedder
from recon_graphrag.embeddings.entities import EntityEmbedder
from recon_graphrag.entity_text import entity_embedding_fingerprint
from recon_graphrag.extraction.assembler import GraphDocumentAssembler
from recon_graphrag.extraction.chunking import TextChunk
from recon_graphrag.extraction.structured import ColumnEntity, RowMapping
from recon_graphrag.extraction.types import ExtractedNode, GraphExtraction
from recon_graphrag.graphdb.neo4j.store import Neo4jGraphStore
from recon_graphrag.pipelines.graphrag_pipeline import GraphBuilderPipeline
from recon_graphrag.pipelines.writer import CypherGraphWriter
from tests.integration.factories import get_memgraph_store, get_neo4j_store
from tests.integration.support import cleanup_graph, require_integration_env
from tests.pipelines.test_writer_replacement import build_document

pytestmark = [pytest.mark.integration, pytest.mark.database]


@pytest.fixture(scope="module", params=["neo4j", "memgraph"])
def backend(request):
    prefix = request.param.upper()
    require_integration_env(
        "RUN_DATABASE_INTEGRATION_TESTS",
        [f"{prefix}_URL"],
        "replacement/identity persistence regressions",
        fail_on_missing=True,
    )
    store = get_neo4j_store() if request.param == "neo4j" else get_memgraph_store()
    store.create_indexes()
    try:
        yield store
    finally:
        store.driver.close()


@pytest.fixture
def case(backend):
    graph = f"blocker-regression-{uuid4().hex}"
    try:
        yield backend, graph
    finally:
        cleanup_graph(backend, graph)
        cleanup_graph(backend, graph + "-other")


def read_entities(store, graph):
    return store.execute_query(
        "MATCH (n:__Entity__ {graph_name:$g}) RETURN properties(n) AS props",
        {"g": graph},
    )


def read_edges(store, graph):
    return store.execute_query(
        "MATCH (:__Entity__ {graph_name:$g})-[r:WORKS_AT]->() RETURN properties(r) AS props",
        {"g": graph},
    )


@pytest.mark.parametrize("reverse", [False, True], ids=["outgoing", "incoming"])
def test_resolve_then_replace_preserves_ownership_strength_and_uid(case, reverse):
    store, graph = case
    writer = CypherGraphWriter(store)
    for doc, person, strength in [("A", "p1", 9.0), ("B", "p2", 3.0)]:
        source, target = ("target", person) if reverse else (person, "target")
        document = build_document(
            chunks=[doc + ":c0"], document_id=doc, graph_name=graph,
            entities=[(person, "Alice", doc + " text"), ("target", "Acme", "company")],
            relationships=[("WORKS_AT", source, target, "identical relationship text")],
        )
        document.relationships[0].strength = strength
        writer.write_graph_document(document)

    result = asyncio.run(store.resolve_entities_normalized(graph_name=graph))
    assert result["merged_groups"] == 1 and not result["skipped"]
    person = next(row["props"] for row in read_entities(store, graph)
                  if row["props"]["id"] != "target")
    assert person["uid"] == f"{len(graph)}:{graph}:{person['id']}"
    assert {tuple(json.loads(value)) for value in person["description_sources"]} == {
        ("A", "A text"), ("B", "B text")}
    edge = read_edges(store, graph)[0]["props"]
    assert {tuple(json.loads(value)) for value in edge["description_sources"]} == {
        ("A", "identical relationship text"), ("B", "identical relationship text")}

    with pytest.raises(Neo4jError, match="(?i)constraint|unique|violat"):
        store.execute_query(
            "CREATE (:__Entity__:Person {graph_name:$g,id:$id,uid:$uid})",
            {"g": graph, "id": person["id"], "uid": person["uid"]},
        )
    store.execute_query(
        "MATCH (n:__Entity__ {graph_name:$g,id:$id}) "
        "SET n.description='obsolete A+B summary',n.description_summary_status='success'",
        {"g": graph, "id": person["id"]},
    )
    writer.write_graph_document(build_document(chunks=["A:c0"], document_id="A", graph_name=graph))
    person = next(row["props"] for row in read_entities(store, graph)
                  if row["props"]["id"] != "target")
    assert person["description"] == "B text"
    assert person["descriptions"] == ["B text"]
    assert person.get("description_summary_status") is None
    assert [json.loads(value) for value in person["description_sources"]] == [["B", "B text"]]
    edge = read_edges(store, graph)[0]["props"]
    assert edge["description"] == "identical relationship text"
    assert [json.loads(value) for value in edge["description_sources"]] == [
        ["B", "identical relationship text"]]
    assert edge["source_chunk_ids"] == ["B:c0"]
    assert edge["observation_count"] == 1 and edge["weight"] == 1.0
    assert edge["strength"] == 3.0
    assert [json.loads(value) for value in edge["strength_sources"]] == [["B", 3.0]]


def test_shorter_document_replaces_chunks_evidence_claims_and_orphans(case):
    store, graph = case
    writer = CypherGraphWriter(store)
    writer.write_graph_document(build_document(
        chunks=["A:c0", "A:c1"], document_id="A", graph_name=graph,
        entities=[("p", "Alice", "A text"), ("target", "Acme", "company"),
                  ("orphan", "Only A", "obsolete")],
        relationships=[("WORKS_AT", "p", "target", "A edge")],
        evidence=[("A:c0", "p"), ("A:c1", "p"), ("A:c1", "orphan"), ("A:c0", "target")],
        claims=[("old-A", "p", "A:c1")],
    ))
    writer.write_graph_document(build_document(
        chunks=["B:c0"], document_id="B", graph_name=graph,
        entities=[("p", "Alice", "B text"), ("target", "Acme", "company")],
        relationships=[("WORKS_AT", "p", "target", "B edge")],
        claims=[("B-claim", "p", "B:c0")],
    ))
    writer.write_graph_document(build_document(
        chunks=["A:c0"], document_id="A", graph_name=graph,
        entities=[("p", "Alice", "updated A text")],
        claims=[("new-A", "p", "A:c0")],
    ))
    chunks = store.execute_query("MATCH (n:Chunk {graph_name:$g}) RETURN n.id AS id", {"g": graph})
    assert {row["id"] for row in chunks} == {"A:c0", "B:c0"}
    assert {row["props"]["id"] for row in read_entities(store, graph)} == {"p", "target"}
    claims = store.execute_query("MATCH (n:Claim {graph_name:$g}) RETURN n.id AS id", {"g": graph})
    assert {row["id"] for row in claims} == {"new-A", "B-claim"}
    assert store.execute_query(
        "MATCH (n:Chunk {graph_name:$g,id:'A:c1'})--() RETURN count(n) AS count", {"g": graph}
    )[0]["count"] == 0
    person = next(row["props"] for row in read_entities(store, graph) if row["props"]["id"] == "p")
    assert set(person["descriptions"]) == {"B text", "updated A text"}
    edge = read_edges(store, graph)[0]["props"]
    assert edge["descriptions"] == ["B edge"]
    assert edge["source_chunk_ids"] == ["B:c0"]
    assert edge["weight"] == edge["observation_count"] == 1


def test_resolved_entities_can_be_embedded_without_description_summarization(case):
    store, graph = case
    writer = CypherGraphWriter(store)
    for doc, entity_id, description in [("A", "p1", "A text"), ("B", "p2", "B text")]:
        writer.write_graph_document(build_document(
            chunks=[doc + ":c0"], document_id=doc, graph_name=graph,
            entities=[(entity_id, "Alice", description)],
        ))
    asyncio.run(store.resolve_entities_normalized(graph_name=graph))
    provider = MagicMock(spec=BaseEmbedder)
    provider.model = "local-regression"
    provider.async_embed_query = AsyncMock(return_value=[0.2] * 1536)
    embedder = EntityEmbedder(store, provider, graph_name=graph)
    asyncio.run(embedder.embed_entities())
    assert provider.async_embed_query.await_count == 1
    assert store.get_unembedded_entities(graph_name=graph, model_fingerprint=embedder.model_fingerprint) == []
    node = read_entities(store, graph)[0]["props"]
    assert len(node["embedding"]) == 1536
    assert node["embedding_input_fingerprint"] == node["embedding_content_fingerprint"]
    assert node["embedding_model_fingerprint"] == embedder.model_fingerprint


def test_replacing_retained_edge_can_lower_or_remove_strength(case):
    store, graph = case
    writer = CypherGraphWriter(store)

    def write(doc, strength):
        document = build_document(
            chunks=[doc + ":c0"], document_id=doc, graph_name=graph,
            entities=[("p", "Alice", "person"), ("target", "Acme", "company")],
            relationships=[("WORKS_AT", "p", "target", "relationship")],
        )
        document.relationships[0].strength = strength
        writer.write_graph_document(document)

    write("A", 9.0)
    write("B", 3.0)
    write("A", 2.0)
    assert read_edges(store, graph)[0]["props"]["strength"] == 3.0
    writer.write_graph_document(build_document(chunks=["B:c0"], document_id="B", graph_name=graph))
    assert read_edges(store, graph)[0]["props"]["strength"] == 2.0
    write("A", None)
    assert read_edges(store, graph)[0]["props"].get("strength") is None


def test_identical_legacy_description_is_not_adopted(case):
    store, graph = case
    store.execute_query(
        "CREATE (:__Entity__:Person {graph_name:$g,id:'legacy',name:'Alice',"
        "description:'same text',descriptions:['same text']})", {"g": graph},
    )
    with pytest.raises(ValueError, match="(?i)ownership|provenance|re-ingest"):
        CypherGraphWriter(store).write_graph_document(build_document(
            chunks=["A:c0"], document_id="A", graph_name=graph,
            entities=[("legacy", "Alice", "same text")],
        ))
    assert store.execute_query(
        "MATCH (n:Document {graph_name:$g}) RETURN count(n) AS count", {"g": graph}
    )[0]["count"] == 0
    assert read_entities(store, graph)[0]["props"].get("description_sources") is None


def create_vector_entity(store, graph):
    id_function = "elementId" if isinstance(store, Neo4jGraphStore) else "id"
    row = store.execute_query(
        "CREATE (n:__Entity__:Person {graph_name:$g,id:'vector',name:'Alice',description:'A'}) "
        f"RETURN {id_function}(n) AS id, labels(n) AS labels", {"g": graph},
    )[0]
    return row["id"], row["labels"]


def fingerprint(labels, description):
    return entity_embedding_fingerprint(labels=labels, name="Alice", description=description)


@pytest.mark.parametrize("null_expected", [False, True], ids=["writer", "summary-null"])
def test_vector_invalidation_fresh_write_and_uncertified_legacy_write(case, null_expected):
    store, graph = case
    node_id, labels = create_vector_entity(store, graph)
    fetched_a, current_b = fingerprint(labels, "A"), fingerprint(labels, "B")
    store.execute_query(
        "MATCH (n:__Entity__ {graph_name:$g}) "
        "SET n.description='B',n.embedding_input_fingerprint=$expected",
        {"g": graph, "expected": None if null_expected else current_b},
    )
    store.upsert_vectors([node_id], "embedding", [[0.1] * 1536],
                         content_fingerprints=[fetched_a], model_fingerprint="old-model")
    assert read_entities(store, graph)[0]["props"].get("embedding") is None
    store.upsert_vectors([node_id], "embedding", [[0.2] * 1536],
                         content_fingerprints=[current_b], model_fingerprint="new-model")
    node = read_entities(store, graph)[0]["props"]
    assert node["embedding_content_fingerprint"] == node["embedding_input_fingerprint"] == current_b
    assert node["embedding_model_fingerprint"] == "new-model"
    store.upsert_vectors([node_id], "embedding", [[0.3] * 1536])
    node = read_entities(store, graph)[0]["props"]
    assert len(node["embedding"]) == 1536
    assert node["embedding_input_fingerprint"] == current_b
    assert node.get("embedding_content_fingerprint") is None
    assert node.get("embedding_model_fingerprint") is None


def test_null_vector_snapshot_is_rechecked_in_write(case):
    store, graph = case
    node_id, labels = create_vector_entity(store, graph)
    original = store.execute_query
    interleaved = False

    def mutate_after_snapshot(query, parameters=None):
        nonlocal interleaved
        result = original(query, parameters)
        if "AS labels" in query and "embedding_input_fingerprint IS NULL" in query:
            original("MATCH (n:__Entity__ {graph_name:$g}) SET n.description='B'", {"g": graph})
            interleaved = True
        return result

    store.execute_query = mutate_after_snapshot
    try:
        store.upsert_vectors([node_id], "embedding", [[0.2] * 1536],
                             content_fingerprints=[fingerprint(labels, "A")], model_fingerprint="old")
    finally:
        store.execute_query = original
    assert interleaved
    node = read_entities(store, graph)[0]["props"]
    assert node["description"] == "B" and node.get("embedding") is None
    assert node.get("embedding_input_fingerprint") is None


def test_neo4j_vector_guard_reads_after_waiting_for_writer_lock(case):
    store, graph = case
    if not isinstance(store, Neo4jGraphStore):
        pytest.skip("Neo4j lock-before-predicate regression; Memgraph surfaces write conflicts")
    node_id, labels = create_vector_entity(store, graph)
    fetched = fingerprint(labels, "A")
    store.execute_query(
        "MATCH (n:__Entity__ {graph_name:$g}) SET n.embedding_input_fingerprint=$fp",
        {"g": graph, "fp": fetched},
    )
    store.upsert_vectors([node_id], "embedding", [[0.1] * 1536],
                         content_fingerprints=[fetched], model_fingerprint="first")
    with store.driver.session() as session, ThreadPoolExecutor(max_workers=1) as pool:
        tx = session.begin_transaction()
        try:
            tx.run("MATCH (n:__Entity__ {graph_name:$g}) "
                   "SET n.description='B',n.embedding_input_fingerprint='newer'", g=graph).consume()
            future = pool.submit(store.upsert_vectors, [node_id], "embedding", [[0.2] * 1536],
                                 content_fingerprints=[fetched], model_fingerprint="stale")
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                statuses = store.execute_query("SHOW TRANSACTIONS YIELD currentQuery, status "
                    "WHERE currentQuery CONTAINS 'setNodeVectorProperty' RETURN status")
                if any("Blocked" in row["status"] for row in statuses):
                    break
                if future.done():
                    future.result()
                    pytest.fail("Vector write unexpectedly finished while writer held its lock")
                time.sleep(0.01)
            else:
                pytest.fail("Vector query did not reach the held writer lock")
        finally:
            tx.commit()
        future.result(timeout=10)
    node = read_entities(store, graph)[0]["props"]
    assert node["embedding_input_fingerprint"] == "newer"
    assert node["embedding_model_fingerprint"] == "first"
    assert node["embedding_content_fingerprint"] == fetched


def test_chunk_local_ids_do_not_bypass_conflict_aware_resolution(case):
    store, graph = case
    observations = [("Alice", "1970-01-01"), ("Bob", "1980-01-01"),
                    ("Alex", "1990-01-01"), ("Alex", "2000-01-01")]
    chunks = [TextChunk(id=f"A:c{i}", text=name, index=i, metadata={})
              for i, (name, _) in enumerate(observations)]
    extractions = {
        chunk.id: GraphExtraction(nodes=[ExtractedNode(
            id="p1", label="Person", properties={"name": name, "birth_date": birth_date},
        )], relationships=[])
        for chunk, (name, birth_date) in zip(chunks, observations, strict=True)
    }
    document = GraphDocumentAssembler().assemble(
        document_id="A", text_hash="test", chunks=chunks, chunk_extractions=extractions,
        metadata={}, graph_name=graph,
    )
    assert len(document.entities) == 4
    CypherGraphWriter(store).write_graph_document(document)
    result = asyncio.run(store.resolve_entities_normalized(
        graph_name=graph, conflict_properties=["birth_date"],
    ))
    assert result["merged_groups"] == 0
    assert len(read_entities(store, graph)) == 4
    assert sorted(row["props"]["name"] for row in read_entities(store, graph)) == [
        "Alex", "Alex", "Alice", "Bob"]


async def test_merge_retargets_claim_objects_outside_merge_group_and_not_other_graphs(case):
    store, graph = case
    store.execute_query(
        "CREATE (:__Entity__:Person {graph_name:$g,id:'subject',name:'Reporter'}) "
        "CREATE (:__Entity__:Person {graph_name:$g,id:'a',name:'Alice'}) "
        "CREATE (:__Entity__:Person {graph_name:$g,id:'b',name:'ALICE'})",
        {"g": graph},
    )
    for scope in (graph, graph + "-other"):
        store.execute_query(
            "UNWIND ['a', 'b', 'unrelated'] AS object_id "
            "CREATE (c:Claim {graph_name:$scope,id:object_id,object_entity_id:object_id}) "
            "WITH c MATCH (s:__Entity__ {graph_name:$g,id:'subject'}) "
            "CREATE (s)-[:SUBJECT_OF]->(c)",
            {"g": graph, "scope": scope},
        )

    result = await store.resolve_entities_normalized(graph_name=graph)
    assert not result["skipped"] and result["merged_groups"] == 1
    entities = read_entities(store, graph)
    assert len(entities) == 2
    canonical = next(row["props"]["id"] for row in entities
                     if row["props"]["id"] != "subject")
    for scope in (graph, graph + "-other"):
        claims = store.execute_query(
            "MATCH (s:__Entity__ {graph_name:$g,id:'subject'})-[:SUBJECT_OF]->"
            "(c:Claim {graph_name:$scope}) "
            "RETURN c.id AS id,c.object_entity_id AS object_id",
            {"g": graph, "scope": scope},
        )
        expected = {"a": canonical, "b": canonical, "unrelated": "unrelated"}
        if scope != graph:
            expected = {key: key for key in expected}
        assert {row["id"]: row["object_id"] for row in claims} == expected


async def test_structured_identities_conflicts_and_schema_lists_survive_write(case):
    store, graph = case
    pipeline = GraphBuilderPipeline(
        graph_store=store, graph_name=graph, llm=MagicMock(), embedder=MagicMock(),
        perform_entity_resolution=False, embed_entities=False,
        summarize_descriptions=False,
    )
    mapping = RowMapping([ColumnEntity(
        "Person", "key", {"name": "name", "contact": "contact", "tags": "tags"},
        types={"record_key": "INTEGER", "tags": "LIST"}, identity_property="record_key",
    )])
    result = await pipeline.build_from_rows([
        {"key": "01", "name": "Alice", "contact": "one", "tags": ["a"]},
        {"key": "1", "name": "Alice", "contact": "two", "tags": ["b"]},
        {"key": "2", "name": "Alice", "contact": "three", "tags": ["c"]},
    ], mapping, metadata={"source": "live-identity-regression"}, finalize=False)
    people = {row["props"]["record_key"]: row["props"] for row in read_entities(store, graph)}
    assert set(people) == {1, 2}
    assert people[1]["canonical_key"] == "Person:1"
    assert people[1]["contact"] == "one"
    assert people[1]["tags"] == ["a", "b"]
    assert people[2]["canonical_key"] == "Person:2"
    assert people[2]["contact"] == "three"
    assert people[2]["tags"] == ["c"]
    assert result["extraction"]["assembly_review_groups"][0]["decision"] == "identity_retained"
    original_ids = {key: person["id"] for key, person in people.items()}
    await pipeline.build_from_rows([
        {"key": "1", "name": "Alice", "contact": "one", "tags": ["a"]},
    ], mapping, metadata={"source": "another-live-document"}, finalize=False)
    assert {row["props"]["record_key"]: row["props"]["id"]
            for row in read_entities(store, graph)} == original_ids


def test_concurrent_identical_identity_ingestion_and_graph_isolation(case):
    store, graph = case
    barrier = threading.Barrier(2)

    def ingest():
        barrier.wait(timeout=10)
        try:
            CypherGraphWriter(store).write_graph_document(build_document(
                chunks=["same:c0"], document_id="same", graph_name=graph,
                entities=[("same", "Alice", "same text")],
            ))
        except Neo4jError as exc:
            assert any(word in str(exc).lower() for word in
                       ("constraint", "unique", "conflict", "deadlock", "serialization")), str(exc)
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert any(list(pool.map(lambda _: ingest(), range(2))))
    CypherGraphWriter(store).write_graph_document(build_document(
        chunks=["same:c0"], document_id="same", graph_name=graph + "-other",
        entities=[("same", "Bob", "other text")],
    ))
    for current in (graph, graph + "-other"):
        for label in ("Document", "Chunk", "__Entity__"):
            count = store.execute_query(
                f"MATCH (n:{label} {{graph_name:$g}}) RETURN count(n) AS count", {"g": current}
            )[0]["count"]
            assert count == 1, (label, current, count)
