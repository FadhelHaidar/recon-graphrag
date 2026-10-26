"""F4: re-ingesting a document replaces what that document contributed."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pytest

from recon_graphrag.extraction.types import (
    ChunkRecord,
    DocumentRecord,
    EntityRecord,
    EvidenceLink,
    GraphDocument,
    RelationshipRecord,
)
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.models.artifacts import ClaimRecord, SourceReference
from recon_graphrag.pipelines.writer import CypherGraphWriter
from tests.pipelines.writer_scenarios import StatefulFakeGraphStore


def build_document(
    *,
    chunks: list[str],
    entities: Sequence[tuple[str, str, str | None]] = (),
    relationships: Sequence[tuple[str, str, str, str | None]] = (),
    claims: Sequence[tuple[str, str, str]] = (),
    document_id: str = "doc:1",
    graph_name: str = "entity-graph",
    evidence: list[tuple[str, str]] | None = None,
) -> GraphDocument:
    """Build a document. Entities: (id, name, description). Relationships:
    (type, source, target, description). Claims: (id, entity_id, chunk_id)."""
    evidence = (
        [(chunks[0], entity_id) for entity_id, _, _ in entities]
        if evidence is None
        else evidence
    )
    return GraphDocument(
        document=DocumentRecord(
            id=document_id, text_hash="hash", graph_name=graph_name
        ),
        chunks=[
            ChunkRecord(
                id=chunk_id,
                document_id=document_id,
                text="text",
                index=index,
                graph_name=graph_name,
            )
            for index, chunk_id in enumerate(chunks)
        ],
        entities=[
            EntityRecord(
                id=entity_id,
                type="Person",
                properties={
                    "name": name,
                    **({"description": description} if description else {}),
                },
                graph_name=graph_name,
            )
            for entity_id, name, description in entities
        ],
        relationships=[
            RelationshipRecord(
                id=f"{source}:{rel_type}:{target}",
                source_id=source,
                target_id=target,
                type=rel_type,
                properties={
                    "source_chunk_ids": [chunks[0]],
                    **({"description": description} if description else {}),
                },
                graph_name=graph_name,
            )
            for rel_type, source, target, description in relationships
        ],
        evidence_links=[
            EvidenceLink(chunk_id=chunk_id, entity_id=entity_id, graph_name=graph_name)
            for chunk_id, entity_id in evidence
        ],
        claims=[
            ClaimRecord(
                id=claim_id,
                entity_id=entity_id,
                claim_type="status",
                description="claim text",
                source=SourceReference(document_id=document_id, chunk_id=chunk_id),
                graph_name=graph_name,
            )
            for claim_id, entity_id, chunk_id in claims
        ],
    )


def make_writer(store: StatefulFakeGraphStore) -> CypherGraphWriter:
    return CypherGraphWriter(cast(GraphStore, store))


def test_shorter_reingestion_removes_chunks_evidence_and_claims() -> None:
    store = StatefulFakeGraphStore()
    writer = make_writer(store)
    writer.write_graph_document(
        build_document(
            chunks=["doc:1:c0", "doc:1:c1"],
            entities=[("e1", "Alice", "Alice leads Acme.")],
            evidence=[("doc:1:c0", "e1"), ("doc:1:c1", "e1")],
            claims=[("claim:1", "e1", "doc:1:c1")],
        )
    )
    assert store.chunk_ids == {"doc:1:c0", "doc:1:c1"}

    writer.write_graph_document(
        build_document(
            chunks=["doc:1:c0"],
            entities=[("e1", "Alice", "Alice leads Acme.")],
            evidence=[("doc:1:c0", "e1")],
        )
    )

    assert store.chunk_ids == {"doc:1:c0"}
    assert store.evidence == {("entity-graph", "doc:1:c0", "e1")}
    assert store.get_claim("claim:1") is None


def test_relationship_keeps_other_documents_chunk_evidence() -> None:
    store = StatefulFakeGraphStore()
    writer = make_writer(store)
    writer.write_graph_document(
        build_document(
            document_id="doc:a",
            chunks=["a:c0"],
            entities=[("e1", "Alice", None), ("e2", "Acme", None)],
            relationships=[("WORKS_AT", "e1", "e2", "Alice works at Acme.")],
        )
    )
    writer.write_graph_document(
        build_document(
            document_id="doc:b",
            chunks=["b:c0"],
            entities=[("e1", "Alice", None), ("e2", "Acme", None)],
            relationships=[("WORKS_AT", "e1", "e2", "Alice founded Acme.")],
        )
    )
    rel = store.get_relationship("e1", "e2", "WORKS_AT")
    assert rel is not None
    assert rel["source_chunk_ids"] == ["a:c0", "b:c0"]

    writer.write_graph_document(
        build_document(
            document_id="doc:a",
            chunks=["a:c0"],
            entities=[("e1", "Alice", None), ("e2", "Acme", None)],
        )
    )

    rel = store.get_relationship("e1", "e2", "WORKS_AT")
    assert rel is not None
    assert rel["source_chunk_ids"] == ["b:c0"]
    assert rel["observation_count"] == 1
    assert rel["weight"] == 1.0
    assert rel["descriptions"] == ["Alice founded Acme."]
    assert rel["description"] == "Alice founded Acme."


def test_relationship_is_deleted_when_its_only_document_drops_it() -> None:
    store = StatefulFakeGraphStore()
    writer = make_writer(store)
    writer.write_graph_document(
        build_document(
            chunks=["a:c0"],
            entities=[("e1", "Alice", None), ("e2", "Acme", None)],
            relationships=[("WORKS_AT", "e1", "e2", "Alice works at Acme.")],
        )
    )
    assert store.relationship_count == 1

    writer.write_graph_document(
        build_document(
            chunks=["a:c0"],
            entities=[("e1", "Alice", None), ("e2", "Acme", None)],
        )
    )

    assert store.relationship_count == 0


def test_shared_entity_survives_with_the_other_documents_description() -> None:
    store = StatefulFakeGraphStore()
    writer = make_writer(store)
    writer.write_graph_document(
        build_document(
            document_id="doc:a",
            chunks=["a:c0"],
            entities=[("e1", "Alice", "Alice leads Acme.")],
        )
    )
    writer.write_graph_document(
        build_document(
            document_id="doc:b",
            chunks=["b:c0"],
            entities=[("e1", "Alice", "Alice founded Beta.")],
        )
    )

    writer.write_graph_document(
        build_document(
            document_id="doc:a",
            chunks=["a:c0"],
            entities=[],
            evidence=[],
        )
    )

    node = store.get_node("e1")
    assert node is not None
    assert node["descriptions"] == ["Alice founded Beta."]
    assert node["description"] == "Alice founded Beta."
    assert node["observation_count"] == 1


def test_replace_drops_only_the_replaced_documents_description() -> None:
    store = StatefulFakeGraphStore()
    writer = make_writer(store)
    writer.write_graph_document(
        build_document(
            chunks=["a:c0"], entities=[("e1", "Alice", "Old fact about Alice.")]
        )
    )
    writer.write_graph_document(
        build_document(
            chunks=["a:c0"], entities=[("e1", "Alice", "New fact about Alice.")]
        )
    )

    node = store.get_node("e1")
    assert node is not None
    assert node["descriptions"] == ["New fact about Alice."]
    assert node["description"] == "New fact about Alice."


def test_identical_description_shared_by_two_documents_is_not_lost() -> None:
    store = StatefulFakeGraphStore()
    writer = make_writer(store)
    writer.write_graph_document(
        build_document(
            document_id="doc:a",
            chunks=["a:c0"],
            entities=[("e1", "Alice", "Alice leads Acme.")],
        )
    )
    writer.write_graph_document(
        build_document(
            document_id="doc:b",
            chunks=["b:c0"],
            entities=[("e1", "Alice", "Alice leads Acme.")],
        )
    )
    writer.write_graph_document(
        build_document(document_id="doc:a", chunks=["a:c0"], entities=[], evidence=[])
    )

    node = store.get_node("e1")
    assert node is not None
    assert node["descriptions"] == ["Alice leads Acme."]


def test_changed_description_invalidates_summary_and_embedding_fingerprint() -> None:
    store = StatefulFakeGraphStore()
    writer = make_writer(store)
    writer.write_graph_document(
        build_document(
            chunks=["a:c0"], entities=[("e1", "Alice", "Old fact about Alice.")]
        )
    )
    before = store.get_node("e1")
    assert before is not None
    assert before["description_summary_status"] is None
    before_fingerprint = before["embedding_input_fingerprint"]

    writer.write_graph_document(
        build_document(
            chunks=["a:c0"], entities=[("e1", "Alice", "New fact about Alice.")]
        )
    )

    node = store.get_node("e1")
    assert node is not None
    assert node["description_summary_status"] is None
    assert node["embedding_input_fingerprint"] != before_fingerprint


def test_reingest_leaves_other_graph_names_untouched() -> None:
    store = StatefulFakeGraphStore()
    writer = make_writer(store)
    writer.write_graph_document(
        build_document(
            graph_name="graph-a",
            chunks=["a:c0", "a:c1"],
            entities=[("e1", "Alice", None)],
        )
    )
    writer.write_graph_document(
        build_document(graph_name="graph-b", chunks=["b:c0"], entities=[])
    )
    writer.write_graph_document(
        build_document(graph_name="graph-b", chunks=[], entities=[], evidence=[])
    )

    assert store.chunk_ids == {"a:c0", "a:c1"}


def test_stored_descriptions_without_provenance_fail_loudly() -> None:
    store = StatefulFakeGraphStore()
    writer = make_writer(store)
    writer.write_graph_document(
        build_document(chunks=["a:c0"], entities=[("e1", "Alice", "Legacy fact.")])
    )
    legacy_node = store.get_node("e1")
    assert legacy_node is not None
    legacy_node["description_sources"] = None

    with pytest.raises(ValueError, match="re-ingest"):
        writer.write_graph_document(
            build_document(
                chunks=["a:c0"], entities=[("e1", "Alice", "Replacement fact.")]
            )
        )


def test_uid_written_for_document_chunk_and_entity() -> None:
    store = StatefulFakeGraphStore()
    writer = make_writer(store)
    writer.write_graph_document(
        build_document(chunks=["doc:1:c0"], entities=[("p1", "John", None)])
    )

    # Length-prefixed uid: "entity-graph" has 12 characters.
    stored_doc = store._documents[("entity-graph", "doc:1")]
    assert stored_doc["uid"] == "12:entity-graph:doc:1"
    stored_chunk = store._chunks[("entity-graph", "doc:1:c0")]
    assert stored_chunk["uid"] == "12:entity-graph:doc:1:c0"
    node = store.get_node("p1")
    assert node is not None
    assert node["uid"] == "12:entity-graph:p1"


def test_uid_is_a_reserved_property_key() -> None:
    store = StatefulFakeGraphStore()
    writer = make_writer(store)
    document = build_document(chunks=["a:c0"], entities=[("e1", "Alice", None)])
    document.chunks[0].metadata["uid"] = "user-supplied"

    with pytest.raises(ValueError, match="uid"):
        writer.write_graph_document(document)
