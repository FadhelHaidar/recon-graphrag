"""Tests for backend writer MERGE behavior."""

from __future__ import annotations

import copy
from typing import cast

import pytest

from recon_graphrag.extraction.types import EntityRecord
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.pipelines import writer as writer_module
from recon_graphrag.pipelines.writer import CypherGraphWriter
from tests.pipelines.writer_scenarios import (
    FakeGraphStore,
    StatefulFakeGraphStore,
    make_entity_graph_document,
    make_graph_document,
    make_relationship_graph_document,
)


@pytest.fixture
def writer_cls():
    return CypherGraphWriter


def test_entity_merge_replaces_properties(writer_cls):
    store = StatefulFakeGraphStore()
    writer = writer_cls(store)

    writer.write_graph_document(
        make_entity_graph_document("e1", "Alice", document_id="doc:a")
    )
    writer.write_graph_document(
        make_entity_graph_document("e1", "Alicia", document_id="doc:b")
    )

    node = store.get_node("e1")
    assert node is not None
    assert node["name"] == "Alicia"


def test_entity_merge_unions_descriptions(writer_cls):
    store = StatefulFakeGraphStore()
    writer = writer_cls(store)

    writer.write_graph_document(
        make_entity_graph_document(
            "e1", "Alice", document_id="doc:a", description="Alice leads Acme."
        )
    )
    writer.write_graph_document(
        make_entity_graph_document(
            "e1", "Alicia", document_id="doc:b", description="Alice founded Beta."
        )
    )

    node = store.get_node("e1")
    assert node is not None
    assert node["descriptions"] == ["Alice leads Acme.", "Alice founded Beta."]
    assert node["observation_count"] == 2
    assert node["description"] == "Alice leads Acme.\nAlice founded Beta."


def test_relationship_merge_unions_source_chunks_and_weight(writer_cls):
    store = StatefulFakeGraphStore()
    writer = writer_cls(store)

    writer.write_graph_document(
        make_relationship_graph_document(
            "person:alice",
            "org:acme",
            "WORKS_AT",
            weight=1.0,
            source_chunk_ids=["chunk:a"],
            document_id="doc:a",
        )
    )
    writer.write_graph_document(
        make_relationship_graph_document(
            "person:alice",
            "org:acme",
            "WORKS_AT",
            weight=1.0,
            source_chunk_ids=["chunk:b"],
            document_id="doc:b",
        )
    )

    rel = store.get_relationship("person:alice", "org:acme", "WORKS_AT")
    assert rel is not None
    assert rel["source_chunk_ids"] == ["chunk:a", "chunk:b"]
    assert rel["observation_count"] == 2
    assert rel["weight"] == 2.0


_COMMON_RESERVED = (
    "id",
    "graph_name",
    "created",
    "updated",
    "embedding",
    "description_summary_status",
    "description_input_fingerprint",
    "description_summary_updated",
    "description_summary_error",
    "description_summary_fallback",
    "description_sources",
    "strength_sources",
    "embedding_input_fingerprint",
    "embedding_content_fingerprint",
    "embedding_model_fingerprint",
    "uid",
)
_MAP_CASES = [
    ("DocumentRecord.metadata", _COMMON_RESERVED + ("text_hash",)),
    ("ChunkRecord.metadata", _COMMON_RESERVED + ("text", "index")),
    (
        "EntityRecord.properties",
        _COMMON_RESERVED + ("type", "descriptions", "observation_count"),
    ),
    ("RelationshipRecord.properties", _COMMON_RESERVED + ("descriptions",)),
]


def _record_maps(graph_document, record_map):
    if record_map == "DocumentRecord.metadata":
        return [(graph_document.document, graph_document.document.metadata)]
    if record_map == "ChunkRecord.metadata":
        return [(chunk, chunk.metadata) for chunk in graph_document.chunks]
    if record_map == "EntityRecord.properties":
        return [(entity, entity.properties) for entity in graph_document.entities]
    return [(rel, rel.properties) for rel in graph_document.relationships]


@pytest.mark.parametrize(("record_map", "reserved_keys"), _MAP_CASES)
@pytest.mark.parametrize("collision", ["null", "same-value"])
def test_reserved_key_rejected_before_any_query(record_map, reserved_keys, collision):
    for key in reserved_keys:
        graph_document = make_graph_document()
        record, target_map = _record_maps(graph_document, record_map)[0]
        target_map[key] = (
            None if collision == "null" else getattr(record, key, "collision")
        )
        submitted = copy.deepcopy(target_map)
        store = FakeGraphStore()

        with pytest.raises(ValueError) as excinfo:
            CypherGraphWriter(cast(GraphStore, store)).write_graph_document(
                graph_document
            )

        message = str(excinfo.value)
        assert store.queries == []
        assert record_map in message
        assert key in message
        assert target_map == submitted


def test_late_invalid_record_rejects_whole_artifact_before_queries():
    graph_document = make_graph_document()
    graph_document.entities.append(
        EntityRecord(
            id="person:9",
            type="Person",
            properties={"name": "Eve", "embedding": [0.1, 0.2]},
        )
    )
    store = FakeGraphStore()

    with pytest.raises(ValueError, match="EntityRecord.properties"):
        CypherGraphWriter(cast(GraphStore, store)).write_graph_document(graph_document)

    assert store.queries == []


def test_reserved_key_error_is_deterministic_and_safe():
    graph_document = make_graph_document()
    graph_document.document.metadata.update({"text_hash": "hash-x", "id": "doc:1"})

    messages = []
    for _ in range(2):
        with pytest.raises(ValueError) as excinfo:
            CypherGraphWriter(cast(GraphStore, FakeGraphStore())).write_graph_document(
                graph_document
            )
        messages.append(str(excinfo.value))

    assert messages[0] == messages[1]
    assert "id, text_hash" in messages[0]
    assert "doc:1" not in messages[0]
    assert "hash-x" not in messages[0]
    assert "document_id" in messages[0]


def _write_rows(store, query_fragment):
    for query, params in zip(store.queries, store.params, strict=True):
        if query_fragment in query:
            return next(iter(params.values()))
    raise AssertionError(f"no writer query matching {query_fragment!r}")


def test_reserved_key_test_table_matches_writer_policy():
    expected = {
        "DocumentRecord.metadata": writer_module._DOCUMENT_METADATA_RESERVED,
        "ChunkRecord.metadata": writer_module._CHUNK_METADATA_RESERVED,
        "EntityRecord.properties": writer_module._ENTITY_PROPERTIES_RESERVED,
        "RelationshipRecord.properties": writer_module._RELATIONSHIP_PROPERTIES_RESERVED,
    }
    for record_map, reserved_keys in _MAP_CASES:
        assert len(reserved_keys) == len(set(reserved_keys))
        assert set(reserved_keys) == expected[record_map]


def test_supported_document_and_chunk_metadata_pass_through():
    graph_document = make_graph_document()
    graph_document.document.metadata.update(
        {"document_id": "doc:1", "source": "report.pdf", "title": "Report"}
    )
    chunk_metadata = {
        "document_id": "doc:1",
        "source": "report.pdf",
        "char_start": 0,
        "char_end": 5,
        "page_start": 2,
        "page_end": 2,
    }
    graph_document.chunks[0].metadata.update(chunk_metadata)
    store = FakeGraphStore()

    stats = CypherGraphWriter(cast(GraphStore, store)).write_graph_document(
        graph_document
    )

    assert stats == {
        "documents": 1,
        "chunks": 1,
        "entities": 3,
        "relationships": 2,
        "evidence_links": 1,
        "claims": 0,
    }
    assert _write_rows(store, "UNWIND $documents")[0]["metadata"] == (
        graph_document.document.metadata
    )
    assert _write_rows(store, "UNWIND $chunks")[0]["metadata"] == chunk_metadata
    assert _write_rows(store, "UNWIND $chunks")[0]["text"] == "hello"
    assert _write_rows(store, "UNWIND $chunks")[0]["index"] == 0


def test_entity_supported_properties_pass_through():
    graph_document = make_graph_document()
    properties = {
        "name": "Alice",
        "title": "Dr. Alice",
        "description": "Leads Acme.",
        "canonical_key": "person:alice",
        "human_readable_id": "person:alice",
        "employee_id": "E-7",
    }
    graph_document.entities[0].properties = dict(properties)
    graph_document.entities[0].canonical_key = "person:alice"
    graph_document.entities[0].human_readable_id = "person:alice"
    store = FakeGraphStore()

    CypherGraphWriter(cast(GraphStore, store)).write_graph_document(graph_document)

    row = _write_rows(store, "UNWIND $entities")[0]
    assert row["canonical_key"] == "person:alice"
    assert row["human_readable_id"] == "person:alice"
    assert row["description_text"] == "Leads Acme."
    assert row["properties"] == {
        key: value for key, value in properties.items() if key != "description"
    }


def test_relationship_evidence_canonical_ids_and_weight_normalization():
    graph_document = make_graph_document()
    rel = graph_document.relationships[0]
    rel.properties = {
        "weight": 3.0,
        "source_chunk_ids": ["chunk:1", "chunk:1", "chunk:2"],
        "observation_count": 9,
        "canonical_key": "person:1:DIRECTED:movie:1",
        "human_readable_id": "person:1:DIRECTED:movie:1",
        "source_canonical_key": "person:1",
        "target_canonical_key": "movie:1",
    }
    rel.observation_count = 2
    rel.strength = 0.6
    store = FakeGraphStore()

    CypherGraphWriter(cast(GraphStore, store)).write_graph_document(graph_document)

    row = _write_rows(store, "UNWIND $relationships")[0]
    assert row["source_chunk_ids"] == ["chunk:1", "chunk:2"]
    assert row["observation_count"] == 2
    assert row["weight"] == 2.0
    assert row["strength"] == 0.6
    assert row["properties"] == {
        "canonical_key": "person:1:DIRECTED:movie:1",
        "human_readable_id": "person:1:DIRECTED:movie:1",
        "source_canonical_key": "person:1",
        "target_canonical_key": "movie:1",
    }
