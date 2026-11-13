"""B07: a changed chunk text must not keep the previous vector."""

from __future__ import annotations

from typing import cast

from recon_graphrag.entity_text import text_fingerprint
from recon_graphrag.extraction.types import (
    ChunkRecord,
    DocumentRecord,
    GraphDocument,
)
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.pipelines.writer import CypherGraphWriter
from tests.pipelines.writer_scenarios import StatefulFakeGraphStore

GRAPH = "entity-graph"


def _document(texts: dict[str, str], *, document_id: str = "doc:1") -> GraphDocument:
    return GraphDocument(
        document=DocumentRecord(id=document_id, text_hash="hash", graph_name=GRAPH),
        chunks=[
            ChunkRecord(
                id=chunk_id,
                document_id=document_id,
                text=text,
                index=index,
                graph_name=GRAPH,
            )
            for index, (chunk_id, text) in enumerate(texts.items())
        ],
        entities=[],
        relationships=[],
        evidence_links=[],
    )


def _writer() -> tuple[StatefulFakeGraphStore, CypherGraphWriter]:
    store = StatefulFakeGraphStore()
    return store, CypherGraphWriter(cast(GraphStore, store))


def _certify(store: StatefulFakeGraphStore, chunk_id: str, text: str) -> None:
    store._chunks[(GRAPH, chunk_id)].update(
        {
            "embedding": [1.0, 0.0],
            "embedding_content_fingerprint": text_fingerprint(text),
            "embedding_model_fingerprint": "model-v1",
        }
    )


def test_changed_chunk_text_invalidates_stored_vector() -> None:
    store, writer = _writer()
    writer.write_graph_document(_document({"doc:1:c0": "alpha"}))
    _certify(store, "doc:1:c0", "alpha")

    writer.write_graph_document(_document({"doc:1:c0": "beta"}))

    chunk = store._chunks[(GRAPH, "doc:1:c0")]
    assert chunk["text"] == "beta"
    assert chunk["embedding"] is None
    assert chunk["embedding_content_fingerprint"] is None
    assert chunk["embedding_model_fingerprint"] is None
    assert chunk["embedding_input_fingerprint"] == text_fingerprint("beta")


def test_unchanged_chunk_text_keeps_stored_vector() -> None:
    store, writer = _writer()
    writer.write_graph_document(_document({"doc:1:c0": "alpha"}))
    _certify(store, "doc:1:c0", "alpha")

    writer.write_graph_document(_document({"doc:1:c0": "alpha"}))

    chunk = store._chunks[(GRAPH, "doc:1:c0")]
    assert chunk["embedding"] == [1.0, 0.0]
    assert chunk["embedding_content_fingerprint"] == text_fingerprint("alpha")
    assert chunk["embedding_model_fingerprint"] == "model-v1"


def test_only_the_changed_chunk_loses_its_vector() -> None:
    store, writer = _writer()
    writer.write_graph_document(
        _document({"doc:1:c0": "alpha", "doc:1:c1": "beta"})
    )
    _certify(store, "doc:1:c0", "alpha")
    _certify(store, "doc:1:c1", "beta")

    writer.write_graph_document(
        _document({"doc:1:c0": "alpha", "doc:1:c1": "gamma"})
    )

    assert store._chunks[(GRAPH, "doc:1:c0")]["embedding"] == [1.0, 0.0]
    assert store._chunks[(GRAPH, "doc:1:c1")]["embedding"] is None


def test_new_chunk_row_carries_the_text_fingerprint() -> None:
    store, writer = _writer()

    writer.write_graph_document(_document({"doc:1:c0": "alpha"}))

    chunk = store._chunks[(GRAPH, "doc:1:c0")]
    assert chunk["embedding"] is None
    assert chunk["embedding_input_fingerprint"] == text_fingerprint("alpha")


def test_chunk_write_query_clears_vector_only_for_changed_text() -> None:
    store, writer = _writer()

    writer.write_graph_document(_document({"doc:1:c0": "alpha"}))

    chunk_query = next(q for q in store.queries if "UNWIND $chunks AS row" in q)
    assert "WITH c, d, row, c.text AS previous_text" in chunk_query
    # The vector, not the fingerprint, is what a stale write would keep forever.
    for preserved in (
        "c.embedding",
        "c.embedding_content_fingerprint",
        "c.embedding_model_fingerprint",
    ):
        assert f"WHEN previous_text = row.text THEN {preserved}" in chunk_query
