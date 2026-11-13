"""B04: entity identity is separate from the domain label.

The writer used to merge on ``(:__Entity__:{label} {id, graph_name})``. A node
that lacked the incoming domain label, or carried an older label, was therefore
invisible and a second node appeared for one identity.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from recon_graphrag.entity_text import entity_embedding_fingerprint
from recon_graphrag.extraction.types import GraphDocument
from recon_graphrag.graphdb._provenance import encode_contributions
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.pipelines.writer import CypherGraphWriter
from tests.pipelines.test_writer_replacement import build_document
from tests.pipelines.writer_scenarios import StatefulFakeGraphStore

GRAPH = "entity-graph"


class DuplicateIdentityStore(StatefulFakeGraphStore):
    """Fake store that reports every affected identity twice when asked."""

    duplicate_reads = False

    def execute_query(self, query: str, parameters: dict | None = None) -> Any:
        result = super().execute_query(query, parameters)
        if (
            self.duplicate_reads
            and "UNWIND $entity_ids AS id" in query
            and "DETACH DELETE e" not in query
        ):
            rows = list(result)
            return rows + rows
        return result


def make_writer(store: StatefulFakeGraphStore) -> CypherGraphWriter:
    return CypherGraphWriter(cast(GraphStore, store))


def person_document(graph_name: str = GRAPH) -> GraphDocument:
    return build_document(
        chunks=["doc:1:c0"],
        entities=[("e1", "Alice", None)],
        graph_name=graph_name,
    )


def entity_queries(store: StatefulFakeGraphStore) -> list[tuple[str, list[dict]]]:
    return [
        (query, params["entities"])
        for query, params in zip(store.queries, store.params, strict=False)
        if "UNWIND $entities AS row" in query
    ]


def test_entity_identity_merge_excludes_the_domain_label() -> None:
    store = StatefulFakeGraphStore()
    make_writer(store).write_graph_document(person_document())

    ((query, _rows),) = entity_queries(store)

    assert "MERGE (e:__Entity__ {id: row.id, graph_name: row.graph_name})" in query
    assert "SET e:`Person`," in query
    assert "MERGE (e:__Entity__:`Person`" not in query


def test_entity_write_reuses_untyped_existing_identity() -> None:
    store = StatefulFakeGraphStore()
    store._entities[(GRAPH, "e1")] = {"id": "e1", "graph_name": GRAPH}

    receipt = make_writer(store).write_graph_document(person_document())

    assert receipt["entities"] == 1
    assert list(store._entities) == [(GRAPH, "e1")]
    assert store._entities[(GRAPH, "e1")]["labels"] == ["__Entity__", "Person"]


def test_entity_type_partition_preserves_repeated_input_rows() -> None:
    store = StatefulFakeGraphStore()
    document = build_document(
        chunks=["doc:1:c0"],
        entities=[("e1", "Alice", None)],
        graph_name=GRAPH,
    )
    document.entities.append(
        type(document.entities[0])(
            id="e1",
            type="Organization",
            graph_name=GRAPH,
            properties={"name": "Alice"},
        )
    )

    receipt = make_writer(store).write_graph_document(document)

    # The pre-existing receipt counts confirmed rows per type query, so a
    # cross-type id counts once per group. Identity stays a single node.
    assert len(entity_queries(store)) == 2
    assert receipt["entities"] == 4
    assert list(store._entities) == [(GRAPH, "e1")]


def test_entity_write_rejects_ambiguous_existing_identity() -> None:
    store = DuplicateIdentityStore()
    writer = make_writer(store)
    writer.write_graph_document(person_document())
    before = store.get_node("e1")
    mark = len(store.queries)
    store.duplicate_reads = True

    with pytest.raises(ValueError) as error:
        writer.write_graph_document(person_document())

    message = str(error.value)
    assert "ambiguous" in message
    assert f"'{GRAPH}'" in message
    assert "'e1' (2 nodes)" in message
    assert "does not merge duplicate entity nodes" in message
    assert store.get_node("e1") == before
    assert not any(
        "UNWIND $documents AS row" in query for query in store.queries[mark:]
    )


def test_entity_label_is_escaped_without_changing_identity() -> None:
    store = StatefulFakeGraphStore()
    document = build_document(
        chunks=["doc:1:c0"],
        entities=[("e1", "Alice", None)],
        graph_name=GRAPH,
    )
    document.entities[0].type = "Data Source"

    make_writer(store).write_graph_document(document)

    ((query, _rows),) = entity_queries(store)
    assert "SET e:`Data Source`," in query
    assert "MERGE (e:__Entity__ {id: row.id, graph_name: row.graph_name})" in query


def test_rebuilt_fingerprint_uses_the_recorded_entity_type() -> None:
    """A retained old label must not decide the embedding input."""
    store = StatefulFakeGraphStore()
    writer = make_writer(store)
    document = build_document(
        chunks=["c1"], entities=[], graph_name=GRAPH, document_id="doc:1"
    )
    labels = ["__Entity__", "Person", "Organization"]
    state = SimpleNamespace(
        chunk_ids=[],
        evidence=[],
        claims=[],
        relationships=[],
        entities={
            "e1": {
                "id": "e1",
                "type": "Organization",
                "labels": labels,
                "name": "Acme",
                "descriptions": ["old", "old"],
                "description_sources": encode_contributions(
                    [["doc:1", "old"], ["doc:2", "old"]]
                ),
                "description": "old",
                "description_summary_status": None,
            }
        },
    )

    (row,) = writer._rebuild_entity_rows(
        document, cast(Any, state), []
    )

    recorded_type = entity_embedding_fingerprint(
        labels=labels,
        name="Acme",
        description=row["description_text"],
        entity_type="Organization",
    )
    label_order = entity_embedding_fingerprint(
        labels=labels,
        name="Acme",
        description=row["description_text"],
    )
    assert row["embedding_input_fingerprint"] == recorded_type
    assert row["embedding_input_fingerprint"] != label_order
