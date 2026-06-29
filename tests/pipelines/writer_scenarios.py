"""Shared contract scenarios for backend graph writers."""

from __future__ import annotations

import re

from recon_graphrag.extraction.types import (
    ChunkRecord,
    DocumentRecord,
    EntityRecord,
    EvidenceLink,
    GraphDocument,
    RelationshipRecord,
)


class FakeGraphStore:
    def __init__(self):
        self.queries: list[str] = []
        self.params: list[dict] = []

    def execute_query(self, query: str, parameters: dict | None = None):
        self.queries.append(query.strip())
        self.params.append(parameters or {})
        return []


class StatefulFakeGraphStore:
    """Minimal in-memory graph store that simulates writer MERGE/SET behavior.

    This is intentionally narrow: it understands the Cypher shape emitted by
    ``Neo4jGraphWriter`` and ``MemgraphGraphWriter`` well enough to test
    cross-document aggregation and graph-name scoping. It is not a general
    Cypher interpreter.
    """

    def __init__(self):
        self.queries: list[str] = []
        self.params: list[dict] = []
        self._nodes: dict[str, dict] = {}
        self._relationships: dict[tuple[str, str, str], dict] = {}

    def execute_query(self, query: str, parameters: dict | None = None):
        self.queries.append(query.strip())
        params = parameters or {}
        self.params.append(params)

        if "UNWIND $entities AS row" in query:
            self._merge_entities(params.get("entities", []))
        elif "UNWIND $relationships AS row" in query:
            self._merge_relationships(params.get("relationships", []))

        return []

    def _merge_entities(self, rows: list[dict]) -> None:
        label_match = re.search(r"MERGE \(e:__Entity__:([^ ]+) \{id: row\.id\}\)", self.queries[-1])
        domain_label = label_match.group(1) if label_match else "Unknown"
        for row in rows:
            eid = row["id"]
            existing = self._nodes.setdefault(
                eid,
                {
                    "id": eid,
                    "type": row.get("type"),
                    "domain_label": domain_label,
                },
            )
            existing.update(row.get("properties", {}))
            existing["type"] = row.get("type", existing.get("type"))
            existing["graph_name"] = row.get("graph_name", existing.get("graph_name"))
            description = row.get("description") or ""
            descriptions = existing.setdefault("descriptions", [])
            if description and description not in descriptions:
                descriptions.append(description)
            existing["observation_count"] = len(descriptions)
            existing["description"] = "\n".join(descriptions)

    def _merge_relationships(self, rows: list[dict]) -> None:
        type_match = re.search(r"MERGE \(source\)-\[r:([^ ]+)\]->\(target\)", self.queries[-1])
        rel_type = type_match.group(1) if type_match else "RELATED_TO"
        rel_type = rel_type.strip("`")  # writers escape dynamic labels
        for row in rows:
            key = (row["source_id"], row["target_id"], rel_type)
            existing = self._relationships.setdefault(
                key,
                {
                    "source_id": row["source_id"],
                    "target_id": row["target_id"],
                    "type": rel_type,
                },
            )
            existing.update(row.get("properties", {}))
            existing["graph_name"] = row.get("graph_name", existing.get("graph_name"))
            source_chunk_ids = existing.setdefault("source_chunk_ids", [])
            for chunk_id in row.get("source_chunk_ids", []):
                if chunk_id not in source_chunk_ids:
                    source_chunk_ids.append(chunk_id)
            existing["observation_count"] = len(source_chunk_ids)
            existing["weight"] = float(len(source_chunk_ids))
            strength = row.get("strength")
            if strength is not None:
                existing["strength"] = max(strength, existing.get("strength", strength))

    def get_node(self, entity_id: str) -> dict | None:
        return self._nodes.get(entity_id)

    def get_relationship(self, source_id: str, target_id: str, rel_type: str) -> dict | None:
        return self._relationships.get((source_id, target_id, rel_type))

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def relationship_count(self) -> int:
        return len(self._relationships)


def make_graph_document() -> GraphDocument:
    return GraphDocument(
        document=DocumentRecord(id="doc:1", text_hash="hash"),
        chunks=[ChunkRecord(id="chunk:1", document_id="doc:1", text="hello", index=0)],
        entities=[
            EntityRecord(id="person:1", type="Person", properties={"name": "Alice"}),
            EntityRecord(id="person:2", type="Person", properties={"name": "Bob"}),
            EntityRecord(id="movie:1", type="Movie", properties={"name": "Inception"}),
        ],
        relationships=[
            RelationshipRecord(
                id="rel:1",
                source_id="person:1",
                target_id="movie:1",
                type="DIRECTED",
            ),
            RelationshipRecord(
                id="rel:2",
                source_id="person:2",
                target_id="movie:1",
                type="ACTED_IN",
            ),
        ],
        evidence_links=[EvidenceLink(chunk_id="chunk:1", entity_id="person:1")],
    )


def make_entity_graph_document(
    entity_id: str,
    name: str,
    graph_name: str = "entity-graph",
    document_id: str = "doc:1",
    description: str | None = None,
) -> GraphDocument:
    """Single-entity GraphDocument for merge/replacement tests."""
    return GraphDocument(
        document=DocumentRecord(id=document_id, text_hash="hash", graph_name=graph_name),
        chunks=[
            ChunkRecord(
                id=f"{document_id}:chunk:1",
                document_id=document_id,
                text="hello",
                index=0,
                graph_name=graph_name,
            )
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
        ],
        relationships=[],
        evidence_links=[
            EvidenceLink(
                chunk_id=f"{document_id}:chunk:1",
                entity_id=entity_id,
                graph_name=graph_name,
            )
        ],
    )


def make_relationship_graph_document(
    source_id: str,
    target_id: str,
    rel_type: str,
    weight: float,
    source_chunk_ids: list[str] | None = None,
    graph_name: str = "entity-graph",
    document_id: str = "doc:1",
) -> GraphDocument:
    """Single-relationship GraphDocument for merge/replacement tests."""
    source_chunk_ids = source_chunk_ids or [f"{document_id}:chunk:1"]
    return GraphDocument(
        document=DocumentRecord(id=document_id, text_hash="hash", graph_name=graph_name),
        chunks=[
            ChunkRecord(
                id=f"{document_id}:chunk:1",
                document_id=document_id,
                text="hello",
                index=0,
                graph_name=graph_name,
            )
        ],
        entities=[
            EntityRecord(
                id=source_id,
                type="Person",
                properties={"name": "Alice"},
                graph_name=graph_name,
            ),
            EntityRecord(
                id=target_id,
                type="Organization",
                properties={"name": "Acme"},
                graph_name=graph_name,
            ),
        ],
        relationships=[
            RelationshipRecord(
                id=f"{source_id}:{rel_type}:{target_id}",
                source_id=source_id,
                target_id=target_id,
                type=rel_type,
                properties={
                    "weight": weight,
                    "source_chunk_ids": source_chunk_ids,
                },
                graph_name=graph_name,
            )
        ],
        evidence_links=[],
    )


def assert_writer_stats_and_query_shape(writer_cls) -> None:
    store = FakeGraphStore()
    stats = writer_cls(store).write_graph_document(make_graph_document())

    assert stats == {
        "documents": 1,
        "chunks": 1,
        "entities": 3,
        "relationships": 2,
        "evidence_links": 1,
        "claims": 0,
    }
    query_text = "\n".join(store.queries)
    for fragment in (
        "MERGE (d:Document",
        "MERGE (c:Chunk",
        "MERGE (e:__Entity__:",
        "MERGE (c)-[r:FROM_CHUNK]",
        "MERGE (source)-[r:",
    ):
        assert fragment in query_text


def assert_writer_groups_entities_by_type(writer_cls) -> None:
    store = FakeGraphStore()
    writer_cls(store).write_graph_document(make_graph_document())

    entity_queries = [query for query in store.queries if "MERGE (e:__Entity__" in query]
    assert len(entity_queries) == 2


def assert_writer_groups_relationships_by_type(writer_cls) -> None:
    store = FakeGraphStore()
    writer_cls(store).write_graph_document(make_graph_document())

    relationship_queries = [
        query for query in store.queries if "MERGE (source)-[r:" in query
    ]
    assert len(relationship_queries) == 2
