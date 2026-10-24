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
    """In-memory graph store that answers the queries the writer emits.

    This is intentionally narrow: it recognizes the exact query shapes
    ``CypherGraphWriter`` produces (state reads, MERGE writes, reconciliation
    deletes) and applies their effects to plain dicts. It is not a general
    Cypher interpreter, so a new writer query must be added here.
    """

    def __init__(self):
        self.queries: list[str] = []
        self.params: list[dict] = []
        self._documents: dict[tuple[str, str], dict] = {}
        self._chunks: dict[tuple[str, str], dict] = {}
        self._entities: dict[tuple[str, str], dict] = {}
        self._relationships: dict[tuple[str, str, str, str], dict] = {}
        self._evidence: set[tuple[str, str, str]] = set()
        self._claims: dict[tuple[str, str], dict] = {}

    def execute_query(self, query: str, parameters: dict | None = None):
        self.queries.append(query.strip())
        params = parameters or {}
        self.params.append(params)
        return self._dispatch(query, params)

    def _dispatch(self, query: str, params: dict):
        if "UNWIND $documents AS row" in query:
            return self._write_documents(params["documents"])
        if "UNWIND $chunks AS row" in query:
            return self._write_chunks(params["chunks"])
        if "SET e.description_sources = row.description_sources" in query:
            return self._rebuild_entities(params["entities"], params["graph_name"])
        if "UNWIND $entities AS row" in query:
            return self._write_entities(params["entities"])
        if "UNWIND $relationships AS row" in query:
            if "MERGE (source)-[r:" in query:
                label = re.search(r"MERGE \(source\)-\[r:`?(\w+)`?\]", query)
                return self._write_relationships(
                    params["relationships"], label.group(1) if label else "RELATED_TO"
                )
            return self._rebuild_relationships(
                params["relationships"], params["graph_name"]
            )
        if "UNWIND $links AS row" in query:
            if "DELETE r" in query:
                return self._delete_evidence(params["links"], params["graph_name"])
            return self._write_evidence(params["links"], None)
        if "UNWIND $claims AS row" in query:
            return self._write_claims(params["claims"])
        if "UNWIND $claim_ids AS id" in query:
            return self._delete_claims(params["claim_ids"], params["graph_name"])
        if "UNWIND $chunk_ids AS id" in query:
            return self._delete_chunks(params["chunk_ids"], params["graph_name"])
        if "UNWIND $rel_ids AS id" in query:
            return self._delete_empty_relationships(
                params["rel_ids"], params["graph_name"]
            )
        if "UNWIND $entity_ids AS id" in query:
            if "DETACH DELETE e" in query:
                return self._delete_orphan_entities(
                    params["entity_ids"], params["graph_name"]
                )
            return self._read_entities(params["entity_ids"], params["graph_name"])
        if "-[:FROM_CHUNK]->(e:__Entity__)" in query:
            return self._read_evidence(params)
        if "(cl:Claim" in query:
            return self._read_claims(params)
        if "RETURN c.id AS id" in query:
            return self._read_document_chunks(params)
        if "type(r) AS type" in query:
            return self._read_relationships(params)
        raise AssertionError(f"StatefulFakeGraphStore cannot emulate query:\n{query}")

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def _write_documents(self, rows: list[dict]) -> list:
        for row in rows:
            self._documents[(row["graph_name"], row["id"])] = dict(row)
        return []

    def _write_chunks(self, rows: list[dict]) -> list:
        for row in rows:
            key = (row["graph_name"], row["id"])
            chunk = self._chunks.setdefault(key, {})
            chunk.update(row)
        return []

    def _write_entities(self, rows: list[dict]) -> list:
        for row in rows:
            key = (row["graph_name"], row["id"])
            node = self._entities.setdefault(
                key, {"id": row["id"], "graph_name": row["graph_name"]}
            )
            node.update(row.get("properties") or {})
            node["type"] = row["type"]
            node["labels"] = ["__Entity__", row["type"]]
            node["uid"] = row.get("uid")
            node["canonical_key"] = row.get("canonical_key")
            node["human_readable_id"] = row.get("human_readable_id")
            node["name"] = (
                row["properties"].get("name")
                or row["properties"].get("title")
                or row.get("human_readable_id")
                or row["id"]
            )
            node["descriptions"] = list(row["descriptions"])
            node["description_sources"] = row["description_sources"]
            node["observation_count"] = len(node["descriptions"])
            node["embedding_input_fingerprint"] = row["embedding_input_fingerprint"]
            if row["description_text"] is not None:
                node["description"] = row["description_text"]
            if row["descriptions_changed"]:
                node["description_summary_status"] = None
        return []

    def _rebuild_entities(self, rows: list[dict], graph_name: str) -> list:
        for row in rows:
            node = self._entities.get((graph_name, row["id"]))
            if node is None:
                continue
            node["descriptions"] = list(row["descriptions"])
            node["description_sources"] = row["description_sources"]
            node["observation_count"] = len(node["descriptions"])
            node["description"] = row["description_text"]
            node["embedding_input_fingerprint"] = row["embedding_input_fingerprint"]
            node["description_summary_status"] = None
        return []

    def _write_relationships(self, rows: list[dict], rel_type: str) -> list:
        for row in rows:
            key = (row["graph_name"], row["source_id"], row["target_id"], rel_type)
            rel = self._relationships.setdefault(key, {})
            rel.update(row.get("properties") or {})
            rel.update(
                {
                    "id": row["id"],
                    "graph_name": row["graph_name"],
                    "source_id": row["source_id"],
                    "target_id": row["target_id"],
                    "type": rel_type,
                    "source_chunk_ids": list(row["source_chunk_ids"]),
                    "observation_count": row["observation_count"],
                    "weight": row["weight"],
                    "strength": row["strength"],
                    "descriptions": list(row["descriptions"]),
                    "description_sources": row["description_sources"],
                }
            )
            if row["description_text"] is not None:
                rel["description"] = row["description_text"]
            if row["descriptions_changed"]:
                rel["description_summary_status"] = None
        return []

    def _rebuild_relationships(self, rows: list[dict], graph_name: str) -> list:
        by_id = {
            rel["id"]: rel
            for rel in self._relationships.values()
            if rel["graph_name"] == graph_name
        }
        for row in rows:
            rel = by_id.get(row["id"])
            if rel is None:
                continue
            rel["source_chunk_ids"] = list(row["source_chunk_ids"])
            rel["observation_count"] = row["observation_count"]
            rel["weight"] = row["weight"]
            rel["descriptions"] = list(row["descriptions"])
            rel["description_sources"] = row["description_sources"]
            rel["description"] = row["description_text"]
            if row["descriptions_changed"]:
                rel["description_summary_status"] = None
        return []

    def _write_evidence(self, rows: list[dict], _graph_name: str | None) -> list:
        for row in rows:
            self._evidence.add((row["graph_name"], row["chunk_id"], row["entity_id"]))
        return []

    def _delete_evidence(self, rows: list[dict], graph_name: str) -> list:
        for row in rows:
            self._evidence.discard((graph_name, row["chunk_id"], row["entity_id"]))
        return []

    def _write_claims(self, rows: list[dict]) -> list:
        for row in rows:
            self._claims[(row["graph_name"], row["id"])] = dict(row)
        return []

    def _delete_claims(self, claim_ids: list[str], graph_name: str) -> list:
        for claim_id in claim_ids:
            self._claims.pop((graph_name, claim_id), None)
        return []

    def _delete_chunks(self, chunk_ids: list[str], graph_name: str) -> list:
        for chunk_id in chunk_ids:
            self._chunks.pop((graph_name, chunk_id), None)
            for evidence in list(self._evidence):
                if evidence[0] == graph_name and evidence[1] == chunk_id:
                    self._evidence.discard(evidence)
        return []

    def _delete_empty_relationships(self, rel_ids: list[str], graph_name: str) -> list:
        for key, rel in list(self._relationships.items()):
            if rel["graph_name"] != graph_name or rel["id"] not in rel_ids:
                continue
            if not rel["source_chunk_ids"]:
                del self._relationships[key]
        return []

    def _delete_orphan_entities(self, entity_ids: list[str], graph_name: str) -> list:
        connected = {
            entity_id
            for _, source_id, target_id, _ in self._relationships
            for entity_id in (source_id, target_id)
        }
        connected |= {entity_id for _, _, entity_id in self._evidence}
        connected |= {claim["entity_id"] for claim in self._claims.values()}
        for entity_id in entity_ids:
            if entity_id in connected:
                continue
            self._entities.pop((graph_name, entity_id), None)
        return []

    # ------------------------------------------------------------------
    # State reads
    # ------------------------------------------------------------------
    def _document_chunk_ids(self, params: dict) -> set[str]:
        return {
            chunk["id"]
            for chunk in self._chunks.values()
            if chunk["graph_name"] == params["graph_name"]
            and chunk["document_id"] == params["document_id"]
        }

    def _read_document_chunks(self, params: dict) -> list[dict]:
        return [{"id": chunk_id} for chunk_id in sorted(self._document_chunk_ids(params))]

    def _read_evidence(self, params: dict) -> list[dict]:
        chunk_ids = self._document_chunk_ids(params)
        return [
            {"chunk_id": chunk_id, "entity_id": entity_id}
            for graph_name, chunk_id, entity_id in sorted(self._evidence)
            if graph_name == params["graph_name"] and chunk_id in chunk_ids
        ]

    def _read_claims(self, params: dict) -> list[dict]:
        chunk_ids = self._document_chunk_ids(params)
        return [
            {
                "id": claim["id"],
                "chunk_id": claim["chunk_id"],
                "entity_id": claim.get("entity_id"),
            }
            for claim in self._claims.values()
            if claim["graph_name"] == params["graph_name"]
            and claim["chunk_id"] in chunk_ids
        ]

    def _read_relationships(self, params: dict) -> list[dict]:
        rel_ids = set(params["rel_ids"])
        chunk_ids = set(params["chunk_ids"])
        fields = (
            "id",
            "source_id",
            "target_id",
            "type",
            "source_chunk_ids",
            "descriptions",
            "description_sources",
            "description",
            "description_summary_status",
            "strength",
        )
        return [
            {field: rel.get(field) for field in fields}
            for rel in self._relationships.values()
            if rel["graph_name"] == params["graph_name"]
            and (rel["id"] in rel_ids or chunk_ids & set(rel["source_chunk_ids"]))
        ]

    def _read_entities(self, entity_ids: list[str], graph_name: str) -> list[dict]:
        fields = (
            "id",
            "descriptions",
            "description_sources",
            "description",
            "description_summary_status",
            "name",
            "labels",
            "embedding_input_fingerprint",
        )
        rows = []
        for entity_id in entity_ids:
            node = self._entities.get((graph_name, entity_id))
            if node is not None:
                rows.append({field: node.get(field) for field in fields})
        return rows

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    def get_node(self, entity_id: str) -> dict | None:
        for (_, stored_id), node in self._entities.items():
            if stored_id == entity_id:
                return node
        return None

    def get_relationship(
        self, source_id: str, target_id: str, rel_type: str
    ) -> dict | None:
        for (_, source, target, stored_type), rel in self._relationships.items():
            if (source, target, stored_type) == (source_id, target_id, rel_type):
                return rel
        return None

    def get_claim(self, claim_id: str) -> dict | None:
        for (_, stored_id), claim in self._claims.items():
            if stored_id == claim_id:
                return claim
        return None

    @property
    def chunk_ids(self) -> set[str]:
        return {chunk_id for (_, chunk_id) in self._chunks}

    @property
    def evidence(self) -> set[tuple[str, str, str]]:
        return set(self._evidence)

    @property
    def node_count(self) -> int:
        return len(self._entities)

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
