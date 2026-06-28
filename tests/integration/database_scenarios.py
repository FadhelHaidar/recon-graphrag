"""Shared live-database scenarios for GraphStore implementations."""

from __future__ import annotations

from recon_graphrag.extraction.types import (
    ChunkRecord,
    DocumentRecord,
    EntityRecord,
    EvidenceLink,
    GraphDocument,
)
from tests.integration.support import cleanup_graph, single_count


def seed_entities(
    store,
    graph_name: str,
    domain_label: str,
    rows: list[dict],
) -> None:
    if not domain_label.isidentifier():
        raise ValueError(f"Invalid test domain label: {domain_label!r}")
    store.execute_query(
        f"""
        UNWIND $rows AS row
        CREATE (e:__Entity__:{domain_label})
        SET e.id = row.id,
            e.graph_name = $graph_name,
            e.name = row.name,
            e.description = coalesce(row.description, '')
        """,
        {"graph_name": graph_name, "rows": rows},
    )


def entity_count(store, graph_name: str) -> int:
    return single_count(
        store,
        "MATCH (e:__Entity__ {graph_name: $graph_name}) RETURN count(e) AS count",
        graph_name,
    )


def assert_graph_document_write(store, graph_name: str) -> None:
    document_id = f"{graph_name}:doc-1"
    chunk_id = f"{graph_name}:chunk-1"
    entity_id = f"{graph_name}:entity-1"
    graph_document = GraphDocument(
        document=DocumentRecord(
            id=document_id,
            text_hash="hash",
            graph_name=graph_name,
            metadata={},
        ),
        chunks=[
            ChunkRecord(
                id=chunk_id,
                document_id=document_id,
                text="Alice works at Acme.",
                index=0,
                graph_name=graph_name,
                metadata={},
            )
        ],
        entities=[
            EntityRecord(
                id=entity_id,
                type="Person",
                graph_name=graph_name,
                properties={"name": "Alice"},
            )
        ],
        relationships=[],
        evidence_links=[
            EvidenceLink(
                chunk_id=chunk_id,
                entity_id=entity_id,
                graph_name=graph_name,
            )
        ],
    )

    store.write_graph_document(graph_document)
    queries = [
        "MATCH (:Document {graph_name: $graph_name}) RETURN count(*) AS count",
        "MATCH (:Chunk {graph_name: $graph_name}) RETURN count(*) AS count",
        "MATCH (:__Entity__ {graph_name: $graph_name}) RETURN count(*) AS count",
        """
        MATCH (:Chunk {graph_name: $graph_name})-[r:FROM_CHUNK]->
              (:__Entity__ {graph_name: $graph_name})
        RETURN count(r) AS count
        """,
    ]
    assert all(single_count(store, query, graph_name) == 1 for query in queries)


async def assert_normalized_entity_resolution(store, graph_name: str) -> None:
    seed_entities(
        store,
        graph_name,
        "Organization",
        [
            {"id": f"{graph_name}:openai-a", "name": "OpenAI"},
            {"id": f"{graph_name}:openai-b", "name": "openai"},
        ],
    )
    result = await store.resolve_entities(
        graph_name=graph_name,
        strategy="normalized",
    )
    assert result["skipped"] is False
    assert result["merged_groups"] == 1
    assert entity_count(store, graph_name) == 1


async def assert_hybrid_alias_dry_run(store, graph_name: str) -> None:
    seed_entities(
        store,
        graph_name,
        "Organization",
        [
            {"id": f"{graph_name}:ibm", "name": "IBM"},
            {
                "id": f"{graph_name}:international-business-machines",
                "name": "International Business Machines",
            },
        ],
    )
    result = await store.resolve_entities(
        graph_name=graph_name,
        strategy="hybrid",
        dry_run=True,
        aliases={"Organization": {"IBM": ["International Business Machines"]}},
    )
    assert result["skipped"] is False
    assert result["signals"]["aliases"] == "used"
    assert result["merged_groups"] == 1
    assert entity_count(store, graph_name) == 2


async def assert_hybrid_ai_review(
    store,
    graph_name: str,
    llm,
    embedder,
) -> None:
    seed_entities(
        store,
        graph_name,
        "Person",
        [
            {
                "id": f"{graph_name}:john-smith",
                "name": "John Smith",
                "description": "A person named John Smith.",
            },
            {
                "id": f"{graph_name}:jon-smith",
                "name": "Jon Smith",
                "description": "A person named Jon Smith.",
            },
        ],
    )
    result = await store.resolve_entities(
        graph_name=graph_name,
        strategy="hybrid",
        dry_run=True,
        merge_threshold=95.0,
        review_threshold=85.0,
        embedder=embedder,
        llm=llm,
        llm_guidance=(
            "This is an integration test. Return JSON only. Treat the pair "
            "as a review candidate unless the evidence is conclusive."
        ),
    )
    assert result["skipped"] is False
    assert result["signals"]["embeddings"] == "used"
    assert result["signals"]["llm"] == "used"
    assert result["review_groups"]
    review = result["review_groups"][0]
    assert review["scores"]["embedding"] is not None
    assert "llm_review" in review
    assert "error" not in review["llm_review"]
    assert entity_count(store, graph_name) == 2


def assert_weighted_community_detection(store, graph_name: str) -> None:
    store.execute_query(
        """
        UNWIND $nodes AS row
        CREATE (e:__Entity__:TestEntity)
        SET e.id = row.id,
            e.graph_name = $graph_name,
            e.name = row.name,
            e.description = ''
        """,
        {
            "graph_name": graph_name,
            "nodes": [
                {"id": f"{graph_name}:a", "name": "A"},
                {"id": f"{graph_name}:b", "name": "B"},
                {"id": f"{graph_name}:c", "name": "C"},
                {"id": f"{graph_name}:d", "name": "D"},
            ],
        },
    )
    store.execute_query(
        """
        UNWIND $relationships AS row
        MATCH (source:__Entity__ {id: row.source_id, graph_name: $graph_name})
        MATCH (target:__Entity__ {id: row.target_id, graph_name: $graph_name})
        CREATE (source)-[r:RELATED_TO]->(target)
        SET r.graph_name = $graph_name, r.weight = row.weight
        """,
        {
            "graph_name": graph_name,
            "relationships": [
                {
                    "source_id": f"{graph_name}:a",
                    "target_id": f"{graph_name}:b",
                    "weight": 10.0,
                },
                {
                    "source_id": f"{graph_name}:c",
                    "target_id": f"{graph_name}:d",
                    "weight": 10.0,
                },
                {
                    "source_id": f"{graph_name}:b",
                    "target_id": f"{graph_name}:c",
                    "weight": 0.1,
                },
            ],
        },
    )
    stats = store.detect_communities(
        graph_name=graph_name,
        relationship_types=["RELATED_TO"],
        relationship_weight_property="weight",
        max_levels=2,
        random_seed=42,
    )
    assert stats
    assert single_count(
        store,
        """
        MATCH (:__Entity__ {graph_name: $graph_name})-[:IN_COMMUNITY]->
              (:Community {graph_name: $graph_name})
        RETURN count(*) AS count
        """,
        graph_name,
    ) >= 4


def _make_simple_graph_document(
    graph_name: str,
    document_id: str,
    entity_name: str = "Alice",
) -> GraphDocument:
    chunk_id = f"{document_id}:chunk-1"
    entity_id = f"{document_id}:entity-1"
    return GraphDocument(
        document=DocumentRecord(
            id=document_id,
            text_hash="hash",
            graph_name=graph_name,
            metadata={},
        ),
        chunks=[
            ChunkRecord(
                id=chunk_id,
                document_id=document_id,
                text=f"{entity_name} works at Acme.",
                index=0,
                graph_name=graph_name,
                metadata={},
            )
        ],
        entities=[
            EntityRecord(
                id=entity_id,
                type="Person",
                graph_name=graph_name,
                properties={"name": entity_name},
            )
        ],
        relationships=[],
        evidence_links=[
            EvidenceLink(
                chunk_id=chunk_id,
                entity_id=entity_id,
                graph_name=graph_name,
            )
        ],
    )


def assert_cross_document_rerun_idempotent(store, graph_name: str) -> None:
    """Rerunning the same document should not inflate entity counts."""
    doc = _make_simple_graph_document(graph_name, f"{graph_name}:doc-rerun")
    store.write_graph_document(doc)
    store.write_graph_document(doc)

    assert entity_count(store, graph_name) == 1
