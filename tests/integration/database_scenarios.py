"""Shared live-database scenarios for GraphStore implementations."""

from __future__ import annotations

from uuid import uuid4

from recon_graphrag.communities.context import (
    build_packed_reference_ids,
    build_reference_ids,
    pack_community_context,
    parse_community_context,
    render_community_context,
)
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
    result = await store.resolve_entities_normalized(
        graph_name=graph_name,
    )
    assert result["skipped"] is False
    assert result["merged_groups"] == 1
    assert entity_count(store, graph_name) == 1


async def assert_resolution_keeps_list_source_chunk_ids(
    store, graph_name: str
) -> None:
    """Regression for source_chunk_ids collapsing to a scalar on merge.

    Two duplicate entities that each hold a same-type edge to the same neighbour,
    both extracted from a single shared chunk, produce two identical one-element
    ``source_chunk_ids`` lists. ``apoc.refactor.mergeNodes(properties: 'combine')``
    dedupes those to one value and unwraps the list into a scalar string, which
    the resolver's own ``apoc.coll.flatten`` (and downstream list consumers) then
    reject. The resolver must keep the property a list.
    """
    store.execute_query(
        """
        CREATE (a:__Entity__:Organization {
            id: $a, graph_name: $g, name: 'Acme Corp', description: ''
        })
        CREATE (b:__Entity__:Organization {
            id: $b, graph_name: $g, name: 'acme corp', description: ''
        })
        CREATE (c:__Entity__:Quote {
            id: $c, graph_name: $g, name: 'Q009', description: ''
        })
        CREATE (a)-[:OFFERED {source_chunk_ids: ['chunk-1']}]->(c)
        CREATE (b)-[:OFFERED {source_chunk_ids: ['chunk-1']}]->(c)
        """,
        {
            "a": f"{graph_name}:acme-a",
            "b": f"{graph_name}:acme-b",
            "c": f"{graph_name}:quote",
            "g": graph_name,
        },
    )

    result = await store.resolve_entities_normalized(graph_name=graph_name)
    assert result["skipped"] is False
    assert result["merged_groups"] == 1

    rows = store.execute_query(
        """
        MATCH (:__Entity__ {graph_name: $g})-[r:OFFERED]->
              (:__Entity__ {graph_name: $g})
        RETURN apoc.meta.cypher.type(r.source_chunk_ids) AS ptype,
               r.source_chunk_ids AS chunks
        """,
        {"g": graph_name},
    )
    assert len(rows) == 1, f"expected one merged OFFERED edge, got {len(rows)}"
    assert "LIST" in rows[0]["ptype"], (
        "source_chunk_ids collapsed to a scalar after merge: "
        f"{rows[0]['chunks']!r}"
    )


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
    result = await store.resolve_entities_hybrid(
        graph_name=graph_name,
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
    result = await store.resolve_entities_hybrid(
        graph_name=graph_name,
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


def assert_directed_community_context(store, graph_name: str, *, id_func: str) -> None:
    """Exercise the real ranked query with backend-ordered endpoints and loops."""
    graph_name = f"{graph_name}-direction-{uuid4().hex}"
    params = {
        "graph_name": graph_name,
        "cid": f"{graph_name}:community",
        "isolated": f"{graph_name}:isolated",
        "outside": f"{graph_name}:outside",
    }
    try:
        seed_entities(store, graph_name, "Person", [
            {"id": f"{graph_name}:a", "name": "Alice"},
            {"id": f"{graph_name}:b", "name": "Bob"},
        ])
        # Query actual backend ordering, including elementId string order;
        # creation order and public IDs do not determine the regression.
        ordered = store.execute_query(
            f"""
            MATCH (e:__Entity__ {{graph_name: $graph_name}})
            RETURN e.id AS id, e.name AS name, {id_func}(e) AS internal_id
            ORDER BY internal_id DESC
            """,
            params,
        )
        source, target = ordered
        assert source["internal_id"] > target["internal_id"]
        params.update({"source": source["id"], "target": target["id"]})
        store.execute_query(
            """
            MATCH (source:__Entity__ {graph_name: $graph_name, id: $source}),
                  (target:__Entity__ {graph_name: $graph_name, id: $target})
            CREATE (c:Community {graph_name: $graph_name, id: $cid, level: 0})
            CREATE (source)-[:IN_COMMUNITY]->(c)
            CREATE (target)-[:IN_COMMUNITY]->(c)
            CREATE (:__Entity__:Person {
                graph_name: $graph_name, id: $isolated, name: 'Isolated'
            })-[:IN_COMMUNITY]->(c)
            CREATE (outside:__Entity__:Person {
                graph_name: $graph_name, id: $outside, name: 'Outside'
            })
            CREATE (source)-[:REPORTS_TO {
                description: 'Higher internal ID reports to lower', observation_count: 3
            }]->(target)
            CREATE (source)-[:KNOWS]->(target)
            CREATE (target)-[:KNOWS]->(source)
            CREATE (source)-[:KNOWS]->(source)
            CREATE (source)-[:OUTSIDE_COMMUNITY]->(outside)
            """,
            params,
        )
        rows = store.get_community_ranked_context(graph_name, params["cid"], level=0)
        expected_edges = {
            (source["id"], "REPORTS_TO", target["id"]),
            (source["id"], "KNOWS", target["id"]),
            (target["id"], "KNOWS", source["id"]),
            (source["id"], "KNOWS", source["id"]),
        }
        edge_rows = [row for row in rows if row["rel_type"] is not None]
        assert len(edge_rows) == 4
        assert {(r["e_id"], r["rel_type"], r["other_id"]) for r in edge_rows} == expected_edges
        assert [r["e_id"] for r in rows if r["rel_type"] is None] == [params["isolated"]]
        reports_to = next(r for r in edge_rows if r["rel_type"] == "REPORTS_TO")
        assert reports_to["rel_description"] == "Higher internal ID reports to lower"
        assert reports_to["observation_count"] == 3
        assert all(r["combined_degree"] == r["e_degree"] + r["other_degree"] for r in edge_rows)

        context = parse_community_context(params["cid"], 0, rows)
        assert {(e.source.id, e.relationship_type, e.target.id) for e in context.edges} == expected_edges
        assert len(context.edges) == 4
        assert [e.id for e in context.entities] == [params["isolated"]]
        packed = pack_community_context(context, max_tokens=10_000)
        assert packed.included_edges == 4
        assert packed.included_entities == 1
        expected_keys = {f"{s}:{rel}:{t}" for s, rel, t in expected_edges}
        expected_ids = {source["id"], target["id"], params["isolated"]} | expected_keys
        assert set(build_reference_ids(context)) == expected_ids
        assert set(build_packed_reference_ids(context, packed)) == expected_ids
        names = {source["id"]: source["name"], target["id"]: target["name"]}
        for text in (render_community_context(context), packed.text):
            for s, rel, t in expected_edges:
                assert f"{names[s]} --[{rel}]--> {names[t]}" in text
            assert f"{target['name']} --[REPORTS_TO]--> {source['name']}" not in text
            assert "Outside" not in text
    finally:
        cleanup_graph(store, graph_name)
