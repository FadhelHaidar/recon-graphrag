"""Opt-in retrieval regressions on disposable backends, without providers.

Checks the two retained optimizations against real Cypher on both backends:
overlapping and duplicate references must need a single chunk-metadata
materialization and must return exactly what per-type resolution returned.
Also pins that publishing a citation is still read-only.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

import pytest

from recon_graphrag.retrieval.citations import (
    resolve_claim_citations,
    resolve_entity_citations,
    resolve_reference_citations,
    resolve_relationship_citations,
)
from tests.integration.factories import get_memgraph_store, get_neo4j_store
from tests.integration.support import cleanup_graph, require_integration_env

pytestmark = [pytest.mark.integration, pytest.mark.database]

ALICE_KEY = "doc1:c1:Person:p1"
BOB_KEY = "doc1:c2:Person:p1"
CLAIM_KEY = "claim:1"
REL_KEY = f"{ALICE_KEY}:KNOWS:{BOB_KEY}"

REFS = [
    {"target_id": ALICE_KEY, "target_type": "entity"},
    {"target_id": ALICE_KEY, "target_type": "entity"},
    {"target_id": BOB_KEY, "target_type": "entity"},
    {"target_id": CLAIM_KEY, "target_type": "claim"},
    {"target_id": REL_KEY, "target_type": "relationship"},
]


@pytest.fixture(params=["neo4j", "memgraph"])
def live_store(request):
    prefix = request.param.upper()
    require_integration_env(
        "RUN_DATABASE_INTEGRATION_TESTS",
        [f"{prefix}_URL"],
        "retrieval citation regressions",
        fail_on_missing=True,
    )
    store = get_neo4j_store() if request.param == "neo4j" else get_memgraph_store()
    graph = f"f9-retrieval-citations-{uuid4().hex}"
    try:
        yield store, graph, request.param
    finally:
        cleanup_graph(store, graph)
        store.driver.close()


@pytest.fixture
def citation_graph(live_store):
    store, graph, _backend = live_store
    store.execute_query(
        """
        CREATE (alice:__Entity__:Person {graph_name:$g, id:'e-alice',
                 canonical_key:$alice, human_readable_id:'p1', name:'Alice'})
        CREATE (bob:__Entity__:Person {graph_name:$g, id:'e-bob',
                 canonical_key:$bob, human_readable_id:'p1', name:'Bob'})
        CREATE (claim:Claim {graph_name:$g, id:$claim, claim_type:'fact',
                 description:'Alice knows Bob'})
        CREATE (shared:Chunk {graph_name:$g, id:'chunk:shared', text:'shared text',
                 page_start:1, page_end:1})
        CREATE (own:Chunk {graph_name:$g, id:'chunk:alice', text:'Alice text',
                 page_start:2, page_end:2})
        CREATE (other:Chunk {graph_name:$g, id:'chunk:bob', text:'Bob text',
                 page_start:3, page_end:3})
        CREATE (claim_chunk:Chunk {graph_name:$g, id:'chunk:claim', text:'claim text',
                 page_start:4, page_end:4})
        CREATE (rel_chunk:Chunk {graph_name:$g, id:'chunk:rel', text:'rel text',
                 page_start:5, page_end:5})
        CREATE (own)-[:FROM_CHUNK]->(alice)
        CREATE (shared)-[:FROM_CHUNK]->(alice)
        CREATE (other)-[:FROM_CHUNK]->(bob)
        CREATE (claim)-[:SOURCED_FROM]->(claim_chunk)
        CREATE (claim)-[:SOURCED_FROM]->(shared)
        CREATE (alice)-[r:KNOWS {weight:1.0,
                 source_chunk_ids:['chunk:rel', 'chunk:shared']}]->(bob)
        SET r.canonical_key = $rel, r.human_readable_id = $rel
        """,
        {
            "g": graph,
            "alice": ALICE_KEY,
            "bob": BOB_KEY,
            "claim": CLAIM_KEY,
            "rel": REL_KEY,
        },
    )
    return store, graph, _backend


def _dedupe(citations: list) -> list:
    seen: set[tuple[str, str, int | None, int | None]] = set()
    deduped = []
    for citation in citations:
        key = (
            citation.document_id,
            citation.chunk_id,
            citation.page_start,
            citation.page_end,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(citation)
    return deduped


def _count_materializations(store: Any) -> list[list[str]]:
    calls: list[list[str]] = []
    original = store.resolve_chunk_citations

    def recording(graph_name: str, chunk_ids: list[str]) -> list[dict]:
        calls.append(list(chunk_ids))
        return original(graph_name, chunk_ids)

    store.resolve_chunk_citations = recording
    return calls


def test_overlapping_references_materialize_once(citation_graph):
    store, graph, _backend = citation_graph
    calls = _count_materializations(store)

    citations = resolve_reference_citations(store, graph, REFS)

    assert len(calls) == 1
    # Overlapping chunks appear once in the materialization request.
    assert len(calls[0]) == len(set(calls[0]))
    assert {c.chunk_id for c in citations} == {
        "chunk:alice",
        "chunk:shared",
        "chunk:bob",
        "chunk:claim",
        "chunk:rel",
    }


def test_batched_resolution_matches_per_type_resolution(citation_graph):
    store, graph, _backend = citation_graph

    batched = resolve_reference_citations(store, graph, REFS)

    per_type = _dedupe(
        resolve_entity_citations(cast(Any, store), [ALICE_KEY, BOB_KEY], graph)
        + resolve_claim_citations(cast(Any, store), graph, [CLAIM_KEY])
        + resolve_relationship_citations(cast(Any, store), graph, [REL_KEY])
    )

    assert [(c.chunk_id, c.page_start) for c in batched] == [
        (c.chunk_id, c.page_start) for c in per_type
    ]


def test_readable_id_still_resolves_to_every_matching_entity(citation_graph):
    """Release behavior: readable ids are matched wherever they appear."""
    store, graph, _backend = citation_graph

    assert {c.chunk_id for c in resolve_entity_citations(store, ["p1"], graph)} == {
        "chunk:alice",
        "chunk:shared",
        "chunk:bob",
    }


def test_resolution_does_not_write(citation_graph):
    store, graph, _backend = citation_graph
    before = store.execute_query(
        """
        MATCH (n) WHERE n.graph_name = $g
        RETURN collect(DISTINCT [n.id, labels(n), n.canonical_key,
                                 n.human_readable_id, n.description]) AS nodes
        """,
        {"g": graph},
    )
    rel_before = store.execute_query(
        """
        MATCH (s:__Entity__ {graph_name:$g})-[r]->(t:__Entity__ {graph_name:$g})
        RETURN type(r) AS kind, r.canonical_key AS key,
               r.weight AS weight, r.source_chunk_ids AS chunks
        """,
        {"g": graph},
    )

    resolve_reference_citations(store, graph, REFS)

    after = store.execute_query(
        """
        MATCH (n) WHERE n.graph_name = $g
        RETURN collect(DISTINCT [n.id, labels(n), n.canonical_key,
                                 n.human_readable_id, n.description]) AS nodes
        """,
        {"g": graph},
    )
    rel_after = store.execute_query(
        """
        MATCH (s:__Entity__ {graph_name:$g})-[r]->(t:__Entity__ {graph_name:$g})
        RETURN type(r) AS kind, r.canonical_key AS key,
               r.weight AS weight, r.source_chunk_ids AS chunks
        """,
        {"g": graph},
    )

    assert after == before
    assert rel_after == rel_before
