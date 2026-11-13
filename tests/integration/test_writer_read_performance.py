"""Opt-in writer state-read profiling: real databases, provider-free.

Run with::

    RUN_DATABASE_INTEGRATION_TESTS=1 uv run pytest \
        tests/integration/test_writer_read_performance.py -m database

The harness seeds one graph per backend, compares the baseline combined-OR
relationship read against the specialized predicates on the same seeded state,
then reports alternating warm timings for the reads and for fresh, small, and
large replacement writes. Timing results are informational; equivalence is
asserted.
"""

from __future__ import annotations

import json
import statistics
import time
from typing import Any, cast
from uuid import uuid4

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
from recon_graphrag.pipelines.writer import CypherGraphWriter
from tests.integration.factories import get_memgraph_store, get_neo4j_store
from tests.integration.support import cleanup_graph, require_integration_env
from tests.pipelines.test_writer_replacement import build_document

pytestmark = [pytest.mark.integration, pytest.mark.database]

PROJECTION = """
RETURN r.id AS id, s.id AS source_id, t.id AS target_id,
       type(r) AS type,
       coalesce(r.source_chunk_ids, []) AS source_chunk_ids,
       r.descriptions AS descriptions,
       r.description_sources AS description_sources,
       r.description AS description,
       r.description_summary_status AS description_summary_status,
       r.strength AS strength,
       r.strength_sources AS strength_sources,
       r.properties AS properties
"""

ENVELOPE = (
    "MATCH (s:__Entity__ {graph_name: $graph_name})-[r]->"
    "(t:__Entity__ {graph_name: $graph_name}) "
    "WHERE r.graph_name = $graph_name AND "
)

BASELINE_OR = f"{ENVELOPE}(r.id IN $rel_ids OR any(chunk IN coalesce(r.source_chunk_ids, [])\n                               WHERE chunk IN $chunk_ids))\n{PROJECTION}"
REL_ONLY = f"{ENVELOPE}r.id IN $rel_ids\n{PROJECTION}"
CHUNK_ONLY = f"{ENVELOPE}any(chunk IN coalesce(r.source_chunk_ids, []) WHERE chunk IN $chunk_ids)\n{PROJECTION}"

CHUNK_COUNT = 30
ENTITY_COUNT = 200
OWNED_RELS = 1200
UNRELATED_RELS = 1500
PARALLEL_PAIRS = 40
READ_REPEATS = 6
FOREIGN_ENTITIES = 2
# Matching self-loop, NULL-id, and foreign-endpoint rows; 'seed-noown' has an
# id but no ownership list.
EDGE_RELS = 4
# Edge rows owned by an even (kept) chunk: the matching self-loop and the
# NULL-id relationship.
EDGE_CHUNK_ROWS = 2

SEEDS = {
    "documents": 2,
    "chunks": CHUNK_COUNT,
    "entities": ENTITY_COUNT,
    "relationships": OWNED_RELS + 2 * PARALLEL_PAIRS + EDGE_RELS,
}


def _seed_graph(store, graph: str, *, include_null_id: bool = True) -> None:
    """Seed chunks, entities, owned, parallel, unrelated, and foreign rows.

    ``include_null_id`` seeds the owned relationship without an id property. The
    reader must return it, but a replacement write rejects it (B05). Write
    differentials therefore seed without it.
    """
    params = {"graph_name": graph}
    store.execute_query(
        "CREATE (:Document {id: 'seed-doc', graph_name: $graph_name, text_hash: 'h'})",
        params,
    )
    store.execute_query(
        "UNWIND range(0, $chunks - 1) AS i "
        "CREATE (:Chunk {id: 'seed-chunk-' + toString(i), graph_name: $graph_name, "
        "text: 'kept ' + toString(i), index: i})",
        {**params, "chunks": CHUNK_COUNT},
    )
    store.execute_query(
        "MATCH (d:Document {id: 'seed-doc', graph_name: $graph_name}) "
        "MATCH (c:Chunk {graph_name: $graph_name}) "
        "CREATE (c)-[:PART_OF]->(d)",
        params,
    )
    store.execute_query(
        "UNWIND range(0, $entities - 1) AS i "
        "CREATE (:__Entity__:Person {id: 'seed-e-' + toString(i), graph_name: $graph_name})",
        {**params, "entities": ENTITY_COUNT},
    )
    store.execute_query(
        "UNWIND range(0, $owned - 1) AS i "
        "MATCH (s:__Entity__ {id: 'seed-e-' + toString(i % $entities), graph_name: $graph_name}) "
        "MATCH (t:__Entity__ {id: 'seed-e-' + toString((i * 3 + 1) % $entities), graph_name: $graph_name}) "
        "CREATE (s)-[:RELATES {graph_name: $graph_name, id: 'seed-rel-' + toString(i), "
        "source_chunk_ids: ['seed-chunk-' + toString(i % $chunks)], strength: toFloat(i), "
        "strength_sources: ['[\"seed-doc\", ' + toString(toFloat(i)) + ']']}]->(t)",
        {
            **params,
            "owned": OWNED_RELS,
            "entities": ENTITY_COUNT,
            "chunks": CHUNK_COUNT,
        },
    )
    store.execute_query(
        "UNWIND range(0, $pairs - 1) AS j "
        "MATCH (s:__Entity__ {id: 'seed-e-' + toString(j % $entities), graph_name: $graph_name}) "
        "MATCH (t:__Entity__ {id: 'seed-e-' + toString((j + 5) % $entities), graph_name: $graph_name}) "
        "CREATE (s)-[:RELATES {graph_name: $graph_name, id: 'seed-par-' + toString(j) + '-a', "
        "source_chunk_ids: ['seed-chunk-' + toString(j % $chunks)], strength: 1000.0 + j, "
        "strength_sources: ['[\"seed-doc\", ' + toString(1000.0 + j) + ']']}]->(t) "
        "CREATE (s)-[:RELATES {graph_name: $graph_name, id: 'seed-par-' + toString(j) + '-b', "
        "source_chunk_ids: ['seed-chunk-' + toString(j % $chunks)], strength: 2000.0 + j, "
        "strength_sources: ['[\"seed-doc\", ' + toString(2000.0 + j) + ']']}]->(t)",
        {**params, "pairs": PARALLEL_PAIRS, "entities": ENTITY_COUNT, "chunks": CHUNK_COUNT},
    )
    store.execute_query(
        "UNWIND range(0, $unrelated - 1) AS i "
        "MATCH (s:__Entity__ {id: 'seed-e-' + toString(i % $entities), graph_name: $graph_name}) "
        "MATCH (t:__Entity__ {id: 'seed-e-' + toString((i * 7 + 3) % $entities), graph_name: $graph_name}) "
        "CREATE (s)-[:RELATES {graph_name: $graph_name, id: 'seed-unrel-' + toString(i), "
        "source_chunk_ids: ['unowned-chunk-' + toString(i)]}]->(t)",
        {**params, "unrelated": UNRELATED_RELS, "entities": ENTITY_COUNT},
    )
    store.execute_query(
        "MATCH (s:__Entity__ {id: 'seed-e-0', graph_name: $graph_name}) "
        "CREATE (s)-[:RELATES {graph_name: $foreign, id: 'seed-foreign', "
        "source_chunk_ids: ['seed-chunk-0']}]->(s)",
        {**params, "foreign": graph + "-foreign"},
    )
    store.execute_query(
        "UNWIND range(0, $foreign_entities - 1) AS i "
        "CREATE (:__Entity__ {id: 'foreign-e-' + toString(i), graph_name: $foreign})",
        {**params, "foreign": graph + "-foreign", "foreign_entities": FOREIGN_ENTITIES},
    )
    store.execute_query(
        "CREATE (:Document {id: 'seed-doc-nochunks', graph_name: $graph_name, text_hash: 'h'})",
        params,
    )
    # Relationship with our graph name but foreign endpoints: the envelope
    # must exclude it from every read of this graph.
    store.execute_query(
        "MATCH (fe:__Entity__ {id: 'foreign-e-0', graph_name: $foreign}) "
        "CREATE (fe)-[:RELATES {graph_name: $graph_name, id: 'seed-fend', "
        "source_chunk_ids: ['seed-chunk-2']}]->(fe)",
        {**params, "foreign": graph + "-foreign"},
    )
    # Matching self-loop owned by an even (kept) chunk.
    store.execute_query(
        "MATCH (s:__Entity__ {id: 'seed-e-1', graph_name: $graph_name}) "
        "CREATE (s)-[:RELATES {graph_name: $graph_name, id: 'seed-self', "
        "source_chunk_ids: ['seed-chunk-0']}]->(s)",
        params,
    )
    # Owned relationship without an id property. 'shadow-chunk' is not one of
    # the document's chunks, so a replacement never empties this row's
    # ownership list (the writer cannot remove an id-less row by id).
    if include_null_id:
        store.execute_query(
            "MATCH (s:__Entity__ {id: 'seed-e-2', graph_name: $graph_name}) "
            "MATCH (t:__Entity__ {id: 'seed-e-3', graph_name: $graph_name}) "
            "CREATE (s)-[:RELATES {graph_name: $graph_name, "
            "source_chunk_ids: ['seed-chunk-0', 'shadow-chunk']}]->(t)",
            params,
        )
    # Relationship with an id but no ownership list.
    store.execute_query(
        "MATCH (s:__Entity__ {id: 'seed-e-4', graph_name: $graph_name}) "
        "MATCH (t:__Entity__ {id: 'seed-e-5', graph_name: $graph_name}) "
        "CREATE (s)-[:RELATES {graph_name: $graph_name, id: 'seed-noown'}]->(t)",
        params,
    )


def _read_params(graph: str) -> dict:
    rel_ids = [f"seed-rel-{i}" for i in range(0, OWNED_RELS, 8)]
    rel_ids += [f"seed-par-{j}-a" for j in range(0, PARALLEL_PAIRS, 2)]
    rel_ids += ["seed-self", "seed-noown"]
    chunk_ids = [f"seed-chunk-{i}" for i in range(0, CHUNK_COUNT, 2)]
    return {"graph_name": graph, "rel_ids": rel_ids, "chunk_ids": chunk_ids}


def _timed(store, query: str, params: dict) -> tuple:
    started = time.perf_counter()
    rows = list(store.execute_query(query, params))
    return rows, time.perf_counter() - started


def _alternating_timings(store, variants, repeats):
    """Measure variants in alternating rounds; return {label: [seconds]}."""
    samples: dict[str, list[float]] = {label: [] for label, _, _ in variants}
    rows_by_label: dict[str, list] = {}
    for _ in range(repeats):
        for label, query, query_params in variants:
            rows, elapsed = _timed(store, query, query_params)
            samples[label].append(elapsed)
            rows_by_label[label] = rows
    return samples, rows_by_label


def _report(label: str, samples: list[float]) -> str:
    return (
        f"{label:26s} min={min(samples) * 1000:7.2f}ms "
        f"med={statistics.median(samples) * 1000:7.2f}ms "
        f"max={max(samples) * 1000:7.2f}ms "
        f"stdev={statistics.pstdev(samples) * 1000:6.2f}ms"
    )


def _server_info(store) -> str:
    try:
        info = store.driver.get_server_info()
        return f"{info.address} agent={info.agent}"
    except (AttributeError, OSError, RuntimeError):
        return "unavailable"


@pytest.fixture(params=["neo4j", "memgraph"])
def perf_backend(request):
    prefix = request.param.upper()
    require_integration_env(
        "RUN_DATABASE_INTEGRATION_TESTS",
        [f"{prefix}_URL"],
        "writer state-read profiling",
        fail_on_missing=True,
    )
    store = get_neo4j_store() if request.param == "neo4j" else get_memgraph_store()
    yield store, request.param
    store.driver.close()


@pytest.fixture
def perf_graph(perf_backend):
    store, backend = perf_backend
    graph = f"writer-read-perf-{backend}-{uuid4().hex[:8]}"
    try:
        yield store, backend, graph
    finally:
        cleanup_graph(store, graph)
        cleanup_graph(store, graph + "-foreign")


def test_relationship_state_read_equivalence_on_seeded_state(perf_graph):
    """Baseline OR and specialized predicates return identical rows in order."""
    store, _backend, graph = perf_graph
    _seed_graph(store, graph)
    params = _read_params(graph)

    rel_only_params = {**params, "chunk_ids": []}
    chunk_only_params = {**params, "rel_ids": []}
    baseline_rel = list(store.execute_query(BASELINE_OR, rel_only_params))
    rel_only = list(store.execute_query(REL_ONLY, rel_only_params))
    baseline_chunk = list(store.execute_query(BASELINE_OR, chunk_only_params))
    chunk_only = list(store.execute_query(CHUNK_ONLY, chunk_only_params))

    assert rel_only == baseline_rel
    assert chunk_only == baseline_chunk
    assert len(rel_only) == len(params["rel_ids"])
    expected_chunk_rows = (
        (CHUNK_COUNT // 2) * (OWNED_RELS // CHUNK_COUNT)
        + PARALLEL_PAIRS
        + EDGE_CHUNK_ROWS
    )
    assert len(chunk_only) == expected_chunk_rows
    assert all(row["id"] != "seed-foreign" for row in rel_only + chunk_only)
    assert all(row["id"] != "seed-fend" for row in rel_only + chunk_only)
    chunk_only_ids = {row["id"] for row in chunk_only}
    assert "seed-self" in chunk_only_ids
    assert any(row["id"] is None for row in chunk_only)
    assert "seed-noown" in {row["id"] for row in rel_only}
    par_only = [row for row in rel_only if row["id"].startswith("seed-par")]
    par_base = [row for row in baseline_rel if row["id"].startswith("seed-par")]
    assert par_only == par_base
    assert len(par_only) == len([rid for rid in params["rel_ids"] if "par" in rid])


BASELINE_PREDICATE = (
    "(r.id IN $rel_ids OR any(chunk IN coalesce(r.source_chunk_ids, [])"
    " WHERE chunk IN $chunk_ids))"
)
VOLATILE_PROPS = frozenset({"uid", "updated", "created", "graph_name"})


def _stable_props(props: dict[str, Any]) -> dict[str, Any]:
    """Drop volatile properties so two graph dumps can be compared."""
    return {key: value for key, value in props.items() if key not in VOLATILE_PROPS}


def _dump_graph(store: Any, graph: str) -> list[str]:
    """Serialize every node and relationship of one graph, order-free."""
    nodes = store.execute_query(
        "MATCH (n {graph_name: $graph_name}) "
        "RETURN labels(n) AS labels, properties(n) AS props",
        {"graph_name": graph},
    )
    rels = store.execute_query(
        "MATCH (s {graph_name: $graph_name})-[r]->(t {graph_name: $graph_name}) "
        "RETURN type(r) AS type, s.id AS source_id, t.id AS target_id, "
        "properties(r) AS props",
        {"graph_name": graph},
    )
    entries = [
        json.dumps(
            {"labels": row["labels"], "props": _stable_props(row["props"])},
            sort_keys=True,
            default=str,
        )
        for row in nodes
    ]
    entries += [
        json.dumps(
            {
                "type": row["type"],
                "source_id": row["source_id"],
                "target_id": row["target_id"],
                "props": _stable_props(row["props"]),
            },
            sort_keys=True,
            default=str,
        )
        for row in rels
    ]
    return sorted(entries)


class BaselineRelationshipStore:
    """Proxy that rewrites production relationship reads to the baseline OR."""

    def __init__(self, store: Any, predicate: str) -> None:
        self._store = store
        self._predicate = predicate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def execute_query(self, query: str, params: dict) -> list[dict[str, Any]]:
        if "type(r) AS type" in query and "$rel_ids" in query:
            head, rest = query.split("AND ", 1)
            _predicate_line, tail = rest.split("\n", 1)
            query = f"{head}AND {self._predicate}\n{tail}"
        return self._store.execute_query(query, params)


def _seed_relationship_record(graph: str, relationship_id: str) -> RelationshipRecord:
    """Build a record whose id and endpoints exist in the seeded graph."""
    if relationship_id.startswith("seed-par-"):
        pair = int(relationship_id.split("-")[2])
        source = f"seed-e-{pair % ENTITY_COUNT}"
        target = f"seed-e-{(pair + 5) % ENTITY_COUNT}"
    elif relationship_id == "seed-self":
        source = target = "seed-e-1"
    elif relationship_id == "seed-noown":
        source, target = "seed-e-4", "seed-e-5"
    else:
        index = int(relationship_id.removeprefix("seed-rel-"))
        source = f"seed-e-{index % ENTITY_COUNT}"
        target = f"seed-e-{(index * 3 + 1) % ENTITY_COUNT}"
    return RelationshipRecord(
        id=relationship_id,
        source_id=source,
        target_id=target,
        type="RELATES",
        graph_name=graph,
    )


def _reader_document(
    graph: str, document_id: str, relationships: list[RelationshipRecord]
) -> GraphDocument:
    """GraphDocument for _read_document_state; ids exist in the seeded graph."""
    return GraphDocument(
        document=DocumentRecord(id=document_id, text_hash="h", graph_name=graph),
        chunks=[],
        entities=[
            EntityRecord(
                id=f"seed-e-{index}",
                type="Person",
                properties={"name": f"S{index}"},
                graph_name=graph,
            )
            for index in range(2)
        ],
        relationships=list(relationships),
        evidence_links=[],
        claims=[],
    )


def _replacement_document(graph: str) -> GraphDocument:
    """Replacement keeping two chunks and one edge of each even parallel pair."""
    entities = []
    for pair in range(0, PARALLEL_PAIRS, 2):
        for endpoint in (pair % ENTITY_COUNT, (pair + 5) % ENTITY_COUNT):
            entities.append(
                EntityRecord(
                    id=f"seed-e-{endpoint}",
                    type="Person",
                    properties={"name": f"S{endpoint}"},
                    graph_name=graph,
                )
            )
    return GraphDocument(
        document=DocumentRecord(id="seed-doc", text_hash="h2", graph_name=graph),
        chunks=[
            ChunkRecord(
                id=f"seed-chunk-{index}",
                document_id="seed-doc",
                text=f"kept {index}",
                index=index,
                graph_name=graph,
            )
            for index in range(2)
        ],
        entities=entities,
        relationships=[
            RelationshipRecord(
                id=f"seed-par-{pair}-a",
                source_id=f"seed-e-{pair % ENTITY_COUNT}",
                target_id=f"seed-e-{(pair + 5) % ENTITY_COUNT}",
                type="RELATES",
                # Realistic replacement: the kept version still attributes
                # chunk 0, so the row keeps nonempty provenance ownership.
                properties={"kept": True, "source_chunk_ids": ["seed-chunk-0"]},
                graph_name=graph,
                strength=5000.0 + pair,
            )
            for pair in range(0, PARALLEL_PAIRS, 2)
        ],
        evidence_links=[
            EvidenceLink(
                chunk_id=f"seed-chunk-{index}",
                entity_id=entity.id,
                graph_name=graph,
            )
            for index in range(2)
            for entity in entities
        ],
        claims=[],
    )


def _seed_write_state(store, graph: str) -> None:
    """Seed one edge per logical key: a state a replacement write accepts."""
    store.execute_query(
        "UNWIND range(0, $entities - 1) AS i "
        "CREATE (:__Entity__:Person {id: 'seed-e-' + toString(i), "
        "graph_name: $graph_name, name: 'S' + toString(i)})",
        {"graph_name": graph, "entities": ENTITY_COUNT},
    )
    store.execute_query(
        "CREATE (d:Document {id: 'seed-doc', graph_name: $graph_name, text_hash: 'h'}) "
        "WITH d "
        "UNWIND range(0, $chunks - 1) AS i "
        "CREATE (c:Chunk {id: 'seed-chunk-' + toString(i), graph_name: $graph_name, "
        "text: 'kept ' + toString(i), index: i}) "
        "CREATE (c)-[:PART_OF]->(d)",
        {"graph_name": graph, "chunks": 2},
    )
    store.execute_query(
        "UNWIND range(0, $pairs - 1) AS j "
        "MATCH (s:__Entity__ {id: 'seed-e-' + toString(j % $entities), graph_name: $graph_name}) "
        "MATCH (t:__Entity__ {id: 'seed-e-' + toString((j + 5) % $entities), graph_name: $graph_name}) "
        "CREATE (s)-[:RELATES {graph_name: $graph_name, id: 'seed-par-' + toString(j) + '-a', "
        "source_chunk_ids: ['seed-chunk-' + toString(j % $chunks)], strength: 1000.0 + j, "
        "strength_sources: ['[\"seed-doc\", ' + toString(1000.0 + j) + ']']}]->(t)",
        {
            "graph_name": graph,
            "pairs": PARALLEL_PAIRS,
            "entities": ENTITY_COUNT,
            "chunks": 2,
        },
    )


def test_relationship_state_read_differential_on_seeded_state(perf_graph):
    """Production reader rows and full writes match the baseline reader."""
    store, _backend, graph = perf_graph
    _seed_graph(store, graph)

    writer = CypherGraphWriter(store)
    baseline_writer = CypherGraphWriter(
        cast(GraphStore, BaselineRelationshipStore(store, BASELINE_PREDICATE))
    )
    rel_ids = [f"seed-rel-{i}" for i in range(0, OWNED_RELS, 16)]
    rel_ids += [f"seed-par-{j}-a" for j in range(0, PARALLEL_PAIRS, 2)]
    rel_ids += ["seed-self", "seed-noown"]
    rel_records = [
        _seed_relationship_record(graph, relationship_id)
        for relationship_id in rel_ids
    ]

    # Reader differential: production predicate selection versus the forced
    # baseline OR, compared as exact row lists (no sorting or normalization).
    both_doc = _reader_document(graph, "seed-doc", rel_records)
    assert writer._read_document_state(both_doc).relationships == (
        baseline_writer._read_document_state(both_doc).relationships
    )

    rel_only_doc = _reader_document(graph, "seed-doc-nochunks", rel_records)
    rel_only_state = writer._read_document_state(rel_only_doc)
    assert rel_only_state.chunk_ids == []
    assert rel_only_state.relationships == (
        baseline_writer._read_document_state(rel_only_doc).relationships
    )

    chunk_only_doc = _reader_document(graph, "seed-doc", [])
    chunk_state = writer._read_document_state(chunk_only_doc)
    assert chunk_state.relationships == (
        baseline_writer._read_document_state(chunk_only_doc).relationships
    )

    chunk_only_ids = {row["id"] for row in chunk_state.relationships}
    assert "seed-self" in chunk_only_ids
    assert any(row["id"] is None for row in chunk_state.relationships)
    rel_only_ids = {row["id"] for row in rel_only_state.relationships}
    assert "seed-noown" in rel_only_ids
    for rows in (rel_only_state.relationships, chunk_state.relationships):
        assert {"seed-fend", "seed-foreign"}.isdisjoint(row["id"] for row in rows)

    # Write differential: identical replacement documents written by both
    # writers on identically seeded graphs. Receipts and final graphs must
    # match, including which parallel-edge row survives the last-wins rebuild.
    graph_prod = f"{graph}-prod"
    graph_baseline = f"{graph}-baseline"
    try:
        # The write phases need a state that relationship identity accepts:
        # one stored edge per logical key, see B02.
        _seed_write_state(store, graph_prod)
        _seed_write_state(store, graph_baseline)
        production_receipt = writer.write_graph_document(
            _replacement_document(graph_prod)
        )
        baseline_receipt = baseline_writer.write_graph_document(
            _replacement_document(graph_baseline)
        )
        assert production_receipt == baseline_receipt
        assert production_receipt["chunks"] == 2
        # Relationship receipts count matched rels per MERGE (SET applies to
        # every existing rel on the endpoint pair), so they exceed the row
        # count when parallel edges share endpoints. Equality between the two
        # writers is the contract under test; the dump equality pins rows.
        assert _dump_graph(store, graph_prod) == _dump_graph(store, graph_baseline)
    finally:
        cleanup_graph(store, graph_prod)
        cleanup_graph(store, graph_prod + "-foreign")
        cleanup_graph(store, graph_baseline)
        cleanup_graph(store, graph_baseline + "-foreign")


def test_relationship_state_read_profiles(perf_graph, capsys):
    """Report alternating warm timings for reads and replacement writes."""
    store, backend, graph = perf_graph
    _seed_graph(store, graph)
    params = _read_params(graph)

    rel_only_params = {**params, "chunk_ids": []}
    chunk_only_params = {**params, "rel_ids": []}
    variants = [
        ("baseline-both", BASELINE_OR, params),
        ("baseline-rel-only", BASELINE_OR, rel_only_params),
        ("rel-only r.id IN", REL_ONLY, rel_only_params),
        ("baseline-chunk-only", BASELINE_OR, chunk_only_params),
        ("chunk-only any()", CHUNK_ONLY, chunk_only_params),
    ]

    samples, rows_by_label = _alternating_timings(store, variants, READ_REPEATS)
    assert rows_by_label["rel-only r.id IN"] == rows_by_label["baseline-rel-only"]
    assert rows_by_label["chunk-only any()"] == rows_by_label["baseline-chunk-only"]
    assert len(rows_by_label["baseline-both"]) == (
        (CHUNK_COUNT // 2) * (OWNED_RELS // CHUNK_COUNT)
        + PARALLEL_PAIRS
        + EDGE_CHUNK_ROWS
        + 1  # 'seed-noown' matches by id only in the combined predicate.
    )

    lines = [f"--- {backend} server={_server_info(store)}", f"graph sizes: {SEEDS} + {UNRELATED_RELS} unrelated rels, rel_ids={len(params['rel_ids'])}, chunk_ids={len(params['chunk_ids'])}"]
    for label, _, _ in variants:
        lines.append(_report(label, samples[label]))

    writer = CypherGraphWriter(store)
    # The write workload uses ids disjoint from the seeded read state so the
    # fresh write starts from an empty document state.
    entities = [(f"perf-e-{i}", "Person", None) for i in range(ENTITY_COUNT)]
    owned = [
        ("LINK", f"perf-e-{i % ENTITY_COUNT}", f"perf-e-{(i * 3 + 1) % ENTITY_COUNT}", None)
        for i in range(OWNED_RELS)
    ]
    fresh = build_document(
        chunks=[f"perf-chunk-{i}" for i in range(CHUNK_COUNT)],
        document_id="perf-doc",
        graph_name=graph,
        entities=entities,
        relationships=owned,
    )
    started = time.perf_counter()
    fresh_receipt = writer.write_graph_document(fresh)
    lines.append(f"fresh ingestion          total={((time.perf_counter() - started) * 1000):7.2f}ms receipt={fresh_receipt}")

    small = build_document(
        chunks=["perf-chunk-0"],
        document_id="perf-doc",
        graph_name=graph,
        entities=entities[:1],
        relationships=owned[:1],
    )
    started = time.perf_counter()
    small_receipt = writer.write_graph_document(small)
    lines.append(f"small replacement        total={((time.perf_counter() - started) * 1000):7.2f}ms receipt={small_receipt}")

    large = build_document(
        chunks=["perf-chunk-0"],
        document_id="perf-doc",
        graph_name=graph,
        entities=entities,
        relationships=owned,
    )
    started = time.perf_counter()
    large_receipt = writer.write_graph_document(large)
    lines.append(f"large replacement        total={((time.perf_counter() - started) * 1000):7.2f}ms receipt={large_receipt}")

    assert fresh_receipt["chunks"] == CHUNK_COUNT
    assert fresh_receipt["relationships"] == OWNED_RELS
    assert fresh_receipt["evidence_links"] == ENTITY_COUNT
    assert small_receipt["chunks"] == 1
    assert small_receipt["relationships"] == 1
    assert large_receipt["chunks"] == 1
    assert large_receipt["relationships"] == OWNED_RELS

    print("\n".join(lines))
