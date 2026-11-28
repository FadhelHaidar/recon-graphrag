"""Unit tests for Memgraph entity-resolution merge behavior."""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest  # pyright: ignore[reportMissingImports]

from recon_graphrag.graphdb.memgraph.entity_resolution import (
    _EntityRecord,
    _MemgraphEntityResolver,
)


class FakeTransaction:
    def __init__(self, store: FakeGraphStore):
        self._store = store
        self.committed = False
        self.rolled_back = False

    def run(self, query: str, parameters: dict | None = None):
        return self._store.execute_query(query, parameters)

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


class FakeTransactionExecutor:
    def __init__(self, transaction: FakeTransaction):
        self._transaction = transaction

    def execute_query(self, query: str, parameters: dict | None = None):
        return self._transaction.run(query, parameters)


class FakeGraphStore:
    def __init__(self, rows=None, node_props=None):
        self.queries: list[tuple[str, dict]] = []
        self._rows = rows or []
        self.node_props = dict(node_props or {})
        self.transactions: list[FakeTransaction] = []
        self._active_transaction: FakeTransaction | None = None

    @contextmanager
    def _transaction(self):
        transaction = FakeTransaction(self)
        self.transactions.append(transaction)
        self._active_transaction = transaction
        try:
            yield FakeTransactionExecutor(transaction)
        except BaseException:
            transaction.rolled_back = True
            raise
        else:
            transaction.committed = True
        finally:
            self._active_transaction = None

    def execute_query(self, query: str, parameters: dict | None = None):
        return self._execute(query, parameters)

    def _execute(self, query: str, parameters: dict | None = None):
        params = parameters or {}
        self.queries.append((query.strip(), params))

        if "MATCH (e:__Entity__)" in query and "id(e) AS node_id" in query:
            return self._rows

        if "AS scope_type" in query:
            return []

        if "properties(r) AS rel_props" in query:
            return []

        if "RETURN DISTINCT type(r) AS rel_type" in query:
            return [{"rel_type": "ACTED_IN", "node_id": 2}]

        if "RETURN id(n) AS node_id, properties(n) AS props" in query:
            node_ids = params.get("node_ids", [])
            if self.node_props:
                return [
                    {"node_id": node_id, "props": dict(self.node_props[node_id])}
                    for node_id in node_ids
                    if node_id in self.node_props
                ]
            return [
                {
                    "node_id": row.get("node_id"),
                    "props": {
                        **(row.get("properties") or {}),
                        "id": row.get("entity_id"),
                        "graph_name": row.get("graph_name"),
                    },
                }
                for row in self._rows
                if row.get("node_id") in node_ids
            ]

        if "RETURN count(*) AS deleted" in query:
            return [{"deleted": len(params.get("other_ids", []))}]

        return [{"merged_id": 1}]


def test_combine_properties_combines_conflicts_and_aliases():
    resolver = _MemgraphEntityResolver(FakeGraphStore(node_props={
        1: {"name": "Christopher Nolan", "aliases": ["C. Nolan"], "source": "a"},
        2: {"name": "Chris Nolan", "aliases": ["Nolan"], "source": "b"},
    }))

    props = resolver._combine_properties([1, 2])

    assert props["name"] == ["Christopher Nolan", "Chris Nolan"]
    assert props["aliases"] == ["C. Nolan", "Nolan"]
    assert props["source"] == ["a", "b"]


def test_rewire_relationships_uses_merge_and_self_loop_guards():
    store = FakeGraphStore()
    resolver = _MemgraphEntityResolver(store)

    resolver._rewire_relationships(canonical_id=1, other_ids=[2])

    query_text = "\n".join(query for query, _ in store.queries)
    assert "MERGE (canonical)-[new_r:`ACTED_IN`]->(target)" in query_text
    assert "MERGE (source)-[new_r:`ACTED_IN`]->(canonical)" in query_text
    assert "WHERE NOT id(target) IN $merged_ids" in query_text
    assert "WHERE NOT id(source) IN $merged_ids" in query_text
    assert "AND id(target) IN $merged_ids" in query_text
    assert "AND id(source) IN $merged_ids" in query_text


def test_merge_groups_reports_neo4j_like_processed_node_count():
    store = FakeGraphStore(node_props={
        1: {
            "id": "person-1",
            "graph_name": "entity-graph",
            "name": "Christopher Nolan",
            "aliases": ["C. Nolan"],
            "source": "a",
        },
        2: {
            "id": "person-2",
            "graph_name": "entity-graph",
            "name": "Chris Nolan",
            "aliases": ["Nolan"],
            "source": "b",
        },
    })
    resolver = _MemgraphEntityResolver(store)
    group = [
        _EntityRecord(1, "person-1", "entity-graph", "Person", "Christopher Nolan", "christophernolan"),
        _EntityRecord(2, "person-2", "entity-graph", "Person", "Chris Nolan", "chrisnolan"),
    ]

    merged_nodes = resolver._merge_groups([group], "name")

    assert merged_nodes == 2
    assert len(store.transactions) == 1
    assert store.transactions[0].committed is True
    props = next(params["props"] for query, params in store.queries if "SET n += $props" in query)
    assert props["name"] == "Christopher Nolan"
    assert props["aliases"] == ["C. Nolan", "Nolan", "Chris Nolan"]


@pytest.mark.asyncio
async def test_memgraph_resolver_normalized_merges_case_variants():
    rows = [
        {
            "node_id": 1,
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "OpenAI",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
        {
            "node_id": 2,
            "entity_id": "e2",
            "graph_name": "g1",
            "resolve_value": "openai",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
    ]
    resolver = _MemgraphEntityResolver(FakeGraphStore(rows=rows))

    result = await resolver.resolve(strategy="normalized", dry_run=True)

    assert result["merged_groups"] == 1


@pytest.mark.asyncio
async def test_memgraph_resolver_does_not_merge_across_graphs():
    rows = [
        {
            "node_id": 1,
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "OpenAI",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
        {
            "node_id": 2,
            "entity_id": "e2",
            "graph_name": "g2",
            "resolve_value": "OpenAI",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
    ]
    resolver = _MemgraphEntityResolver(FakeGraphStore(rows=rows))

    result = await resolver.resolve(strategy="exact", dry_run=True)

    assert result["merged_groups"] == 0


@pytest.mark.asyncio
async def test_memgraph_resolver_fuzzy_produces_review_candidate():
    rows = [
        {
            "node_id": 1,
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "John Smith",
            "labels": ["__Entity__", "Person"],
            "properties": {},
        },
        {
            "node_id": 2,
            "entity_id": "e2",
            "graph_name": "g1",
            "resolve_value": "Jon Smith",
            "labels": ["__Entity__", "Person"],
            "properties": {},
        },
    ]
    resolver = _MemgraphEntityResolver(FakeGraphStore(rows=rows))

    result = await resolver.resolve(
        strategy="fuzzy",
        dry_run=True,
        merge_threshold=95.0,
        review_threshold=85.0,
    )

    assert result["merged_groups"] == 0
    assert result["review_groups"][0]["decision"] == "review"


@pytest.mark.asyncio
async def test_memgraph_resolver_hybrid_skips_missing_ai_dependencies():
    rows = [
        {
            "node_id": 1,
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "Acme",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        }
    ]
    resolver = _MemgraphEntityResolver(FakeGraphStore(rows=rows))

    result = await resolver.resolve(strategy="hybrid", dry_run=True)

    assert result["signals"]["embeddings"] == "skipped_no_embedder"
    assert result["signals"]["llm"] == "skipped_no_llm"


@pytest.mark.asyncio
async def test_memgraph_resolver_hybrid_keeps_llm_review_when_auto_merge_disabled():
    rows = [
        {
            "node_id": 1,
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "John Smith",
            "labels": ["__Entity__", "Person"],
            "properties": {},
        },
        {
            "node_id": 2,
            "entity_id": "e2",
            "graph_name": "g1",
            "resolve_value": "Jon Smith",
            "labels": ["__Entity__", "Person"],
            "properties": {},
        },
    ]
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content=(
                '{"same_entity": true, "confidence": 0.96, '
                '"reason": "Names refer to the same person.", '
                '"merge_allowed": true}'
            )
        )
    )
    resolver = _MemgraphEntityResolver(FakeGraphStore(rows=rows))

    result = await resolver.resolve(
        strategy="hybrid",
        merge_threshold=95.0,
        review_threshold=85.0,
        llm=llm,
        allow_ai_auto_merge=False,
    )

    assert result["merged_groups"] == 0
    assert result["merged_nodes"] == 0
    assert len(result["review_groups"]) == 1
    assert result["review_groups"][0]["llm_review"]["merge_allowed"] is True
    assert result["ai_merged_review_groups"] == []


@pytest.mark.asyncio
async def test_memgraph_resolver_hybrid_sends_property_context_to_llm():
    rows = [
        {
            "node_id": 1,
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "John Smith",
            "labels": ["__Entity__", "Person"],
            "properties": {
                "name": "John Smith",
                "description": "Senior engineer on the Apollo API team.",
                "canonical_key": "person:john-smith",
                "department": "Platform",
                "embedding": [0.1, 0.2],
            },
        },
        {
            "node_id": 2,
            "entity_id": "e2",
            "graph_name": "g1",
            "resolve_value": "Jon Smith",
            "labels": ["__Entity__", "Person"],
            "properties": {
                "name": "Jon Smith",
                "description": "Apollo API platform engineer.",
                "department": "Platform",
            },
        },
    ]
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content='{"same_entity": true, "confidence": 0.91, "merge_allowed": false}'
        )
    )
    resolver = _MemgraphEntityResolver(FakeGraphStore(rows=rows))

    result = await resolver.resolve(
        strategy="hybrid",
        dry_run=True,
        merge_threshold=95.0,
        review_threshold=85.0,
        llm=llm,
        context_properties={"Person": ["department"]},
    )

    payload = json.loads(llm.ainvoke.await_args.args[0])
    profiles = payload["candidate"]["entities"]
    assert profiles[0]["description"] == "Senior engineer on the Apollo API team."
    assert profiles[0]["properties"] == {"department": "Platform"}
    assert "embedding" not in json.dumps(profiles)
    assert result["review_groups"][0]["entities"] == profiles


@pytest.mark.asyncio
async def test_memgraph_resolver_hybrid_conflict_properties_block_auto_merge():
    rows = [
        {
            "node_id": 1,
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "Titanic",
            "labels": ["__Entity__", "Movie"],
            "properties": {"name": "Titanic", "year": "1953"},
        },
        {
            "node_id": 2,
            "entity_id": "e2",
            "graph_name": "g1",
            "resolve_value": "Titanic",
            "labels": ["__Entity__", "Movie"],
            "properties": {"name": "Titanic", "year": "1997"},
        },
    ]
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content='{"same_entity": true, "confidence": 1.0, "merge_allowed": true}'
        )
    )
    resolver = _MemgraphEntityResolver(FakeGraphStore(rows=rows))

    result = await resolver.resolve(
        strategy="hybrid",
        merge_threshold=95.0,
        review_threshold=85.0,
        llm=llm,
        allow_ai_auto_merge=True,
        conflict_properties={"Movie": ["year"]},
    )

    llm.ainvoke.assert_not_awaited()
    assert result["merged_groups"] == 0
    assert result["review_groups"][0]["decision"] == "blocked"
    assert result["review_groups"][0]["conflicts"][0]["property"] == "year"


@pytest.mark.asyncio
async def test_memgraph_resolver_hybrid_auto_merges_llm_approved_review():
    rows = [
        {
            "node_id": 1,
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "John Smith",
            "labels": ["__Entity__", "Person"],
            "properties": {"id": "e1", "graph_name": "g1"},
        },
        {
            "node_id": 2,
            "entity_id": "e2",
            "graph_name": "g1",
            "resolve_value": "Jon Smith",
            "labels": ["__Entity__", "Person"],
            "properties": {"id": "e2", "graph_name": "g1"},
        },
    ]
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content=(
                '{"same_entity": true, "confidence": 0.96, '
                '"reason": "Names refer to the same person.", '
                '"merge_allowed": true}'
            )
        )
    )
    store = FakeGraphStore(rows=rows)
    resolver = _MemgraphEntityResolver(store)

    result = await resolver.resolve(
        strategy="hybrid",
        merge_threshold=95.0,
        review_threshold=85.0,
        llm=llm,
        allow_ai_auto_merge=True,
    )

    assert result["merged_groups"] == 1
    assert result["merged_nodes"] == 2
    assert result["review_groups"] == []
    assert result["ai_merged_review_groups"][0]["decision"] == "merge"
    merge_query_text = "\n".join(query for query, _ in store.queries)
    assert "WHERE id(n) IN $other_ids" in merge_query_text
    assert "DELETE n" in merge_query_text


@pytest.mark.asyncio
async def test_memgraph_resolver_normalized_builds_groups_once(monkeypatch):
    """Normalized resolution must not repeat the grouping pass."""
    rows = [
        {
            "node_id": 1,
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "OpenAI",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
        {
            "node_id": 2,
            "entity_id": "e2",
            "graph_name": "g1",
            "resolve_value": "openai",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
    ]
    resolver = _MemgraphEntityResolver(FakeGraphStore(rows=rows))
    calls: list[int] = []
    original = resolver._build_normalized_groups

    def counting(entities):
        calls.append(len(entities))
        return original(entities)

    monkeypatch.setattr(resolver, "_build_normalized_groups", counting)

    result = await resolver.resolve(strategy="normalized", dry_run=True)

    assert calls == [2]
    assert result["merged_groups"] == 1


def test_memgraph_rewire_protects_structural_edges_and_preserves_domain_edge_identity():
    class RelTypeFakeStore(FakeGraphStore):
        def execute_query(self, query, parameters=None):
            if "RETURN DISTINCT type(r) AS rel_type" in query:
                self.queries.append((query, parameters))
                return [
                    {"rel_type": "FROM_CHUNK", "node_id": 2},
                    {"rel_type": "IN_COMMUNITY", "node_id": 2},
                    {"rel_type": "WORKS_AT", "node_id": 2},
                ]
            return super().execute_query(query, parameters)

    store = RelTypeFakeStore(rows=[])
    resolver = _MemgraphEntityResolver(store)
    resolver._rewire_relationships(canonical_id=1, other_ids=[2])

    structural_queries = [
        query for query, _ in store.queries
        if ":`FROM_CHUNK`" in query or ":`IN_COMMUNITY`" in query
    ]
    assert structural_queries
    for query in structural_queries:
        assert "source_chunk_ids" not in query
        assert "observation_count" not in query
        assert "description_summary_fallback" not in query

    domain_rewires = [
        query for query, _ in store.queries
        if ":`WORKS_AT`" in query and "SET new_r += rel_props" in query
    ]
    assert len(domain_rewires) == 2
    for query in domain_rewires:
        assert query.index("new_r.id AS existing_id") < query.index("SET new_r += rel_props")
        assert query.index("new_r.created AS existing_created") < query.index("SET new_r += rel_props")
        assert query.index("new_r.canonical_key AS existing_canonical_key") < query.index("SET new_r += rel_props")
        assert "new_r.id = coalesce(existing_id, rel_props.id)" in query
        assert "new_r.created = coalesce(existing_created, rel_props.created)" in query
        assert "new_r.canonical_key = coalesce(existing_canonical_key, rel_props.canonical_key)" in query


def test_memgraph_merge_preserves_scalar_bookkeeping_and_clears_embedding_fingerprints():
    from recon_graphrag.graphdb.entity_resolution import _EntityRecord

    store = FakeGraphStore(rows=[])
    resolver = _MemgraphEntityResolver(store)
    group = [
        _EntityRecord(
            1,
            "e1",
            "g1",
            "Organization",
            "OpenAI Research",
            "openairesearch",
            {
                "id": "e1",
                "graph_name": "g1",
                "title": "OpenAI Research",
                "created": "2025-01-01T00:00:00Z",
                "updated": "2025-01-02T00:00:00Z",
                "embedding_content_fingerprint": "old-content-fp",
                "embedding_model_fingerprint": "old-model-fp",
            },
        ),
        _EntityRecord(
            2,
            "e2",
            "g1",
            "Organization",
            "OpenAI",
            "openai",
            {
                "id": "e2",
                "graph_name": "g1",
                "title": "OpenAI",
                "created": "2025-02-01T00:00:00Z",
                "updated": "2025-02-02T00:00:00Z",
                "embedding_content_fingerprint": "other-content-fp",
                "embedding_model_fingerprint": "other-model-fp",
            },
        ),
    ]

    resolver._merge_group_records(group, "name")

    set_call = next(
        params for query, params in store.queries if "SET n += $props" in query
    )
    props = set_call["props"]
    assert props["title"] == "OpenAI Research"
    assert props["created"] == "2025-01-01T00:00:00Z"
    assert props["updated"] == "2025-01-02T00:00:00Z"
    assert props["embedding"] is None
    assert props["embedding_input_fingerprint"] is None
    assert props["embedding_content_fingerprint"] is None
    assert props["embedding_model_fingerprint"] is None
