"""Tests for Neo4j entity resolution strategies."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from recon_graphrag.graphdb.neo4j import entity_resolution as neo4j_entity_resolution
from recon_graphrag.graphdb.neo4j.entity_resolution import _Neo4jEntityResolver


class FakeGraphStore:
    def __init__(self, apoc_available=True, rows=None):
        self.calls = []
        self.apoc_available = apoc_available
        self._rows = rows or []

    def execute_query(self, query, parameters=None):
        self.calls.append((query.strip(), parameters or {}))
        if "apoc.version" in query and not self.apoc_available:
            raise RuntimeError("There is no procedure with the name apoc.version")
        if "apoc.version" in query:
            return [{"version": "5.0"}]
        if "RETURN count(node) AS merged_groups" in query:
            return [{"merged_groups": 2}]
        if "apoc.refactor.mergeNodes" in query:
            return [{"merged_id": "4:abc"}]
        if "properties(n) AS props" in query:
            # Hydration stage: the loader no longer projects properties, so the
            # fake answers the batched property lookup from the same rows.
            properties_by_id = {
                row.get("node_id"): row.get("properties", {}) for row in self._rows
            }
            return [
                {"node_id": node_id, "props": properties_by_id.get(node_id, {})}
                for node_id in (parameters or {}).get("node_ids", [])
            ]
        if "MATCH (e:__Entity__)" in query and "elementId(e) AS node_id" in query:
            # Honour the keyset cursor and page size like the real store.
            after = (parameters or {}).get("after")
            if after is None:
                return self._rows
            page_size = (parameters or {}).get("page_size", len(self._rows))
            return [row for row in self._rows if row["node_id"] > after][:page_size]
        return []

    def create_vector_index(self, **kwargs):
        pass

    def create_fulltext_index(self, **kwargs):
        pass

    def upsert_vectors(self, **kwargs):
        pass


# ------------------------------------------------------------------
# _Neo4jEntityResolver — exact strategy
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_neo4j_resolver_pages_entity_load_across_all_nodes(monkeypatch):
    monkeypatch.setattr(neo4j_entity_resolution, "_LOAD_PAGE_SIZE", 2)

    class PagingStore(FakeGraphStore):
        """Fails loudly if the loader keeps asking for pages, instead of hanging."""

        max_pages = 4

        def execute_query(self, query, parameters=None):
            if "ORDER BY node_id" in query:
                served = sum(1 for q, _ in self.calls if "ORDER BY node_id" in q)
                if served >= self.max_pages:
                    raise AssertionError("loader never stopped paging")
            return super().execute_query(query, parameters)

    names = ["Alpha", "Alpha", "Beta", "Beta", "Gamma", "Gamma"]
    rows = [
        {
            "node_id": f"4:{index}",
            "entity_id": f"e{index}",
            "graph_name": "g1",
            "resolve_value": name,
            "labels": ["__Entity__", "Organization"],
            "properties": {"name": name},
        }
        for index, name in enumerate(names)
    ]
    store = PagingStore(rows=rows)
    resolver = _Neo4jEntityResolver(store)

    result = await resolver.resolve(strategy="exact", dry_run=True)

    load_pages = [q for q, _ in store.calls if "ORDER BY node_id" in q]
    # Three data pages plus the short page that ends the loop.
    assert len(load_pages) == 4
    # One group per name: nodes from every page reached the grouping stage.
    assert result["merged_groups"] == 3


@pytest.mark.asyncio
async def test_neo4j_resolver_exact_groups_by_raw_name():
    rows = [
        {
            "node_id": "4:a",
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "OpenAI",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
        {
            "node_id": "4:b",
            "entity_id": "e2",
            "graph_name": "g1",
            "resolve_value": "OpenAI",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
        {
            "node_id": "4:c",
            "entity_id": "e3",
            "graph_name": "g1",
            "resolve_value": "openai",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
    ]
    store = FakeGraphStore(apoc_available=True, rows=rows)
    resolver = _Neo4jEntityResolver(store)
    result = await resolver.resolve(strategy="exact", dry_run=True)
    assert result["skipped"] is False
    assert result["merged_groups"] == 1
    assert result["merged_nodes"] == 0


@pytest.mark.asyncio
async def test_neo4j_resolver_exact_does_not_merge_across_graphs():
    rows = [
        {
            "node_id": "4:a",
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "OpenAI",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
        {
            "node_id": "4:b",
            "entity_id": "e2",
            "graph_name": "g2",
            "resolve_value": "OpenAI",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
    ]
    store = FakeGraphStore(apoc_available=True, rows=rows)
    resolver = _Neo4jEntityResolver(store)
    result = await resolver.resolve(strategy="exact", dry_run=True)
    assert result["merged_groups"] == 0


# ------------------------------------------------------------------
# _Neo4jEntityResolver — normalized strategy
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_neo4j_resolver_normalized_merges_case_variants():
    rows = [
        {
            "node_id": "4:a",
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "OpenAI",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
        {
            "node_id": "4:b",
            "entity_id": "e2",
            "graph_name": "g1",
            "resolve_value": "openai",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
    ]
    store = FakeGraphStore(apoc_available=True, rows=rows)
    resolver = _Neo4jEntityResolver(store)
    result = await resolver.resolve(strategy="normalized", dry_run=True)
    assert result["merged_groups"] == 1


@pytest.mark.asyncio
async def test_neo4j_resolver_normalized_merges_whitespace_variants():
    rows = [
        {
            "node_id": "4:a",
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "Open AI",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
        {
            "node_id": "4:b",
            "entity_id": "e2",
            "graph_name": "g1",
            "resolve_value": "OpenAI",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
    ]
    store = FakeGraphStore(apoc_available=True, rows=rows)
    resolver = _Neo4jEntityResolver(store)
    result = await resolver.resolve(strategy="normalized", dry_run=True)
    assert result["merged_groups"] == 1


@pytest.mark.asyncio
async def test_neo4j_resolver_normalized_merges_org_suffix_variants():
    rows = [
        {
            "node_id": "4:a",
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "Microsoft Corp.",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
        {
            "node_id": "4:b",
            "entity_id": "e2",
            "graph_name": "g1",
            "resolve_value": "Microsoft Corporation",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
    ]
    store = FakeGraphStore(apoc_available=True, rows=rows)
    resolver = _Neo4jEntityResolver(store)
    result = await resolver.resolve(strategy="normalized", dry_run=True)
    assert result["merged_groups"] == 1


@pytest.mark.asyncio
async def test_neo4j_resolver_normalized_does_not_merge_across_domain_labels():
    rows = [
        {
            "node_id": "4:a",
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "Apple",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
        {
            "node_id": "4:b",
            "entity_id": "e2",
            "graph_name": "g1",
            "resolve_value": "Apple",
            "labels": ["__Entity__", "Product"],
            "properties": {},
        },
    ]
    store = FakeGraphStore(apoc_available=True, rows=rows)
    resolver = _Neo4jEntityResolver(store)
    result = await resolver.resolve(strategy="normalized", dry_run=True)
    assert result["merged_groups"] == 0


# ------------------------------------------------------------------
# _Neo4jEntityResolver — fuzzy strategy
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_neo4j_resolver_fuzzy_produces_review_candidate():
    rows = [
        {
            "node_id": "4:a",
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "John Smith",
            "labels": ["__Entity__", "Person"],
            "properties": {},
        },
        {
            "node_id": "4:b",
            "entity_id": "e2",
            "graph_name": "g1",
            "resolve_value": "Jon Smith",
            "labels": ["__Entity__", "Person"],
            "properties": {},
        },
    ]
    store = FakeGraphStore(apoc_available=True, rows=rows)
    resolver = _Neo4jEntityResolver(store)
    result = await resolver.resolve(
        strategy="fuzzy",
        dry_run=True,
        merge_threshold=95.0,
        review_threshold=85.0,
    )
    # "john smith" vs "jon smith" fuzzy ratio is typically ~91
    assert result["merged_groups"] == 0
    assert len(result["review_groups"]) == 1
    assert result["review_groups"][0]["decision"] == "review"


@pytest.mark.asyncio
async def test_neo4j_resolver_fuzzy_merges_more_than_two_variants_in_one_group():
    # Three variants of one person that all match the anchor above the merge
    # threshold must collapse into a single group, not just an isolated pair.
    rows = [
        {
            "node_id": "4:a",
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "Jonathan Smith",
            "labels": ["__Entity__", "Person"],
            "properties": {},
        },
        {
            "node_id": "4:b",
            "entity_id": "e2",
            "graph_name": "g1",
            "resolve_value": "Jonathan Smithe",
            "labels": ["__Entity__", "Person"],
            "properties": {},
        },
        {
            "node_id": "4:c",
            "entity_id": "e3",
            "graph_name": "g1",
            "resolve_value": "Jonathan Smyth",
            "labels": ["__Entity__", "Person"],
            "properties": {},
        },
    ]
    store = FakeGraphStore(apoc_available=True, rows=rows)
    resolver = _Neo4jEntityResolver(store)
    entities = resolver._load_entities("g1", "name")
    groups, review_groups, _ = resolver._build_fuzzy_groups(
        entities,
        merge_threshold=90.0,
        review_threshold=80.0,
        max_candidates_per_entity=20,
    )
    assert len(groups) == 1
    assert {e.node_id for e in groups[0]} == {"4:a", "4:b", "4:c"}


@pytest.mark.asyncio
async def test_neo4j_resolver_fuzzy_does_not_merge_short_different_names():
    rows = [
        {
            "node_id": "4:a",
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "ABC",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
        {
            "node_id": "4:b",
            "entity_id": "e2",
            "graph_name": "g1",
            "resolve_value": "ABD",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
    ]
    store = FakeGraphStore(apoc_available=True, rows=rows)
    resolver = _Neo4jEntityResolver(store)
    result = await resolver.resolve(
        strategy="fuzzy",
        dry_run=True,
        merge_threshold=95.0,
        review_threshold=85.0,
    )
    assert result["merged_groups"] == 0
    assert len(result["review_groups"]) == 0


# ------------------------------------------------------------------
# _Neo4jEntityResolver — alias / hybrid strategy
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_neo4j_resolver_hybrid_with_simple_alias():
    rows = [
        {
            "node_id": "4:a",
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "IBM",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
        {
            "node_id": "4:b",
            "entity_id": "e2",
            "graph_name": "g1",
            "resolve_value": "International Business Machines",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
    ]
    store = FakeGraphStore(apoc_available=True, rows=rows)
    resolver = _Neo4jEntityResolver(store)
    result = await resolver.resolve(
        strategy="hybrid",
        dry_run=True,
        aliases={"IBM": ["International Business Machines"]},
    )
    assert result["merged_groups"] == 1
    assert result["signals"]["aliases"] == "used"


@pytest.mark.asyncio
async def test_neo4j_resolver_hybrid_with_domain_scoped_alias():
    rows = [
        {
            "node_id": "4:a",
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "IBM",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
        {
            "node_id": "4:b",
            "entity_id": "e2",
            "graph_name": "g1",
            "resolve_value": "International Business Machines",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
    ]
    store = FakeGraphStore(apoc_available=True, rows=rows)
    resolver = _Neo4jEntityResolver(store)
    result = await resolver.resolve(
        strategy="hybrid",
        dry_run=True,
        aliases={"Organization": {"IBM": ["International Business Machines"]}},
    )
    assert result["merged_groups"] == 1


@pytest.mark.asyncio
async def test_neo4j_resolver_hybrid_skips_ai_signals_when_deps_absent():
    rows = [
        {
            "node_id": "4:a",
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "Acme",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
    ]
    store = FakeGraphStore(apoc_available=True, rows=rows)
    resolver = _Neo4jEntityResolver(store)
    result = await resolver.resolve(strategy="hybrid", dry_run=True)
    assert result["signals"]["embeddings"] == "skipped_no_embedder"
    assert result["signals"]["llm"] == "skipped_no_llm"


@pytest.mark.asyncio
async def test_neo4j_resolver_hybrid_uses_embedder_for_review_scoring():
    rows = [
        {
            "node_id": "4:a",
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "John Smith",
            "labels": ["__Entity__", "Person"],
            "properties": {},
        },
        {
            "node_id": "4:b",
            "entity_id": "e2",
            "graph_name": "g1",
            "resolve_value": "Jon Smith",
            "labels": ["__Entity__", "Person"],
            "properties": {},
        },
    ]
    embedder = MagicMock()
    embedder.async_embed_query = AsyncMock(side_effect=[[1.0, 0.0], [0.8, 0.6]])
    store = FakeGraphStore(apoc_available=True, rows=rows)
    resolver = _Neo4jEntityResolver(store)

    result = await resolver.resolve(
        strategy="hybrid",
        dry_run=True,
        merge_threshold=95.0,
        review_threshold=85.0,
        embedder=embedder,
    )

    assert result["signals"]["embeddings"] == "used"
    assert embedder.async_embed_query.await_count == 2
    assert result["review_groups"][0]["scores"]["embedding"] == 0.8


@pytest.mark.asyncio
async def test_neo4j_resolver_hybrid_uses_llm_with_aliases_and_guidance():
    rows = [
        {
            "node_id": "4:a",
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "John Smith",
            "labels": ["__Entity__", "Person"],
            "properties": {},
        },
        {
            "node_id": "4:b",
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
                '{"same_entity": true, "confidence": 0.91, '
                '"reason": "Provided aliases support the match.", '
                '"merge_allowed": false}'
            )
        )
    )
    aliases = {"Person": {"John Smith": ["Jon Smith"]}}
    guidance = "Nicknames may match, but require clear evidence."
    store = FakeGraphStore(apoc_available=True, rows=rows)
    resolver = _Neo4jEntityResolver(store)

    result = await resolver.resolve(
        strategy="hybrid",
        dry_run=True,
        merge_threshold=95.0,
        review_threshold=85.0,
        aliases=aliases,
        llm=llm,
        llm_guidance=guidance,
    )

    llm.ainvoke.assert_awaited_once()
    prompt = llm.ainvoke.await_args.args[0]
    assert "entity_deduplication_review" in prompt
    assert "John Smith" in prompt
    assert "Jon Smith" in prompt
    assert guidance in prompt
    assert result["signals"]["llm"] == "used"
    assert result["review_groups"][0]["scores"]["llm"] == 0.91
    assert result["review_groups"][0]["llm_review"]["same_entity"] is True


@pytest.mark.asyncio
async def test_neo4j_resolver_hybrid_sends_property_context_to_llm():
    rows = [
        {
            "node_id": "4:a",
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "John Smith",
            "labels": ["__Entity__", "Person"],
            "properties": {
                "name": "John Smith",
                "title": "John Smith",
                "description": "Senior engineer on the Apollo API team.",
                "canonical_key": "person:john-smith",
                "human_readable_id": "person:john-smith",
                "department": "Platform",
                "embedding": [0.1, 0.2],
                "graph_name": "g1",
            },
        },
        {
            "node_id": "4:b",
            "entity_id": "e2",
            "graph_name": "g1",
            "resolve_value": "Jon Smith",
            "labels": ["__Entity__", "Person"],
            "properties": {
                "name": "Jon Smith",
                "description": "Apollo API platform engineer.",
                "canonical_key": "person:jon-smith",
                "department": "Platform",
            },
        },
    ]
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content=(
                '{"same_entity": true, "confidence": 0.91, '
                '"reason": "Descriptions and department match.", '
                '"merge_allowed": false}'
            )
        )
    )
    resolver = _Neo4jEntityResolver(FakeGraphStore(apoc_available=True, rows=rows))

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
    assert profiles[0]["canonical_key"] == "person:john-smith"
    assert profiles[0]["properties"] == {"department": "Platform"}
    assert "embedding" not in json.dumps(profiles)
    assert "graph_name" not in profiles[0]
    assert result["review_groups"][0]["entities"] == profiles


@pytest.mark.asyncio
async def test_neo4j_resolver_hybrid_config_only_context_mode_limits_properties():
    rows = [
        {
            "node_id": "4:a",
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "John Smith",
            "labels": ["__Entity__", "Person"],
            "properties": {
                "name": "John Smith",
                "description": "Senior engineer on the Apollo API team.",
                "department": "Platform",
                "role": "Engineer",
            },
        },
        {
            "node_id": "4:b",
            "entity_id": "e2",
            "graph_name": "g1",
            "resolve_value": "Jon Smith",
            "labels": ["__Entity__", "Person"],
            "properties": {
                "name": "Jon Smith",
                "description": "Apollo API platform engineer.",
                "department": "Platform",
                "role": "Engineer",
            },
        },
    ]
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content='{"same_entity": true, "confidence": 0.91, "merge_allowed": false}'
        )
    )
    resolver = _Neo4jEntityResolver(FakeGraphStore(apoc_available=True, rows=rows))

    await resolver.resolve(
        strategy="hybrid",
        dry_run=True,
        merge_threshold=95.0,
        review_threshold=85.0,
        llm=llm,
        context_properties=["department"],
        context_mode="config_only",
    )

    payload = json.loads(llm.ainvoke.await_args.args[0])
    profiles = payload["candidate"]["entities"]
    assert profiles[0]["description"] == "Senior engineer on the Apollo API team."
    assert profiles[0]["properties"] == {"department": "Platform"}
    assert "role" not in profiles[0]["properties"]


@pytest.mark.asyncio
async def test_neo4j_resolver_hybrid_truncates_long_property_context():
    long_description = "x" * 700
    rows = [
        {
            "node_id": "4:a",
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "John Smith",
            "labels": ["__Entity__", "Person"],
            "properties": {"name": "John Smith", "description": long_description},
        },
        {
            "node_id": "4:b",
            "entity_id": "e2",
            "graph_name": "g1",
            "resolve_value": "Jon Smith",
            "labels": ["__Entity__", "Person"],
            "properties": {"name": "Jon Smith", "description": long_description},
        },
    ]
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content='{"same_entity": false, "confidence": 0.5, "merge_allowed": false}'
        )
    )
    resolver = _Neo4jEntityResolver(FakeGraphStore(apoc_available=True, rows=rows))

    await resolver.resolve(
        strategy="hybrid",
        dry_run=True,
        merge_threshold=95.0,
        review_threshold=85.0,
        llm=llm,
    )

    payload = json.loads(llm.ainvoke.await_args.args[0])
    assert len(payload["candidate"]["entities"][0]["description"]) == 500


@pytest.mark.asyncio
async def test_neo4j_resolver_hybrid_conflict_properties_block_auto_merge():
    rows = [
        {
            "node_id": "4:a",
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "Titanic",
            "labels": ["__Entity__", "Movie"],
            "properties": {"name": "Titanic", "year": "1953"},
        },
        {
            "node_id": "4:b",
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
    resolver = _Neo4jEntityResolver(FakeGraphStore(apoc_available=True, rows=rows))

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
    assert result["review_groups"][0]["llm_review"]["merge_allowed"] is False


@pytest.mark.asyncio
async def test_neo4j_resolver_hybrid_matching_conflict_property_allows_llm_review():
    rows = [
        {
            "node_id": "4:a",
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "John Smith",
            "labels": ["__Entity__", "Person"],
            "properties": {"name": "John Smith", "country": "US"},
        },
        {
            "node_id": "4:b",
            "entity_id": "e2",
            "graph_name": "g1",
            "resolve_value": "Jon Smith",
            "labels": ["__Entity__", "Person"],
            "properties": {"name": "Jon Smith", "country": "us"},
        },
    ]
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content='{"same_entity": true, "confidence": 0.91, "merge_allowed": false}'
        )
    )
    resolver = _Neo4jEntityResolver(FakeGraphStore(apoc_available=True, rows=rows))

    result = await resolver.resolve(
        strategy="hybrid",
        dry_run=True,
        merge_threshold=95.0,
        review_threshold=85.0,
        llm=llm,
        conflict_properties={"Person": ["country"]},
    )

    llm.ainvoke.assert_awaited_once()
    assert result["review_groups"][0]["decision"] == "review"


@pytest.mark.asyncio
async def test_neo4j_resolver_hybrid_keeps_llm_review_when_auto_merge_disabled():
    rows = [
        {
            "node_id": "4:a",
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "John Smith",
            "labels": ["__Entity__", "Person"],
            "properties": {},
        },
        {
            "node_id": "4:b",
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
    store = FakeGraphStore(apoc_available=True, rows=rows)
    resolver = _Neo4jEntityResolver(store)

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
async def test_neo4j_resolver_hybrid_auto_merges_llm_approved_review():
    rows = [
        {
            "node_id": "4:a",
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "John Smith",
            "labels": ["__Entity__", "Person"],
            "properties": {},
        },
        {
            "node_id": "4:b",
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
    store = FakeGraphStore(apoc_available=True, rows=rows)
    resolver = _Neo4jEntityResolver(store)

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
    merge_call = next(
        call for call in store.calls
        if "apoc.refactor.mergeNodes" in call[0] and "SET node.`name`" in call[0]
    )
    assert merge_call[1]["node_ids"] == ["4:a", "4:b"]


# ------------------------------------------------------------------
# Dry run
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_neo4j_resolver_dry_run_does_not_call_apoc_merge():
    rows = [
        {
            "node_id": "4:a",
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "OpenAI",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
        {
            "node_id": "4:b",
            "entity_id": "e2",
            "graph_name": "g1",
            "resolve_value": "openai",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
    ]
    store = FakeGraphStore(apoc_available=True, rows=rows)
    resolver = _Neo4jEntityResolver(store)
    result = await resolver.resolve(strategy="normalized", dry_run=True)
    assert result["skipped"] is False
    assert result["merged_nodes"] == 0
    merge_calls = [c for c in store.calls if "apoc.refactor.mergeNodes" in c[0]]
    assert len(merge_calls) == 0


# ------------------------------------------------------------------
# APOC unavailable
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_neo4j_resolver_skips_when_apoc_unavailable():
    store = FakeGraphStore(apoc_available=False)
    resolver = _Neo4jEntityResolver(store)
    result = await resolver.resolve()
    assert result["skipped"] is True
    assert "merged_groups" in result
    assert result["merged_groups"] == 0
