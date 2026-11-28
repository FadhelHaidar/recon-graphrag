"""Shared entity-resolution normalization contract tests."""

import pytest

from recon_graphrag.graphdb.entity_resolution import (
    BaseEntityResolver,
    _EntityRecord,
)
from recon_graphrag.graphdb.memgraph.entity_resolution import (
    _normalize_name as normalize_memgraph_name,
)
from recon_graphrag.graphdb.neo4j.entity_resolution import (
    _normalize_name as normalize_neo4j_name,
)


@pytest.mark.parametrize(
    "normalize_name",
    [normalize_neo4j_name, normalize_memgraph_name],
)
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("OpenAI", "openai"),
        ("Open AI", "openai"),
        ("U.S.A.", "usa"),
        ("Microsoft Corp.", "microsoft"),
        ("Microsoft Corporation", "microsoft"),
        ("Acme Inc.", "acme"),
        ("Acme Ltd", "acme"),
    ],
)
def test_normalize_name_contract(normalize_name, value, expected):
    assert normalize_name(value) == expected


class ContractResolver(BaseEntityResolver):
    def __init__(self, rows):
        super().__init__(graph_store=None)
        self.rows = rows
        self.merged_groups = []

    def _load_entities(self, graph_name: str, resolve_property: str):
        return self.rows

    def _merge_groups(self, groups, resolve_property: str) -> int:
        self.merged_groups = groups
        return sum(len(group) for group in groups)


@pytest.mark.asyncio
async def test_shared_resolver_preserves_hybrid_auto_merge_result_shape():
    rows = [
        _EntityRecord(1, "e1", "g1", "Person", "John Smith", "johnsmith", {}),
        _EntityRecord(2, "e2", "g1", "Person", "Jon Smith", "jonsmith", {}),
    ]

    class LLM:
        async def ainvoke(self, prompt):
            class Response:
                content = (
                    '{"same_entity": true, "confidence": 0.96, '
                    '"reason": "Names refer to the same person.", '
                    '"merge_allowed": true}'
                )

            return Response()

    resolver = ContractResolver(rows)
    result = await resolver.resolve(
        strategy="hybrid",
        merge_threshold=95.0,
        review_threshold=85.0,
        llm=LLM(),
        allow_ai_auto_merge=True,
    )

    assert result["skipped"] is False
    assert result["strategy"] == "hybrid"
    assert result["merged_groups"] == 1
    assert result["merged_nodes"] == 2
    assert result["candidate_groups"] == 1
    assert result["review_groups"] == []
    assert result["ai_merged_review_groups"][0]["decision"] == "merge"
    assert result["signals"] == {
        "normalized": "used",
        "fuzzy": "used",
        "aliases": "skipped_no_aliases",
        "embeddings": "skipped_no_embedder",
        "llm": "used",
    }


@pytest.mark.asyncio
async def test_normalized_resolution_builds_groups_once(monkeypatch):
    """Normalized resolution must not repeat the grouping pass."""
    rows = [
        _EntityRecord(1, "e1", "g1", "Person", "OpenAI", "openai", {}),
        _EntityRecord(2, "e2", "g1", "Person", "openai", "openai", {}),
    ]
    resolver = ContractResolver(rows)
    calls: list[int] = []
    original = resolver._build_normalized_groups

    def counting(entities):
        calls.append(len(entities))
        return original(entities)

    monkeypatch.setattr(resolver, "_build_normalized_groups", counting)

    result = await resolver.resolve_normalized(graph_name="g1", dry_run=True)

    assert calls == [2]
    assert result["merged_groups"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("conflicts", [["birth_year"], {"Person": ["birth_year"]}])
async def test_normalized_dispatch_preserves_conflict_configuration(conflicts):
    rows = [
        _EntityRecord(1, "one", "graph", "Person", "Alice", "alice", {"birth_year": 1980}),
        _EntityRecord(2, "two", "graph", "Person", "ALICE", "alice", {"birth_year": 2000}),
    ]
    resolver = ContractResolver(rows)
    direct = await resolver.resolve_normalized(graph_name="graph", conflict_properties=conflicts, dry_run=True)
    dispatched = await resolver.resolve(graph_name="graph", strategy="normalized", conflict_properties=conflicts, dry_run=True)
    assert dispatched == direct
    assert dispatched["merged_groups"] == 0
    assert dispatched["review_groups"]


def test_build_alias_groups_merges_multiple_aliases_and_alias_to_alias():
    from recon_graphrag.graphdb.entity_resolution import _normalize_name

    # Case 1: canonical + multiple aliases merge in a single group.
    entities_with_canonical = [
        _EntityRecord(1, "e1", "g1", "Organization", "IBM", _normalize_name("IBM"), {}),
        _EntityRecord(2, "e2", "g1", "Organization", "Big Blue", _normalize_name("Big Blue"), {}),
        _EntityRecord(
            3,
            "e3",
            "g1",
            "Organization",
            "International Business Machines",
            _normalize_name("International Business Machines"),
            {},
        ),
    ]
    resolver = ContractResolver(entities_with_canonical)
    groups, _ = resolver._build_alias_groups(
        entities_with_canonical,
        set(),
        {"IBM": ["Big Blue", "International Business Machines"]},
    )
    assert len(groups) == 1
    assert [e.entity_id for e in groups[0]] == ["e1", "e2", "e3"]

    # Case 2: two aliases without the canonical node still merge together.
    entities_aliases_only = entities_with_canonical[1:]
    groups_alias_only, _ = resolver._build_alias_groups(
        entities_aliases_only,
        set(),
        {"Organization": {"IBM": ["Big Blue", "International Business Machines"]}},
    )
    assert len(groups_alias_only) == 1
    assert {e.entity_id for e in groups_alias_only[0]} == {"e2", "e3"}


def test_acronym_match_and_candidate_index_parity():
    from recon_graphrag.graphdb.entity_resolution import (
        _FuzzyCandidateIndex,
        _normalize_name,
        _scan_candidates,
    )

    assert BaseEntityResolver._is_acronym_match("International Business Machines", "IBM")
    assert BaseEntityResolver._is_acronym_match("U.S.A.", "United States America")
    assert not BaseEntityResolver._is_acronym_match("International Business Machines", "NASA")

    pool = [
        _EntityRecord(1, "e1", "g1", "Organization", "IBM", _normalize_name("IBM"), {}),
        _EntityRecord(
            2,
            "e2",
            "g1",
            "Organization",
            "International Business Machines",
            _normalize_name("International Business Machines"),
            {},
        ),
        _EntityRecord(3, "e3", "g1", "Organization", "NASA", _normalize_name("NASA"), {}),
    ]
    index = _FuzzyCandidateIndex(pool)
    for anchor in pool:
        assert index.candidates(
            anchor, 20, set(), BaseEntityResolver._is_acronym_match
        ) == _scan_candidates(anchor, pool, 20, set(), BaseEntityResolver._is_acronym_match)

    ibm_candidates = index.candidates(
        pool[0], 20, set(), BaseEntityResolver._is_acronym_match
    )
    assert [c.entity_id for c in ibm_candidates] == ["e2"]


def test_rescue_dropped_pairs_is_bounded_and_rescues_acronyms():
    from recon_graphrag.graphdb.entity_resolution import _normalize_name

    resolver = ContractResolver([])
    anchor = _EntityRecord(1, "e1", "g1", "Organization", "Alpha", _normalize_name("Alpha"), {})
    low_noise = _EntityRecord(2, "e2", "g1", "Organization", "Azzzz", _normalize_name("Azzzz"), {})
    plausible = [
        _EntityRecord(10 + i, f"p{i}", "g1", "Organization", f"Alpha{i}", _normalize_name(f"Alpha{i}"), {})
        for i in range(6)
    ]
    ibm = _EntityRecord(20, "ibm", "g1", "Organization", "IBM", _normalize_name("IBM"), {})
    ibm_full = _EntityRecord(
        21,
        "ibm_full",
        "g1",
        "Organization",
        "International Business Machines",
        _normalize_name("International Business Machines"),
        {},
    )
    dropped = [
        (anchor, low_noise, 20.0),
        *[(anchor, other, 65.0 + i) for i, other in enumerate(plausible)],
        (ibm, ibm_full, 18.75),
    ]

    rescued = resolver._rescue_dropped_pairs(dropped, set())

    rescued_ids = [tuple(rg["node_ids"]) for rg in rescued]
    assert (1, 2) not in rescued_ids  # below score floor
    assert (20, 21) in rescued_ids  # multi-word acronym rescued despite low fuzz ratio
    anchor_rescues = [pair for pair in rescued_ids if 1 in pair]
    assert len(anchor_rescues) == 3  # capped per entity to top 3


@pytest.mark.asyncio
async def test_hybrid_merge_skips_redundant_llm_merge_summarization():
    from recon_graphrag.graphdb.entity_resolution import _normalize_name

    rows = [
        _EntityRecord(
            1,
            "e1",
            "g1",
            "Organization",
            "OpenAI",
            _normalize_name("OpenAI"),
            {"descriptions": ["AI lab in San Francisco"]},
        ),
        _EntityRecord(
            2,
            "e2",
            "g1",
            "Organization",
            "openai",
            _normalize_name("openai"),
            {"descriptions": ["Creator of GPT models"]},
        ),
    ]

    calls = 0

    class CountingLLM:
        async def ainvoke(self, prompt):
            nonlocal calls
            calls += 1

            class Response:
                content = "summary"

            return Response()

    resolver = ContractResolver(rows)
    result = await resolver.resolve(strategy="hybrid", llm=CountingLLM())

    assert result["merged_groups"] == 1
    assert calls == 0
