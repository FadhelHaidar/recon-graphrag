"""F9 Phase 2: indexed fuzzy candidate discovery without changing outcomes.

The fuzzy pass used to scan the whole singleton pool per anchor and only then
apply ``max_candidates_per_entity``. The index must return exactly what the scan
returned -- same gates, same pool order, same cap -- while inspecting only its
gate-compatible block.
"""

from __future__ import annotations

import random

import pytest

from recon_graphrag.graphdb import entity_resolution as er
from recon_graphrag.graphdb.entity_resolution import (
    BaseEntityResolver,
    _EntityRecord,
    _FuzzyCandidateIndex,
    _scan_candidates,
)

_acronym = BaseEntityResolver._is_acronym_match


def record(node_id: str, name: str, *, graph="g1", domain="Person") -> _EntityRecord:
    return _EntityRecord(
        node_id=node_id,
        entity_id=f"e-{node_id}",
        graph_name=graph,
        domain_label=domain,
        resolve_value=name,
        normalized_value=er._normalize_name(name),
        properties={},
    )


def edge_case_pool() -> list[_EntityRecord]:
    """Names that exercise every gate: equal, same lead, other lead, lengths."""
    names = [
        "Alice",
        "Alice",  # identical normalized value
        "Alicia",  # same lead, different value
        "Alyce",  # same lead, shorter
        "Al",  # very short, same lead
        "A",  # single character
        "Bob",
        "Bobby",
        "Robert",
        "",  # empty normalized value skips the leading-character gate
        "Abcdefghijklmnop",
        "Abcdefghijklmno",
        "Zoe",
    ]
    pool = [record(f"n{i}", name) for i, name in enumerate(names)]
    # Same names in a second graph and domain must never be candidates.
    pool.append(record("other-graph", "Alice", graph="g2"))
    pool.append(record("other-domain", "Alice", domain="Organization"))
    return pool


@pytest.mark.parametrize("cap", [1, 2, 3, 20, 1000])
def test_index_returns_exactly_what_the_full_scan_returns(cap):
    pool = edge_case_pool()
    index = _FuzzyCandidateIndex(pool)
    for anchor in pool:
        assert index.candidates(anchor, cap, set(), _acronym) == _scan_candidates(
            anchor, pool, cap, set(), _acronym
        )


def test_index_respects_used_exclusion_before_the_cap():
    pool = edge_case_pool()
    index = _FuzzyCandidateIndex(pool)
    anchor = pool[0]
    used = {pool[1].node_id, pool[2].node_id}

    assert index.candidates(anchor, 2, used, _acronym) == _scan_candidates(
        anchor, pool, 2, used, _acronym
    )


def test_index_matches_full_scan_on_a_larger_deterministic_pool():
    rng = random.Random(20240517)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    pool = []
    for i in range(400):
        length = rng.randint(0, 14)
        name = "".join(rng.choice(alphabet) for _ in range(length))
        pool.append(record(f"n{i}", name))
    index = _FuzzyCandidateIndex(pool)
    for anchor in pool[::7]:
        assert index.candidates(anchor, 5, set(), _acronym) == _scan_candidates(
            anchor, pool, 5, set(), _acronym
        )


def test_index_inspects_less_than_a_full_scan_per_anchor():
    """Indexed inspections must beat the old one-pool-scan-per-anchor baseline."""
    rng = random.Random(7)
    alphabet = "abcdefghijklmnopqrstuvwxyz"

    def build(count: int) -> list[_EntityRecord]:
        pool = []
        for i in range(count):
            length = rng.randint(4, 12)
            name = rng.choice(alphabet) + "".join(
                rng.choice(alphabet) for _ in range(length - 1)
            )
            pool.append(record(f"n{i}", name))
        return pool

    pool = build(2000)
    anchors = pool[:50]
    full_scan_inspections = len(anchors) * len(pool)  # old behavior, by construction

    index = _FuzzyCandidateIndex(pool)
    for anchor in anchors:
        index.candidates(anchor, 20, set(), _acronym)

    assert index.inspections < full_scan_inspections / 10, (
        index.inspections,
        full_scan_inspections,
    )


def test_inspections_track_the_block_when_all_names_share_length_and_lead():
    """Documented residual: identical length + leading char means one block.

    The gate-compatible set is then a large fraction of the pool, so per-anchor
    inspections scale with that block -- an inherent property of the current
    gates, not something indexing can remove.
    """
    pool = [record(f"n{i}", f"a{i:04d}") for i in range(300)]
    index = _FuzzyCandidateIndex(pool)
    for anchor in pool[:10]:
        index.candidates(anchor, 20, set(), _acronym)

    # Every anchor sees the same single block of 300 records, not 300 per anchor
    # ... capped only by the fact that the block is the whole pool here.
    assert index.inspections == 10 * len(pool)


def test_index_falls_back_when_values_stop_being_single_tokens():
    pool = [
        _EntityRecord(
            node_id="a",
            entity_id="e-a",
            graph_name="g1",
            domain_label="Person",
            resolve_value="Alice Wonder",
            normalized_value="alice wonder",  # hypothetical future normalization
            properties={},
        ),
        record("b", "Alice Wonderland"),
    ]
    index = _FuzzyCandidateIndex(pool)
    assert index.candidates(pool[0], 10, set(), _acronym) == _scan_candidates(
        pool[0], pool, 10, set(), _acronym
    )


def test_fuzzy_groups_are_identical_to_the_full_scan_reference(monkeypatch):
    """Merging, review groups, and dropped pairs must not change with the index."""
    pool = edge_case_pool()
    fuzzy_entities = [
        record("x1", "Jonathan Smith"),
        record("x2", "Jonathan Smithe"),
        record("x3", "Jonathan Smyth"),
        record("x4", "Zoe"),
    ]
    entities = pool + fuzzy_entities

    indexed_groups, indexed_review, indexed_dropped = _build_groups(entities)
    assert indexed_groups

    monkeypatch.setattr(
        er._FuzzyCandidateIndex,
        "candidates",
        lambda self, entity, cap, used, acronym: _scan_candidates(
            entity, self._pool, cap, used, acronym
        ),
    )
    scan_groups, scan_review, scan_dropped = _build_groups(entities)

    assert _shape(indexed_groups) == _shape(scan_groups)
    assert indexed_review == scan_review
    assert _shape(indexed_dropped) == _shape(scan_dropped)


def _build_groups(entities):
    resolver = _MinimalResolver()
    return resolver._build_fuzzy_groups(
        entities, merge_threshold=85.0, review_threshold=70.0, max_candidates_per_entity=20
    )


def _shape(groups):
    """Order-sensitive view of a group/pair result for equality assertions."""
    return [[e.node_id for e in group] if isinstance(group, list) else [e.node_id for e in group[:2]]
            for group in groups]


class _MinimalResolver(BaseEntityResolver):
    """Concrete resolver used only for shared grouping logic."""

    def __init__(self):
        super().__init__(graph_store=None)

    def _load_entities(self, graph_name: str, resolve_property: str):
        return []

    def _merge_groups(self, groups, resolve_property: str) -> int:
        return 0
