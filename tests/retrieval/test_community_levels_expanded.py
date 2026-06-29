"""Expanded fake-store tests for community level resolution and hierarchy shape.

These tests document the current behavior of ``resolve_community_level`` and
related graph-store community queries with deterministic fake stores.
"""

from __future__ import annotations

import pytest

from recon_graphrag.retrieval.community_levels import resolve_community_level


class FakeGraphStore:
    def __init__(self, communities=None):
        self._communities = communities or []

    def execute_query(self, query, parameters=None):
        if "RETURN max(c.level) AS level" in query:
            levels = [c["level"] for c in self._communities if c.get("level") is not None]
            return [{"level": max(levels) if levels else None}]
        return []

    def get_communities(self, graph_name, level=None):
        communities = [c for c in self._communities if c.get("graph_name") == graph_name]
        if level is not None:
            communities = [c for c in communities if c.get("level") == level]
        return sorted(communities, key=lambda c: c.get("level", 0))


def _communities(levels: list[int], graph_name: str = "entity-graph") -> list[dict]:
    return [
        {"id": f"c{i}", "level": lvl, "graph_name": graph_name, "summary": f"level {lvl}"}
        for i, lvl in enumerate(levels)
    ]


def test_finest_resolves_to_max_stored():
    """After reversal: finest = highest level number."""
    store = FakeGraphStore(_communities([0, 1, 2]))
    assert resolve_community_level(store, "entity-graph", "finest") == 2


def test_coarsest_resolves_to_zero():
    """After reversal: coarsest = level 0."""
    store = FakeGraphStore(_communities([0, 1, 2]))
    assert resolve_community_level(store, "entity-graph", "coarsest") == 0


def test_explicit_level_passes_through():
    store = FakeGraphStore(_communities([0, 1, 2]))
    assert resolve_community_level(store, "entity-graph", 1) == 1


def test_none_and_all_resolve_to_none():
    store = FakeGraphStore(_communities([0, 1, 2]))
    assert resolve_community_level(store, "entity-graph", None) is None
    assert resolve_community_level(store, "entity-graph", "all") is None


def test_negative_level_raises():
    with pytest.raises(ValueError):
        resolve_community_level(FakeGraphStore(), "entity-graph", -1)


def test_invalid_string_raises():
    with pytest.raises(ValueError):
        resolve_community_level(FakeGraphStore(), "entity-graph", "middle")


def test_no_community_graph_returns_zero_for_coarsest():
    """After reversal: coarsest is always level 0."""
    store = FakeGraphStore([])
    assert resolve_community_level(store, "entity-graph", "coarsest") == 0


def test_no_community_graph_returns_none_for_finest():
    """After reversal: finest requires querying max level."""
    store = FakeGraphStore([])
    assert resolve_community_level(store, "entity-graph", "finest") is None


def test_single_level_graph_returns_zero_for_coarsest():
    store = FakeGraphStore(_communities([0]))
    assert resolve_community_level(store, "entity-graph", "coarsest") == 0


def test_level_ordering_ascending():
    store = FakeGraphStore(_communities([2, 0, 1]))
    levels = [c["level"] for c in store.get_communities("entity-graph")]
    assert levels == [0, 1, 2]


def test_parent_child_direction():
    """After reversal: hierarchy edges point from child (higher level) to parent (lower level).

    Level 0 = coarsest (top-level parent), higher levels = finer sub-communities.
    """
    store = FakeGraphStore(
        [
            {"id": "c0", "level": 0, "graph_name": "entity-graph", "parent": None},
            {"id": "c1", "level": 1, "graph_name": "entity-graph", "parent": "c0"},
        ]
    )
    child = store.get_communities("entity-graph", level=1)[0]
    parent = store.get_communities("entity-graph", level=0)[0]
    assert child["level"] - 1 == parent["level"]
    assert child["parent"] == parent["id"]
