"""Shared templates use recorded types and explicit domain relationship scope."""

import pytest

from recon_graphrag.retrieval._queries import (
    COMMUNITY_MEMBER_GRAPH_QUERY,
    drift_retrieval_query,
    local_retrieval_query,
    ranked_context_query,
)


@pytest.mark.parametrize("factory", [local_retrieval_query, drift_retrieval_query])
@pytest.mark.parametrize("ordered", [False, True])
def test_local_and_drift_types_and_relationship_scope(factory, ordered):
    query = factory(order_by_score=ordered)
    assert "labels(node)[-1]" not in query
    assert "CASE WHEN node.type <> '' THEN node.type" in query
    assert "head([label IN labels(node) WHERE label <> '__Entity__'])" in query
    assert query.count("AND r.graph_name = node.graph_name") == 2
    assert "source_rel.graph_name IS NULL OR source_rel.graph_name = node.graph_name" in query


def test_community_domain_edges_require_graph_ownership():
    for query in [COMMUNITY_MEMBER_GRAPH_QUERY, ranked_context_query(id_func="id")]:
        assert "r.graph_name = $graph_name" in query
        assert "r1.graph_name = $graph_name" in query
