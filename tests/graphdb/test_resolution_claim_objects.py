"""Entity merges must retarget scalar claim objects, regardless of subject."""

import pytest

from recon_graphrag.graphdb.entity_resolution import _EntityRecord
from recon_graphrag.graphdb.memgraph.entity_resolution import _MemgraphEntityResolver
from recon_graphrag.graphdb.neo4j.entity_resolution import _Neo4jEntityResolver
from tests.graphdb.memgraph.test_memgraph_entity_resolution import (
    FakeGraphStore as MemgraphStore,
)
from tests.graphdb.neo4j.test_neo4j_entity_resolution import (
    FakeGraphStore as Neo4jStore,
)


def merge_group():
    return [
        _EntityRecord(2, "duplicate-a", "g:1", "Person", "Alice", "alice"),
        _EntityRecord(1, "canonical", "g:1", "Person", "Alice Smith", "alicesmith"),
        _EntityRecord(3, "duplicate-b", "g:1", "Person", "A. Smith", "asmith"),
    ]


@pytest.mark.parametrize("resolver_class,store_class", [
    (_Neo4jEntityResolver, Neo4jStore),
    (_MemgraphEntityResolver, MemgraphStore),
])
def test_claim_object_rewrite_is_graph_scoped_and_independent_of_subject(
    resolver_class, store_class,
):
    store = store_class()
    assert resolver_class(store)._merge_groups([merge_group()], "name") == 3
    calls = store.calls if hasattr(store, "calls") else store.queries
    rewrites = [(query, params) for query, params in calls
                if "SET c.object_entity_id" in query]
    assert len(rewrites) == 1
    query, params = rewrites[0]
    graph_parameter = "canonical_graph_name" if store_class is Neo4jStore else "graph_name"
    # The standalone Claim match also covers subjects outside the merge group.
    assert f"MATCH (c:Claim {{graph_name: ${graph_parameter}}})" in query
    assert "WHERE c.object_entity_id IN $losing_entity_ids" in query
    assert "SET c.object_entity_id = $canonical_entity_id" in query
    assert "SUBJECT_OF" not in query
    assert params[graph_parameter] == "g:1"
    assert params["canonical_entity_id"] == "canonical"
    assert set(params["losing_entity_ids"]) == {"duplicate-a", "duplicate-b"}
    assert "canonical" not in params["losing_entity_ids"]


def test_neo4j_claim_rewrite_is_atomic_with_merge_and_preserves_no_claim_result():
    store = Neo4jStore()
    _Neo4jEntityResolver(store)._merge_groups([merge_group()], "name")
    query, _ = next(call for call in store.calls if "apoc.refactor.mergeNodes" in call[0])
    assert query.index("apoc.refactor.mergeNodes") < query.index("SET c.object_entity_id")
    assert "OPTIONAL MATCH (c:Claim" in query
    assert "WITH node, collect(c) AS claims" in query
    assert "FOREACH (c IN claims | SET c.object_entity_id = $canonical_entity_id)" in query
    assert query.index("SET c.object_entity_id") < query.index("RETURN elementId(node)")


def test_memgraph_claim_rewrite_precedes_deletion_and_failure_stops_deletion():
    class FailingStore(MemgraphStore):
        def execute_query(self, query, parameters=None):
            result = super().execute_query(query, parameters)
            if "SET c.object_entity_id" in query:
                raise RuntimeError("claim rewrite failed")
            return result

    store = MemgraphStore()
    _MemgraphEntityResolver(store)._merge_groups([merge_group()], "name")
    queries = [query for query, _ in store.queries]
    rewrite_index = next(i for i, query in enumerate(queries) if "SET c.object_entity_id" in query)
    delete_index = next(i for i, query in enumerate(queries) if "DELETE n" in query)
    assert rewrite_index < delete_index

    failing_store = FailingStore()
    with pytest.raises(RuntimeError, match="claim rewrite failed"):
        _MemgraphEntityResolver(failing_store)._merge_groups([merge_group()], "name")
    assert not any("DELETE n" in query for query, _ in failing_store.queries)
