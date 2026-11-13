"""Resolvers check incident graph ownership before either backend mutates data."""

from unittest.mock import MagicMock

import pytest

from recon_graphrag.graphdb._provenance import preflight_merge_ownership
from recon_graphrag.graphdb.memgraph.entity_resolution import _MemgraphEntityResolver
from recon_graphrag.graphdb.neo4j.entity_resolution import _Neo4jEntityResolver
from tests.graphdb.test_resolution_provenance import group


@pytest.mark.parametrize("resolver_type", [_Neo4jEntityResolver, _MemgraphEntityResolver])
@pytest.mark.parametrize("recorded_type,expected", [("Organization", "Organization"), (None, "Person"), ("", "Person")])
def test_load_entities_prefers_recorded_type(resolver_type, recorded_type, expected):
    store = MagicMock()
    store.execute_query.return_value = [{
        "labels": ["__Entity__", "Person", "Organization"],
        "properties": {"type": recorded_type}, "resolve_value": "Acme",
    }]
    assert resolver_type(store)._load_entities("g", "name")[0].domain_label == expected


@pytest.mark.parametrize("resolver_type", [_Neo4jEntityResolver, _MemgraphEntityResolver])
@pytest.mark.parametrize("edge_graph,target_graph,rel_type,source_labels,target_labels,allowed", [
    (None, "g:1", "KNOWS", ["__Entity__"], ["__Entity__"], False),
    ("foreign", "g:1", "KNOWS", ["__Entity__"], ["__Entity__"], False),
    ("g:1", "foreign", "KNOWS", ["__Entity__"], ["__Entity__"], False),
    (None, "g:1", "FROM_CHUNK", ["Chunk"], ["__Entity__"], True),
    (None, "g:1", "SUBJECT_OF", ["Claim"], ["__Entity__"], True),
    (None, "g:1", "IN_COMMUNITY", ["__Entity__"], ["Community"], True),
    (None, "g:1", "FROM_CHUNK", ["__Entity__"], ["__Entity__"], False),
    ("foreign", "g:1", "FROM_CHUNK", ["Chunk"], ["__Entity__"], False),
])
def test_incident_scope_checked_before_mutation(resolver_type, edge_graph, target_graph, rel_type, source_labels, target_labels, allowed):
    store = MagicMock()
    row = {"source_graph": "g:1", "target_graph": target_graph, "scope_type": rel_type,
           "source_labels": source_labels, "target_labels": target_labels, "relationship_graph": edge_graph}
    store.execute_query.side_effect = lambda query, params: [row] if "AS scope_type" in query else []
    if allowed:
        preflight_merge_ownership(store, [group()], "id")
    else:
        with pytest.raises(ValueError, match="foreign or unscoped"):
            resolver_type(store)._merge_groups([group()], "name")
    assert all("SET " not in call.args[0] and "DELETE " not in call.args[0] for call in store.execute_query.call_args_list)


@pytest.mark.parametrize("resolver_type", [_Neo4jEntityResolver, _MemgraphEntityResolver])
def test_all_groups_are_scope_checked_before_first_merge(resolver_type):
    from copy import deepcopy

    first = group()
    second = deepcopy(first)
    for entity in second:
        entity.node_id += 10
    store = MagicMock()

    def execute(query, params):
        if "AS scope_type" in query and params["node_ids"] == [11, 12]:
            return [{"source_graph": "g:1", "target_graph": "g:1",
                     "source_labels": ["__Entity__"], "target_labels": ["__Entity__"],
                     "scope_type": "KNOWS", "relationship_graph": None}]
        return []

    store.execute_query.side_effect = execute
    with pytest.raises(ValueError, match="foreign or unscoped"):
        resolver_type(store)._merge_groups([first, second], "name")
    assert all("SET " not in call.args[0] and "DELETE " not in call.args[0] for call in store.execute_query.call_args_list)
