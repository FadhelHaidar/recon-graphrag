"""Backend resolution must not lose ownership or canonical identity."""

from copy import deepcopy

import pytest

from recon_graphrag.graphdb._provenance import (
    encode_contributions,
    raw_descriptions,
    reject_mixed_ownership,
)
from recon_graphrag.graphdb.entity_resolution import _EntityRecord
from recon_graphrag.graphdb.memgraph.entity_resolution import _MemgraphEntityResolver
from recon_graphrag.graphdb.neo4j.entity_resolution import _Neo4jEntityResolver
from tests.graphdb.memgraph.test_memgraph_entity_resolution import (
    FakeGraphStore as MemgraphStore,
)
from tests.graphdb.neo4j.test_neo4j_entity_resolution import (
    FakeGraphStore as Neo4jStore,
)


def group():
    return [
        _EntityRecord(1, 'canonical', 'g:1', 'Person', 'Alice Smith', 'alicesmith', {
            'uid': '3:g:1:canonical', 'descriptions': ['shared'], 'description': 'Old summary',
            'embedding_input_fingerprint': 'before-merge-A',
            'description_sources': encode_contributions([['a', 'shared']]),
        }),
        _EntityRecord(2, 'duplicate', 'g:1', 'Person', 'Alice', 'alice', {
            'uid': '3:g:1:duplicate', 'descriptions': ['shared'],
            'embedding_input_fingerprint': 'before-merge-B',
            'description_sources': encode_contributions([['b', 'shared']]),
        }),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize('resolver_class,store_class', [
    (_Neo4jEntityResolver, Neo4jStore), (_MemgraphEntityResolver, MemgraphStore),
])
async def test_merge_summaries_leave_owned_raw_descriptions_unchanged(resolver_class, store_class):
    entities = group()
    before = deepcopy([entity.properties for entity in entities])
    resolver = resolver_class(store_class())
    await resolver._prepare_merge_summaries([entities], None)
    assert [entity.properties for entity in entities] == before


def test_neo4j_merge_preserves_scalar_canonical_uid_and_normalizes_provenance():
    store = Neo4jStore()
    resolver = _Neo4jEntityResolver(store)
    resolver._merge_groups([group()], 'name')
    query, params = next(call for call in store.calls if 'apoc.refactor.mergeNodes' in call[0])
    assert params['canonical_uid'] == '3:g:1:canonical'
    assert params['canonical_graph_name'] == 'g:1'
    assert 'node.uid = $canonical_uid' in query
    assert 'node.embedding_input_fingerprint = null' in query
    assert "uid: 'discard'" in query
    assert "`.*`: 'combine'" in query
    assert 'ORDER BY apoc.coll.indexOf($node_ids, elementId(n))' in query
    assert 'r.strength_sources = CASE WHEN r.strength_sources = [] THEN null' in query
    normalize = next(query for query, _ in store.calls if 'SET n.description_sources' in query)
    assert 'apoc.convert.toList(n.description_sources)' in normalize
    assert 'apoc.convert.toList(r.description_sources)' in normalize
    assert 'apoc.convert.toList(r.strength_sources)' in normalize


def test_memgraph_merge_unions_node_and_both_edge_direction_provenance():
    entities = group()

    class Store(MemgraphStore):
        def execute_query(self, query, parameters=None):
            result = super().execute_query(query, parameters)
            if 'RETURN id(n) AS node_id, properties(n) AS props' in query:
                return [{'node_id': entity.node_id, 'props': deepcopy(entity.properties)} for entity in entities]
            return result

    store = Store()
    _MemgraphEntityResolver(store)._merge_groups([entities], 'name')
    props = next(params['props'] for query, params in store.queries if 'SET n += $props' in query)
    assert props['uid'] == '3:g:1:canonical'
    assert props['graph_name'] == 'g:1'
    assert props['embedding_input_fingerprint'] is None
    assert props['description_sources'] == encode_contributions([['a', 'shared'], ['b', 'shared']])
    rewires = [query for query, _ in store.queries if 'SET new_r += rel_props' in query]
    assert len(rewires) == 2
    for query in rewires:
        assert 'existing_description_sources + coalesce(rel_props.description_sources, [])' in query
        assert 'existing_strength_sources + coalesce(rel_props.strength_sources, [])' in query
        assert 'coalesce(rel_props.descriptions,' in query
        assert query.index('AS existing_description_sources') < query.index('SET new_r += rel_props')
        assert query.index('SET new_r.description_sources') > query.index('SET new_r += rel_props')


@pytest.mark.parametrize('resolver_class,store_class', [
    (_Neo4jEntityResolver, Neo4jStore), (_MemgraphEntityResolver, MemgraphStore),
])
@pytest.mark.parametrize('incident_edges', [False, True])
@pytest.mark.parametrize('strength', [False, True])
def test_mixed_legacy_ownership_rejected_before_mutation(
    resolver_class, store_class, incident_edges, strength,
):
    entities = group()
    if strength:
        tracked = {'strength': 9.0, 'strength_sources': encode_contributions([['a', 9.0]])}
        legacy = {'strength': 9.0}
    else:
        tracked = entities[0].properties
        legacy = {'descriptions': ['shared'], 'description': 'same text, unknown owner'}
    if incident_edges:
        rows = [
            {'source_node_id': 1, 'target_node_id': 3, 'rel_type': 'KNOWS', 'rel_props': tracked},
            {'source_node_id': 2, 'target_node_id': 3, 'rel_type': 'KNOWS', 'rel_props': legacy},
        ]
    else:
        entities[0].properties = tracked
        entities[1].properties = legacy
        rows = []

    class Store(store_class):
        def execute_query(self, query, parameters=None):
            result = super().execute_query(query, parameters)
            if 'properties(r) AS rel_props' in query:
                return rows
            return result

    store = Store()
    with pytest.raises(ValueError, match='mixed tracked/legacy'):
        resolver_class(store)._merge_groups([entities], 'name')
    calls = store.calls if hasattr(store, 'calls') else store.queries
    assert all('SET ' not in query and 'DELETE ' not in query for query, _ in calls)


def test_empty_raw_descriptions_do_not_adopt_summary_as_owned_text():
    props = {'descriptions': [], 'description': 'stale summary', 'description_sources': []}
    assert raw_descriptions(props) == []
    reject_mixed_ownership([props, group()[0].properties])
