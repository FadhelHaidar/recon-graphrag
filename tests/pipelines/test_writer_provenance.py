"""Writer provenance stays property-compatible and replacement-owned."""

import json
from unittest.mock import MagicMock

import pytest

from recon_graphrag.graphdb._provenance import encode_contributions
from recon_graphrag.pipelines.writer import (
    CypherGraphWriter,
    _DocumentState,
    _accessible_entries,
    _description_entries,
    _merge_contributions,
)
from tests.pipelines.test_writer_replacement import build_document


def test_description_provenance_json_roundtrip_and_apoc_singleton():
    entries = [['doc:"a', 'A\nquoted "fact" — text']]
    stored = encode_contributions(entries)
    assert all(isinstance(value, str) for value in stored)
    assert json.loads(stored[0]) == entries[0]
    assert _description_entries(stored) == entries
    assert _description_entries(stored[0]) == entries


@pytest.mark.parametrize('value', ['not json', ['["only-owner"]'], ['["doc", {}]']])
def test_description_provenance_rejects_malformed_records(value):
    with pytest.raises(ValueError, match='re-ingest'):
        _description_entries(value)


def test_identical_text_retains_both_owners_until_each_is_removed():
    entries, descriptions = _merge_contributions([['a', 'shared']], 'b', ['shared'])
    assert entries == [['a', 'shared'], ['b', 'shared']]
    assert descriptions == ['shared']
    entries, descriptions = _merge_contributions(entries, 'a', [])
    assert entries == [['b', 'shared']]
    assert descriptions == ['shared']


@pytest.mark.parametrize('new_descriptions', [['same'], ['new'], []])
def test_legacy_description_ownership_is_never_adopted(new_descriptions):
    with pytest.raises(ValueError, match='per-document provenance'):
        _accessible_entries(None, ['same'], new_descriptions, 'doc', 'entity')


def test_removing_contribution_rebuilds_raw_text_not_stale_summary():
    document = build_document(document_id='a', chunks=['a:c0'])
    stored = {
        'id': 'e1', 'name': 'Alice', 'labels': ['__Entity__', 'Person'],
        'descriptions': ['old', 'surviving'],
        'description_sources': encode_contributions([['a', 'old'], ['b', 'surviving']]),
        'description': 'Old summary containing removed facts',
        'description_summary_status': 'complete',
        'embedding_input_fingerprint': 'old-fingerprint',
    }
    row, = CypherGraphWriter(MagicMock())._rebuild_entity_rows(
        document, _DocumentState(entities={'e1': stored}), []
    )
    assert row['description_text'] == 'surviving'
    assert row['descriptions_changed'] is True
    assert row['embedding_input_fingerprint'] != 'old-fingerprint'
    assert _description_entries(row['description_sources']) == [['b', 'surviving']]


def relationship_state():
    return _DocumentState(chunk_ids=['a:c0'], relationships=[{
        'id': 'e1:WORKS_AT:e2', 'source_id': 'e1', 'target_id': 'e2', 'type': 'WORKS_AT',
        'source_chunk_ids': ['a:c0', 'b:c0'], 'descriptions': ['shared'],
        'description_sources': encode_contributions([['a', 'shared'], ['b', 'shared']]),
        'strength': 9.0, 'strength_sources': encode_contributions([['a', 9.0], ['b', 3.0]]),
    }])


def test_lowered_strength_uses_surviving_document_maximum():
    document = build_document(document_id='a', chunks=['a:c0'],
                              relationships=[('WORKS_AT', 'e1', 'e2', 'shared')])
    document.relationships[0].strength = 1.0
    row, = CypherGraphWriter(MagicMock())._relationship_rows(document, relationship_state())
    assert row['strength'] == 3.0
    assert [json.loads(value) for value in row['strength_sources']] == [['b', 3.0], ['a', 1.0]]


@pytest.mark.parametrize('drop_relationship', [False, True])
def test_removed_strength_cannot_survive_replacement(drop_relationship):
    document = build_document(document_id='a', chunks=['a:c0'], relationships=(
        [] if drop_relationship else [('WORKS_AT', 'e1', 'e2', 'shared')]
    ))
    writer = CypherGraphWriter(MagicMock())
    if drop_relationship:
        row, = writer._rebuild_relationship_rows(document, relationship_state(), [])
    else:
        row, = writer._relationship_rows(document, relationship_state())
    assert row['strength'] == 3.0
    assert [json.loads(value) for value in row['strength_sources']] == [['b', 3.0]]


def test_last_strength_removal_clears_value():
    document = build_document(document_id='a', chunks=['a:c0'],
                              relationships=[('WORKS_AT', 'e1', 'e2', 'shared')])
    state = relationship_state()
    state.relationships[0]['strength_sources'] = encode_contributions([['a', 9.0]])
    row, = CypherGraphWriter(MagicMock())._relationship_rows(document, state)
    assert row['strength'] is None
    assert row['strength_sources'] == []


def test_unknown_legacy_strength_is_rejected_even_if_identical():
    document = build_document(document_id='a', chunks=['a:c0'],
                              relationships=[('WORKS_AT', 'e1', 'e2', 'shared')])
    document.relationships[0].strength = 9.0
    state = relationship_state()
    state.relationships[0]['strength_sources'] = None
    with pytest.raises(ValueError, match='strength ownership is unknown'):
        CypherGraphWriter(MagicMock())._relationship_rows(document, state)


def test_legacy_removed_description_fails_before_any_mutation():
    from tests.pipelines.test_writer_replacement import make_writer
    from tests.pipelines.writer_scenarios import StatefulFakeGraphStore

    store = StatefulFakeGraphStore()
    writer = make_writer(store)
    writer.write_graph_document(build_document(chunks=['a:c0'], entities=[('e1', 'Alice', 'legacy')]))
    node = store.get_node('e1')
    assert node is not None
    node['description_sources'] = None
    store.queries.clear()
    with pytest.raises(ValueError, match='re-ingest'):
        writer.write_graph_document(build_document(chunks=['a:c0']))
    assert all('SET ' not in query and 'DELETE ' not in query for query in store.queries)
