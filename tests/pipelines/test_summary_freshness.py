"""Summary freshness follows effective prompt inputs, not raw text alone."""

import pytest

from tests.pipelines.test_writer_replacement import build_document, make_writer
from tests.pipelines.writer_scenarios import StatefulFakeGraphStore


@pytest.mark.parametrize("updates", [
    {"role": "CEO"}, {"role": None}, {"name": "Alicia"}, {"aliases_domain": ["new"]},
])
def test_property_only_change_invalidates_successful_summary(updates):
    store = StatefulFakeGraphStore()
    writer = make_writer(store)
    document = build_document(chunks=["chunk"], entities=[("alice", "Alice", "Source fact")])
    document.entities[0].properties["role"] = "Engineer"
    writer.write_graph_document(document)
    node = store.get_node("alice")
    assert node is not None
    node.update(description="Previous summary", description_summary_status="success")
    document.entities[0].properties.update(updates)
    writer.write_graph_document(document)
    assert node.get("description_summary_status") is None


def test_omitted_properties_preserve_existing_successful_summary():
    store = StatefulFakeGraphStore()
    writer = make_writer(store)
    document = build_document(chunks=["chunk"], entities=[("alice", "Alice", "Source fact")])
    document.entities[0].properties["role"] = "Engineer"
    writer.write_graph_document(document)
    node = store.get_node("alice")
    assert node is not None
    node.update(description="Previous summary", description_summary_status="success")
    del document.entities[0].properties["role"]
    writer.write_graph_document(document)
    assert node["description_summary_status"] == "success"
