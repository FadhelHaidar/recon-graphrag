from recon_graphrag.graphdb.store_base import BaseGraphStore
from recon_graphrag.models.artifacts import CommunityReport


class FakeBaseStore(BaseGraphStore):
    def __init__(self):
        self.calls = []

    def execute_query(self, query: str, parameters: dict | None = None):
        self.calls.append((query.strip(), parameters or {}))
        return []


def test_base_store_get_community_stats_uses_shared_result_shape():
    store = FakeBaseStore()

    store.get_community_stats("graph-a")

    query, params = store.calls[-1]
    assert "c.id AS community_id" in query
    assert "child_community_count" in query
    assert params == {"graph_name": "graph-a"}


def test_base_store_get_communities_reads_level_scope():
    store = FakeBaseStore()

    store.get_communities("graph-a", level=1)

    query, params = store.calls[-1]
    assert "MATCH (c:`Community` {graph_name: $graph_name, level: $level})" in query
    assert "child_community_count" in query
    assert params == {"graph_name": "graph-a", "level": 1}


def test_base_store_store_community_summary_sets_summary():
    store = FakeBaseStore()

    store.store_community_summary("community:1", 0, "summary", "graph-a")

    query, params = store.calls[-1]
    assert "SET c.summary = $summary" in query
    assert params == {
        "graph_name": "graph-a",
        "cid": "community:1",
        "level": 0,
        "summary": "summary",
    }


def test_base_store_store_community_report_sets_structured_fields():
    store = FakeBaseStore()
    report = CommunityReport(
        id="report:community:1:0",
        community_id="community:1",
        level=0,
        title="Title",
        summary="Summary",
    )

    store.store_community_report(report, "graph-a")

    query, params = store.calls[-1]
    assert "c.report_json" in query
    assert "c.report_status = 'success'" in query
    assert params["graph_name"] == "graph-a"
    assert params["cid"] == "community:1"
    assert params["title"] == "Title"


def test_base_store_mark_community_report_failed_sets_status():
    store = FakeBaseStore()

    store.mark_community_report_failed("graph-a", "community:1", 0, "bad json")

    query, params = store.calls[-1]
    assert "c.report_status = 'failed'" in query
    assert params == {
        "graph_name": "graph-a",
        "cid": "community:1",
        "level": 0,
        "error": "bad json",
    }


def test_base_store_claim_and_citation_reads_are_graph_scoped():
    store = FakeBaseStore()

    store.get_claims_for_entities("graph-a", ["person:alice"])
    query, params = store.calls[-1]
    assert "graph_name: $graph_name" in query
    assert params["graph_name"] == "graph-a"

    store.resolve_chunk_citations("graph-a", ["chunk:1"])
    query, params = store.calls[-1]
    assert "Chunk {id: cid, graph_name: $graph_name}" in query
    assert "properties(c) AS chunk_metadata" in query
    assert "properties(d) AS document_metadata" in query
    assert params == {"graph_name": "graph-a", "chunk_ids": ["chunk:1"]}
