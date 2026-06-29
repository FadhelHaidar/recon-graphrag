import pytest

from recon_graphrag.embeddings.community_reports import CommunityReportEmbedder


class FakeStore:
    def __init__(self):
        self.reports = [
            {"id": "c1", "level": 0, "title": "Good", "report_text": "good"},
            {"id": "c1", "level": 1, "title": "Bad", "report_text": "bad"},
        ]
        self.embedded = set()
        self.failed = set()
        self.upserts = []

    def get_unembedded_community_reports(self, graph_name, limit=500):
        return [
            report for report in self.reports
            if (report["id"], report["level"]) not in self.embedded | self.failed
        ][:limit]

    def upsert_community_report_vectors(
        self, node_ids, vectors, graph_name, levels
    ):
        self.upserts.append((node_ids, vectors, graph_name, levels))
        self.embedded.update(zip(node_ids, levels))

    def execute_query(self, query, parameters=None):
        params = parameters or {}
        self.failed.add((params["id"], params["level"]))
        return []


class PartialEmbedder:
    async def async_embed_query(self, text):
        if "bad" in text.lower():
            raise RuntimeError("provider failure")
        return [0.1, 0.2]


@pytest.mark.asyncio
async def test_embed_reports_scopes_keys_and_terminates_after_failure():
    store = FakeStore()
    count = await CommunityReportEmbedder(
        store, PartialEmbedder(), "graph-a"
    ).embed_reports()

    assert count == 1
    assert store.upserts == [(["c1"], [[0.1, 0.2]], "graph-a", [0])]
    assert store.failed == {("c1", 1)}
