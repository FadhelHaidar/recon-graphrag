from typing import Any, cast

import pytest

from recon_graphrag.embeddings.community_reports import CommunityReportEmbedder


class FakeStore:
    """In-memory community report store.

    Honours failure marks and vector writes so ``embed_reports`` converges the
    way the real shared store does.
    """

    # A fix that never converges should fail fast instead of hanging the suite.
    max_fetches = 10

    def __init__(self, reports=None):
        self.reports = reports if reports is not None else [
            {"id": "c1", "level": 0, "title": "Good", "report_text": "good"},
            {"id": "c1", "level": 1, "title": "Bad", "report_text": "bad"},
        ]
        self.embedded = set()
        self.failed = set()
        self.upserts = []
        self.fetches = 0
        self.fail_writes = False

    def get_unembedded_community_reports(self, graph_name, limit=500):
        self.fetches += 1
        assert self.fetches <= self.max_fetches, (
            "embed_reports did not converge; it refetched the same reports"
        )
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
        if self.fail_writes:
            raise RuntimeError("store rejected the failure mark")
        params = parameters or {}
        self.failed.add((params["id"], params["level"]))
        return []


class NoopMarkStore(FakeStore):
    """Store that acknowledges nothing: marks and writes are silently dropped."""

    def execute_query(self, query, parameters=None):
        return []

    def upsert_community_report_vectors(
        self, node_ids, vectors, graph_name, levels
    ):
        self.upserts.append((node_ids, vectors, graph_name, levels))


class PartialEmbedder:
    async def async_embed_query(self, text):
        if "bad" in text.lower():
            raise RuntimeError("provider failure")
        return [0.1, 0.2]


class CountingEmbedder:
    def __init__(self):
        self.calls = []

    async def async_embed_query(self, text):
        self.calls.append(text)
        return [0.1, 0.2]


def _blank_report(report_id="c0", level=0):
    return {"id": report_id, "level": level, "title": "", "report_text": "   \n\t  "}


def _embedder(store, embedder, graph_name="graph-a") -> CommunityReportEmbedder:
    """The fakes implement only the members this test exercises."""
    return CommunityReportEmbedder(cast(Any, store), cast(Any, embedder), graph_name)


@pytest.mark.asyncio
async def test_embed_reports_scopes_keys_and_terminates_after_failure():
    store = FakeStore()
    count = await _embedder(store, PartialEmbedder()).embed_reports()

    assert count == 1
    assert store.upserts == [(["c1"], [[0.1, 0.2]], "graph-a", [0])]
    assert store.failed == {("c1", 1)}


@pytest.mark.asyncio
async def test_blank_report_is_marked_failed_without_calling_the_provider():
    store = FakeStore([_blank_report()])
    embedder = CountingEmbedder()

    count = await _embedder(store, embedder).embed_reports()

    assert count == 0
    assert store.failed == {("c0", 0)}
    assert embedder.calls == []
    assert store.fetches == 2  # processed once, then drained


@pytest.mark.asyncio
async def test_blank_report_does_not_discard_embeddable_reports():
    store = FakeStore([
        {"id": "c1", "level": 0, "title": "Good", "report_text": "good"},
        _blank_report("c2"),
    ])
    embedder = CountingEmbedder()

    count = await _embedder(store, embedder).embed_reports()

    assert count == 1
    assert "Good - good" in embedder.calls
    assert store.failed == {("c2", 0)}


@pytest.mark.asyncio
async def test_title_only_report_is_still_embeddable():
    store = FakeStore([
        {"id": "c1", "level": 0, "title": "Only Title", "report_text": "   "},
    ])
    embedder = CountingEmbedder()

    count = await _embedder(store, embedder).embed_reports()

    assert count == 1
    assert store.failed == set()


@pytest.mark.asyncio
async def test_unchanged_batch_stops_when_the_store_ignores_marks_and_writes():
    store = NoopMarkStore([_blank_report()])
    embedder = CountingEmbedder()

    count = await _embedder(store, embedder).embed_reports()

    assert count == 0
    assert store.fetches == 2  # second identical batch stops the loop
    assert embedder.calls == []


@pytest.mark.asyncio
async def test_failure_mark_error_keeps_the_running_count():
    store = FakeStore([_blank_report()])
    store.fail_writes = True

    count = await _embedder(store, CountingEmbedder()).embed_reports()

    assert count == 0
    assert store.failed == set()


@pytest.mark.asyncio
async def test_corrected_report_is_embedded_after_the_error_is_cleared():
    store = FakeStore([_blank_report()])
    embedder = CountingEmbedder()
    project = _embedder(store, embedder)

    assert await project.embed_reports() == 0
    assert store.failed == {("c0", 0)}

    # An operator fixes the externally written row and clears the mark.
    store.reports[0]["report_text"] = "recovered"
    store.failed.clear()

    assert await project.embed_reports() == 1
    assert store.embedded == {("c0", 0)}
