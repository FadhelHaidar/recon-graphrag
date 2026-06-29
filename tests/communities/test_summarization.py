"""Tests for community summarization context formatting."""

from __future__ import annotations

import json

import pytest

from recon_graphrag.llm.base import LLMResponse
from recon_graphrag.communities.summarization import CommunitySummarizer


class FakeReportGraphStore:
    def __init__(self):
        self.stored_reports = []
        self.failed_reports = []

    def get_communities(self, graph_name, level=None):
        return [{"id": "community:1", "level": 0, "entity_count": 2}]

    def get_community_ranked_context(self, graph_name, community_id, level=0):
        return [
            {
                "e_id": "person:alice",
                "e_name": "Alice",
                "e_description": "CEO",
                "e_labels": ["Person"],
                "e_degree": 1,
                "rel_type": "WORKS_AT",
                "rel_description": "Alice works at Acme",
                "observation_count": 1,
                "combined_degree": 2,
                "other_id": "org:acme",
                "other_name": "Acme",
                "other_description": "Company",
                "other_labels": ["Organization"],
                "other_degree": 1,
            }
        ]

    def get_claims_for_entities(self, graph_name, entity_ids):
        return []

    def store_community_report(self, report, graph_name):
        self.stored_reports.append((graph_name, report))

    def mark_community_report_failed(self, graph_name, community_id, level, error):
        self.failed_reports.append((graph_name, community_id, level, error))


class FakeReportLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, prompt):
        self.calls += 1
        return LLMResponse(
            content=json.dumps({"title": "Bad", "summary": "No findings", "findings": []})
        )


class FakeSummaryLLM:
    def __init__(self, response: str = "Alice and Acme are key entities in this community."):
        self.calls = 0
        self._response = response

    async def ainvoke(self, prompt):
        self.calls += 1
        return LLMResponse(content=self._response)


class FakeSuccessGraphStore(FakeReportGraphStore):
    def __init__(self):
        super().__init__()
        self.summaries: list[tuple] = []

    def store_community_summary(self, community_id, level, summary, graph_name):
        self.summaries.append((graph_name, community_id, level, summary))

    def get_community_child_summary_context(self, graph_name, community_id, level, child_level):
        return []


@pytest.mark.asyncio
async def test_report_generation_failure_is_marked_not_stored():
    store = FakeReportGraphStore()
    llm = FakeReportLLM()
    summarizer = CommunitySummarizer(store, llm=llm, use_reports=True)

    results, stats = await summarizer.summarize_all(level=0)

    assert results == []
    assert stats.failed == 1
    assert stats.succeeded == 0
    assert store.stored_reports == []
    assert len(store.failed_reports) == 1
    assert store.failed_reports[0][:3] == ("entity-graph", "community:1", 0)


@pytest.mark.asyncio
async def test_plain_text_summary_success_is_stored():
    store = FakeSuccessGraphStore()
    llm = FakeSummaryLLM()
    summarizer = CommunitySummarizer(store, llm=llm, use_reports=False)

    results, stats = await summarizer.summarize_all(level=0)

    assert stats.succeeded == 1
    assert stats.failed == 0
    assert len(results) == 1
    assert results[0]["summary"] == "Alice and Acme are key entities in this community."
    assert len(store.summaries) == 1
    assert store.summaries[0][:3] == ("entity-graph", "community:1", 0)
    assert llm.calls == 1
