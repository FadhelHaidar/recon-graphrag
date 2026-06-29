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
        self.existing_fingerprint = None

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

    def get_community_reports_by_keys(self, graph_name, keys, top_k):
        if self.existing_fingerprint is None:
            return []
        return [
            {
                "id": keys[0]["id"],
                "level": keys[0]["level"],
                "report_text": "Existing report",
                "input_fingerprint": self.existing_fingerprint,
            }
        ]

    def store_community_report(self, report, graph_name):
        self.stored_reports.append((graph_name, report))

    def mark_community_report_failed(self, graph_name, community_id, level, error):
        self.failed_reports.append((graph_name, community_id, level, error))

    def get_child_community_reports(
        self, graph_name, community_id, level, child_level
    ):
        return []


class FakeReportLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, prompt):
        self.calls += 1
        return LLMResponse(
            content=json.dumps({"title": "Bad", "summary": "No findings", "findings": []})
        )


class FakeValidReportLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, prompt):
        self.calls += 1
        return LLMResponse(
            content=json.dumps(
                {
                    "title": "Alice and Acme",
                    "summary": "Alice works with Acme.",
                    "rating": 7.0,
                    "rating_explanation": "Important relationship.",
                    "findings": [
                        {
                            "description": "Alice is connected to Acme.",
                            "references": [
                                {"target_id": "person:alice", "target_type": "entity"}
                            ],
                        }
                    ],
                }
            )
        )


class CapturingValidReportLLM(FakeValidReportLLM):
    def __init__(self):
        super().__init__()
        self.prompts = []

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return await super().ainvoke(prompt)


@pytest.mark.asyncio
async def test_report_generation_failure_is_marked_not_stored():
    store = FakeReportGraphStore()
    llm = FakeReportLLM()
    summarizer = CommunitySummarizer(store, llm=llm)

    results, stats = await summarizer.generate_all(level=0)

    assert results == []
    assert stats.failed == 1
    assert stats.succeeded == 0
    assert store.stored_reports == []
    assert len(store.failed_reports) == 1
    assert store.failed_reports[0][:3] == ("entity-graph", "community:1", 0)


@pytest.mark.asyncio
async def test_report_generation_stores_input_fingerprint():
    store = FakeReportGraphStore()
    llm = FakeValidReportLLM()
    summarizer = CommunitySummarizer(store, llm=llm)

    results, stats = await summarizer.generate_all(level=0)

    assert stats.succeeded == 1
    assert results
    _, report = store.stored_reports[0]
    assert report.version.input_fingerprint


@pytest.mark.asyncio
async def test_report_skip_existing_requires_matching_fingerprint():
    store = FakeReportGraphStore()
    llm = FakeValidReportLLM()
    summarizer = CommunitySummarizer(store, llm=llm)
    context = summarizer._fetch_community_context_obj("community:1", 0)
    store.existing_fingerprint = summarizer._context_fingerprint(context)

    results, stats = await summarizer.generate_all(level=0, skip_existing=True)

    assert results == []
    assert stats.skipped == 1
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_parent_report_substitutes_child_reports_when_direct_context_truncates():
    class ParentStore(FakeReportGraphStore):
        def get_community_ranked_context(self, graph_name, community_id, level=0):
            rows = super().get_community_ranked_context(graph_name, community_id, level)
            rows[0]["e_description"] = "A" * 1000
            rows[0]["other_description"] = "B" * 1000
            return rows

        def get_child_community_reports(
            self, graph_name, community_id, level, child_level
        ):
            return [{
                "id": "child:1",
                "level": child_level,
                "report_text": "Important child report.",
                "input_fingerprint": "child-fingerprint",
                "context_tokens_used": 500,
            }]

    llm = CapturingValidReportLLM()
    summarizer = CommunitySummarizer(ParentStore(), llm=llm, max_context_tokens=200)

    report = await summarizer.generate_report("community:1", level=0)

    assert "Important child report." in llm.prompts[0]
    assert report.context_truncated is True
