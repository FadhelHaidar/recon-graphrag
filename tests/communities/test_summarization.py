"""Tests for community summarization context formatting."""

from __future__ import annotations

import json

import pytest

from recon_graphrag.communities.summarization import CommunitySummarizer
from recon_graphrag.llm.base import LLMResponse


class FakeNode(dict):
    def __init__(self, labels, **properties):
        super().__init__(properties)
        self.labels = labels


class FakeObjectNode:
    def __init__(self, labels, **properties):
        self.labels = labels
        self.properties = properties


class FakeGraphStore:
    def __init__(self, rows):
        self.rows = rows

    def get_community_entity_context(self, graph_name, community_id, level=0):
        return self.rows


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


def test_fetch_entity_context_accepts_list_labels():
    store = FakeGraphStore(
        [
            {
                "e": FakeNode(["__Entity__", "Person"], name="Keanu Reeves"),
                "rel_type": "ACTED_IN",
                "other": FakeNode(["__Entity__", "Movie"], name="The Matrix"),
            }
        ]
    )
    summarizer = CommunitySummarizer(store, llm=None)

    context = summarizer._fetch_entity_context("87")

    assert "- [Person] Keanu Reeves" in context
    assert "Keanu Reeves --[ACTED_IN]--> The Matrix" in context


def test_fetch_entity_context_accepts_set_labels():
    store = FakeGraphStore(
        [
            {
                "e": FakeNode({"__Entity__", "Movie"}, name="The Matrix"),
                "rel_type": None,
                "other": None,
            }
        ]
    )
    summarizer = CommunitySummarizer(store, llm=None)

    context = summarizer._fetch_entity_context("88")

    assert context == "- [Movie] The Matrix"


def test_fetch_entity_context_accepts_backend_node_objects_without_get():
    store = FakeGraphStore(
        [
            {
                "e": FakeObjectNode(["__Entity__", "Person"], name="Carrie-Anne Moss"),
                "rel_type": "ACTED_IN",
                "other": FakeObjectNode(["__Entity__", "Movie"], name="The Matrix"),
            }
        ]
    )
    summarizer = CommunitySummarizer(store, llm=None)

    context = summarizer._fetch_entity_context("96")

    assert "- [Person] Carrie-Anne Moss" in context
    assert "Carrie-Anne Moss --[ACTED_IN]--> The Matrix" in context


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
