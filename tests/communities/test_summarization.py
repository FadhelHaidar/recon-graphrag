"""Tests for community summarization context formatting."""

from __future__ import annotations

from recon_graphrag.communities.summarization import CommunitySummarizer


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
