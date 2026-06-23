"""Tests for typed community context records and rendering."""

from __future__ import annotations

from recon_graphrag.communities.context import (
    CommunityContext,
    EdgeContext,
    EntityContext,
    parse_community_context,
    render_community_context,
)


def _make_rows():
    """Simulate query rows from degree-ranked context query."""
    return [
        {
            "e_id": "e1",
            "e_name": "Alice",
            "e_description": "CEO of Acme",
            "e_labels": ["__Entity__", "Person"],
            "e_degree": 5,
            "rel_type": "WORKS_AT",
            "rel_description": "Alice works at Acme",
            "observation_count": 2,
            "combined_degree": 8,
            "other_id": "e2",
            "other_name": "Acme",
            "other_description": "A tech company",
            "other_labels": ["__Entity__", "Organization"],
            "other_degree": 3,
        },
        {
            "e_id": "e1",
            "e_name": "Alice",
            "e_description": "CEO of Acme",
            "e_labels": ["__Entity__", "Person"],
            "e_degree": 5,
            "rel_type": "FRIEND_OF",
            "rel_description": "Alice knows Bob",
            "observation_count": 1,
            "combined_degree": 7,
            "other_id": "e3",
            "other_name": "Bob",
            "other_description": "Engineer",
            "other_labels": ["__Entity__", "Person"],
            "other_degree": 2,
        },
        {
            "e_id": "e4",
            "e_name": "Charlie",
            "e_description": "Intern",
            "e_labels": ["__Entity__", "Person"],
            "e_degree": 0,
            "rel_type": None,
            "rel_description": None,
            "observation_count": None,
            "combined_degree": None,
            "other_id": None,
            "other_name": None,
            "other_description": None,
            "other_labels": None,
            "other_degree": None,
        },
    ]


class TestParseCommunityContext:
    def test_parses_edges_sorted_by_combined_degree(self):
        rows = _make_rows()
        ctx = parse_community_context("c1", 0, rows)

        assert ctx.community_id == "c1"
        assert ctx.level == 0
        assert len(ctx.edges) == 2
        # First edge has higher combined_degree (8 vs 7)
        assert ctx.edges[0].combined_degree == 8
        assert ctx.edges[0].relationship_type == "WORKS_AT"
        assert ctx.edges[1].combined_degree == 7
        assert ctx.edges[1].relationship_type == "FRIEND_OF"

    def test_parses_entity_context(self):
        rows = _make_rows()
        ctx = parse_community_context("c1", 0, rows)

        assert ctx.edges[0].source.id == "e1"
        assert ctx.edges[0].source.name == "Alice"
        assert ctx.edges[0].source.degree == 5
        assert ctx.edges[0].target.id == "e2"
        assert ctx.edges[0].target.name == "Acme"
        assert ctx.edges[0].target.degree == 3

    def test_parses_isolated_entities(self):
        rows = _make_rows()
        ctx = parse_community_context("c1", 0, rows)

        # Charlie has no edges
        assert len(ctx.entities) == 1
        assert ctx.entities[0].id == "e4"
        assert ctx.entities[0].name == "Charlie"

    def test_strips_entity_label(self):
        rows = _make_rows()
        ctx = parse_community_context("c1", 0, rows)

        assert "__Entity__" not in ctx.edges[0].source.labels
        assert "Person" in ctx.edges[0].source.labels

    def test_empty_rows(self):
        ctx = parse_community_context("c1", 0, [])
        assert ctx.entities == []
        assert ctx.edges == []


class TestRenderCommunityContext:
    def test_renders_edges_with_entity_descriptions(self):
        ctx = CommunityContext(
            community_id="c1",
            level=0,
            edges=[
                EdgeContext(
                    source=EntityContext(id="e1", name="Alice", description="CEO", labels=["Person"], degree=5),
                    target=EntityContext(id="e2", name="Acme", description="Tech company", labels=["Organization"], degree=3),
                    relationship_type="WORKS_AT",
                    combined_degree=8,
                ),
            ],
        )
        text = render_community_context(ctx)

        assert "[Person] Alice: CEO" in text
        assert "[Organization] Acme: Tech company" in text
        assert "Alice --[WORKS_AT]--> Acme" in text

    def test_does_not_repeat_entity_descriptions(self):
        ctx = CommunityContext(
            community_id="c1",
            level=0,
            edges=[
                EdgeContext(
                    source=EntityContext(id="e1", name="Alice", description="CEO", labels=["Person"], degree=5),
                    target=EntityContext(id="e2", name="Acme", description="Tech", labels=["Organization"], degree=3),
                    relationship_type="WORKS_AT",
                    combined_degree=8,
                ),
                EdgeContext(
                    source=EntityContext(id="e1", name="Alice", description="CEO", labels=["Person"], degree=5),
                    target=EntityContext(id="e3", name="Bob", description="Engineer", labels=["Person"], degree=2),
                    relationship_type="FRIEND_OF",
                    combined_degree=7,
                ),
            ],
        )
        text = render_community_context(ctx)

        # Alice appears only once with full description
        assert text.count("[Person] Alice: CEO") == 1
        # But Alice is referenced in both relationships
        assert text.count("Alice --[") == 2

    def test_renders_isolated_entities(self):
        ctx = CommunityContext(
            community_id="c1",
            level=0,
            entities=[
                EntityContext(id="e4", name="Charlie", description="Intern", labels=["Person"], degree=0),
            ],
        )
        text = render_community_context(ctx)

        assert "[Person] Charlie: Intern" in text

    def test_empty_context(self):
        ctx = CommunityContext(community_id="c1", level=0)
        assert render_community_context(ctx) == ""

    def test_full_parse_and_render_roundtrip(self):
        rows = _make_rows()
        ctx = parse_community_context("c1", 0, rows)
        text = render_community_context(ctx)

        assert "Alice" in text
        assert "Acme" in text
        assert "Bob" in text
        assert "Charlie" in text
        assert "WORKS_AT" in text
        assert "FRIEND_OF" in text
