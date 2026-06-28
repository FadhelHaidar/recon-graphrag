"""Tests for typed community context records, rendering, and packing."""

from __future__ import annotations

from recon_graphrag.communities.context import (
    CommunityContext,
    EdgeContext,
    EntityContext,
    pack_community_context,
    parse_community_context,
    render_community_context,
)
from recon_graphrag.utils.tokens import ApproximateTokenCounter


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


def _make_large_context(n_edges: int = 10) -> CommunityContext:
    """Create a context with n edges for packing tests."""
    edges = []
    for i in range(n_edges):
        edges.append(
            EdgeContext(
                source=EntityContext(id=f"e{i}", name=f"Entity{i}", description=f"Description {i}", labels=["Person"], degree=10 - i),
                target=EntityContext(id=f"e{i+10}", name=f"Target{i}", description=f"Target desc {i}", labels=["Org"], degree=5),
                relationship_type="RELATES_TO",
                combined_degree=15 - i,
            )
        )
    return CommunityContext(community_id="c1", level=0, edges=edges)


class TestPackCommunityContext:
    def test_all_fit_in_budget(self):
        ctx = _make_large_context(3)
        counter = ApproximateTokenCounter(ratio=4.0)
        full_text = render_community_context(ctx)
        full_tokens = counter.count(full_text)

        packed = pack_community_context(ctx, max_tokens=full_tokens + 100, counter=counter)

        assert packed.included_edges == 3
        assert packed.excluded_edges == 0
        assert packed.truncated is False
        assert packed.used_tokens > 0

    def test_excludes_when_over_budget(self):
        ctx = _make_large_context(10)
        counter = ApproximateTokenCounter(ratio=4.0)

        # Very small budget — only first few edges fit
        packed = pack_community_context(ctx, max_tokens=50, counter=counter)

        assert packed.included_edges < 10
        assert packed.excluded_edges > 0
        assert packed.truncated is True

    def test_ranked_order_preserved(self):
        ctx = _make_large_context(5)
        counter = ApproximateTokenCounter(ratio=4.0)

        # Render full context, then measure the first edge block precisely
        full_text = render_community_context(ctx)
        lines = full_text.split("\n")
        # Each edge produces 3 lines (source, target, relationship)
        first_edge_text = "\n".join(lines[:3])
        budget = counter.count(first_edge_text)

        packed = pack_community_context(ctx, max_tokens=budget, counter=counter)

        # First edge (highest degree) should be included
        assert "Entity0" in packed.text
        assert "RELATES_TO" in packed.text
        assert packed.included_edges == 1

    def test_zero_budget_returns_empty(self):
        ctx = _make_large_context(3)
        packed = pack_community_context(ctx, max_tokens=0)

        assert packed.text == ""
        assert packed.used_tokens == 0
        assert packed.included_edges == 0

    def test_entity_deduplication_saves_tokens(self):
        """Same entity in multiple edges only includes description once."""
        long_desc = "Alice is a very important person with a long description."
        ctx = CommunityContext(
            community_id="c1",
            level=0,
            edges=[
                EdgeContext(
                    source=EntityContext(id="e1", name="Alice", description=long_desc, labels=["Person"], degree=5),
                    target=EntityContext(id="e2", name="Acme", description="Short", labels=["Org"], degree=3),
                    relationship_type="WORKS_AT",
                    combined_degree=8,
                ),
                EdgeContext(
                    source=EntityContext(id="e1", name="Alice", description=long_desc, labels=["Person"], degree=5),
                    target=EntityContext(id="e3", name="Bob", description="Short", labels=["Person"], degree=2),
                    relationship_type="FRIEND_OF",
                    combined_degree=7,
                ),
            ],
        )
        counter = ApproximateTokenCounter(ratio=4.0)
        packed = pack_community_context(ctx, max_tokens=200, counter=counter)

        # Alice's description line appears only once (not repeated for second edge)
        assert packed.text.count("[Person] Alice:") == 1
        # But both relationships reference Alice
        assert packed.text.count("Alice --[") == 2

    def test_isolated_entities_appended(self):
        ctx = CommunityContext(
            community_id="c1",
            level=0,
            edges=[
                EdgeContext(
                    source=EntityContext(id="e1", name="Alice", description="CEO", labels=["Person"], degree=5),
                    target=EntityContext(id="e2", name="Acme", description="Tech", labels=["Org"], degree=3),
                    relationship_type="WORKS_AT",
                    combined_degree=8,
                ),
            ],
            entities=[
                EntityContext(id="e3", name="Charlie", description="Intern", labels=["Person"], degree=0),
            ],
        )
        counter = ApproximateTokenCounter(ratio=4.0)
        full_text = render_community_context(ctx)
        full_tokens = counter.count(full_text)

        packed = pack_community_context(ctx, max_tokens=full_tokens + 100, counter=counter)

        assert "Charlie" in packed.text
        assert packed.included_entities >= 1

    def test_packed_token_count_accurate(self):
        ctx = _make_large_context(5)
        counter = ApproximateTokenCounter(ratio=4.0)

        packed = pack_community_context(ctx, max_tokens=500, counter=counter)

        # Recount from actual text — allow ±1 token rounding difference
        actual_tokens = counter.count(packed.text)
        assert abs(packed.used_tokens - actual_tokens) <= 1
