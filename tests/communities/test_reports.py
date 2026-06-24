"""Tests for structured community report generation."""

from __future__ import annotations

import json

import pytest

from recon_graphrag.communities.context import (
    ClaimContext,
    CommunityContext,
    EdgeContext,
    EntityContext,
    build_reference_ids,
    enrich_context_with_claims,
)
from recon_graphrag.communities.reports import (
    ReportParser,
    ReportRubric,
    ReportValidationError,
    build_repair_prompt,
    build_report_prompt,
    extract_reference_ids,
)
from recon_graphrag.models.artifacts import (
    CommunityFinding,
    CommunityReport,
    FindingReference,
    report_to_json,
    report_to_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context() -> CommunityContext:
    alice = EntityContext(id="person:alice", name="Alice", description="CEO of Acme", labels=["Person"], degree=5)
    bob = EntityContext(id="person:bob", name="Bob", description="CTO of Acme", labels=["Person"], degree=3)
    acme = EntityContext(id="org:acme", name="Acme", description="Tech company", labels=["Organization"], degree=4)
    return CommunityContext(
        community_id="1",
        level=0,
        entities=[acme],
        edges=[
            EdgeContext(source=alice, target=acme, relationship_type="WORKS_AT", description="Alice works at Acme", combined_degree=9),
            EdgeContext(source=bob, target=acme, relationship_type="WORKS_AT", description="Bob works at Acme", combined_degree=7),
        ],
        claims=[
            ClaimContext(id="claim:1", entity_id="person:alice", claim_type="role", description="Alice is the CEO."),
        ],
    )


def _make_valid_report_json() -> str:
    return json.dumps({
        "title": "Acme Leadership",
        "summary": "Alice and Bob lead Acme Corp.",
        "rating": 8.0,
        "rating_explanation": "Key leadership community.",
        "findings": [
            {
                "description": "Alice leads Acme as CEO.",
                "references": [
                    {"target_id": "person:alice", "target_type": "entity"},
                    {"target_id": "claim:1", "target_type": "claim"},
                ],
            },
            {
                "description": "Bob is the CTO.",
                "references": [
                    {"target_id": "person:bob", "target_type": "entity"},
                ],
            },
        ],
    })


# ---------------------------------------------------------------------------
# Prompt builder tests
# ---------------------------------------------------------------------------


class TestBuildReportPrompt:
    def test_prompt_includes_context_and_ids(self):
        context = _make_context()
        ref_ids = build_reference_ids(context)
        prompt = build_report_prompt(
            community_id="1",
            level=0,
            context="some context text",
            reference_ids=ref_ids,
        )
        assert "person:alice" in prompt
        assert "person:bob" in prompt
        assert "org:acme" in prompt
        assert "claim:1" in prompt
        assert "some context text" in prompt
        assert "JSON" in prompt

    def test_prompt_uses_custom_rubric(self):
        rubric = ReportRubric(rating_name="severity", rating_description="How severe", min_rating=1, max_rating=5)
        prompt = build_report_prompt(
            community_id="1", level=0, context="ctx", reference_ids=["a"], rubric=rubric
        )
        assert "severity" in prompt
        assert "1" in prompt
        assert "5" in prompt


# ---------------------------------------------------------------------------
# Reference ID extraction tests
# ---------------------------------------------------------------------------


class TestExtractReferenceIds:
    def test_extracts_all_ids(self):
        ids = extract_reference_ids(
            entity_ids=["e1", "e2"],
            relationship_keys=["e1:REL:e2"],
            claim_ids=["claim:1"],
        )
        assert "e1" in ids
        assert "e2" in ids
        assert "e1:REL:e2" in ids
        assert "claim:1" in ids


class TestBuildReferenceIds:
    def test_from_context(self):
        context = _make_context()
        ids = build_reference_ids(context)
        assert "person:alice" in ids
        assert "person:bob" in ids
        assert "org:acme" in ids
        assert "claim:1" in ids
        # Relationship keys
        assert any("WORKS_AT" in rid for rid in ids)


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestReportParser:
    def test_parse_valid_report(self):
        parser = ReportParser()
        report = parser.parse(
            _make_valid_report_json(),
            community_id="1",
            level=0,
            valid_ids={"person:alice", "person:bob", "org:acme", "claim:1"},
        )
        assert report.community_id == "1"
        assert report.level == 0
        assert report.title == "Acme Leadership"
        assert report.rating == 8.0
        assert len(report.findings) == 2
        assert len(report.findings[0].references) == 2
        assert report.findings[0].references[0].target_type == "entity"

    def test_parse_strips_markdown_fences(self):
        parser = ReportParser()
        fenced = f"```json\n{_make_valid_report_json()}\n```"
        report = parser.parse(fenced, community_id="1", level=0)
        assert report.title == "Acme Leadership"

    def test_parse_rejects_missing_title(self):
        parser = ReportParser()
        data = {"summary": "test", "findings": [{"description": "f", "references": []}]}
        with pytest.raises(ReportValidationError) as exc:
            parser.parse(json.dumps(data), community_id="1", level=0)
        assert "title" in str(exc.value.errors)

    def test_parse_rejects_empty_findings(self):
        parser = ReportParser()
        data = {"title": "T", "summary": "S", "findings": []}
        with pytest.raises(ReportValidationError) as exc:
            parser.parse(json.dumps(data), community_id="1", level=0)
        assert "empty" in str(exc.value.errors).lower()

    def test_parse_filters_invalid_references(self):
        parser = ReportParser()
        data = {
            "title": "T",
            "summary": "S",
            "findings": [
                {
                    "description": "A finding.",
                    "references": [
                        {"target_id": "unknown:id", "target_type": "entity"},
                        {"target_id": "person:alice", "target_type": "entity"},
                    ],
                }
            ],
        }
        report = parser.parse(
            json.dumps(data),
            community_id="1",
            level=0,
            valid_ids={"person:alice"},
        )
        assert len(report.findings) == 1
        assert len(report.findings[0].references) == 1
        assert report.findings[0].references[0].target_id == "person:alice"

    def test_parse_skips_findings_with_no_valid_refs(self):
        parser = ReportParser()
        data = {
            "title": "T",
            "summary": "S",
            "findings": [
                {
                    "description": "Bad finding.",
                    "references": [{"target_id": "unknown", "target_type": "entity"}],
                },
                {
                    "description": "Good finding.",
                    "references": [{"target_id": "person:alice", "target_type": "entity"}],
                },
            ],
        }
        report = parser.parse(
            json.dumps(data),
            community_id="1",
            level=0,
            valid_ids={"person:alice"},
        )
        assert len(report.findings) == 1
        assert report.findings[0].description == "Good finding."

    def test_parse_no_validation_when_valid_ids_none(self):
        parser = ReportParser()
        data = {
            "title": "T",
            "summary": "S",
            "findings": [
                {
                    "description": "Any finding.",
                    "references": [{"target_id": "anything", "target_type": "entity"}],
                }
            ],
        }
        report = parser.parse(json.dumps(data), community_id="1", level=0)
        assert len(report.findings) == 1


# ---------------------------------------------------------------------------
# Repair tests
# ---------------------------------------------------------------------------


class TestRepair:
    def test_build_repair_prompt(self):
        prompt = build_repair_prompt(
            raw_content="bad json",
            errors=["Missing title.", "Empty findings."],
            valid_ids=["e1", "e2"],
        )
        assert "Missing title" in prompt
        assert "Empty findings" in prompt
        assert "e1" in prompt
        assert "bad json" in prompt


# ---------------------------------------------------------------------------
# Canonical text rendering tests
# ---------------------------------------------------------------------------


class TestReportRendering:
    def test_report_to_json_includes_new_fields(self):
        report = CommunityReport(
            id="r1",
            community_id="1",
            level=0,
            title="Test Title",
            summary="Test summary.",
            rating=7.5,
            rating_explanation="Because.",
            findings=[
                CommunityFinding(
                    id="f1",
                    description="A finding.",
                    references=[FindingReference(target_id="e1", target_type="entity")],
                ),
            ],
        )
        json_str = report_to_json(report)
        data = json.loads(json_str)
        assert data["title"] == "Test Title"
        assert data["rating"] == 7.5
        assert data["findings"][0]["references"][0]["target_id"] == "e1"

    def test_report_to_text_includes_title_and_rating(self):
        report = CommunityReport(
            id="r1",
            community_id="1",
            level=0,
            title="Test Title",
            summary="Summary text.",
            rating=5.0,
            findings=[],
        )
        text = report_to_text(report)
        assert "Test Title" in text
        assert "Rating: 5.0" in text
        assert "Summary text." in text

    def test_report_to_text_omits_rating_when_none(self):
        report = CommunityReport(
            id="r1",
            community_id="1",
            level=0,
            title="Title",
            summary="Summary.",
        )
        text = report_to_text(report)
        assert "Rating" not in text


# ---------------------------------------------------------------------------
# Context enrichment tests
# ---------------------------------------------------------------------------


class TestContextEnrichment:
    def test_enrich_with_claims(self):
        context = _make_context()
        assert len(context.claims) == 1

        new_claims = [
            {"claim_id": "claim:2", "entity_id": "person:bob", "claim_type": "role", "description": "Bob is CTO.", "status": "active"},
        ]
        enriched = enrich_context_with_claims(context, new_claims)
        assert len(enriched.claims) == 1  # new context, only new claims
        assert enriched.claims[0].id == "claim:2"

    def test_enrich_skips_invalid_rows(self):
        context = _make_context()
        bad_rows = [
            {"claim_id": "", "entity_id": "e1"},
            {"entity_id": "e1"},  # missing claim_id
        ]
        enriched = enrich_context_with_claims(context, bad_rows)
        assert len(enriched.claims) == 0
