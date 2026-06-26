"""Tests for versioned artifact types and canonical rendering."""

from __future__ import annotations

import json

from recon_graphrag.models.artifacts import (
    ARTIFACT_PROMPT_VERSION,
    ARTIFACT_SCHEMA_VERSION,
    ArtifactVersion,
    Citation,
    ClaimRecord,
    CommunityFinding,
    CommunityReport,
    FindingReference,
    DescriptionObservation,
    DocumentSource,
    SourceReference,
    citation_excerpt,
    report_to_json,
    report_to_text,
    source_ref,
)
from recon_graphrag.models.types import SearchResult


class TestSourceReference:
    def test_frozen_fields(self):
        ref = SourceReference(
            document_id="doc:1",
            chunk_id="chunk:1",
            document_name="Test Doc",
            page_start=1,
            page_end=3,
            char_start=0,
            char_end=100,
        )
        assert ref.document_id == "doc:1"
        assert ref.chunk_id == "chunk:1"
        assert ref.document_name == "Test Doc"
        assert ref.page_start == 1
        assert ref.page_end == 3
        assert ref.char_start == 0
        assert ref.char_end == 100

    def test_defaults_are_none(self):
        ref = SourceReference(document_id="doc:1", chunk_id="chunk:1")
        assert ref.document_name is None
        assert ref.page_start is None
        assert ref.page_end is None
        assert ref.char_start is None
        assert ref.char_end is None


class TestCitation:
    def test_frozen_fields(self):
        c = Citation(
            document_id="doc:1",
            chunk_id="chunk:1",
            document_name="Test",
            excerpt="hello",
        )
        assert c.document_id == "doc:1"
        assert c.excerpt == "hello"

    def test_defaults_are_none(self):
        c = Citation(document_id="doc:1", chunk_id="chunk:1")
        assert c.document_name is None
        assert c.page_start is None
        assert c.page_end is None
        assert c.excerpt is None
        assert c.metadata == {}

    def test_metadata_passthrough(self):
        c = Citation(
            document_id="doc:1",
            chunk_id="chunk:1",
            metadata={"record_id": "row-42", "kind": "support-ticket"},
        )
        assert c.metadata["record_id"] == "row-42"


class TestDocumentSource:
    def test_construction(self):
        ds = DocumentSource(
            document_id="doc:1",
            document_name="Test",
            chunk_list=[
                Citation(document_id="doc:1", chunk_id="c1"),
                Citation(document_id="doc:1", chunk_id="c2"),
            ],
        )
        assert len(ds.chunk_list) == 2

    def test_default_empty_chunk_list(self):
        ds = DocumentSource(document_id="doc:1", document_name="Test")
        assert ds.chunk_list == []


class TestDescriptionObservation:
    def test_construction(self):
        source = SourceReference(document_id="doc:1", chunk_id="c1")
        obs = DescriptionObservation(
            entity_id="e1",
            description="Alice works at Acme",
            source=source,
        )
        assert obs.entity_id == "e1"
        assert obs.source.document_id == "doc:1"



class TestClaimRecord:
    def test_construction(self):
        source = SourceReference(document_id="doc:1", chunk_id="c1")
        claim = ClaimRecord(
            id="claim:1",
            entity_id="e1",
            claim_type="founded",
            description="Alice founded Acme in 2020",
            source=source,
        )
        assert claim.status == "active"
        assert claim.id == "claim:1"


class TestCommunityFinding:
    def test_construction(self):
        f = CommunityFinding(id="f1", description="Alice leads the team", rank=0.8)
        assert f.rank == 0.8

    def test_default_rank(self):
        f = CommunityFinding(id="f1", description="test")
        assert f.rank == 0.0


class TestArtifactVersion:
    def test_defaults(self):
        v = ArtifactVersion()
        assert v.schema_version == ARTIFACT_SCHEMA_VERSION
        assert v.prompt_version == ARTIFACT_PROMPT_VERSION
        assert v.input_fingerprint is None
        assert v.created_at is None


class TestCommunityReport:
    def test_construction(self):
        findings = [
            CommunityFinding(id="f1", description="finding 1", rank=0.9),
            CommunityFinding(id="f2", description="finding 2", rank=0.5),
        ]
        report = CommunityReport(
            id="r1",
            community_id="c1",
            level=0,
            findings=findings,
            summary="Community summary",
        )
        assert len(report.findings) == 2
        assert report.version.schema_version == ARTIFACT_SCHEMA_VERSION

    def test_default_empty_findings(self):
        report = CommunityReport(id="r1", community_id="c1", level=0)
        assert report.findings == []
        assert report.summary == ""



class TestCanonicalRendering:
    def test_source_ref(self):
        ref = SourceReference(document_id="doc:1", chunk_id="c1")
        assert source_ref(ref) == "[Source:doc:1:c1]"

    def test_report_to_json_sorted_keys(self):
        report = CommunityReport(
            id="r1",
            community_id="c1",
            level=0,
            findings=[CommunityFinding(id="f1", description="d", rank=0.5)],
            summary="summary",
        )
        text = report_to_json(report)
        data = json.loads(text)
        assert list(data.keys()) == sorted(data.keys())
        assert data["id"] == "r1"
        assert data["findings"][0]["id"] == "f1"

    def test_report_to_text_includes_findings_and_summary(self):
        report = CommunityReport(
            id="r1",
            community_id="c1",
            level=2,
            findings=[
                CommunityFinding(id="f2", description="second", rank=0.5),
                CommunityFinding(id="f1", description="first", rank=0.9),
            ],
            summary="Overall summary.",
        )
        text = report_to_text(report)
        assert "Community c1 (level 2)" in text
        # Higher rank comes first
        lines = text.split("\n")
        first_idx = next(i for i, l in enumerate(lines) if "first" in l)
        second_idx = next(i for i, l in enumerate(lines) if "second" in l)
        assert first_idx < second_idx
        assert "Overall summary." in text

    def test_report_to_text_includes_finding_references(self):
        report = CommunityReport(
            id="r1",
            community_id="c1",
            level=0,
            findings=[
                CommunityFinding(
                    id="f1",
                    description="Alice leads Acme.",
                    references=[
                        FindingReference(
                            target_id="person:alice",
                            target_type="entity",
                        ),
                        FindingReference(
                            target_id="claim:1",
                            target_type="claim",
                        ),
                    ],
                )
            ],
        )

        text = report_to_text(report)

        assert "entity:person:alice" in text
        assert "claim:claim:1" in text

    def test_report_to_text_empty_findings(self):
        report = CommunityReport(id="r1", community_id="c1", level=0, summary="just summary")
        text = report_to_text(report)
        assert "Community c1 (level 0)" in text
        assert "just summary" in text

    def test_citation_excerpt_short_text(self):
        assert citation_excerpt("hello", max_chars=100) == "hello"

    def test_citation_excerpt_long_text(self):
        text = "a" * 300
        result = citation_excerpt(text, max_chars=200)
        assert len(result) == 203  # 200 + "..."
        assert result.endswith("...")


class TestSearchResultCitations:
    def test_default_citations_empty(self):
        result = SearchResult(query="q", mode="global", answer="a")
        assert result.citations == []
        assert result.sources == []

    def test_sources_groups_by_document(self):
        result = SearchResult(
            query="q",
            mode="global",
            answer="a",
            citations=[
                Citation(
                    document_id="doc:1",
                    chunk_id="c1",
                    document_name="Doc One",
                    excerpt="hello",
                ),
                Citation(
                    document_id="doc:1",
                    chunk_id="c2",
                    document_name="Doc One",
                    excerpt="world",
                ),
                Citation(
                    document_id="doc:2",
                    chunk_id="c3",
                    document_name="Doc Two",
                    excerpt="other",
                ),
            ],
        )
        sources = result.sources
        assert len(sources) == 2
        doc1 = next(s for s in sources if s.document_id == "doc:1")
        assert len(doc1.chunk_list) == 2
        assert doc1.document_name == "Doc One"

    def test_str_unchanged(self):
        result = SearchResult(query="q", mode="global", answer="answer text")
        assert str(result) == "[GLOBAL] answer text"


class TestGraphDocumentClaims:
    def test_default_claims_empty(self):
        from recon_graphrag.extraction.types import (
            ChunkRecord,
            DocumentRecord,
            EntityRecord,
            GraphDocument,
        )

        doc = GraphDocument(
            document=DocumentRecord(id="d1", text_hash="h"),
            chunks=[ChunkRecord(id="c1", document_id="d1", text="t", index=0)],
            entities=[EntityRecord(id="e1", type="Person")],
            relationships=[],
            evidence_links=[],
        )
        assert doc.claims == []

    def test_with_claims(self):
        from recon_graphrag.extraction.types import (
            ChunkRecord,
            DocumentRecord,
            EntityRecord,
            GraphDocument,
        )

        source = SourceReference(document_id="d1", chunk_id="c1")
        doc = GraphDocument(
            document=DocumentRecord(id="d1", text_hash="h"),
            chunks=[ChunkRecord(id="c1", document_id="d1", text="t", index=0)],
            entities=[EntityRecord(id="e1", type="Person")],
            relationships=[],
            evidence_links=[],
            claims=[
                ClaimRecord(
                    id="claim:1",
                    entity_id="e1",
                    claim_type="founded",
                    description="founded Acme",
                    source=source,
                )
            ],
        )
        assert len(doc.claims) == 1
        assert doc.claims[0].id == "claim:1"


class TestVersionConstants:
    def test_schema_version_is_semver_like(self):
        parts = ARTIFACT_SCHEMA_VERSION.split(".")
        assert len(parts) >= 2
        assert all(p.isdigit() for p in parts)

    def test_prompt_version_string(self):
        assert isinstance(ARTIFACT_PROMPT_VERSION, str)
