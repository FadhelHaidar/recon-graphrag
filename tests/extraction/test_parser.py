"""Tests for graph extraction JSON parser."""

import json

import pytest

from recon_graphrag.extraction.parser import ClaimParser, GraphExtractionParser


def test_parse_valid_json():
    parser = GraphExtractionParser()
    data = {
        "nodes": [
            {"id": "p1", "label": "Person", "properties": {"name": "Alice"}}
        ],
        "relationships": [
            {
                "source_id": "p1",
                "target_id": "p2",
                "type": "KNOWS",
                "properties": {},
            }
        ],
    }
    result = parser.parse(json.dumps(data))
    assert len(result.nodes) == 1
    assert result.nodes[0].id == "p1"
    assert result.nodes[0].label == "Person"
    assert len(result.relationships) == 1
    assert result.relationships[0].type == "KNOWS"


def test_parse_markdown_wrapped_json():
    parser = GraphExtractionParser()
    data = {
        "nodes": [{"id": "p1", "label": "Person", "properties": {}}],
        "relationships": [],
    }
    wrapped = f"```json\n{json.dumps(data)}\n```"
    result = parser.parse(wrapped)
    assert len(result.nodes) == 1


def test_parse_malformed_json_raises():
    parser = GraphExtractionParser()
    with pytest.raises(json.JSONDecodeError):
        parser.parse("not json at all")


def test_parse_missing_arrays():
    parser = GraphExtractionParser()
    result = parser.parse('{"nodes": null, "relationships": null}')
    assert result.nodes == []
    assert result.relationships == []


def test_parse_skips_nodes_without_id_or_label():
    parser = GraphExtractionParser()
    data = {
        "nodes": [
            {"id": "p1", "label": "Person"},
            {"id": "", "label": "Person"},
            {"label": "Person"},
            {"id": "p2"},
        ],
        "relationships": [],
    }
    result = parser.parse(json.dumps(data))
    assert len(result.nodes) == 1
    assert result.nodes[0].id == "p1"


def test_parse_skips_relationships_without_required_fields():
    parser = GraphExtractionParser()
    data = {
        "nodes": [
            {"id": "p1", "label": "Person"},
            {"id": "p2", "label": "Person"},
        ],
        "relationships": [
            {"source_id": "p1", "target_id": "p2", "type": "KNOWS"},
            {"source_id": "p1", "type": "KNOWS"},
            {"target_id": "p2", "type": "KNOWS"},
            {"source_id": "p1", "target_id": "p2"},
        ],
    }
    result = parser.parse(json.dumps(data))
    assert len(result.relationships) == 1


# ---------------------------------------------------------------------------
# ClaimParser tests
# ---------------------------------------------------------------------------


class TestClaimParser:
    def test_parse_valid_claims(self):
        parser = ClaimParser()
        claims_data = [
            {
                "subject_entity_id": "person:alice",
                "claim_type": "role",
                "description": "Alice is the CEO of Acme Corp.",
                "status": "active",
            },
            {
                "subject_entity_id": "person:alice",
                "claim_type": "opinion",
                "description": "Alice supports the new policy.",
                "status": "active",
                "start_date": "2024-01-01",
            },
        ]
        result = parser.parse(
            json.dumps(claims_data),
            valid_entity_ids={"person:alice", "org:acme"},
        )
        assert len(result) == 2
        assert result[0].subject_entity_id == "person:alice"
        assert result[0].claim_type == "role"
        assert result[1].start_date == "2024-01-01"

    def test_parse_claims_from_wrapped_json_object(self):
        parser = ClaimParser()
        data = {
            "claims": [
                {
                    "subject_entity_id": "person:bob",
                    "claim_type": "status",
                    "description": "Bob was promoted.",
                }
            ]
        }
        result = parser.parse(json.dumps(data), valid_entity_ids={"person:bob"})
        assert len(result) == 1
        assert result[0].claim_type == "status"

    def test_parse_skips_claims_with_unknown_entity(self):
        parser = ClaimParser()
        claims_data = [
            {
                "subject_entity_id": "person:unknown",
                "claim_type": "role",
                "description": "Unknown person did something.",
            },
            {
                "subject_entity_id": "person:alice",
                "claim_type": "role",
                "description": "Alice is the CEO.",
            },
        ]
        result = parser.parse(
            json.dumps(claims_data),
            valid_entity_ids={"person:alice"},
        )
        assert len(result) == 1
        assert result[0].subject_entity_id == "person:alice"

    def test_parse_skips_claims_missing_required_fields(self):
        parser = ClaimParser()
        claims_data = [
            {"subject_entity_id": "person:alice", "description": "Valid claim."},
            {"subject_entity_id": "", "description": "Missing subject."},
            {"subject_entity_id": "person:alice", "description": ""},
            {"claim_type": "role", "description": "Missing subject entirely."},
        ]
        result = parser.parse(
            json.dumps(claims_data),
            valid_entity_ids={"person:alice"},
        )
        assert len(result) == 1
        assert result[0].description == "Valid claim."

    def test_parse_defaults_claim_type_to_general(self):
        parser = ClaimParser()
        claims_data = [
            {
                "subject_entity_id": "person:alice",
                "description": "Something about Alice.",
            }
        ]
        result = parser.parse(
            json.dumps(claims_data),
            valid_entity_ids={"person:alice"},
        )
        assert result[0].claim_type == "general"

    def test_parse_defaults_status_to_active(self):
        parser = ClaimParser()
        claims_data = [
            {
                "subject_entity_id": "person:alice",
                "claim_type": "role",
                "description": "Alice is the CEO.",
            }
        ]
        result = parser.parse(
            json.dumps(claims_data),
            valid_entity_ids={"person:alice"},
        )
        assert result[0].status == "active"

    def test_parse_empty_array_returns_empty(self):
        parser = ClaimParser()
        result = parser.parse("[]", valid_entity_ids={"person:alice"})
        assert result == []

    def test_parse_no_entity_filter_accepts_all(self):
        parser = ClaimParser()
        claims_data = [
            {
                "subject_entity_id": "person:anyone",
                "claim_type": "role",
                "description": "Anyone can be claimed.",
            }
        ]
        result = parser.parse(json.dumps(claims_data))
        assert len(result) == 1
