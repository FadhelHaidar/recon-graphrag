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


@pytest.mark.parametrize(
    "payload",
    [
        '{"error": "temporary failure"}',
        '{"error": {"code": 503, "message": "private value"}}',
        '{"detail": "private value"}',
        "{}",
    ],
)
def test_parse_rejects_object_without_extraction_envelope(payload):
    with pytest.raises(ValueError) as error:
        GraphExtractionParser().parse(payload)
    assert str(error.value) == (
        "extraction: expected an object with nodes or relationships"
    )


@pytest.mark.parametrize("payload", ["[]", '"private value"', "null", "42"])
def test_parse_rejects_non_object_json(payload):
    with pytest.raises(ValueError) as error:
        GraphExtractionParser().parse(payload)
    assert str(error.value) == (
        "extraction: expected an object with nodes or relationships"
    )


@pytest.mark.parametrize("kind", ["nodes", "relationships"])
@pytest.mark.parametrize("container", [{}, "private value", 0, False, 1.5])
def test_parse_rejects_non_array_containers(kind, container):
    with pytest.raises(ValueError) as error:
        GraphExtractionParser().parse(json.dumps({kind: container}))
    assert str(error.value) == f"{kind}: expected an array or null"


@pytest.mark.parametrize(
    "payload, node_count, relationship_count",
    [
        ('{"nodes": []}', 0, 0),
        ('{"relationships": []}', 0, 0),
        ('{"nodes": null}', 0, 0),
        ('{"relationships": null}', 0, 0),
        ('{"nodes": [{"id": "n", "label": "Thing"}]}', 1, 0),
        (
            '{"relationships": [{"source_id": "a", "target_id": "b", '
            '"type": "LINK"}]}',
            0,
            1,
        ),
    ],
)
def test_parse_accepts_supported_envelope_forms(
    payload, node_count, relationship_count
):
    result = GraphExtractionParser().parse(payload)
    assert len(result.nodes) == node_count
    assert len(result.relationships) == relationship_count


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


@pytest.mark.parametrize("kind", ["nodes", "relationships"])
@pytest.mark.parametrize("properties", [[], [["name", "private value"]], "private value", "", 0, False])
def test_parse_rejects_nonobject_properties_with_safe_field_path(kind, properties):
    item = ({"id": "private ID", "label": "Thing"} if kind == "nodes" else
            {"source_id": "private ID", "target_id": "private ID", "type": "LINK"})
    item["properties"] = properties
    with pytest.raises(ValueError) as error:
        GraphExtractionParser().parse(json.dumps({kind: [item]}))
    assert str(error.value) == f"{kind}[0].properties: expected an object"


@pytest.mark.parametrize("kind", ["nodes", "relationships"])
@pytest.mark.parametrize("properties", [None, {}])
def test_parse_preserves_null_and_empty_properties_as_empty_object(kind, properties):
    item = ({"id": "n", "label": "Thing"} if kind == "nodes" else
            {"source_id": "n", "target_id": "n", "type": "LINK"})
    item["properties"] = properties
    result = GraphExtractionParser().parse(json.dumps({kind: [item]}))
    assert getattr(result, kind)[0].properties == {}


def test_parse_leaves_nested_values_for_schema_filtering_and_validation():
    properties = {"extra": {"nested": [["private value"]]}}
    result = GraphExtractionParser().parse(json.dumps({"nodes": [
        {"id": "n", "label": "Thing", "properties": properties},
    ]}))
    assert result.nodes[0].properties == properties


def test_parse_property_error_retains_original_index_after_skipped_nodes():
    with pytest.raises(ValueError) as error:
        GraphExtractionParser().parse(json.dumps({"nodes": [
            {"label": "Thing"},
            {"id": "private ID", "label": "Thing", "properties": "private value"},
        ]}))
    assert str(error.value) == "nodes[1].properties: expected an object"


def test_parser_preserves_default_explicit_key_trust():
    result = GraphExtractionParser().parse(json.dumps({"nodes": [{
        "id": "a", "label": "Thing", "canonical_key_trusted": False,
        "properties": {"canonical_key": "Thing:A"},
    }]}))
    # Trust metadata belongs to the SDK, not an LLM-controlled output field.
    assert result.nodes[0].canonical_key_trusted is True
