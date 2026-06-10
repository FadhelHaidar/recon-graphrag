"""Tests for graph extraction JSON parser."""

import json

import pytest

from recon_graphrag.extraction.parser import GraphExtractionParser
from recon_graphrag.extraction.types import ExtractedNode, ExtractedRelationship


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
