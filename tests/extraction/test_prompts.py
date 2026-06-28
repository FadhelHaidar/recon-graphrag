"""Tests for schema prompt builder."""

from recon_graphrag.extraction.prompts import SchemaPromptBuilder
from recon_graphrag.extraction.schema import (
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
)


def test_prompt_includes_labels_and_patterns():
    schema = GraphSchema(
        node_types=[
            NodeType(
                label="Person",
                description="A human",
                properties=[PropertyType(name="name", type="STRING")],
            ),
            NodeType(
                label="Movie",
                description="A film",
                properties=[PropertyType(name="title", type="STRING")],
                identity_property="title",
            ),
        ],
        relationship_types=[
            RelationshipType(label="DIRECTED", description="Directed a movie")
        ],
        patterns=[("Person", "DIRECTED", "Movie")],
    )

    builder = SchemaPromptBuilder()
    prompt = builder.build_prompt("Christopher Nolan directed Inception.", schema)

    assert "Person" in prompt
    assert "Movie" in prompt
    assert "DIRECTED" in prompt
    assert "Identity property: title" in prompt
    assert "Person -[DIRECTED]-> Movie" in prompt
    assert "Christopher Nolan directed Inception." in prompt
    assert "Return valid JSON only" in prompt
    assert 'numeric "weight" property' in prompt
    assert '"weight": 1.0' in prompt


def test_claim_prompt_includes_entity_ids():
    entity_ids = ["person:alice", "org:acme"]
    prompt = SchemaPromptBuilder.build_claim_prompt(
        text="Alice runs Acme Corp.",
        entity_ids=entity_ids,
    )
    assert "person:alice" in prompt
    assert "org:acme" in prompt
    assert "Alice runs Acme Corp." in prompt
    assert "subject_entity_id" in prompt
    assert "claim_type" in prompt
    assert "valid json only" in prompt.lower()


def test_claim_prompt_lists_all_entity_ids():
    entity_ids = [f"person:p{i}" for i in range(5)]
    prompt = SchemaPromptBuilder.build_claim_prompt(text="test", entity_ids=entity_ids)
    for eid in entity_ids:
        assert eid in prompt
