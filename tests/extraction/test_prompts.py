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
