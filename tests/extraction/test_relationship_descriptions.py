"""Preserve unique relationship observations without changing strength semantics."""

import pytest

from recon_graphrag.extraction.assembler import GraphDocumentAssembler
from recon_graphrag.extraction.chunking import TextChunk
from recon_graphrag.extraction.types import (
    ExtractedNode,
    ExtractedRelationship,
    GraphExtraction,
)


def relationship_document(descriptions, *, document_id="document", graph_name="quality"):
    chunks = [TextChunk(id=f"{document_id}:{i}", text="source", index=i, metadata={}) for i in range(len(descriptions))]
    return GraphDocumentAssembler().assemble(
        document_id=document_id, text_hash="hash", chunks=chunks, metadata={}, graph_name=graph_name,
        chunk_extractions={chunk.id: GraphExtraction(nodes=[
            ExtractedNode("alice", "Person", {"canonical_key": "alice"}),
            ExtractedNode("company", "Company", {"canonical_key": "company"}),
        ], relationships=[ExtractedRelationship("alice", "company", "LEADS", {"description": description, "weight": i + 1})])
            for i, (chunk, description) in enumerate(zip(chunks, descriptions, strict=True))},
    )


@pytest.mark.parametrize("descriptions", [
    ["First fact", "Second fact"], ["", "Later fact"],
    ["Repeated", "Repeated", "Additional"], ["Long " * 600, "Last fact"],
])
def test_relationships_preserve_all_unique_descriptions(descriptions):
    relationship = relationship_document(descriptions).relationships[0]
    expected = "\n".join(dict.fromkeys(text.strip() for text in descriptions if text.strip()))
    assert relationship.properties["description"] == expected
    assert relationship.observation_count == len(descriptions)
    assert relationship.properties["weight"] == len(descriptions)
    assert relationship.strength == 1
