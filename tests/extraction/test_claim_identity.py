"""Generated assertion identity regressions."""

from dataclasses import replace

import pytest

from recon_graphrag.extraction.assembler import GraphDocumentAssembler
from recon_graphrag.extraction.chunking import TextChunk
from recon_graphrag.extraction.types import (
    ExtractedClaim,
    ExtractedNode,
    GraphExtraction,
)


def assemble_claims(claims, *, local_id="alice", graph_name="quality-claims"):
    return GraphDocumentAssembler().assemble(
        document_id="document", text_hash="hash",
        chunks=[TextChunk(id="chunk", text="source", index=0, metadata={})],
        chunk_extractions={"chunk": GraphExtraction(nodes=[
            ExtractedNode(id=local_id, label="Person", properties={"canonical_key": "person:alice"}),
            ExtractedNode(id="acme", label="Company", properties={"canonical_key": "company:acme"}),
            ExtractedNode(id="beta", label="Company", properties={"canonical_key": "company:beta"}),
        ])},
        chunk_claims={"chunk": claims}, metadata={}, graph_name=graph_name,
    )


@pytest.mark.parametrize("changes", [
    {"object_entity_id": "beta"}, {"start_date": "2024-01-01"},
    {"end_date": "2025-01-01"},
])
def test_distinct_assertions_do_not_share_generated_ids(changes):
    first = ExtractedClaim("alice", "role", "Alice leads the company.", object_entity_id="acme")
    document = assemble_claims([first, replace(first, **changes)])
    assert len({claim.id for claim in document.claims}) == 2


def test_generated_identity_uses_persistent_endpoints_not_model_local_names():
    first = ExtractedClaim("alice", "role", "Alice leads the company.", object_entity_id="acme")
    before = assemble_claims([first]).claims[0]
    after = assemble_claims([replace(first, subject_entity_id="person-one")], local_id="person-one").claims[0]
    assert before.id == after.id


def test_status_and_source_metadata_do_not_change_assertion_identity():
    first = ExtractedClaim("alice", "role", "Alice leads the company.", object_entity_id="acme")
    claims = assemble_claims([first, replace(first, status="resolved", source_text="another excerpt")]).claims
    assert claims[0].id == claims[1].id
    assert len(claims[0].id.rsplit(":", 1)[-1]) == 64
