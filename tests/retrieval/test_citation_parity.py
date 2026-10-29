"""Parity tests for reference citation resolution.

Batching the chunk metadata lookup must not change which citations come back or
the order they come in, and it must not repeat work for duplicate references.
"""

from __future__ import annotations

from typing import Any, cast

from recon_graphrag.retrieval.citations import (
    resolve_claim_citations,
    resolve_entity_citations,
    resolve_reference_citations,
    resolve_relationship_citations,
)


class RecordingStore:
    """Graph store fake that records selections and metadata materializations."""

    def __init__(self, chunks_by_ref: dict[str, list[str]]):
        self.chunks_by_ref = chunks_by_ref
        self.selections: list[tuple[str, Any]] = []
        self.materializations: list[list[str]] = []

    def execute_query(self, query: str, params: dict | None = None) -> list[dict]:
        params = params or {}
        if "entity_ids" in params:
            kind, refs = "entity", params["entity_ids"]
        elif "claim_ids" in params:
            kind, refs = "claim", params["claim_ids"]
        elif "relationship_keys" in params:
            kind, refs = "relationship", params["relationship_keys"]
        else:  # pragma: no cover - guards the fake against schema drift
            raise AssertionError(f"unexpected selection query: {query[:60]}")
        self.selections.append((kind, list(refs)))
        return [
            {"chunk_id": chunk_id}
            for ref in refs
            for chunk_id in self.chunks_by_ref.get(ref, [])
        ]

    def resolve_chunk_citations(self, graph_name: str, chunk_ids: list[str]) -> list[dict]:
        self.materializations.append(list(chunk_ids))
        return [
            {
                "chunk_id": chunk_id,
                "document_id": f"doc:{chunk_id}",
                "document_name": f"name:{chunk_id}",
                "page_start": 1,
                "page_end": 2,
                "excerpt": f"text of {chunk_id}",
                "chunk_metadata": {},
                "document_metadata": {},
            }
            for chunk_id in chunk_ids
        ]


class StorageOrderStore(RecordingStore):
    """Materializes metadata in a fixed storage order, ignoring input order."""

    STORAGE = ["c-rel", "c-claim", "c-bob", "c-alice", "c-shared"]

    def resolve_chunk_citations(self, graph_name: str, chunk_ids: list[str]) -> list[dict]:
        wanted = set(chunk_ids)
        return super().resolve_chunk_citations(
            graph_name, [chunk_id for chunk_id in self.STORAGE if chunk_id in wanted]
        )


CHUNKS = {
    "person:alice": ["c-shared", "c-alice"],
    "person:bob": ["c-bob"],
    "claim:1": ["c-shared", "c-claim"],
    "person:alice:KNOWS:person:bob": ["c-rel", "c-shared"],
}

REFS = [
    {"target_id": "person:alice", "target_type": "entity"},
    {"target_id": "person:alice", "target_type": "entity"},
    {"target_id": "person:bob", "target_type": "entity"},
    {"target_id": "claim:1", "target_type": "claim"},
    {"target_id": "person:alice:KNOWS:person:bob", "target_type": "relationship"},
]


def _dedupe(citations: list) -> list:
    seen: set[tuple[str, str, int | None, int | None]] = set()
    deduped = []
    for citation in citations:
        key = (
            citation.document_id,
            citation.chunk_id,
            citation.page_start,
            citation.page_end,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(citation)
    return deduped


def test_duplicate_references_select_and_materialize_once():
    store = RecordingStore(CHUNKS)

    citations = resolve_reference_citations(cast(Any, store), "graph", REFS)

    assert store.selections == [
        ("entity", ["person:alice", "person:bob"]),
        ("claim", ["claim:1"]),
        ("relationship", ["person:alice:KNOWS:person:bob"]),
    ]
    assert store.materializations == [
        ["c-shared", "c-alice", "c-bob", "c-claim", "c-rel"]
    ]
    assert [c.chunk_id for c in citations] == [
        "c-shared",
        "c-alice",
        "c-bob",
        "c-claim",
        "c-rel",
    ]


def test_batched_resolution_matches_per_type_resolution():
    refs = [ref for ref in REFS if ref["target_id"] != "person:alice" or ref["target_type"] != "entity"]
    entity_ids = [r["target_id"] for r in refs if r["target_type"] == "entity"]
    claim_ids = [r["target_id"] for r in refs if r["target_type"] == "claim"]
    relationship_keys = [r["target_id"] for r in refs if r["target_type"] == "relationship"]

    batched = resolve_reference_citations(cast(Any, RecordingStore(CHUNKS)), "graph", refs)

    per_type_store = RecordingStore(CHUNKS)
    per_type = _dedupe(
        resolve_entity_citations(cast(Any, per_type_store), entity_ids, "graph")
        + resolve_claim_citations(cast(Any, per_type_store), "graph", claim_ids)
        + resolve_relationship_citations(cast(Any, per_type_store), "graph", relationship_keys)
    )

    assert batched == per_type
    assert len(per_type_store.materializations) == 3


def test_storage_row_order_is_kept_inside_a_category():
    """Release returned metadata row order within a category; batching must too."""
    refs = [
        {"target_id": "person:alice", "target_type": "entity"},
        {"target_id": "person:bob", "target_type": "entity"},
        {"target_id": "claim:1", "target_type": "claim"},
        {"target_id": "person:alice:KNOWS:person:bob", "target_type": "relationship"},
    ]

    batched = resolve_reference_citations(cast(Any, StorageOrderStore(CHUNKS)), "graph", refs)

    per_type_store = StorageOrderStore(CHUNKS)
    per_type = _dedupe(
        resolve_entity_citations(cast(Any, per_type_store), ["person:alice", "person:bob"], "graph")
        + resolve_claim_citations(cast(Any, per_type_store), "graph", ["claim:1"])
        + resolve_relationship_citations(
            cast(Any, per_type_store), "graph", ["person:alice:KNOWS:person:bob"]
        )
    )

    assert batched == per_type
    # Storage order decides within a category; category precedence still wins.
    assert [c.chunk_id for c in batched] == [
        "c-bob",
        "c-alice",
        "c-shared",
        "c-claim",
        "c-rel",
    ]
