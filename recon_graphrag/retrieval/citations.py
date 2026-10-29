"""Citation resolution for search results.

Resolves entity/claim/chunk evidence into user-facing Citation objects
with document identity, page ranges, and bounded excerpts.
"""

from __future__ import annotations

import json

from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.models.artifacts import Citation, citation_excerpt

_INTERNAL_METADATA_KEYS = {
    "embedding",
    "graph_name",
    "id",
    "text",
    "text_hash",
    "created",
    "updated",
}


def resolve_entity_citations(
    graph_store: GraphStore,
    entity_ids: list[str],
    graph_name: str,
) -> list[Citation]:
    """Resolve entity IDs to source chunk citations.

    Looks up evidence links (FROM_CHUNK) for each entity, then resolves
    chunk metadata (document, page, excerpt).

    Args:
        graph_store: Graph store backend.
        entity_ids: Entity IDs to resolve.
        graph_name: Graph scope.

    Returns:
        Deduplicated list of Citation objects.
    """
    return resolve_chunk_citations(
        graph_store,
        graph_name,
        _entity_chunk_ids(graph_store, graph_name, entity_ids),
    )


def _entity_chunk_ids(
    graph_store: GraphStore,
    graph_name: str,
    entity_ids: list[str],
) -> list[str]:
    """Entity references to source chunk ids."""
    if not entity_ids:
        return []

    rows = graph_store.execute_query(
        """
        UNWIND $entity_ids AS eid
        MATCH (c:Chunk {graph_name: $graph_name})-[:FROM_CHUNK]->
              (e:__Entity__ {graph_name: $graph_name})
        WHERE e.id = eid OR e.canonical_key = eid OR e.human_readable_id = eid
        RETURN DISTINCT c.id AS chunk_id
        """,
        {"entity_ids": entity_ids, "graph_name": graph_name},
    )
    return [r["chunk_id"] for r in rows if r.get("chunk_id")]


def resolve_chunk_citations(
    graph_store: GraphStore,
    graph_name: str,
    chunk_ids: list[str],
) -> list[Citation]:
    """Resolve chunk IDs to Citation objects.

    Args:
        graph_store: Graph store backend.
        chunk_ids: Chunk IDs to resolve.

    Returns:
        Deduplicated list of Citation objects.
    """
    if not chunk_ids:
        return []

    rows = graph_store.resolve_chunk_citations(graph_name, chunk_ids)
    seen: set[str] = set()
    citations: list[Citation] = []

    for row in rows:
        cid = row.get("chunk_id", "")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        citations.append(
            Citation(
                document_id=row.get("document_id", ""),
                chunk_id=cid,
                document_name=row.get("document_name"),
                page_start=row.get("page_start"),
                page_end=row.get("page_end"),
                excerpt=(
                    citation_excerpt(row["excerpt"]) if row.get("excerpt") else None
                ),
                metadata=_citation_metadata(row),
            )
        )

    return citations


def resolve_claim_citations(
    graph_store: GraphStore,
    graph_name: str,
    claim_ids: list[str],
) -> list[Citation]:
    """Resolve claim IDs to source chunk citations via SOURCED_FROM edges.

    Args:
        graph_store: Graph store backend.
        claim_ids: Claim IDs to resolve.

    Returns:
        Deduplicated list of Citation objects.
    """
    return resolve_chunk_citations(
        graph_store,
        graph_name,
        _claim_chunk_ids(graph_store, graph_name, claim_ids),
    )


def _claim_chunk_ids(
    graph_store: GraphStore,
    graph_name: str,
    claim_ids: list[str],
) -> list[str]:
    """Claim ids to source chunk ids."""
    if not claim_ids:
        return []

    query = """
    UNWIND $claim_ids AS cid
    MATCH (c:Claim {id: cid, graph_name: $graph_name})-[:SOURCED_FROM]->
          (ch:Chunk {graph_name: $graph_name})
    RETURN DISTINCT ch.id AS chunk_id
    """
    rows = graph_store.execute_query(
        query,
        {"graph_name": graph_name, "claim_ids": claim_ids},
    )
    return [r["chunk_id"] for r in rows if r.get("chunk_id")]


def resolve_relationship_citations(
    graph_store: GraphStore,
    graph_name: str,
    relationship_keys: list[str],
) -> list[Citation]:
    """Resolve relationship keys to citations via source_chunk_ids."""
    return resolve_chunk_citations(
        graph_store,
        graph_name,
        _relationship_chunk_ids(graph_store, graph_name, relationship_keys),
    )


def _relationship_chunk_ids(
    graph_store: GraphStore,
    graph_name: str,
    relationship_keys: list[str],
) -> list[str]:
    """Relationship references to source chunk ids."""
    if not relationship_keys:
        return []

    query = """
    MATCH (source:__Entity__ {graph_name: $graph_name})-[r]->
          (target:__Entity__ {graph_name: $graph_name})
    WITH r,
         coalesce(source.human_readable_id, source.canonical_key, source.id)
           + ':' + type(r) + ':'
           + coalesce(target.human_readable_id, target.canonical_key, target.id)
         AS endpoint_key
    WHERE r.human_readable_id IN $relationship_keys
       OR r.canonical_key IN $relationship_keys
       OR r.id IN $relationship_keys
       OR endpoint_key IN $relationship_keys
    UNWIND coalesce(r.source_chunk_ids, []) AS chunk_id
    RETURN DISTINCT chunk_id
    """
    rows = graph_store.execute_query(
        query,
        {"graph_name": graph_name, "relationship_keys": relationship_keys},
    )
    return [r["chunk_id"] for r in rows if r.get("chunk_id")]


def resolve_reference_citations(
    graph_store: GraphStore,
    graph_name: str,
    references: list[dict],
) -> list[Citation]:
    """Resolve validated entity, claim, and relationship references."""
    entity_ids: list[str] = []
    claim_ids: list[str] = []
    relationship_keys: list[str] = []
    # Duplicate references would only repeat the same lookups; first-seen order
    # decides the citation order.
    seen_refs: set[tuple[str, str]] = set()
    for ref in references:
        if not isinstance(ref, dict):
            continue
        target_id = str(ref.get("target_id", "")).strip()
        target_type = str(ref.get("target_type", "")).strip()
        if not target_id or (target_type, target_id) in seen_refs:
            continue
        seen_refs.add((target_type, target_id))
        if target_type == "entity":
            entity_ids.append(target_id)
        elif target_type == "claim":
            claim_ids.append(target_id)
        elif target_type == "relationship":
            relationship_keys.append(target_id)

    # Overlapping chunks are materialized once, after all references resolved.
    entity_chunk_ids = _entity_chunk_ids(graph_store, graph_name, entity_ids)
    claim_chunk_ids = _claim_chunk_ids(graph_store, graph_name, claim_ids)
    relationship_chunk_ids = _relationship_chunk_ids(
        graph_store, graph_name, relationship_keys
    )
    chunk_ids = list(dict.fromkeys(
        entity_chunk_ids + claim_chunk_ids + relationship_chunk_ids
    ))
    citations = resolve_chunk_citations(graph_store, graph_name, chunk_ids)
    # Metadata rows arrive in backend order, which is the order release returned
    # within a category. Only category precedence (entities, then claims, then
    # relationships) is ours to fix, so sort by category and leave each
    # category's row order alone.
    chunk_category: dict[str, int] = {}
    for category, category_chunk_ids in enumerate(
        (entity_chunk_ids, claim_chunk_ids, relationship_chunk_ids)
    ):
        for chunk_id in category_chunk_ids:
            chunk_category.setdefault(chunk_id, category)
    citations.sort(key=lambda citation: chunk_category.get(citation.chunk_id, 3))

    seen: set[tuple[str, str, int | None, int | None]] = set()
    deduped: list[Citation] = []
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


def _citation_metadata(row: dict) -> dict:
    document_metadata = _clean_metadata(row.get("document_metadata"))
    chunk_metadata = _clean_metadata(row.get("chunk_metadata"))
    metadata = {**document_metadata, **chunk_metadata}

    # Preserve convenience fields in metadata for vector-store-style callers.
    for key in ("document_id", "chunk_id", "document_name", "page_start", "page_end"):
        value = row.get(key)
        if value is not None:
            metadata.setdefault(key, value)
    return metadata


def _clean_metadata(value) -> dict:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if key not in _INTERNAL_METADATA_KEYS and item is not None
    }


def format_citation_metadata(
    citations: list[Citation],
    metadata_keys: list[str] | None = None,
) -> list[str]:
    """Render ``chunk_id: {metadata}`` lines for citation-aware prompting."""
    lines = []
    key_filter = set(metadata_keys or [])
    for citation in citations:
        metadata = dict(citation.metadata or {})
        if key_filter:
            metadata = {
                key: value for key, value in metadata.items() if key in key_filter
            }
        if not metadata:
            continue
        metadata_text = json.dumps(
            metadata, ensure_ascii=True, sort_keys=True, default=str
        )
        lines.append(f"{citation.chunk_id}: {metadata_text}")
    return lines
