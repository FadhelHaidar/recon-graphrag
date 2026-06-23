"""Citation resolution for search results.

Resolves entity/claim/chunk evidence into user-facing Citation objects
with document identity, page ranges, and bounded excerpts.
"""

from __future__ import annotations

from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.models.artifacts import Citation, citation_excerpt


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
    if not entity_ids:
        return []

    # Get chunk IDs from evidence links
    query = """
    UNWIND $entity_ids AS eid
    MATCH (c:Chunk)-[:FROM_CHUNK]->(e:__Entity__ {id: eid})
    RETURN DISTINCT c.id AS chunk_id
    """
    rows = graph_store.execute_query(
        query, {"entity_ids": entity_ids, "graph_name": graph_name}
    )
    chunk_ids = [r["chunk_id"] for r in rows if r.get("chunk_id")]
    return resolve_chunk_citations(graph_store, chunk_ids)


def resolve_chunk_citations(
    graph_store: GraphStore,
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

    rows = graph_store.resolve_chunk_citations(chunk_ids)
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
                excerpt=citation_excerpt(row.get("excerpt", "")),
            )
        )

    return citations


def resolve_claim_citations(
    graph_store: GraphStore,
    claim_ids: list[str],
) -> list[Citation]:
    """Resolve claim IDs to source chunk citations via SOURCED_FROM edges.

    Args:
        graph_store: Graph store backend.
        claim_ids: Claim IDs to resolve.

    Returns:
        Deduplicated list of Citation objects.
    """
    if not claim_ids:
        return []

    query = """
    UNWIND $claim_ids AS cid
    MATCH (c:Claim {id: cid})-[:SOURCED_FROM]->(ch:Chunk)
    RETURN DISTINCT ch.id AS chunk_id
    """
    rows = graph_store.execute_query(query, {"claim_ids": claim_ids})
    chunk_ids = [r["chunk_id"] for r in rows if r.get("chunk_id")]
    return resolve_chunk_citations(graph_store, chunk_ids)
