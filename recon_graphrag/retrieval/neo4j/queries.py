"""Neo4j-specific Cypher query templates used by the Neo4j graph store.

Shared templates live in ``retrieval._queries``; this module binds the Neo4j
dialect choices (``elementId`` internal-id, no vector-order tie-break).
Generic retrieval consumers should call GraphStore methods instead of importing
backend query templates directly.
"""

from recon_graphrag.retrieval._queries import (
    COMMUNITY_CHILD_REPORT_QUERY,
    COMMUNITY_REPORTS_BY_KEY_QUERY,
    drift_retrieval_query,
    local_retrieval_query,
    ranked_context_query,
)

DEFAULT_LOCAL_RETRIEVAL_QUERY = local_retrieval_query(order_by_score=False)
DEFAULT_DRIFT_RETRIEVAL_QUERY = drift_retrieval_query(order_by_score=False)
LOCAL_RETRIEVAL_QUERY_RANKED_SOURCES = local_retrieval_query(
    order_by_score=False, rank_sources=True
)
DRIFT_RETRIEVAL_QUERY_RANKED_SOURCES = drift_retrieval_query(
    order_by_score=False, rank_sources=True
)
COMMUNITY_RANKED_CONTEXT_QUERY = ranked_context_query(id_func="elementId")

__all__ = [
    "DEFAULT_LOCAL_RETRIEVAL_QUERY",
    "DEFAULT_DRIFT_RETRIEVAL_QUERY",
    "LOCAL_RETRIEVAL_QUERY_RANKED_SOURCES",
    "DRIFT_RETRIEVAL_QUERY_RANKED_SOURCES",
    "COMMUNITY_CHILD_REPORT_QUERY",
    "COMMUNITY_REPORTS_BY_KEY_QUERY",
    "COMMUNITY_RANKED_CONTEXT_QUERY",
]
