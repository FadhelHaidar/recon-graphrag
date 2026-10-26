"""Memgraph-specific Cypher query templates used by the Memgraph graph store.

Shared templates live in ``retrieval._queries``; this module binds the Memgraph
dialect choices (``id`` internal-id, plus a ``score`` tie-break on the
local/DRIFT queries where vector-search order is not stable).
"""

from recon_graphrag.retrieval._queries import (
    COMMUNITY_CHILD_REPORT_QUERY,
    COMMUNITY_MEMBER_GRAPH_QUERY,
    COMMUNITY_REPORTS_BY_KEY_QUERY,
    drift_retrieval_query,
    local_retrieval_query,
    ranked_context_query,
)

DEFAULT_LOCAL_RETRIEVAL_QUERY = local_retrieval_query(order_by_score=True)
DEFAULT_DRIFT_RETRIEVAL_QUERY = drift_retrieval_query(order_by_score=True)
LOCAL_RETRIEVAL_QUERY_RANKED_SOURCES = local_retrieval_query(
    order_by_score=True, rank_sources=True
)
DRIFT_RETRIEVAL_QUERY_RANKED_SOURCES = drift_retrieval_query(
    order_by_score=True, rank_sources=True
)
COMMUNITY_RANKED_CONTEXT_QUERY = ranked_context_query(id_func="id")

__all__ = [
    "DEFAULT_LOCAL_RETRIEVAL_QUERY",
    "DEFAULT_DRIFT_RETRIEVAL_QUERY",
    "LOCAL_RETRIEVAL_QUERY_RANKED_SOURCES",
    "DRIFT_RETRIEVAL_QUERY_RANKED_SOURCES",
    "COMMUNITY_CHILD_REPORT_QUERY",
    "COMMUNITY_REPORTS_BY_KEY_QUERY",
    "COMMUNITY_RANKED_CONTEXT_QUERY",
    "COMMUNITY_MEMBER_GRAPH_QUERY",
]
