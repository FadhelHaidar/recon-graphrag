"""Helpers for semantic community-level selection."""

from __future__ import annotations

from typing import Literal, Optional, TypeAlias

from recon_graphrag.graph.base import GraphStore


CommunityLevelSelector: TypeAlias = Optional[
    int | Literal["all", "finest", "coarsest"]
]


def resolve_community_level(
    graph_store: GraphStore,
    graph_name: str,
    selector: CommunityLevelSelector,
) -> Optional[int]:
    """Resolve semantic community-level aliases to stored integer levels.

    Recon-GraphRAG stores level 0 as the finest community level. Higher
    integer levels are broader/coarser parent communities.
    """
    if selector is None or selector == "all":
        return None
    if selector == "finest":
        return 0
    if selector == "coarsest":
        result = graph_store.execute_query(
            """
            MATCH (c:Community {graph_name: $graph_name})
            WHERE c.level IS NOT NULL
            RETURN max(c.level) AS level
            """,
            {"graph_name": graph_name},
        )
        if not result:
            return None
        return result[0].get("level")
    if isinstance(selector, int):
        if selector < 0:
            raise ValueError("community level must be >= 0")
        return selector

    raise ValueError(
        "community level must be an integer, None, 'all', 'finest', or 'coarsest'"
    )
