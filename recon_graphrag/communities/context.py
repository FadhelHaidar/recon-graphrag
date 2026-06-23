"""Typed community context records and rendering.

Replaces raw backend node objects with plain typed records at the summarizer
boundary. Edges are sorted by combined endpoint degree (paper-compatible).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EntityContext:
    """Typed entity record for community context."""

    id: str
    name: str
    description: str
    labels: list[str] = field(default_factory=list)
    degree: int = 0


@dataclass(frozen=True)
class EdgeContext:
    """Typed relationship record for community context."""

    source: EntityContext
    target: EntityContext
    relationship_type: str
    description: str = ""
    observation_count: int = 1
    combined_degree: int = 0  # source.degree + target.degree


@dataclass
class CommunityContext:
    """Typed community context with degree-ranked edges."""

    community_id: str
    level: int
    entities: list[EntityContext] = field(default_factory=list)
    edges: list[EdgeContext] = field(default_factory=list)


def render_community_context(context: CommunityContext) -> str:
    """Render community context as structured text for summarization.

    Entities are listed once with their full description. Subsequent references
    use just the name. Relationships are rendered inline with their endpoints.
    """
    lines: list[str] = []
    seen_entities: set[str] = set()

    for edge in context.edges:
        # Source entity
        if edge.source.id not in seen_entities:
            label = edge.source.labels[0] if edge.source.labels else "Entity"
            lines.append(f"- [{label}] {edge.source.name}: {edge.source.description}")
            seen_entities.add(edge.source.id)

        # Target entity
        if edge.target.id not in seen_entities:
            label = edge.target.labels[0] if edge.target.labels else "Entity"
            lines.append(f"- [{label}] {edge.target.name}: {edge.target.description}")
            seen_entities.add(edge.target.id)

        # Relationship
        lines.append(
            f"  {edge.source.name} --[{edge.relationship_type}]--> {edge.target.name}"
        )

    # Add isolated entities (no edges)
    for entity in context.entities:
        if entity.id not in seen_entities:
            label = entity.labels[0] if entity.labels else "Entity"
            lines.append(f"- [{label}] {entity.name}: {entity.description}")
            seen_entities.add(entity.id)

    return "\n".join(lines)


def parse_community_context(
    community_id: str,
    level: int,
    rows: list[dict],
) -> CommunityContext:
    """Parse raw query rows into a typed CommunityContext.

    Expects rows from the degree-ranked context query with keys:
    - e_id, e_name, e_description, e_labels, e_degree
    - rel_type, rel_description, observation_count, combined_degree
    - other_id, other_name, other_description, other_labels, other_degree

    Rows where rel_type is NULL represent isolated entities (no edges).
    """
    entities: dict[str, EntityContext] = {}
    edges: list[EdgeContext] = []
    edge_entity_ids: set[str] = set()

    for row in rows:
        e_id = row.get("e_id")
        other_id = row.get("other_id")
        rel_type = row.get("rel_type")

        # Source entity
        if e_id and e_id not in entities:
            entities[e_id] = EntityContext(
                id=e_id,
                name=row.get("e_name", ""),
                description=row.get("e_description", ""),
                labels=_parse_labels(row.get("e_labels", [])),
                degree=row.get("e_degree", 0),
            )

        if rel_type and other_id:
            # Target entity
            if other_id not in entities:
                entities[other_id] = EntityContext(
                    id=other_id,
                    name=row.get("other_name", ""),
                    description=row.get("other_description", ""),
                    labels=_parse_labels(row.get("other_labels", [])),
                    degree=row.get("other_degree", 0),
                )

            edge = EdgeContext(
                source=entities[e_id],
                target=entities[other_id],
                relationship_type=rel_type,
                description=row.get("rel_description", ""),
                observation_count=row.get("observation_count", 1),
                combined_degree=row.get("combined_degree", 0),
            )
            edges.append(edge)
            edge_entity_ids.add(e_id)
            edge_entity_ids.add(other_id)

    # Sort edges by combined_degree DESC, observation_count DESC, type ASC
    edges.sort(
        key=lambda e: (-e.combined_degree, -e.observation_count, e.relationship_type)
    )

    # Isolated entities (in community but not part of any edge)
    isolated = [e for eid, e in entities.items() if eid not in edge_entity_ids]

    return CommunityContext(
        community_id=community_id,
        level=level,
        entities=isolated,
        edges=edges,
    )


def _parse_labels(raw) -> list[str]:
    """Parse labels from various backend formats (list, string, etc.)."""
    if isinstance(raw, list):
        return [str(l) for l in raw if l != "__Entity__"]
    if isinstance(raw, str):
        return [raw] if raw != "__Entity__" else []
    return []
