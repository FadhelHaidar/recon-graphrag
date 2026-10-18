"""Typed community context records, rendering, and token-bounded packing.

Replaces raw backend node objects with plain typed records at the summarizer
boundary. Edges are sorted by combined endpoint degree (paper-compatible).
Packing ensures context fits within LLM token budgets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from recon_graphrag.utils.tokens import (
    ApproximateTokenCounter,
    TokenCounter,
)


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


@dataclass(frozen=True)
class ClaimContext:
    """Typed claim record for community context."""

    id: str
    entity_id: str
    claim_type: str
    description: str
    status: str = "active"
    object_entity_id: str | None = None
    source_text: str | None = None
    text_unit_id: str | None = None
    start_date: str | None = None
    end_date: str | None = None


@dataclass
class CommunityContext:
    """Typed community context with degree-ranked edges."""

    community_id: str
    level: int
    entities: list[EntityContext] = field(default_factory=list)
    edges: list[EdgeContext] = field(default_factory=list)
    claims: list[ClaimContext] = field(default_factory=list)


@dataclass
class PackedCommunityContext:
    """Result of packing community context into a token budget."""

    community_id: str
    level: int
    text: str
    used_tokens: int
    max_tokens: int
    included_edges: int
    excluded_edges: int
    included_entities: int
    excluded_entities: int
    truncated: bool


@dataclass(frozen=True)
class _CommunityEvidenceSelection:
    """Rendered direct evidence; child report narratives grant no IDs."""

    packed: PackedCommunityContext
    edges: tuple[EdgeContext, ...]
    isolated_entities: tuple[EntityContext, ...]
    rendered_claims: tuple[ClaimContext, ...] = ()


def _render_entity(entity: EntityContext) -> str:
    label = entity.labels[0] if entity.labels else "Entity"
    return f"- [{label}] {entity.name}: {entity.description}"


def _render_edge_unit(edge: EdgeContext, seen_entities: set[str]) -> str:
    # Snapshot membership preserves the packed self-loop's duplicate entity line.
    lines = [
        _render_entity(entity)
        for entity in (edge.source, edge.target)
        if entity.id not in seen_entities
    ]
    line = f"  {edge.source.name} --[{edge.relationship_type}]--> {edge.target.name}"
    if edge.description:
        line += f": {edge.description}"
    return "\n".join([*lines, line])


def _render_claim(claim: ClaimContext) -> str:
    parts = [f"status={claim.status}"]
    if claim.start_date or claim.end_date:
        parts.append(f"dates={claim.start_date or '?'}..{claim.end_date or '?'}")
    if claim.object_entity_id:
        parts.append(f"object={claim.object_entity_id}")
    line = f"  [{claim.id}] ({claim.claim_type}; {', '.join(parts)}) {claim.description}"
    if claim.source_text:
        line += f" Source: {claim.source_text[:200]}"
    return line


def render_community_context(context: CommunityContext) -> str:
    """Render community context as structured text for summarization.

    Entities are listed once with their full description. Subsequent references
    use just the name. Relationships are rendered inline with their endpoints.
    Claims are listed after relationships.
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
        rel_line = f"  {edge.source.name} --[{edge.relationship_type}]--> {edge.target.name}"
        if edge.description:
            rel_line += f": {edge.description}"
        lines.append(rel_line)

    # Add isolated entities (no edges)
    for entity in context.entities:
        if entity.id not in seen_entities:
            label = entity.labels[0] if entity.labels else "Entity"
            lines.append(f"- [{label}] {entity.name}: {entity.description}")
            seen_entities.add(entity.id)

    # Add claims
    if context.claims:
        lines.append("")
        lines.append("Claims:")
        lines.extend(_render_claim(claim) for claim in context.claims)

    return "\n".join(lines)


def pack_community_context(
    context: CommunityContext,
    max_tokens: int,
    counter: TokenCounter | None = None,
) -> PackedCommunityContext:
    """Pack community context into a token budget.

    Iterates ranked edges in order, including each if it fits. Entity
    descriptions are only included on first occurrence (deduplication).
    Remaining entities, then atomic claims with admitted subjects, are appended
    if space remains. Headings and separators share the same budget.

    Args:
        context: Ranked community context from parse_community_context.
        max_tokens: Maximum tokens for the rendered context.
        counter: Token counter (defaults to ApproximateTokenCounter).

    Returns:
        PackedCommunityContext with rendered text and telemetry.
    """
    return _pack_community_evidence(context, max_tokens, counter).packed


def _pack_community_evidence(
    context: CommunityContext,
    max_tokens: int,
    counter: TokenCounter | None = None,
    *,
    prefix_text: str = "",
) -> _CommunityEvidenceSelection:
    """Pack evidence against the complete text, including any fixed parent prefix."""
    counter = counter or ApproximateTokenCounter()

    if max_tokens <= 0 or counter.count(prefix_text) > max_tokens:
        return _CommunityEvidenceSelection(
            packed=PackedCommunityContext(
                community_id=context.community_id,
                level=context.level,
                text="",
                used_tokens=0,
                max_tokens=max_tokens,
                included_edges=0,
                excluded_edges=len(context.edges),
                included_entities=0,
                excluded_entities=len(context.entities),
                truncated=False,
            ),
            edges=(),
            isolated_entities=(),
        )

    lines: list[str] = []
    seen_entities: set[str] = set()
    def compose(direct_text: str) -> str:
        return "\n\n".join(part for part in (prefix_text, direct_text) if part)

    truncated = False
    included_edges = 0
    included_entities = 0
    selected_isolates: list[EntityContext] = []

    for edge in context.edges:
        unit_text = _render_edge_unit(edge, seen_entities)
        candidate = compose("\n".join([*lines, unit_text]))
        if counter.count(candidate) <= max_tokens:
            lines.append(unit_text)
            seen_entities.add(edge.source.id)
            seen_entities.add(edge.target.id)
            included_edges += 1
        else:
            # Stop at the first edge that doesn't fit so included_edges stays a
            # true prefix of the ranked list. build_packed_reference_ids rebuilds
            # the allowlist as edges[:included_edges]; a keep-trying fill would
            # desync the two and drop valid findings that cite a shown edge.
            truncated = True
            break

    excluded_edges = len(context.edges) - included_edges

    # Isolated entities
    for entity in context.entities:
        if entity.id in seen_entities:
            continue
        entity_line = _render_entity(entity)
        candidate = compose("\n".join([*lines, entity_line]))
        if counter.count(candidate) <= max_tokens:
            lines.append(entity_line)
            seen_entities.add(entity.id)
            included_entities += 1
            selected_isolates.append(entity)
        else:
            truncated = True
            break

    direct_text = "\n".join(lines)
    selected_claims: list[ClaimContext] = []
    for claim in context.claims:
        if claim.entity_id not in seen_entities:
            continue
        separator = "\n" if selected_claims else "\n\nClaims:\n"
        candidate = direct_text + separator + _render_claim(claim)
        if counter.count(compose(candidate)) > max_tokens:
            truncated = True
            break
        direct_text = candidate
        selected_claims.append(claim)

    text = compose(direct_text)
    total_entities = len({e.id for e in _all_entities(context)})
    excluded_entities = total_entities - len(seen_entities)

    return _CommunityEvidenceSelection(
        packed=PackedCommunityContext(
            community_id=context.community_id,
            level=context.level,
            text=text,
            used_tokens=counter.count(text),
            max_tokens=max_tokens,
            included_edges=included_edges,
            excluded_edges=excluded_edges,
            included_entities=included_entities,
            excluded_entities=excluded_entities,
            truncated=truncated,
        ),
        edges=tuple(context.edges[:included_edges]),
        isolated_entities=tuple(selected_isolates),
        rendered_claims=tuple(selected_claims),
    )


def _build_selected_reference_ids(selection: _CommunityEvidenceSelection) -> list[str]:
    """Authorize only selected direct evidence, retaining reference ordering."""
    ids: list[str] = []
    seen_entities: set[str] = set()
    for edge in selection.edges:
        for entity in (edge.source, edge.target):
            if entity.id not in seen_entities:
                ids.append(entity.id)
                seen_entities.add(entity.id)
    for edge in selection.edges:
        ids.append(f"{edge.source.id}:{edge.relationship_type}:{edge.target.id}")
    for entity in selection.isolated_entities:
        ids.append(entity.id)
    ids.extend(claim.id for claim in selection.rendered_claims)
    return ids


def _all_entities(context: CommunityContext) -> list[EntityContext]:
    """Collect all unique entities from edges and isolated list."""
    seen: set[str] = set()
    result: list[EntityContext] = []
    for edge in context.edges:
        if edge.source.id not in seen:
            result.append(edge.source)
            seen.add(edge.source.id)
        if edge.target.id not in seen:
            result.append(edge.target)
            seen.add(edge.target.id)
    for entity in context.entities:
        if entity.id not in seen:
            result.append(entity)
            seen.add(entity.id)
    return result


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


def enrich_context_with_claims(
    context: CommunityContext,
    claim_rows: list[dict],
) -> CommunityContext:
    """Add claims to an existing CommunityContext.

    Args:
        context: Parsed community context.
        claim_rows: Rows from get_claims_for_entities() with keys:
            claim_id, entity_id, claim_type, description, status, chunk_id.

    Returns:
        New CommunityContext with claims added.
    """
    claims = [
        ClaimContext(
            id=row["claim_id"],
            entity_id=row["entity_id"],
            claim_type=row.get("claim_type", "general"),
            description=row.get("description", ""),
            status=row.get("status", "active"),
            object_entity_id=row.get("object_entity_id"),
            source_text=row.get("source_text"),
            text_unit_id=row.get("text_unit_id"),
            start_date=row.get("start_date"),
            end_date=row.get("end_date"),
        )
        for row in claim_rows
        if row.get("claim_id") and row.get("entity_id")
    ]
    return CommunityContext(
        community_id=context.community_id,
        level=context.level,
        entities=context.entities,
        edges=context.edges,
        claims=claims,
    )


def build_reference_ids(context: CommunityContext) -> list[str]:
    """Build the reference ID allowlist from a CommunityContext.

    Returns entity IDs, relationship keys (source:type:target), and claim IDs.
    """
    ids: list[str] = []

    # Entity IDs
    seen_entities: set[str] = set()
    for edge in context.edges:
        if edge.source.id not in seen_entities:
            ids.append(edge.source.id)
            seen_entities.add(edge.source.id)
        if edge.target.id not in seen_entities:
            ids.append(edge.target.id)
            seen_entities.add(edge.target.id)
    for entity in context.entities:
        if entity.id not in seen_entities:
            ids.append(entity.id)
            seen_entities.add(entity.id)

    # Relationship keys
    for edge in context.edges:
        rel_key = f"{edge.source.id}:{edge.relationship_type}:{edge.target.id}"
        ids.append(rel_key)

    # Claim IDs
    for claim in context.claims:
        ids.append(claim.id)

    return ids


def build_packed_reference_ids(
    context: CommunityContext,
    packed: PackedCommunityContext,
) -> list[str]:
    """Recover references only from an exact canonical packed rendering.

    No counter is available here. Telemetry determines the graph prefix; the
    entire remaining text must equal a complete eligible claim prefix. Searching
    for claim blocks would mistake multiline descriptions/sources for evidence.
    Internal callers retain explicit selections instead of recovering from text.
    """
    edges = tuple(context.edges[:packed.included_edges])
    seen_entities: set[str] = set()
    lines: list[str] = []
    for edge in edges:
        lines.append(_render_edge_unit(edge, seen_entities))
        seen_entities.update((edge.source.id, edge.target.id))

    isolates: list[EntityContext] = []
    for entity in context.entities:
        if entity.id in seen_entities:
            continue
        if len(isolates) >= packed.included_entities:
            break
        isolates.append(entity)
        lines.append(_render_entity(entity))
        seen_entities.add(entity.id)

    selection = _CommunityEvidenceSelection(packed, edges, tuple(isolates))
    text = "\n".join(lines)
    if text == packed.text:
        return _build_selected_reference_ids(selection)

    claims: list[ClaimContext] = []
    for claim in context.claims:
        if claim.entity_id not in seen_entities:
            continue
        text += ("\n" if claims else "\n\nClaims:\n") + _render_claim(claim)
        claims.append(claim)
        if text == packed.text:
            return _build_selected_reference_ids(
                _CommunityEvidenceSelection(packed, edges, tuple(isolates), tuple(claims))
            )
    # Noncanonical or partial evidence cannot safely grant references.
    return []
