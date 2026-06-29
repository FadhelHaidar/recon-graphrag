"""Mixed-context builder for local search.

Collects five candidate types from the entity subgraph — seed entities,
relationships, text units (chunks), community reports, and claims — then
ranks, token-packs, and renders them into a single context string for LLM
synthesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.models.artifacts import Citation
from recon_graphrag.retrieval.citations import resolve_chunk_citations
from recon_graphrag.retrieval.community_levels import (
    CommunityLevelSelector,
    resolve_community_level,
)
from recon_graphrag.utils.tokens import (
    ApproximateTokenCounter,
    PackItem,
    PackResult,
    TokenCounter,
    pack_items,
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityCandidate:
    id: str
    name: str
    label: str
    description: str
    score: float


@dataclass(frozen=True)
class RelationshipCandidate:
    source_id: str
    source_name: str
    target_id: str
    target_name: str
    relationship_type: str
    description: str
    weight: float


@dataclass(frozen=True)
class TextUnitCandidate:
    chunk_id: str
    text: str
    coverage: int  # number of matched entities this chunk links to


@dataclass(frozen=True)
class CommunityReportCandidate:
    community_id: str
    level: int
    summary: str
    rating: float | None


@dataclass(frozen=True)
class ClaimCandidate:
    claim_id: str
    entity_id: str
    claim_type: str
    description: str
    evidence_count: int


@dataclass
class MixedContextResult:
    context: str
    citations: list[Citation]
    included_entity_ids: list[str]
    included_chunk_ids: list[str]
    included_community_ids: list[str]
    included_claim_ids: list[str]
    used_tokens: int
    max_tokens: int


# ---------------------------------------------------------------------------
# Default allocation ratios
# ---------------------------------------------------------------------------

DEFAULT_ALLOCATION = {
    "text_units": 0.50,
    "community_reports": 0.10,
    "graph_facts": 0.40,
}


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class MixedContextBuilder:
    """Build mixed-context from entity subgraph for local search."""

    def __init__(
        self,
        graph_store: GraphStore,
        graph_name: str = "entity-graph",
        counter: TokenCounter | None = None,
    ):
        self.graph_store = graph_store
        self.graph_name = graph_name
        self.counter = counter or ApproximateTokenCounter()

    def build_context(
        self,
        entity_matches: list[dict],
        entity_context_rows: list[dict],
        token_budget: int,
        allocation: dict[str, float] | None = None,
        community_level: CommunityLevelSelector = "coarsest",
    ) -> MixedContextResult:
        """Build mixed context from entity matches and their context.

        Args:
            entity_matches: Ranked entity matches from hybrid retrieval
                (list of {"id": str, "score": float}).
            entity_context_rows: Context rows from fetch_entity_context().
            token_budget: Total token budget for the context.
            allocation: Ratios per category. Defaults to
                50% text units, 10% community reports, 40% graph facts.
            community_level: Which community level to fetch reports from.

        Returns:
            MixedContextResult with rendered context, citations, and telemetry.
        """
        alloc = allocation or DEFAULT_ALLOCATION

        entity_ids = [m["id"] for m in entity_matches]
        entity_scores = {m["id"]: float(m.get("score", 0.0)) for m in entity_matches}

        # 1. Collect candidates
        entities = self._collect_entities(entity_matches, entity_context_rows)
        relationships = self._collect_relationships(entity_context_rows)
        chunk_ids_from_context = self._collect_chunk_ids(entity_context_rows)
        text_units = self._collect_text_units(chunk_ids_from_context, entity_ids)
        community_reports = self._collect_community_reports(
            entity_ids, community_level
        )
        claims = self._collect_claims(entity_ids)

        # 2. Rank each category
        entities.sort(key=lambda e: (-entity_scores.get(e.id, 0.0), e.id))
        relationships.sort(key=lambda r: (-r.weight, r.source_id, r.target_id))
        text_units.sort(key=lambda t: (-t.coverage, t.chunk_id))
        community_reports.sort(
            key=lambda c: (-(c.rating or 0.0), c.community_id)
        )
        claims.sort(key=lambda c: (-c.evidence_count, c.claim_id))

        # 3. Compute per-category budgets with overflow
        budgets = self._compute_budgets(token_budget, alloc)

        # 4. Pack each category
        entity_items = [
            PackItem(
                id=e.id,
                text=self._render_entity(e),
                priority=entity_scores.get(e.id, 0.0),
            )
            for e in entities
        ]
        rel_items = [
            PackItem(
                id=f"{r.source_id}:{r.relationship_type}:{r.target_id}",
                text=self._render_relationship(r),
                priority=r.weight,
            )
            for r in relationships
        ]
        tu_items = [
            PackItem(
                id=t.chunk_id,
                text=t.text,
                priority=float(t.coverage),
            )
            for t in text_units
        ]
        cr_items = [
            PackItem(
                id=c.community_id,
                text=self._render_community_report(c),
                priority=c.rating or 0.0,
            )
            for c in community_reports
        ]
        claim_items = [
            PackItem(
                id=c.claim_id,
                text=self._render_claim(c),
                priority=float(c.evidence_count),
            )
            for c in claims
        ]

        graph_fact_items = entity_items + rel_items + claim_items

        packed_tu = pack_items(tu_items, budgets["text_units"], self.counter)
        packed_cr = pack_items(cr_items, budgets["community_reports"], self.counter)
        packed_gf = pack_items(
            graph_fact_items, budgets["graph_facts"], self.counter
        )

        # 5. Overflow: redistribute unused tokens
        overflow = (
            budgets["text_units"]
            - packed_tu.used_tokens
            + budgets["community_reports"]
            - packed_cr.used_tokens
            + budgets["graph_facts"]
            - packed_gf.used_tokens
        )
        if overflow > 0:
            # Try to include more text units first
            extra_tu = pack_items(
                [i for i in tu_items if i.id not in {x.id for x in packed_tu.included}],
                overflow,
                self.counter,
            )
            packed_tu = PackResult(
                included=packed_tu.included + extra_tu.included,
                excluded=packed_tu.excluded,
                used_tokens=packed_tu.used_tokens + extra_tu.used_tokens,
                max_tokens=packed_tu.max_tokens,
                truncated_item_ids=packed_tu.truncated_item_ids,
            )

        # 6. Render context
        sections: list[str] = []
        if packed_gf.included:
            sections.append(
                "=== Graph Facts ===\n"
                + "\n\n".join(i.text for i in packed_gf.included)
            )
        if packed_tu.included:
            sections.append(
                "=== Source Text ===\n"
                + "\n\n".join(i.text for i in packed_tu.included)
            )
        if packed_cr.included:
            sections.append(
                "=== Community Reports ===\n"
                + "\n\n".join(i.text for i in packed_cr.included)
            )

        context = "\n\n".join(sections)

        # 7. Collect included IDs for citation resolution
        entity_id_set = {e.id for e in entities}
        rel_key_set = {
            f"{r.source_id}:{r.relationship_type}:{r.target_id}"
            for r in relationships
        }
        claim_id_set = {c.claim_id for c in claims}

        included_entity_ids = [
            i.id for i in packed_gf.included if i.id in entity_id_set
        ]
        included_rel_keys = [
            i.id for i in packed_gf.included if i.id in rel_key_set
        ]
        included_chunk_ids = [i.id for i in packed_tu.included]
        included_community_ids = [i.id for i in packed_cr.included]
        included_claim_ids = [
            i.id for i in packed_gf.included if i.id in claim_id_set
        ]

        # 8. Resolve citations only from included chunks
        citations: list[Citation] = []
        if included_chunk_ids:
            try:
                citations = resolve_chunk_citations(
                    self.graph_store, self.graph_name, included_chunk_ids
                )
            except Exception:
                pass

        total_tokens = (
            packed_tu.used_tokens + packed_cr.used_tokens + packed_gf.used_tokens
        )

        return MixedContextResult(
            context=context,
            citations=citations,
            included_entity_ids=included_entity_ids,
            included_chunk_ids=included_chunk_ids,
            included_community_ids=included_community_ids,
            included_claim_ids=included_claim_ids,
            used_tokens=total_tokens,
            max_tokens=token_budget,
        )

    # ------------------------------------------------------------------
    # Candidate collection
    # ------------------------------------------------------------------

    def _collect_entities(
        self,
        entity_matches: list[dict],
        entity_context_rows: list[dict],
    ) -> list[EntityCandidate]:
        entities: list[EntityCandidate] = []
        seen: set[str] = set()
        for match in entity_matches:
            eid = match["id"]
            if eid in seen:
                continue
            seen.add(eid)
            entities.append(
                EntityCandidate(
                    id=eid,
                    name="",
                    label="Entity",
                    description="",
                    score=float(match.get("score", 0.0)),
                )
            )
        # Enrich from context rows
        for row in entity_context_rows:
            title = row.get("title", "")
            if title and "(" in title:
                parts = title.rsplit("(", 1)
                name = parts[0].strip()
                label = parts[1].rstrip(")").strip() if len(parts) > 1 else "Entity"
                for e in entities:
                    if not e.name:
                        object.__setattr__(e, "name", name)
                        object.__setattr__(e, "label", label)
        return entities

    def _collect_relationships(
        self, entity_context_rows: list[dict]
    ) -> list[RelationshipCandidate]:
        rels: list[RelationshipCandidate] = []
        seen: set[str] = set()
        for row in entity_context_rows:
            for rel_str in row.get("relationships", []):
                if rel_str in seen:
                    continue
                seen.add(rel_str)
                parsed = self._parse_relationship(rel_str)
                if parsed:
                    rels.append(parsed)
        return rels

    def _collect_chunk_ids(self, entity_context_rows: list[dict]) -> list[str]:
        seen: set[str] = set()
        chunk_ids: list[str] = []
        for row in entity_context_rows:
            for cid in row.get("source_chunk_ids", []):
                cid = str(cid).strip()
                if cid and cid not in seen:
                    seen.add(cid)
                    chunk_ids.append(cid)
        return chunk_ids

    def _collect_text_units(
        self, chunk_ids: list[str], entity_ids: list[str]
    ) -> list[TextUnitCandidate]:
        if not chunk_ids:
            return []
        rows = self.graph_store.execute_query(
            """
            UNWIND $chunk_ids AS cid
            MATCH (c:Chunk {id: cid, graph_name: $graph_name})
            OPTIONAL MATCH (c)-[:FROM_CHUNK]->(e:__Entity__ {graph_name: $graph_name})
            WITH c, collect(DISTINCT e.id) AS linked_entities
            RETURN c.id AS chunk_id,
                   c.text AS text,
                   linked_entities
            """,
            {"chunk_ids": chunk_ids, "graph_name": self.graph_name},
        )
        entity_set = set(entity_ids)
        candidates: list[TextUnitCandidate] = []
        for row in rows:
            text = row.get("text", "")
            if not text:
                continue
            linked = set(row.get("linked_entities", []))
            coverage = len(linked & entity_set)
            candidates.append(
                TextUnitCandidate(
                    chunk_id=row["chunk_id"],
                    text=text,
                    coverage=max(coverage, 1),
                )
            )
        return candidates

    def _collect_community_reports(
        self,
        entity_ids: list[str],
        community_level: CommunityLevelSelector,
    ) -> list[CommunityReportCandidate]:
        if not entity_ids:
            return []

        resolved_level = resolve_community_level(
            self.graph_store, self.graph_name, community_level
        )
        if resolved_level is None:
            return []

        rows = self.graph_store.execute_query(
            """
            UNWIND $entity_ids AS eid
            MATCH (e:__Entity__ {graph_name: $graph_name})-[:IN_COMMUNITY]->
                  (c:Community {graph_name: $graph_name, level: $level})
            WHERE e.id = eid
            WITH DISTINCT c
            WHERE coalesce(c.report_text, c.summary, '') <> ''
            RETURN c.id AS community_id,
                   c.level AS level,
                   coalesce(c.report_text, c.summary) AS summary,
                   c.rating AS rating
            """,
            {
                "entity_ids": entity_ids,
                "graph_name": self.graph_name,
                "level": resolved_level,
            },
        )
        return [
            CommunityReportCandidate(
                community_id=row["community_id"],
                level=row["level"],
                summary=row["summary"],
                rating=row.get("rating"),
            )
            for row in rows
        ]

    def _collect_claims(
        self, entity_ids: list[str]
    ) -> list[ClaimCandidate]:
        if not entity_ids:
            return []
        rows = self.graph_store.get_claims_for_entities(
            self.graph_name, entity_ids
        )
        by_claim: dict[str, ClaimCandidate] = {}
        for row in rows:
            cid = row.get("claim_id", "")
            if not cid:
                continue
            if cid in by_claim:
                old = by_claim[cid]
                by_claim[cid] = ClaimCandidate(
                    claim_id=cid,
                    entity_id=old.entity_id,
                    claim_type=old.claim_type,
                    description=old.description,
                    evidence_count=old.evidence_count + 1,
                )
            else:
                by_claim[cid] = ClaimCandidate(
                    claim_id=cid,
                    entity_id=row.get("entity_id", ""),
                    claim_type=row.get("claim_type", "general"),
                    description=row.get("description", ""),
                    evidence_count=1,
                )
        return list(by_claim.values())

    # ------------------------------------------------------------------
    # Budget computation
    # ------------------------------------------------------------------

    def _compute_budgets(
        self, total: int, alloc: dict[str, float]
    ) -> dict[str, int]:
        tu_ratio = alloc.get("text_units", 0.5)
        cr_ratio = alloc.get("community_reports", 0.1)
        gf_ratio = alloc.get("graph_facts", 0.4)
        return {
            "text_units": int(total * tu_ratio),
            "community_reports": int(total * cr_ratio),
            "graph_facts": int(total * gf_ratio),
        }

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_relationship(rel_str: str) -> RelationshipCandidate | None:
        """Parse 'Label: Name -[TYPE]-> Label: Name' format."""
        try:
            left, right = rel_str.split(" -[", 1)
            rel_type, right = right.split("]-> ", 1)
            source_parts = left.split(": ", 1)
            target_parts = right.split(": ", 1)
            source_name = source_parts[1] if len(source_parts) > 1 else source_parts[0]
            target_name = target_parts[1] if len(target_parts) > 1 else target_parts[0]
            return RelationshipCandidate(
                source_id=source_name,
                source_name=source_name,
                target_id=target_name,
                target_name=target_name,
                relationship_type=rel_type.strip(),
                description="",
                weight=1.0,
            )
        except (ValueError, IndexError):
            return None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _render_entity(e: EntityCandidate) -> str:
        label = e.label or "Entity"
        name = e.name or e.id
        desc = e.description
        if desc:
            return f"[{label}] {name}: {desc}"
        return f"[{label}] {name}"

    @staticmethod
    def _render_relationship(r: RelationshipCandidate) -> str:
        line = f"{r.source_name} --[{r.relationship_type}]--> {r.target_name}"
        if r.description:
            line += f": {r.description}"
        return line

    @staticmethod
    def _render_community_report(c: CommunityReportCandidate) -> str:
        header = f"Community {c.community_id} (level {c.level})"
        if c.rating is not None:
            header += f" [rating: {c.rating}]"
        return f"{header}:\n{c.summary}"

    @staticmethod
    def _render_claim(c: ClaimCandidate) -> str:
        return f"[{c.claim_id}] ({c.claim_type}) {c.description}"
