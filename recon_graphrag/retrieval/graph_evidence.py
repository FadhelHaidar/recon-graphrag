"""Structured graph evidence carried on ``SearchResult.graph``.

The payload describes the entity/relationship evidence *included in synthesis*,
never the model's reasoning. ``Community`` nodes are deliberately excluded: a
contributing community report is expanded into its member entities and their
directed relationships, and the report id/level stays in ``provenance``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MEMBER_LIMIT = 25
DEFAULT_EDGE_LIMIT = 50
DEFAULT_MAX_REPORTS = 5
DEFAULT_MAX_NODES = 200
DEFAULT_MAX_EDGES = 500

# Capping keeps retrieved facts first, then community members, then nodes that
# only exist because an edge points at them.
_ORIGIN_RANK = {"retrieved": 0, "community_member": 1, "edge_endpoint": 2}


# -- one-call construction ------------------------------------------------
async def build_evidence(
    *,
    graph_name: str,
    mode: str,
    graph_store: Any = None,
    entities: Iterable[Any] = (),
    relationships: Iterable[Any] = (),
    rows: Iterable[dict] = (),
    top_k_relationships: int | None = None,
    reports: Iterable[dict] = (),
    provenance: dict | None = None,
) -> dict:
    """Build a payload from mixed-context selections and/or entity rows."""
    builder = GraphEvidenceBuilder(
        graph_name=graph_name, mode=mode, graph_store=graph_store
    )
    for entity in entities:
        builder.add_node(
            entity.id,
            name=entity.name,
            labels=entity.labels or (entity.label,),
            description=entity.description,
        )
    for rel in relationships:
        builder.add_edge(
            rel.source_id,
            rel.target_id,
            rel.relationship_type,
            description=rel.description,
            weight=rel.weight,
            source_name=rel.source_name,
            target_name=rel.target_name,
        )
    for row in rows:
        builder.add_context_row(row, top_k_relationships=top_k_relationships)
    builder.set_provenance(direct_facts=builder.direct_facts(), **(provenance or {}))
    await builder.expand_communities(reports)
    return builder.build()


def relationship_key(source: str, rel_type: str, target: str) -> str:
    """Opaque relationship identity, matching mixed_context's convention.

    Compound keys are never split on ':' — entity ids may contain separators.
    """
    return f"{source}:{rel_type}:{target}"


def _as_labels(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        value = [value]
    return [str(label) for label in value if label and label != "__Entity__"]


@dataclass
class GraphEvidenceBuilder:
    """Accumulate graph evidence, then emit a deterministic bounded payload."""

    graph_name: str
    mode: str
    graph_store: Any = None
    member_limit: int = DEFAULT_MEMBER_LIMIT
    edge_limit: int = DEFAULT_EDGE_LIMIT
    max_reports: int = DEFAULT_MAX_REPORTS
    max_nodes: int = DEFAULT_MAX_NODES
    max_edges: int = DEFAULT_MAX_EDGES
    _nodes: dict[str, dict] = field(default_factory=dict)
    _edges: dict[str, dict] = field(default_factory=dict)
    _communities: list[dict] = field(default_factory=list)
    _reasons: list[dict] = field(default_factory=list)
    _warnings: list[dict] = field(default_factory=list)
    _provenance: dict = field(default_factory=dict)
    _dropped_nodes: int = 0
    _dropped_edges: int = 0

    # -- accumulation -----------------------------------------------------
    def add_node(
        self,
        node_id: str,
        *,
        name: str | None = None,
        labels: Any = (),
        description: str = "",
        origin: str = "retrieved",
    ) -> None:
        if not node_id:
            return
        node_id = str(node_id)
        existing = self._nodes.get(node_id)
        if existing is not None:
            # Never downgrade a richer origin, but fill in a missing name.
            if not existing["name"] and name:
                existing["name"] = str(name)
            if not existing["labels"]:
                existing["labels"] = _as_labels(labels)
            return
        self._nodes[node_id] = {
            "id": node_id,
            "name": str(name) if name else node_id,
            "labels": _as_labels(labels),
            "description": description or "",
            "origin": origin,
        }

    def add_edge(
        self,
        source: str,
        target: str,
        rel_type: str,
        *,
        description: str = "",
        weight: float = 1.0,
        origin: str = "retrieved",
        source_name: str | None = None,
        target_name: str | None = None,
    ) -> None:
        source, target = str(source or ""), str(target or "")
        if not source or not target or not rel_type:
            return
        # Endpoint closure: an edge never outlives its endpoints.
        self.add_node(source, name=source_name, origin="edge_endpoint")
        self.add_node(target, name=target_name, origin="edge_endpoint")
        key = relationship_key(source, rel_type, target)
        if key in self._edges:
            return
        self._edges[key] = {
            "key": key,
            "source": source,
            "target": target,
            "type": str(rel_type),
            "description": description or "",
            "weight": float(weight if weight is not None else 1.0),
            "origin": origin,
        }

    def add_relationship_record(self, record: dict, *, origin: str = "retrieved") -> None:
        """Add one ``relationship_records`` row (already direction-normalized)."""
        self.add_edge(
            str(record.get("source_id", "")),
            str(record.get("target_id", "")),
            str(record.get("rel", "")),
            description=str(record.get("description", "") or ""),
            weight=record.get("weight", 1.0) or 1.0,
            origin=origin,
            source_name=str(record.get("source_name", "") or ""),
            target_name=str(record.get("target_name", "") or ""),
        )

    def add_context_row(
        self, row: dict, *, top_k_relationships: int | None = None
    ) -> None:
        """Add the entity of one ``fetch_entity_context`` row plus its records.

        Mirrors the rendering path: relationships are weight-ranked and cut to
        ``top_k_relationships`` exactly as the context text cuts them.
        """
        self.add_node(
            str(row.get("entity_id", "")),
            name=str(row.get("entity_name", "") or ""),
            labels=row.get("entity_labels", ()),
            description=str(row.get("entity_description", "") or ""),
        )
        records = list(row.get("relationship_records", []) or [])
        if top_k_relationships is not None and len(records) > top_k_relationships:
            records.sort(key=lambda r: -float(r.get("weight", 1.0) or 1.0))
            self._reasons.append(
                {
                    "code": "top_k_relationships",
                    "scope": f"entity:{row.get('entity_id', '')}",
                    "kept": top_k_relationships,
                    "dropped": len(records) - top_k_relationships,
                }
            )
            records = records[:top_k_relationships]
        for record in records:
            self.add_relationship_record(record)

    def add_payload(self, payload: dict) -> None:
        """Merge an already-built payload (per-action evidence) into this one."""
        for node in payload.get("nodes", []) or []:
            if "Community" in (node.get("labels") or []):
                continue
            origin = node.get("origin")
            self.add_node(
                str(node.get("id", "")),
                name=str(node.get("name", "") or ""),
                labels=node.get("labels", ()),
                description=str(node.get("description", "") or ""),
                origin=origin if origin in _ORIGIN_RANK else "retrieved",
            )
        for edge in payload.get("edges", []) or []:
            origin = edge.get("origin")
            self.add_edge(
                str(edge.get("source", "")),
                str(edge.get("target", "")),
                str(edge.get("type", "")),
                description=str(edge.get("description", "") or ""),
                weight=edge.get("weight", 1.0) or 1.0,
                origin=origin if origin in _ORIGIN_RANK else "retrieved",
            )
        for community in payload.get("provenance", {}).get("communities", []) or []:
            if community not in self._communities:
                self._communities.append(community)

    def set_provenance(self, **values: Any) -> None:
        self._provenance.update(values)

    def direct_facts(self) -> dict:
        """Ids/keys added before any community expansion."""
        return {
            "entity_ids": sorted(
                n["id"] for n in self._nodes.values() if n["origin"] == "retrieved"
            ),
            "relationship_keys": sorted(
                key for key, edge in self._edges.items() if edge["origin"] == "retrieved"
            ),
        }

    # -- community expansion ---------------------------------------------
    async def expand_communities(
        self,
        reports: Iterable[dict],
        *,
        origin: str = "context",
    ) -> None:
        """Expand contributing community reports into members and edges.

        ``reports`` entries need ``community_id`` (or ``id``) and ``level``.
        Bounded: one query per report, at most ``max_reports`` reports.
        """
        seen: set[tuple[str, int | None]] = set()
        selected: list[tuple[str, int | None]] = []
        for report in reports:
            community_id = str(
                report.get("community_id") or report.get("id") or ""
            ).strip()
            if not community_id:
                continue
            level = report.get("level")
            try:
                level = int(level) if level is not None else None
            except (TypeError, ValueError):
                level = None
            key = (community_id, level)
            if key in seen:
                continue
            seen.add(key)
            selected.append(key)
        if len(selected) > self.max_reports:
            self._reasons.append(
                {
                    "code": "report_limit",
                    "scope": "result",
                    "kept": self.max_reports,
                    "dropped": len(selected) - self.max_reports,
                }
            )
            selected = selected[: self.max_reports]
        if not selected or self.graph_store is None:
            return

        for community_id, level in selected:
            if level is None:
                # Level identifies the report; without it expansion is unsafe.
                self._warnings.append(
                    {"code": "community_level_unknown", "community_id": community_id}
                )
                continue
            try:
                rows = self.graph_store.fetch_community_member_graph(
                    self.graph_name,
                    community_id,
                    level=level,
                    member_limit=self.member_limit,
                )
            except Exception as exc:  # noqa: BLE001 - evidence is best effort
                self._warnings.append(
                    {
                        "code": "community_expansion_failed",
                        "community_id": community_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            row = rows[0] if rows else {}
            members = list(row.get("members", []) or [])
            edges = list(row.get("edges", []) or [])
            for member in members:
                self.add_node(
                    str(member.get("id", "")),
                    name=str(member.get("name", "") or ""),
                    labels=member.get("labels", ()),
                    description=str(member.get("description", "") or ""),
                    origin="community_member",
                )
            # Deterministic edge slice: strongest first, then key order.
            # shortcut: the query has no edge LIMIT, so a very high-degree
            # member still transfers one UNWIND row per edge; add a per-member
            # LIMIT in the query if that becomes a cost problem.
            edges.sort(
                key=lambda e: (
                    -float(e.get("weight", 1.0) or 1.0),
                    str(e.get("source", "")),
                    str(e.get("type", "")),
                    str(e.get("target", "")),
                )
            )
            if len(edges) > self.edge_limit:
                self._reasons.append(
                    {
                        "code": "edge_limit",
                        "scope": f"community:{community_id}",
                        "kept": self.edge_limit,
                        "dropped": len(edges) - self.edge_limit,
                    }
                )
                edges = edges[: self.edge_limit]
            member_names = {
                str(m.get("id", "")): str(m.get("name", "") or "") for m in members
            }
            for edge in edges:
                source = str(edge.get("source", ""))
                target = str(edge.get("target", ""))
                self.add_edge(
                    source,
                    target,
                    str(edge.get("type", "")),
                    description=str(edge.get("description", "") or ""),
                    weight=edge.get("weight", 1.0) or 1.0,
                    origin="community_expansion",
                    source_name=member_names.get(source, ""),
                    target_name=member_names.get(target, ""),
                )
            member_count = int(row.get("member_count", len(members)) or 0)
            edge_count = int(row.get("edge_count", len(edges)) or 0)
            if member_count >= self.member_limit:
                # The cap was reached; the true member total is unknown.
                self._reasons.append(
                    {
                        "code": "member_limit",
                        "scope": f"community:{community_id}",
                        "kept": len(members),
                        "dropped": None,
                    }
                )
            self._communities.append(
                {
                    "id": community_id,
                    "level": level,
                    "origin": origin,
                    "members_selected": len(members),
                    "member_count_known": member_count < self.member_limit,
                    "edges_selected": min(edge_count, self.edge_limit),
                }
            )

    # -- output -----------------------------------------------------------
    def build(self) -> dict:
        node_rank = sorted(
            self._nodes.values(),
            key=lambda n: (_ORIGIN_RANK.get(n["origin"], 9), n["id"]),
        )
        if len(node_rank) > self.max_nodes:
            self._dropped_nodes += len(node_rank) - self.max_nodes
            self._reasons.append(
                {
                    "code": "payload_limit",
                    "scope": "nodes",
                    "kept": self.max_nodes,
                    "dropped": len(node_rank) - self.max_nodes,
                }
            )
            node_rank = node_rank[: self.max_nodes]
        kept_ids = {n["id"] for n in node_rank}

        edges = [
            e
            for e in self._edges.values()
            if e["source"] in kept_ids and e["target"] in kept_ids
        ]
        closed_out = len(self._edges) - len(edges)
        edges.sort(key=lambda e: (e["type"], e["source"], e["target"]))
        if len(edges) > self.max_edges:
            dropped = len(edges) - self.max_edges
            self._dropped_edges += dropped
            self._reasons.append(
                {
                    "code": "payload_limit",
                    "scope": "edges",
                    "kept": self.max_edges,
                    "dropped": dropped,
                }
            )
            edges = edges[: self.max_edges]
        if closed_out:
            self._dropped_edges += closed_out
            self._reasons.append(
                {
                    "code": "payload_limit",
                    "scope": "edges_without_nodes",
                    "kept": len(edges),
                    "dropped": closed_out,
                }
            )

        provenance = {
            "graph_name": self.graph_name,
            "mode": self.mode,
            "communities": sorted(
                self._communities,
                key=lambda c: (str(c["id"]), int(c["level"] or 0)),
            ),
            **self._provenance,
        }
        if self._warnings:
            provenance["warnings"] = list(self._warnings)

        return {
            "nodes": sorted(node_rank, key=lambda n: n["id"]),
            "edges": edges,
            "provenance": provenance,
            "truncation": {
                "applied": bool(self._reasons),
                "dropped_nodes": self._dropped_nodes,
                "dropped_edges": self._dropped_edges,
                "reasons": list(self._reasons),
                "limits": {
                    "member_limit": self.member_limit,
                    "edge_limit": self.edge_limit,
                    "max_reports": self.max_reports,
                    "max_nodes": self.max_nodes,
                    "max_edges": self.max_edges,
                },
            },
        }
