"""Memgraph-specific entity resolution implementation.

Reuses the grouping logic from the Neo4j resolver but replaces the APOC-based
merge with a manual copy-properties/rewire-relationships/delete strategy,
because Memgraph/MAGE does not provide APOC's mergeNodes procedure.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from recon_graphrag.graphdb.memgraph.cypher import escape_cypher_identifier

# ------------------------------------------------------------------
# Normalization helpers
# ------------------------------------------------------------------

_ORG_SUFFIXES = [
    "corp",
    "corporation",
    "inc",
    "incorporated",
    "ltd",
    "limited",
    "llc",
    "co",
    "company",
    "plc",
    "llp",
    "lp",
    "holdings",
    "group",
]

_ORG_SUFFIX_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in _ORG_SUFFIXES) + r")\.?\b",
    re.IGNORECASE,
)

_PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")


def _normalize_name(value: str) -> str:
    """Normalize an entity name for deterministic matching."""
    if not value:
        return ""
    value = unicodedata.normalize("NFC", value)
    value = value.casefold()
    value = _ORG_SUFFIX_PATTERN.sub("", value)
    value = _PUNCTUATION_PATTERN.sub("", value)
    value = " ".join(value.split())
    # Remove spaces so "Open AI" and "OpenAI" both become "openai"
    value = value.replace(" ", "")
    return value.strip()


# ------------------------------------------------------------------
# Internal record
# ------------------------------------------------------------------

@dataclass
class _EntityRecord:
    node_id: int
    entity_id: str
    graph_name: str
    domain_label: str
    resolve_value: str
    normalized_value: str
    properties: dict = field(default_factory=dict)


# ------------------------------------------------------------------
# Internal Memgraph resolver
# ------------------------------------------------------------------

class _MemgraphEntityResolver:
    def __init__(self, graph_store):
        self.graph_store = graph_store

    async def resolve(  # noqa: C901
        self,
        *,
        graph_name: str = "entity-graph",
        strategy: str = "normalized",
        resolve_property: str = "name",
        dry_run: bool = False,
        merge_threshold: float = 95.0,
        review_threshold: float = 85.0,
        max_candidates_per_entity: int = 20,
        aliases: Optional[dict] = None,
        embedder=None,
        llm=None,
        llm_guidance: Optional[str] = None,
        allow_ai_auto_merge: bool = False,
    ) -> dict:
        if strategy not in ("exact", "normalized", "fuzzy", "hybrid"):
            raise ValueError(f"Unknown strategy: {strategy}")

        entities = self._load_entities(graph_name, resolve_property)

        if strategy == "exact":
            groups, review_groups = self._build_exact_groups(entities)
            signals = {"exact": "used"}
        elif strategy == "normalized":
            groups, review_groups = self._build_normalized_groups(entities)
            signals = {"normalized": "used"}
        elif strategy == "fuzzy":
            groups, review_groups = self._build_fuzzy_groups(
                entities,
                merge_threshold=merge_threshold,
                review_threshold=review_threshold,
                max_candidates_per_entity=max_candidates_per_entity,
            )
            signals = {"normalized": "used", "fuzzy": "used"}
        else:  # hybrid
            groups, review_groups = await self._build_hybrid_groups(
                entities,
                merge_threshold=merge_threshold,
                review_threshold=review_threshold,
                max_candidates_per_entity=max_candidates_per_entity,
                aliases=aliases,
                embedder=embedder,
                llm=llm,
                llm_guidance=llm_guidance,
                allow_ai_auto_merge=allow_ai_auto_merge,
            )
            signals = {
                "normalized": "used",
                "fuzzy": "used",
                "aliases": "used" if aliases else "skipped_no_aliases",
                "embeddings": "used" if embedder else "skipped_no_embedder",
                "llm": "used" if llm else "skipped_no_llm",
            }

        merged_nodes = 0
        if not dry_run and groups:
            merged_nodes = self._merge_groups(groups, resolve_property)

        return {
            "skipped": False,
            "strategy": strategy,
            "merged_groups": len(groups),
            "merged_nodes": merged_nodes,
            "candidate_groups": len(groups) + len(review_groups),
            "review_groups": review_groups,
            "signals": signals,
        }

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _load_entities(
        self, graph_name: str, resolve_property: str
    ) -> list[_EntityRecord]:
        prop = escape_cypher_identifier(resolve_property)
        rows = self.graph_store.execute_query(
            f"""
            MATCH (e:__Entity__)
            WHERE e.graph_name = $graph_name
              AND e.{prop} IS NOT NULL
            RETURN
              id(e) AS node_id,
              e.id AS entity_id,
              e.graph_name AS graph_name,
              e.{prop} AS resolve_value,
              labels(e) AS labels,
              properties(e) AS properties
            """,
            {"graph_name": graph_name},
        )

        entities = []
        for row in rows:
            labels = row.get("labels", [])
            domain_labels = [l for l in labels if l != "__Entity__"]
            domain_label = domain_labels[0] if domain_labels else "__Entity__"
            resolve_value = row.get("resolve_value", "")
            entities.append(
                _EntityRecord(
                    node_id=row.get("node_id", 0),
                    entity_id=row.get("entity_id", ""),
                    graph_name=row.get("graph_name", ""),
                    domain_label=domain_label,
                    resolve_value=resolve_value,
                    normalized_value=_normalize_name(resolve_value),
                    properties=row.get("properties", {}),
                )
            )
        return entities

    # ------------------------------------------------------------------
    # Group builders
    # ------------------------------------------------------------------

    def _build_exact_groups(
        self, entities: list[_EntityRecord]
    ) -> tuple[list[list[_EntityRecord]], list[dict]]:
        key_map: dict[tuple[str, str, str], list[_EntityRecord]] = {}
        for e in entities:
            key = (e.graph_name, e.domain_label, e.resolve_value)
            key_map.setdefault(key, []).append(e)
        groups = [g for g in key_map.values() if len(g) > 1]
        return groups, []

    def _build_normalized_groups(
        self, entities: list[_EntityRecord]
    ) -> tuple[list[list[_EntityRecord]], list[dict]]:
        key_map: dict[tuple[str, str, str], list[_EntityRecord]] = {}
        for e in entities:
            key = (e.graph_name, e.domain_label, e.normalized_value)
            key_map.setdefault(key, []).append(e)
        groups = [g for g in key_map.values() if len(g) > 1]
        return groups, []

    def _build_fuzzy_groups(
        self,
        entities: list[_EntityRecord],
        merge_threshold: float,
        review_threshold: float,
        max_candidates_per_entity: int,
    ) -> tuple[list[list[_EntityRecord]], list[dict]]:
        normalized_groups, _ = self._build_normalized_groups(entities)
        merged_ids = {e.node_id for g in normalized_groups for e in g}
        singletons = [e for e in entities if e.node_id not in merged_ids]

        fuzzy_groups: list[list[_EntityRecord]] = []
        review_groups: list[dict] = []
        used = set()
        reviewed_pairs: set[frozenset[int]] = set()

        for e1 in singletons:
            if e1.node_id in used:
                continue
            candidates = self._blocked_candidates(
                e1, singletons, max_candidates_per_entity, used
            )
            for e2 in candidates:
                pair_key = frozenset({e1.node_id, e2.node_id})
                if pair_key in reviewed_pairs:
                    continue
                score = self._fuzzy_score(e1, e2)
                if score >= merge_threshold:
                    fuzzy_groups.append([e1, e2])
                    used.add(e1.node_id)
                    used.add(e2.node_id)
                    reviewed_pairs.add(pair_key)
                    break
                elif score >= review_threshold:
                    review_groups.append(
                        {
                            "domain_label": e1.domain_label,
                            "names": [e1.resolve_value, e2.resolve_value],
                            "reason": "fuzzy_candidate",
                            "scores": {
                                "fuzzy": round(score, 2),
                                "embedding": None,
                                "llm": None,
                            },
                            "decision": "review",
                        }
                    )
                    reviewed_pairs.add(pair_key)
                    break

        return normalized_groups + fuzzy_groups, review_groups

    def _blocked_candidates(
        self,
        entity: _EntityRecord,
        pool: list[_EntityRecord],
        max_candidates: int,
        used: set[int],
    ) -> list[_EntityRecord]:
        candidates = []
        for other in pool:
            if other.node_id == entity.node_id or other.node_id in used:
                continue
            if other.graph_name != entity.graph_name:
                continue
            if other.domain_label != entity.domain_label:
                continue
            len1 = len(entity.normalized_value)
            len2 = len(other.normalized_value)
            if abs(len1 - len2) > max(3, int(min(len1, len2) * 0.4)):
                continue
            tok1 = entity.normalized_value.split()[:1]
            tok2 = other.normalized_value.split()[:1]
            if tok1 and tok2 and tok1[0] != tok2[0]:
                if not self._is_acronym_match(
                    entity.normalized_value, other.normalized_value
                ):
                    continue
            candidates.append(other)
        return candidates[:max_candidates]

    @staticmethod
    def _is_acronym_match(s1: str, s2: str) -> bool:
        words1 = s1.split()
        words2 = s2.split()
        acronym1 = "".join(w[0] for w in words1 if w)
        acronym2 = "".join(w[0] for w in words2 if w)
        return acronym1 == acronym2 or acronym1 == s2 or acronym2 == s1

    @staticmethod
    def _fuzzy_score(e1: _EntityRecord, e2: _EntityRecord) -> float:
        try:
            from rapidfuzz import fuzz  # type: ignore[import-untyped]

            return float(fuzz.ratio(e1.normalized_value, e2.normalized_value))
        except ImportError:
            return 0.0

    async def _build_hybrid_groups(
        self,
        entities: list[_EntityRecord],
        merge_threshold: float,
        review_threshold: float,
        max_candidates_per_entity: int,
        aliases: Optional[dict],
        embedder,
        llm,
        llm_guidance: Optional[str],
        allow_ai_auto_merge: bool,
    ) -> tuple[list[list[_EntityRecord]], list[dict]]:
        groups, review_groups = self._build_fuzzy_groups(
            entities,
            merge_threshold=merge_threshold,
            review_threshold=review_threshold,
            max_candidates_per_entity=max_candidates_per_entity,
        )
        merged_ids = {e.node_id for g in groups for e in g}

        alias_groups, alias_reviews = self._build_alias_groups(
            entities, merged_ids, aliases
        )
        groups.extend(alias_groups)
        review_groups.extend(alias_reviews)
        for g in alias_groups:
            for e in g:
                merged_ids.add(e.node_id)

        if embedder and review_groups:
            review_groups = await self._score_with_embeddings(
                review_groups, embedder, allow_ai_auto_merge, merge_threshold
            )

        if llm and review_groups:
            review_groups = await self._llm_review(
                review_groups,
                llm,
                llm_guidance,
                aliases,
                allow_ai_auto_merge,
                merge_threshold,
            )

        return groups, review_groups

    def _build_alias_groups(
        self,
        entities: list[_EntityRecord],
        merged_ids: set[int],
        aliases: Optional[dict],
    ) -> tuple[list[list[_EntityRecord]], list[dict]]:
        groups: list[list[_EntityRecord]] = []
        review_groups: list[dict] = []
        if not aliases:
            return groups, review_groups

        simple_aliases: dict[str, list[str]] = {}
        domain_aliases: dict[str, dict[str, list[str]]] = {}
        for key, value in aliases.items():
            if isinstance(value, list):
                simple_aliases[key] = value
            elif isinstance(value, dict):
                domain_aliases[key] = value

        for e1 in entities:
            if e1.node_id in merged_ids:
                continue
            for canonical, alias_list in simple_aliases.items():
                norm_canonical = _normalize_name(canonical)
                if e1.normalized_value != norm_canonical:
                    continue
                for e2 in entities:
                    if (
                        e2.node_id in merged_ids
                        or e2.node_id == e1.node_id
                        or e2.graph_name != e1.graph_name
                        or e2.domain_label != e1.domain_label
                    ):
                        continue
                    if any(
                        _normalize_name(alias) == e2.normalized_value
                        for alias in alias_list
                    ):
                        groups.append([e1, e2])
                        merged_ids.add(e1.node_id)
                        merged_ids.add(e2.node_id)
                        break
                if e1.node_id in merged_ids:
                    break

        for domain, domain_alias_dict in domain_aliases.items():
            for e1 in entities:
                if e1.node_id in merged_ids or e1.domain_label != domain:
                    continue
                for canonical, alias_list in domain_alias_dict.items():
                    norm_canonical = _normalize_name(canonical)
                    if e1.normalized_value != norm_canonical:
                        continue
                    for e2 in entities:
                        if (
                            e2.node_id in merged_ids
                            or e2.node_id == e1.node_id
                            or e2.graph_name != e1.graph_name
                            or e2.domain_label != domain
                        ):
                            continue
                        if any(
                            _normalize_name(alias) == e2.normalized_value
                            for alias in alias_list
                        ):
                            groups.append([e1, e2])
                            merged_ids.add(e1.node_id)
                            merged_ids.add(e2.node_id)
                            break
                    if e1.node_id in merged_ids:
                        break

        return groups, review_groups

    async def _score_with_embeddings(
        self,
        review_groups: list[dict],
        embedder,
        allow_ai_auto_merge: bool,
        merge_threshold: float,
    ) -> list[dict]:
        for rg in review_groups:
            names = rg.get("names", [])
            if len(names) < 2:
                rg["scores"]["embedding"] = None
                continue
            try:
                vector_a = await embedder.async_embed_query(str(names[0]))
                vector_b = await embedder.async_embed_query(str(names[1]))
                rg["scores"]["embedding"] = round(
                    self._cosine_similarity(vector_a, vector_b),
                    4,
                )
            except Exception as exc:
                rg["scores"]["embedding"] = None
                rg["embedding_error"] = str(exc)
        return review_groups

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    async def _llm_review(
        self,
        review_groups: list[dict],
        llm,
        llm_guidance: Optional[str],
        aliases: Optional[dict],
        allow_ai_auto_merge: bool,
        merge_threshold: float,
    ) -> list[dict]:
        for rg in review_groups:
            prompt = self._build_llm_review_prompt(rg, aliases, llm_guidance)
            try:
                response = await llm.ainvoke(prompt)
                parsed = self._parse_llm_json(response.content)
                confidence = parsed.get("confidence")
                rg["scores"]["llm"] = confidence
                rg["llm_review"] = {
                    "same_entity": parsed.get("same_entity"),
                    "confidence": confidence,
                    "reason": parsed.get("reason"),
                    "merge_allowed": parsed.get("merge_allowed", False),
                }
            except Exception as exc:
                rg["scores"]["llm"] = None
                rg["llm_review"] = {
                    "error": str(exc),
                }
        return review_groups

    @staticmethod
    def _build_llm_review_prompt(
        review_group: dict,
        aliases: Optional[dict],
        llm_guidance: Optional[str],
    ) -> str:
        payload = {
            "task": "entity_deduplication_review",
            "instruction": (
                "Decide whether the candidate names refer to the same real-world "
                "entity in this graph context. Be conservative. Return only a "
                "JSON object and do not include markdown."
            ),
            "candidate": review_group,
            "user_aliases": aliases or {},
            "llm_guidance": llm_guidance or "",
            "expected_json": {
                "same_entity": "boolean",
                "confidence": "number from 0.0 to 1.0",
                "reason": "short explanation",
                "merge_allowed": "boolean",
            },
        }
        return json.dumps(payload, ensure_ascii=True, indent=2)

    @staticmethod
    def _parse_llm_json(content: str) -> dict:
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(content[start : end + 1])
            raise

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def _merge_groups(
        self, groups: list[list[_EntityRecord]], resolve_property: str
    ) -> int:
        merged_nodes = 0
        for group in groups:
            sorted_group = sorted(
                group,
                key=lambda e: len(e.resolve_value) if e.resolve_value else 0,
                reverse=True,
            )
            node_ids = [e.node_id for e in sorted_group]
            canonical_name = sorted_group[0].resolve_value if sorted_group else ""
            aliases = list(
                {
                    e.resolve_value
                    for e in group
                    if e.resolve_value and e.resolve_value != canonical_name
                }
            )

            canonical_id = node_ids[0]
            other_ids = node_ids[1:]
            if not other_ids:
                continue

            # 1. Combine properties from all nodes onto the canonical node.
            combined_props = self._combine_properties(node_ids)
            combined_props[resolve_property] = canonical_name
            if aliases:
                existing_aliases = combined_props.get("aliases", [])
                if not isinstance(existing_aliases, list):
                    existing_aliases = [existing_aliases]
                combined_props["aliases"] = list(
                    dict.fromkeys(existing_aliases + aliases)
                )

            self.graph_store.execute_query(
                """
                MATCH (n)
                WHERE id(n) = $canonical_id
                SET n += $props
                RETURN id(n) AS merged_id
                """,
                {"canonical_id": canonical_id, "props": combined_props},
            )

            # 2. Rewire relationships from other nodes to the canonical node.
            self._rewire_relationships(canonical_id, other_ids)

            # 3. Delete the other nodes.
            self.graph_store.execute_query(
                """
                MATCH (n)
                WHERE id(n) IN $other_ids
                DELETE n
                """,
                {"other_ids": other_ids},
            )

            merged_nodes += len(node_ids)

        return merged_nodes

    def _combine_properties(self, node_ids: list[int]) -> dict:
        rows = self.graph_store.execute_query(
            """
            MATCH (n)
            WHERE id(n) IN $node_ids
            RETURN id(n) AS node_id, properties(n) AS props
            """,
            {"node_ids": node_ids},
        )
        combined: dict = {}
        for row in rows:
            props = row.get("props", {}) or {}
            for key, value in props.items():
                if key not in combined:
                    combined[key] = value
                elif isinstance(combined[key], list) and isinstance(value, list):
                    for v in value:
                        if v not in combined[key]:
                            combined[key].append(v)
                elif isinstance(combined[key], list):
                    if value not in combined[key]:
                        combined[key].append(value)
                elif isinstance(value, list):
                    combined[key] = [combined[key]] + [
                        item for item in value if item != combined[key]
                    ]
                elif combined[key] != value:
                    combined[key] = [combined[key], value]
        return combined

    def _rewire_relationships(
        self, canonical_id: int, other_ids: list[int]
    ) -> None:
        merged_ids = [canonical_id] + other_ids
        # Find all distinct relationship types touching the nodes to be merged.
        type_rows = self.graph_store.execute_query(
            """
            MATCH (n)-[r]-(other)
            WHERE id(n) IN $other_ids
            RETURN DISTINCT type(r) AS rel_type, id(n) AS node_id
            """,
            {"other_ids": other_ids},
        )

        # Group by relationship type so we can create relationships with a
        # statically-known type in each query.
        types_by_node: dict = {}
        for row in type_rows:
            rel_type = row.get("rel_type")
            node_id = row.get("node_id")
            if rel_type and node_id is not None:
                types_by_node.setdefault(node_id, []).append(rel_type)

        # For each node and relationship type, copy outgoing/incoming
        # relationships to the canonical node and delete the originals.
        for node_id, rel_types in types_by_node.items():
            for rel_type in set(rel_types):
                escaped_rel = escape_cypher_identifier(rel_type)
                # Outgoing relationships
                self.graph_store.execute_query(
                    f"""
                    MATCH (source)-[r:{escaped_rel}]->(target)
                    WHERE id(source) = $node_id
                    WITH r, target, properties(r) AS rel_props
                    MATCH (canonical)
                    WHERE id(canonical) = $canonical_id
                    WITH r, canonical, target, rel_props
                    WHERE NOT id(target) IN $merged_ids
                    MERGE (canonical)-[new_r:{escaped_rel}]->(target)
                    SET new_r += rel_props
                    DELETE r
                    """,
                    {
                        "node_id": node_id,
                        "canonical_id": canonical_id,
                        "merged_ids": merged_ids,
                    },
                )
                self.graph_store.execute_query(
                    f"""
                    MATCH (source)-[r:{escaped_rel}]->(target)
                    WHERE id(source) = $node_id
                      AND id(target) IN $merged_ids
                    DELETE r
                    """,
                    {"node_id": node_id, "merged_ids": merged_ids},
                )
                # Incoming relationships
                self.graph_store.execute_query(
                    f"""
                    MATCH (source)-[r:{escaped_rel}]->(target)
                    WHERE id(target) = $node_id
                    WITH r, source, properties(r) AS rel_props
                    MATCH (canonical)
                    WHERE id(canonical) = $canonical_id
                    WITH r, source, canonical, rel_props
                    WHERE NOT id(source) IN $merged_ids
                    MERGE (source)-[new_r:{escaped_rel}]->(canonical)
                    SET new_r += rel_props
                    DELETE r
                    """,
                    {
                        "node_id": node_id,
                        "canonical_id": canonical_id,
                        "merged_ids": merged_ids,
                    },
                )
                self.graph_store.execute_query(
                    f"""
                    MATCH (source)-[r:{escaped_rel}]->(target)
                    WHERE id(target) = $node_id
                      AND id(source) IN $merged_ids
                    DELETE r
                    """,
                    {"node_id": node_id, "merged_ids": merged_ids},
                )
