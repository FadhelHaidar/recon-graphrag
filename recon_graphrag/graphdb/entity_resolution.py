"""Backend-neutral entity resolution grouping and review logic."""

from __future__ import annotations

import json
import hashlib
import math
import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from recon_graphrag.graphdb.entity_resolution_context import (
    blocked_review_group,
    build_entity_profiles,
    conflict_for_group,
)


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


def _first_property_value(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value


@dataclass
class _EntityRecord:
    node_id: Any
    entity_id: str
    graph_name: str
    domain_label: str
    resolve_value: str
    normalized_value: str
    properties: dict = field(default_factory=dict)


class BaseEntityResolver(ABC):
    """Shared resolution strategy logic.

    Backend subclasses provide entity loading and merge execution, because those
    depend on node ID semantics, Cypher functions, and merge procedures.
    """

    def __init__(self, graph_store):
        self.graph_store = graph_store
        self._merge_summaries: dict[str, dict] = {}

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
        context_properties: Optional[dict[str, list[str]] | list[str]] = None,
        conflict_properties: Optional[dict[str, list[str]] | list[str]] = None,
        context_mode: str = "safe_defaults",
    ) -> dict:
        if strategy not in ("exact", "normalized", "fuzzy", "hybrid"):
            raise ValueError(f"Unknown strategy: {strategy}")

        preflight_result = self._preflight(dry_run=dry_run)
        if preflight_result is not None:
            return preflight_result

        entities = self._load_entities(graph_name, resolve_property)

        ai_merged_review_groups = []

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
        else:
            (
                groups,
                review_groups,
                ai_merged_review_groups,
            ) = await self._build_hybrid_groups(
                entities,
                merge_threshold=merge_threshold,
                review_threshold=review_threshold,
                max_candidates_per_entity=max_candidates_per_entity,
                aliases=aliases,
                embedder=embedder,
                llm=llm,
                llm_guidance=llm_guidance,
                allow_ai_auto_merge=allow_ai_auto_merge,
                context_properties=context_properties,
                conflict_properties=conflict_properties,
                context_mode=context_mode,
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
            self._merge_summaries = await self._prepare_merge_summaries(groups, llm)
            merged_nodes = self._merge_groups(groups, resolve_property)

        return {
            "skipped": False,
            "strategy": strategy,
            "merged_groups": len(groups),
            "merged_nodes": merged_nodes,
            "candidate_groups": len(groups) + len(review_groups),
            "review_groups": review_groups,
            "ai_merged_review_groups": ai_merged_review_groups,
            "signals": signals,
        }

    async def _prepare_merge_summaries(
        self, groups: list[list[_EntityRecord]], llm
    ) -> dict[str, dict]:
        """Normalize observations and optionally summarize merged descriptions."""
        summaries: dict[str, dict] = {}
        for group in groups:
            canonical = max(
                group,
                key=lambda entity: len(entity.resolve_value or ""),
            )
            observations: list[str] = []
            for entity in group:
                values = entity.properties.get("descriptions", [])
                if not isinstance(values, list):
                    values = [values]
                description = entity.properties.get("description")
                if description:
                    values.append(description)
                for value in values:
                    text = str(value).strip()
                    if text and text not in observations:
                        observations.append(text)
            deterministic = "\n".join(observations)
            fingerprint = hashlib.sha256(
                json.dumps(observations, ensure_ascii=True).encode("utf-8")
            ).hexdigest()
            summary = deterministic
            fallback = False
            if llm is not None and len(observations) > 1:
                prompt = (
                    "Consolidate these unique observations into one factual entity "
                    "description. Preserve all non-conflicting facts and return only "
                    f"the description.\n\n{deterministic}"
                )
                try:
                    response = await llm.ainvoke(prompt)
                    summary = response.content.strip()
                    if not summary:
                        raise ValueError("empty entity description summary")
                except Exception:
                    summary = deterministic
                    fallback = True
            summaries[canonical.entity_id] = {
                "descriptions": observations,
                "description": summary,
                "description_input_fingerprint": fingerprint,
                "description_summary_fallback": fallback,
            }
        return summaries

    def _preflight(self, *, dry_run: bool) -> dict | None:
        return None

    @abstractmethod
    def _load_entities(
        self, graph_name: str, resolve_property: str
    ) -> list[_EntityRecord]:
        raise NotImplementedError

    @abstractmethod
    def _merge_groups(
        self, groups: list[list[_EntityRecord]], resolve_property: str
    ) -> int:
        raise NotImplementedError

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
        reviewed_pairs: set[frozenset] = set()

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
                            "node_ids": [e1.node_id, e2.node_id],
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
        used: set,
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
        context_properties: Optional[dict[str, list[str]] | list[str]],
        conflict_properties: Optional[dict[str, list[str]] | list[str]],
        context_mode: str,
    ) -> tuple[list[list[_EntityRecord]], list[dict], list[dict]]:
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

        groups, review_groups = self._apply_conflict_rules(
            groups,
            review_groups,
            entities,
            conflict_properties,
        )
        active_review_groups = [
            rg for rg in review_groups if rg.get("decision") != "blocked"
        ]

        if embedder and active_review_groups:
            active_review_groups = await self._score_with_embeddings(
                active_review_groups, embedder, allow_ai_auto_merge, merge_threshold
            )

        if llm and active_review_groups:
            active_review_groups = await self._llm_review(
                active_review_groups,
                llm,
                llm_guidance,
                aliases,
                allow_ai_auto_merge,
                merge_threshold,
                entities,
                context_properties,
                context_mode,
            )
        review_groups = [
            *active_review_groups,
            *[rg for rg in review_groups if rg.get("decision") == "blocked"],
        ]

        ai_merged_review_groups = []
        if allow_ai_auto_merge and review_groups:
            ai_groups, review_groups, ai_merged_review_groups = self._promote_ai_reviews(
                review_groups=review_groups,
                entities=entities,
                merged_ids=merged_ids,
                min_confidence=merge_threshold / 100,
            )
            groups.extend(ai_groups)

        return groups, review_groups, ai_merged_review_groups

    def _apply_conflict_rules(
        self,
        groups: list[list[_EntityRecord]],
        review_groups: list[dict],
        entities: list[_EntityRecord],
        conflict_properties: Optional[dict[str, list[str]] | list[str]],
    ) -> tuple[list[list[_EntityRecord]], list[dict]]:
        if not conflict_properties:
            return groups, review_groups

        clean_groups = []
        blocked_groups = []
        for group in groups:
            conflicts = conflict_for_group(group, conflict_properties)
            if conflicts:
                blocked_groups.append(blocked_review_group(group, conflicts))
            else:
                clean_groups.append(group)

        entity_by_node_id = {e.node_id: e for e in entities}
        clean_reviews = []
        for review_group in review_groups:
            group = [
                entity_by_node_id[node_id]
                for node_id in review_group.get("node_ids", [])
                if node_id in entity_by_node_id
            ]
            conflicts = conflict_for_group(group, conflict_properties)
            if conflicts:
                review_group["decision"] = "blocked"
                review_group["reason"] = "property_conflict"
                review_group["conflicts"] = conflicts
                review_group["llm_review"] = {
                    "same_entity": False,
                    "confidence": 1.0,
                    "reason": "Configured conflict properties differ.",
                    "merge_allowed": False,
                }
            clean_reviews.append(review_group)

        return clean_groups, clean_reviews + blocked_groups

    def _promote_ai_reviews(
        self,
        review_groups: list[dict],
        entities: list[_EntityRecord],
        merged_ids: set,
        min_confidence: float,
    ) -> tuple[list[list[_EntityRecord]], list[dict], list[dict]]:
        entity_by_node_id = {e.node_id: e for e in entities}
        promoted_groups: list[list[_EntityRecord]] = []
        remaining_reviews: list[dict] = []
        promoted_reviews: list[dict] = []

        for rg in review_groups:
            llm_review = rg.get("llm_review") or {}
            node_ids = rg.get("node_ids") or []
            confidence = llm_review.get("confidence")
            try:
                confidence_value = float(confidence)
            except (TypeError, ValueError):
                confidence_value = 0.0

            can_merge = (
                llm_review.get("same_entity") is True
                and llm_review.get("merge_allowed") is True
                and confidence_value >= min_confidence
                and len(node_ids) >= 2
                and all(node_id not in merged_ids for node_id in node_ids)
            )
            if not can_merge:
                remaining_reviews.append(rg)
                continue

            group = [
                entity_by_node_id[node_id]
                for node_id in node_ids
                if node_id in entity_by_node_id
            ]
            if len(group) < 2:
                remaining_reviews.append(rg)
                continue

            for entity in group:
                merged_ids.add(entity.node_id)
            rg["decision"] = "merge"
            rg["reason"] = "llm_auto_merge"
            promoted_groups.append(group)
            promoted_reviews.append(rg)

        return promoted_groups, remaining_reviews, promoted_reviews

    def _build_alias_groups(
        self,
        entities: list[_EntityRecord],
        merged_ids: set,
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
        entities: list[_EntityRecord],
        context_properties: Optional[dict[str, list[str]] | list[str]],
        context_mode: str,
    ) -> list[dict]:
        entity_by_node_id = {e.node_id: e for e in entities}
        for rg in review_groups:
            rg["entities"] = build_entity_profiles(
                entity_by_node_id,
                rg.get("node_ids", []),
                context_properties=context_properties,
                context_mode=context_mode,
            )
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
                "Decide whether the candidate entity profiles refer to the same "
                "real-world entity in this graph context. Compare names, labels, "
                "descriptions, aliases, and supplied properties. Be conservative. "
                "Return only a "
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
