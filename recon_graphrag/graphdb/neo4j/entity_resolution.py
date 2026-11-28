"""Neo4j-specific entity resolution implementation."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import cast

from recon_graphrag.entity_text import entity_domain_type
from recon_graphrag.graphdb._provenance import (
    build_uid,
    preflight_merge_ownership,
    raw_descriptions,
)
from recon_graphrag.graphdb.entity_resolution import (
    BaseEntityResolver,
    _EntityRecord,
    _first_property_value,
    _normalize_name,
)
from recon_graphrag.graphdb.cypher import escape_cypher_identifier
from recon_graphrag.graphdb.store_base import (
    WRITE_RETRY_ATTEMPTS as _MERGE_RETRY_ATTEMPTS,
    WRITE_RETRY_DELAY_SECONDS as _MERGE_RETRY_DELAY_SECONDS,
    QueryExecutor,
    is_retryable_write_error as _is_retryable_write_error,
)

logger = logging.getLogger(__name__)
_STRUCTURAL_REL_TYPES = ("FROM_CHUNK", "SUBJECT_OF", "IN_COMMUNITY")


class _Neo4jEntityResolver(BaseEntityResolver):
    def _preflight(self, *, dry_run: bool) -> dict | None:
        if dry_run:
            return None
        try:
            self.graph_store.execute_query("RETURN apoc.version() AS version")
        except Exception as exc:
            return {
                "skipped": True,
                "reason": f"APOC is unavailable: {exc}",
                "merged_groups": 0,
                "merged_nodes": 0,
                "candidate_groups": 0,
                "review_groups": [],
                "signals": {"apoc": "unavailable"},
            }
        return None

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
              elementId(e) AS node_id,
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
            domain_label = entity_domain_type(
                labels=labels, entity_type=row.get("properties", {}).get("type"),
                fallback="__Entity__",
            )
            resolve_value = row.get("resolve_value", "")
            entities.append(
                _EntityRecord(
                    node_id=row.get("node_id", ""),
                    entity_id=row.get("entity_id", ""),
                    graph_name=row.get("graph_name", ""),
                    domain_label=domain_label,
                    resolve_value=resolve_value,
                    normalized_value=_normalize_name(resolve_value),
                    properties=row.get("properties", {}),
                )
            )
        return entities

    def _merge_groups(
        self, groups: list[list[_EntityRecord]], resolve_property: str
    ) -> int:
        if not groups:
            return 0
        transaction = getattr(self.graph_store, "_transaction", None)
        if not callable(transaction):
            raise RuntimeError(  # noqa: TRY004 - missing store capability
                "Entity merging requires a store with _transaction() support; "
                "no nodes were merged."
            )

        begin = cast(Callable[[], AbstractContextManager[QueryExecutor]], transaction)
        preflight_merge_ownership(self.graph_store, groups, "elementId")
        merged_nodes = 0
        for group in groups:
            merged = self._merge_group_with_retry(group, resolve_property, begin)
            if merged:
                merged_nodes += len(group)
        return merged_nodes

    def _merge_group_with_retry(
        self,
        group: list[_EntityRecord],
        resolve_property: str,
        begin: Callable[[], AbstractContextManager[QueryExecutor]],
    ) -> bool:
        for attempt in range(1, _MERGE_RETRY_ATTEMPTS + 1):
            try:
                with begin() as executor:
                    return self._merge_group(group, resolve_property, executor)
            except Exception as error:
                if (
                    not _is_retryable_write_error(error)
                    or attempt == _MERGE_RETRY_ATTEMPTS
                ):
                    raise
                logger.warning(
                    "Neo4j merge group rolled back: graph_name=%s "
                    "canonical_entity_id=%s attempt=%s/%s error=%s: %s",
                    group[0].graph_name if group else "",
                    group[0].entity_id if group else "",
                    attempt,
                    _MERGE_RETRY_ATTEMPTS,
                    type(error).__name__,
                    error,
                )
                time.sleep(_MERGE_RETRY_DELAY_SECONDS * attempt)
        raise AssertionError("unreachable: retry loop exhausted without raising")

    def _merge_group(
        self,
        group: list[_EntityRecord],
        resolve_property: str,
        executor: QueryExecutor,
    ) -> bool:
        sorted_group = sorted(
            group,
            key=lambda e: len(e.resolve_value) if e.resolve_value else 0,
            reverse=True,
        )
        node_ids = [e.node_id for e in sorted_group]
        canonical_entity_id = sorted_group[0].entity_id if sorted_group else ""
        canonical_key = _first_property_value(
            sorted_group[0].properties.get("canonical_key")
            if sorted_group
            else None
        )
        canonical_readable_id = _first_property_value(
            sorted_group[0].properties.get("human_readable_id")
            if sorted_group
            else None
        )
        canonical_name = sorted_group[0].resolve_value if sorted_group else ""
        aliases = list(
            {
                e.resolve_value
                for e in group
                if e.resolve_value and e.resolve_value != canonical_name
            }
        )

        result = executor.execute_query(
            f"""
            MATCH (n)
            WHERE elementId(n) IN $node_ids
            OPTIONAL MATCH (n)-[r]-()
            WHERE NOT type(r) IN $structural_rel_types
            WITH n, r ORDER BY apoc.coll.indexOf($node_ids, elementId(n))
            WITH collect(DISTINCT n) AS nodes, collect(DISTINCT r) AS rels
            // APOC combine fails on empty arrays; null carries the same empty ownership.
            FOREACH (n IN nodes |
                SET n.descriptions = CASE WHEN n.descriptions = [] THEN null ELSE n.descriptions END,
                    n.description_sources = CASE WHEN n.description_sources = [] THEN null ELSE n.description_sources END)
            FOREACH (r IN rels |
                SET r.descriptions = CASE WHEN r.descriptions = [] THEN null ELSE r.descriptions END,
                    r.description_sources = CASE WHEN r.description_sources = [] THEN null ELSE r.description_sources END,
                    r.strength_sources = CASE WHEN r.strength_sources = [] THEN null ELSE r.strength_sources END,
                    r.source_chunk_ids = CASE WHEN r.source_chunk_ids = [] THEN null ELSE r.source_chunk_ids END)
            WITH nodes
            CALL apoc.refactor.mergeNodes(
                nodes,
                {{properties: {{uid: 'discard', id: 'discard', graph_name: 'discard',
                  title: 'discard', created: 'discard', updated: 'discard',
                  embedding_content_fingerprint: 'discard',
                  embedding_model_fingerprint: 'discard',
                  `.*`: 'combine'}},
                  mergeRels: true}}
            ) YIELD node
            SET node.{escape_cypher_identifier(resolve_property)} = $canonical_name,
                node.id = $canonical_entity_id,
                node.uid = $canonical_uid,
                node.graph_name = $canonical_graph_name,
                node.canonical_key = coalesce($canonical_key, node.canonical_key),
                node.human_readable_id = coalesce($canonical_readable_id, node.human_readable_id),
                node.embedding = null,
                node.embedding_input_fingerprint = null,
                node.embedding_content_fingerprint = null,
                node.embedding_model_fingerprint = null
            WITH node
            // Claim objects are scalar IDs, not relationships rewired by APOC.
            OPTIONAL MATCH (c:Claim {{graph_name: $canonical_graph_name}})
            WHERE c.object_entity_id IN $losing_entity_ids
            WITH node, collect(c) AS claims
            FOREACH (c IN claims | SET c.object_entity_id = $canonical_entity_id)
            WITH node
            OPTIONAL MATCH (node)-[self_r]->(node)
            WHERE NOT type(self_r) IN $structural_rel_types
            WITH node, collect(DISTINCT self_r) AS self_rels
            FOREACH (sr IN self_rels | DELETE sr)
            RETURN elementId(node) AS merged_id
            """,
            {
                "node_ids": node_ids,
                "structural_rel_types": list(_STRUCTURAL_REL_TYPES),
                "canonical_entity_id": canonical_entity_id,
                "losing_entity_ids": [e.entity_id for e in sorted_group[1:]],
                "canonical_uid": build_uid(sorted_group[0].graph_name, canonical_entity_id),
                "canonical_graph_name": sorted_group[0].graph_name,
                "canonical_key": canonical_key,
                "canonical_readable_id": canonical_readable_id,
                "canonical_name": canonical_name,
            },
        )
        if result:
            merged_id = result[0].get("merged_id")
            if merged_id:
                # APOC can unwrap singleton provenance on nodes and domain edges.
                executor.execute_query(
                    """
                    MATCH (n) WHERE elementId(n) = $node_id
                    SET n.description_sources = apoc.convert.toList(n.description_sources)
                    WITH n
                    MATCH (n)-[r]-()
                    WHERE NOT type(r) IN $structural_rel_types
                    SET r.source_chunk_ids = apoc.convert.toList(r.source_chunk_ids),
                        r.description_sources = apoc.convert.toList(r.description_sources),
                        r.strength_sources = apoc.convert.toList(r.strength_sources)
                    WITH DISTINCT r,
                         apoc.coll.toSet(apoc.convert.toList(r.source_chunk_ids)) AS chunks,
                         apoc.coll.toSet(CASE
                           WHEN r.descriptions IS NOT NULL
                           THEN apoc.convert.toList(r.descriptions)
                           ELSE apoc.convert.toList(r.description) END) AS descriptions
                    SET r.source_chunk_ids = chunks,
                        r.observation_count = size(chunks),
                        r.weight = toFloat(size(chunks)),
                        r.strength = apoc.coll.max(apoc.convert.toList(r.strength)),
                        r.descriptions = descriptions,
                        r.description = apoc.text.join(descriptions, '\n'),
                        r.description_summary_status = null,
                        r.description_input_fingerprint = null,
                        r.description_summary_error = null,
                        r.description_summary_updated = null,
                        r.description_summary_fallback = true
                    """,
                    {
                        "node_id": merged_id,
                        "structural_rel_types": list(_STRUCTURAL_REL_TYPES),
                    },
                )
            summary = self._merge_summaries.get(canonical_entity_id)
            if merged_id and summary:
                executor.execute_query(
                    """
                    MATCH (n) WHERE elementId(n) = $node_id
                    SET n.descriptions = $descriptions,
                        n.description = $description,
                        n.embedding_input_fingerprint = null,
                        n.observation_count = size($descriptions),
                        n.description_input_fingerprint = null,
                        n.description_summary_status = null,
                        n.description_summary_error = null,
                        n.description_summary_updated = null,
                        n.description_summary_fallback = $fallback
                    """,
                    {
                        "node_id": merged_id,
                        "descriptions": list(dict.fromkeys(
                            text for entity in sorted_group
                            for text in raw_descriptions(entity.properties)
                            if text
                        )),
                        "description": summary["description"],
                        "fallback": summary["description_summary_fallback"],
                    },
                )
            if merged_id and aliases:
                executor.execute_query(
                    """
                    MATCH (n)
                    WHERE elementId(n) = $node_id
                    SET n.aliases = coalesce(n.aliases, []) + $aliases
                    RETURN n
                    """,
                    {"node_id": merged_id, "aliases": aliases},
                )
        return bool(result)
