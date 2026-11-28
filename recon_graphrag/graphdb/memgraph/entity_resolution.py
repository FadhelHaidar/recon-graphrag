"""Memgraph-specific entity resolution implementation."""

from __future__ import annotations

import logging
import time
from copy import copy
from dataclasses import replace

from recon_graphrag.entity_text import entity_domain_type
from recon_graphrag.graphdb._provenance import (
    build_uid,
    preflight_merge_ownership,
    property_list,
    raw_descriptions,
)
from recon_graphrag.graphdb.cypher import escape_cypher_identifier
from recon_graphrag.graphdb.entity_resolution import (
    BaseEntityResolver,
    _EntityRecord,
    _first_property_value,
    _normalize_name,
)
from recon_graphrag.graphdb.store_base import (
    WRITE_RETRY_ATTEMPTS as _MERGE_RETRY_ATTEMPTS,
)
from recon_graphrag.graphdb.store_base import (
    WRITE_RETRY_DELAY_SECONDS as _MERGE_RETRY_DELAY_SECONDS,
)
from recon_graphrag.graphdb.store_base import (
    is_retryable_write_error as _is_retryable_write_error,
)

logger = logging.getLogger(__name__)
_STRUCTURAL_REL_TYPES = frozenset({"FROM_CHUNK", "SUBJECT_OF", "IN_COMMUNITY"})
_SCALAR_BOOKKEEPING_KEYS = ("title", "created", "updated")


class _MemgraphEntityResolver(BaseEntityResolver):
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
            domain_label = entity_domain_type(
                labels=labels, entity_type=row.get("properties", {}).get("type"),
                fallback="__Entity__",
            )
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

    def _merge_groups(
        self, groups: list[list[_EntityRecord]], resolve_property: str
    ) -> int:
        transaction = getattr(self.graph_store, "_transaction", None)
        if not callable(transaction):
            raise RuntimeError(  # noqa: TRY004 - missing store capability
                "Entity merging requires a store with _transaction() support; "
                "no nodes were merged."
            )

        # Check every group against the current ownership rules before the
        # first group writes. Each attempt still rechecks its own group, but
        # this preflight does not make the complete run atomic.
        preflight_merge_ownership(self.graph_store, groups, "id")

        merged_nodes = 0
        for group in groups:
            sorted_group = sorted(
                group,
                key=lambda e: len(e.resolve_value) if e.resolve_value else 0,
                reverse=True,
            )
            if len(sorted_group) < 2:
                continue
            if self._merge_group(sorted_group, resolve_property, transaction):
                merged_nodes += len(sorted_group)

        return merged_nodes

    def _merge_group(
        self,
        group: list[_EntityRecord],
        resolve_property: str,
        transaction,
    ) -> bool:
        """Merge one group, re-reading state in each fresh transaction attempt."""
        for attempt in range(1, _MERGE_RETRY_ATTEMPTS + 1):
            try:
                with transaction() as executor:
                    scoped = copy(self)
                    scoped.graph_store = executor
                    merged = scoped._merge_group_once(group, resolve_property)
                return merged
            except Exception as error:
                if (
                    not _is_retryable_write_error(error)
                    or attempt == _MERGE_RETRY_ATTEMPTS
                ):
                    raise
                logger.warning(
                    "merge group rolled back: graph_name=%s "
                    "canonical_entity_id=%s attempt=%s/%s error=%s: %s",
                    group[0].graph_name,
                    group[0].entity_id,
                    attempt,
                    _MERGE_RETRY_ATTEMPTS,
                    type(error).__name__,
                    error,
                )
                time.sleep(_MERGE_RETRY_DELAY_SECONDS * attempt)
        raise AssertionError("unreachable: retry loop exhausted without raising")

    def _merge_group_once(
        self, group: list[_EntityRecord], resolve_property: str
    ) -> bool:
        """Merge one group inside an open transaction, or skip it when stale."""
        node_ids = [entity.node_id for entity in group]
        rows = self.graph_store.execute_query(
            """
            MATCH (n:__Entity__)
            WHERE id(n) IN $node_ids AND n.graph_name = $graph_name
            RETURN id(n) AS node_id, properties(n) AS props
            """,
            {"node_ids": node_ids, "graph_name": group[0].graph_name},
        )
        fresh_group = self._fresh_group(group, rows, resolve_property)
        if fresh_group is None:
            return False

        # Run the ownership preflight on the fresh transaction state, before
        # any mutation. A retry repeats this read and the preflight.
        preflight_merge_ownership(self.graph_store, [fresh_group], "id")

        self._merge_group_records(fresh_group, resolve_property)
        return True

    def _fresh_group(
        self,
        group: list[_EntityRecord],
        rows: list[dict],
        resolve_property: str,
    ) -> list[_EntityRecord] | None:
        expected = {entity.node_id: entity for entity in group}
        rows_by_id = {
            row.get("node_id"): row.get("props") or {}
            for row in rows
        }
        missing = sorted(set(expected) - set(rows_by_id), key=str)
        if missing:
            logger.warning(
                "skipping stale merge group: graph_name=%s "
                "canonical_entity_id=%s missing_node_ids=%s",
                group[0].graph_name,
                group[0].entity_id,
                missing,
            )
            return None
        unexpected = sorted(set(rows_by_id) - set(expected), key=str)
        if unexpected:
            logger.warning(
                "skipping stale merge group: graph_name=%s "
                "canonical_entity_id=%s unexpected_node_ids=%s",
                group[0].graph_name,
                group[0].entity_id,
                unexpected,
            )
            return None

        fresh_group = []
        for entity in group:
            props = rows_by_id[entity.node_id]
            if (
                props.get("id") != entity.entity_id
                or props.get("graph_name") != entity.graph_name
            ):
                logger.warning(
                    "skipping stale merge group: graph_name=%s "
                    "canonical_entity_id=%s node_id=%s",
                    group[0].graph_name,
                    group[0].entity_id,
                    entity.node_id,
                )
                return None
            resolve_value = _first_property_value(props.get(resolve_property))
            resolve_value = "" if resolve_value is None else str(resolve_value)
            fresh_group.append(
                replace(
                    entity,
                    resolve_value=resolve_value,
                    normalized_value=_normalize_name(resolve_value),
                    properties=props,
                )
            )
        return fresh_group

    def _merge_group_records(
        self, sorted_group: list[_EntityRecord], resolve_property: str
    ) -> None:
        node_ids = [e.node_id for e in sorted_group]
        canonical = sorted_group[0]
        canonical_entity_id = canonical.entity_id
        canonical_key = _first_property_value(
            canonical.properties.get("canonical_key")
        )
        canonical_readable_id = _first_property_value(
            canonical.properties.get("human_readable_id")
        )
        canonical_name = canonical.resolve_value
        aliases = list(
            {
                e.resolve_value
                for e in sorted_group
                if e.resolve_value and e.resolve_value != canonical_name
            }
        )

        canonical_id = node_ids[0]
        other_ids = node_ids[1:]
        if not other_ids:
            return

        combined_props = self._combine_property_rows(
            [entity.properties for entity in sorted_group]
        )
        for scalar_key in _SCALAR_BOOKKEEPING_KEYS:
            if scalar_key in combined_props:
                canonical_scalar = _first_property_value(
                    canonical.properties.get(scalar_key)
                )
                combined_props[scalar_key] = (
                    canonical_scalar
                    if canonical_scalar is not None
                    else _first_property_value(combined_props[scalar_key])
                )
        combined_props["id"] = canonical_entity_id
        combined_props["graph_name"] = canonical.graph_name
        combined_props["uid"] = build_uid(canonical.graph_name, canonical_entity_id)
        combined_props["description_sources"] = property_list(combined_props.get("description_sources"))
        # Drop any (now-stale, possibly corrupted) merged embedding so the
        # embedding step re-embeds the merged node from its consolidated
        # description. SET n += {embedding: null} removes the property.
        combined_props["embedding"] = None
        combined_props["embedding_input_fingerprint"] = None
        combined_props["embedding_content_fingerprint"] = None
        combined_props["embedding_model_fingerprint"] = None
        if canonical_key:
            combined_props["canonical_key"] = canonical_key
        if canonical_readable_id:
            combined_props["human_readable_id"] = canonical_readable_id
        combined_props[resolve_property] = canonical_name
        summary = self._merge_summaries.get(canonical_entity_id)
        if summary:
            combined_props.update(summary)
            combined_props["descriptions"] = list(dict.fromkeys(
                text for entity in sorted_group
                for text in raw_descriptions(entity.properties)
                if text
            ))
            combined_props["observation_count"] = len(combined_props["descriptions"])
        combined_props.update({
            "description_summary_status": None,
            "description_input_fingerprint": None,
            "description_summary_error": None,
            "description_summary_updated": None,
        })
        if aliases:
            existing_aliases = combined_props.get("aliases", [])
            if not isinstance(existing_aliases, list):
                existing_aliases = [existing_aliases]
            combined_props["aliases"] = list(
                dict.fromkeys(existing_aliases + aliases)
            )

        written = self.graph_store.execute_query(
            """
            MATCH (n)
            WHERE id(n) = $canonical_id
            SET n += $props
            RETURN id(n) AS merged_id
            """,
            {"canonical_id": canonical_id, "props": combined_props},
        )
        if len(written) != 1:
            raise RuntimeError(
                "Merge invariant violated: canonical entity "
                f"{canonical_entity_id!r} in graph {canonical.graph_name!r} "
                "was not updated; the merge group is stale."
            )

        self._rewire_relationships(canonical_id, other_ids)

        # Claim objects are scalar IDs, not relationships rewired above.
        # Keep the retarget before the loser delete; the delete removes only
        # entity nodes and does not touch Claim nodes.
        self.graph_store.execute_query(
            """
            MATCH (c:Claim {graph_name: $graph_name})
            WHERE c.object_entity_id IN $losing_entity_ids
            SET c.object_entity_id = $canonical_entity_id
            """,
            {
                "graph_name": canonical.graph_name,
                "losing_entity_ids": [e.entity_id for e in sorted_group[1:]],
                "canonical_entity_id": canonical_entity_id,
            },
        )

        self._delete_losers(canonical.graph_name, other_ids)

    def _delete_losers(self, graph_name: str, other_ids: list[int]) -> None:
        rows = self.graph_store.execute_query(
            """
            MATCH (n)
            WHERE id(n) IN $other_ids
            OPTIONAL MATCH (n)-[r]-()
            WITH n, count(r) AS degree
            WHERE degree = 0
            DELETE n
            RETURN count(*) AS deleted
            """,
            {"other_ids": other_ids},
        )
        deleted = rows[0].get("deleted", 0) if rows else 0
        if deleted != len(other_ids):
            raise RuntimeError(
                "Merge invariant violated: expected to delete "
                f"{len(other_ids)} merged entities in graph {graph_name!r}, "
                f"deleted {deleted}; concurrent writes or residual "
                "relationships were detected."
            )

    def _combine_properties(self, node_ids: list[int]) -> dict:
        # Union semantics matching Neo4j apoc.refactor.mergeNodes({properties:'combine'}):
        # conflicting scalar values across merged nodes collapse into a list, so a
        # custom scalar prop (e.g. occupation) can become multi-valued after a merge.
        rows = self.graph_store.execute_query(
            """
            MATCH (n)
            WHERE id(n) IN $node_ids
            RETURN id(n) AS node_id, properties(n) AS props
            """,
            {"node_ids": node_ids},
        )
        return self._combine_property_rows(
            [row.get("props", {}) or {} for row in rows]
        )

    @staticmethod
    def _combine_property_rows(rows: list[dict]) -> dict:
        combined: dict = {}
        for props in rows:
            for key, value in props.items():
                if key not in combined:
                    combined[key] = list(value) if isinstance(value, list) else value
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
        type_rows = self.graph_store.execute_query(
            """
            MATCH (n)-[r]-(other)
            WHERE id(n) IN $other_ids
            RETURN DISTINCT type(r) AS rel_type, id(n) AS node_id
            """,
            {"other_ids": other_ids},
        )

        types_by_node: dict = {}
        for row in type_rows:
            rel_type = row.get("rel_type")
            node_id = row.get("node_id")
            if rel_type and node_id is not None:
                types_by_node.setdefault(node_id, []).append(rel_type)

        for node_id, rel_types in types_by_node.items():
            for rel_type in set(rel_types):
                escaped_rel = escape_cypher_identifier(rel_type)
                if rel_type in _STRUCTURAL_REL_TYPES:
                    self._rewire_structural_relationships(
                        node_id=node_id,
                        canonical_id=canonical_id,
                        merged_ids=merged_ids,
                        escaped_rel=escaped_rel,
                    )
                    continue
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
                    WITH r, new_r, rel_props,
                         new_r.id AS existing_id,
                         new_r.created AS existing_created,
                         new_r.canonical_key AS existing_canonical_key,
                         coalesce(new_r.source_chunk_ids, []) AS existing_chunks,
                         coalesce(new_r.descriptions,
                           CASE WHEN new_r.description IS NULL THEN []
                                ELSE [new_r.description] END) AS existing_descriptions,
                         coalesce(new_r.description_sources, []) AS existing_description_sources,
                         coalesce(new_r.strength_sources, []) AS existing_strength_sources,
                         new_r.strength AS existing_strength
                    SET new_r += rel_props
                    WITH r, new_r, rel_props, existing_id, existing_created, existing_canonical_key,
                         existing_strength,
                         existing_description_sources, existing_strength_sources,
                         reduce(acc = [], item IN existing_chunks +
                           coalesce(rel_props.source_chunk_ids, []) |
                           CASE WHEN item IN acc THEN acc ELSE acc + [item] END
                         ) AS merged_chunks,
                         reduce(acc = [], item IN existing_descriptions +
                           coalesce(rel_props.descriptions,
                             CASE WHEN rel_props.description IS NULL THEN []
                                  ELSE [rel_props.description] END) |
                           CASE WHEN item IN acc THEN acc ELSE acc + [item] END
                         ) AS merged_descriptions
                    SET new_r.description_sources = reduce(
                          acc = [], item IN existing_description_sources + coalesce(rel_props.description_sources, []) |
                          CASE WHEN item IN acc THEN acc ELSE acc + [item] END),
                        new_r.strength_sources = reduce(
                          acc = [], item IN existing_strength_sources + coalesce(rel_props.strength_sources, []) |
                          CASE WHEN item IN acc THEN acc ELSE acc + [item] END),
                        new_r.id = coalesce(existing_id, rel_props.id),
                        new_r.created = coalesce(existing_created, rel_props.created),
                        new_r.canonical_key = coalesce(existing_canonical_key, rel_props.canonical_key),
                        new_r.source_chunk_ids = merged_chunks,
                        new_r.observation_count = size(merged_chunks),
                        new_r.weight = toFloat(size(merged_chunks)),
                        new_r.descriptions = merged_descriptions,
                        new_r.description = reduce(text = '', item IN merged_descriptions |
                          text + CASE WHEN text = '' THEN '' ELSE '\n' END + item),
                        new_r.description_summary_status = null,
                        new_r.description_input_fingerprint = null,
                        new_r.description_summary_error = null,
                        new_r.description_summary_updated = null,
                        new_r.description_summary_fallback = true,
                        new_r.strength = CASE
                          WHEN existing_strength IS NULL THEN rel_props.strength
                          WHEN rel_props.strength IS NULL THEN existing_strength
                          WHEN existing_strength >= rel_props.strength THEN existing_strength
                          ELSE rel_props.strength END
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
                    WITH r, new_r, rel_props,
                         new_r.id AS existing_id,
                         new_r.created AS existing_created,
                         new_r.canonical_key AS existing_canonical_key,
                         coalesce(new_r.source_chunk_ids, []) AS existing_chunks,
                         coalesce(new_r.descriptions,
                           CASE WHEN new_r.description IS NULL THEN []
                                ELSE [new_r.description] END) AS existing_descriptions,
                         coalesce(new_r.description_sources, []) AS existing_description_sources,
                         coalesce(new_r.strength_sources, []) AS existing_strength_sources,
                         new_r.strength AS existing_strength
                    SET new_r += rel_props
                    WITH r, new_r, rel_props, existing_id, existing_created, existing_canonical_key,
                         existing_strength,
                         existing_description_sources, existing_strength_sources,
                         reduce(acc = [], item IN existing_chunks +
                           coalesce(rel_props.source_chunk_ids, []) |
                           CASE WHEN item IN acc THEN acc ELSE acc + [item] END
                         ) AS merged_chunks,
                         reduce(acc = [], item IN existing_descriptions +
                           coalesce(rel_props.descriptions,
                             CASE WHEN rel_props.description IS NULL THEN []
                                  ELSE [rel_props.description] END) |
                           CASE WHEN item IN acc THEN acc ELSE acc + [item] END
                         ) AS merged_descriptions
                    SET new_r.description_sources = reduce(
                          acc = [], item IN existing_description_sources + coalesce(rel_props.description_sources, []) |
                          CASE WHEN item IN acc THEN acc ELSE acc + [item] END),
                        new_r.strength_sources = reduce(
                          acc = [], item IN existing_strength_sources + coalesce(rel_props.strength_sources, []) |
                          CASE WHEN item IN acc THEN acc ELSE acc + [item] END),
                        new_r.id = coalesce(existing_id, rel_props.id),
                        new_r.created = coalesce(existing_created, rel_props.created),
                        new_r.canonical_key = coalesce(existing_canonical_key, rel_props.canonical_key),
                        new_r.source_chunk_ids = merged_chunks,
                        new_r.observation_count = size(merged_chunks),
                        new_r.weight = toFloat(size(merged_chunks)),
                        new_r.descriptions = merged_descriptions,
                        new_r.description = reduce(text = '', item IN merged_descriptions |
                          text + CASE WHEN text = '' THEN '' ELSE '\n' END + item),
                        new_r.description_summary_status = null,
                        new_r.description_input_fingerprint = null,
                        new_r.description_summary_error = null,
                        new_r.description_summary_updated = null,
                        new_r.description_summary_fallback = true,
                        new_r.strength = CASE
                          WHEN existing_strength IS NULL THEN rel_props.strength
                          WHEN rel_props.strength IS NULL THEN existing_strength
                          WHEN existing_strength >= rel_props.strength THEN existing_strength
                          ELSE rel_props.strength END
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

    def _rewire_structural_relationships(
        self,
        *,
        node_id: int,
        canonical_id: int,
        merged_ids: list[int],
        escaped_rel: str,
    ) -> None:
        """Move structural edges without adding domain-relationship properties."""
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
