"""Shared graph store helpers.

Concrete backends keep their own packages and dialect-specific operations.
This base only contains backend-neutral Cypher-shaped helpers whose return
shape is part of the shared GraphStore contract.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, LiteralString, Protocol, cast
from uuid import uuid4

import neo4j

from recon_graphrag.graphdb.cypher import escape_cypher_identifier
from recon_graphrag.models.artifacts import (
    CommunityReport,
    report_to_json,
    report_to_text,
)
from recon_graphrag.report_text import report_embedding_fingerprint

# A failed write runs in its own transaction, so a deadlock or serialization
# abort rolls back cleanly and the same write can be retried. Bounded, because
# a persistent abort must surface.
WRITE_RETRY_ATTEMPTS = 3
WRITE_RETRY_DELAY_SECONDS = 0.05

# A graph-scoped filtered search retries once at these widths when the first
# read returns fewer distinct matches than the store target. The retry reads
# the full cap so duplicate identities cannot hide later scoped matches; the
# caller trims the deduplicated result to its target. The retry never widens a
# read that already used the cap.
_ENTITY_SEARCH_EXPANSION_CAP = 1000
_COMMUNITY_REPORT_SEARCH_EXPANSION_CAP = 500

# A wider search read can fall back to its first-round rows only when the
# database or driver reports the failure. Other errors report a defect and
# must propagate. Both backends use the Neo4j driver over Bolt.
_SEARCH_RETRY_ERRORS = (
    neo4j.exceptions.Neo4jError,
    neo4j.exceptions.DriverError,
)

# The report, vector, and failure writes set a temporary Community property
# first to acquire a write lock before their generation and text checks. The
# same statement removes the property, so a committed graph never keeps the
# token. The property name is __recon_community_write_lock.

# Selection may omit the generation predicate. Guarded writes never do.
_GENERATION_UNGUARDED: object = object()


def _bound_generation(expected_generation: object) -> str | None:
    """Reject the selection wildcard on a guarded community write."""
    if expected_generation is _GENERATION_UNGUARDED:
        raise ValueError("guarded community writes require an explicit generation")
    return cast(str | None, expected_generation)


def _affected_count(rows: list[dict], key: str) -> int:
    """Read one aggregate count from a write result."""
    if not rows:
        return 0
    return int(rows[0].get(key) or 0)


def _unique_search_row_count(
    rows: list[dict], key_fields: tuple[str, ...]
) -> int:
    """Count distinct row identities after the store applied its scope."""
    return len({tuple(row.get(field) for field in key_fields) for row in rows})


def _should_expand_search(
    rows: list[dict],
    *,
    target: int,
    initial_width: int,
    cap: int,
    key_fields: tuple[str, ...],
    graph_scoped: bool,
) -> bool:
    """Return True when one graph-scoped wider read can meet the target."""
    if not graph_scoped or not 0 < initial_width < cap:
        return False
    return _unique_search_row_count(rows, key_fields) < target


def _dedupe_search_rows(
    rows: list[dict], key_fields: tuple[str, ...]
) -> list[dict]:
    """Keep the best native score per identity and order by score.

    Equal scores use identity order, so ties do not depend on database row order.
    """
    best: dict[tuple, dict] = {}
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        current = best.get(key)
        if current is None or _search_row_score(row) > _search_row_score(current):
            best[key] = row
    return sorted(
        best.values(),
        key=lambda row: (
            -_search_row_score(row),
            tuple(row.get(field) for field in key_fields),
        ),
    )


def _search_row_score(row: dict) -> float:
    return float(row.get("score") or 0.0)


def is_retryable_write_error(error: BaseException) -> bool:
    """Return True for an abort where retrying the same write can succeed.

    A unique-constraint failure on ``uid`` is included: two concurrent writers
    MERGE the same entity that does not exist yet, both create it, and the
    loser's rollback leaves the winner's node in place for the retry to match.
    """
    code = str(getattr(error, "code", "") or "")
    if "DeadlockDetected" in code or "SerializationError" in code:
        return True
    if "ConstraintValidationFailed" in code:
        return True
    if "unique constraint violation" in str(error).lower():
        return True
    return type(error).__name__ in {
        "DeadlockDetected",
        "TransientError",
        "SerializationError",
        "TransientTransactionError",
    }


class TransactionExecutor:
    """Run ``execute_query`` on one open driver transaction.

    The writer and entity resolver only need ``execute_query``, so this can
    stand in for the store for the duration of one transaction; the owning
    ``_transaction`` context commits or rolls it back.
    """

    def __init__(self, transaction) -> None:
        self._transaction = transaction

    def execute_query(
        self, query: str, parameters: dict | None = None
    ) -> list[dict]:
        # Driver typing narrows to LiteralString; queries are built from escaped
        # identifiers in trusted code, exactly as in the store's execute_query.
        result = self._transaction.run(cast(LiteralString, query), parameters or {})
        return [dict(record) for record in result]


class QueryExecutor(Protocol):
    """Minimal write surface shared by a graph store and one transaction."""

    def execute_query(
        self, query: str, parameters: dict | None = None
    ) -> list[dict]: ...


@contextmanager
def replacement_transaction(store: Any) -> Iterator[QueryExecutor]:
    """Run a set of writes as one unit so a failure leaves prior state intact.

    Stores must provide an explicit transaction; never degrade replacement to
    separately committed deletion and creation.
    """
    transaction = getattr(store, "_transaction", None)
    if not callable(transaction):
        raise RuntimeError(
            "Community replacement requires a store with _transaction() support; "
            "existing communities were left untouched."
        )
    factory = cast(Callable[[], AbstractContextManager[QueryExecutor]], transaction)
    with factory() as executor:
        yield executor


class BaseGraphStore:
    """Mixin for graph-store methods that are identical across backends."""

    def execute_query(self, query: str, parameters: dict | None = None) -> list[dict]:
        raise NotImplementedError

    def get_entity_count(self) -> int:
        return self._count("MATCH (e:__Entity__) RETURN count(e) AS cnt")

    def get_chunk_count(self) -> int:
        return self._count("MATCH (c:Chunk) RETURN count(c) AS cnt")

    def get_evidence_link_count(self) -> int:
        return self._count(
            "MATCH (:Chunk)-[r:FROM_CHUNK]->(:__Entity__) RETURN count(r) AS cnt"
        )

    def get_relationship_count(self) -> int:
        # Directed traversal: an undirected pattern returns each stored
        # relationship once per direction, so it counts self-loops twice.
        return self._count(
            "MATCH (:__Entity__)-[r]->(:__Entity__) RETURN count(r) AS cnt"
        )

    def validate_graph_build(self) -> dict:
        counts = {
            "entity_count": self.get_entity_count(),
            "chunk_count": self.get_chunk_count(),
            "evidence_link_count": self.get_evidence_link_count(),
            "entity_relationship_count": self.get_relationship_count(),
        }

        for key, query in self._extra_validation_count_queries().items():
            counts[key] = self._count(query)
        return counts

    def _extra_validation_count_queries(self) -> dict[str, str]:
        return {}

    def _count(self, query: str, params: dict | None = None) -> int:
        result = self.execute_query(query, params) if params else self.execute_query(query)
        return result[0]["cnt"] if result else 0

    def get_community_stats(self, graph_name: str) -> list[dict]:
        query = """
        MATCH (c:Community {graph_name: $graph_name})
        OPTIONAL MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
        WITH c, count(DISTINCT e) AS entity_count
        OPTIONAL MATCH (c)<-[:PARENT_COMMUNITY]-(child:Community)
        WITH c, entity_count, count(DISTINCT child) AS child_community_count
        RETURN c.id AS community_id,
               c.level AS level,
               entity_count,
               child_community_count
        ORDER BY c.level, entity_count DESC
        """
        return self.execute_query(query, {"graph_name": graph_name})

    def get_communities(
        self,
        graph_name: str,
        level: int | None = None,
    ) -> list[dict]:
        community_label = escape_cypher_identifier("Community")
        if level is not None:
            query = f"""
            MATCH (c:{community_label} {{graph_name: $graph_name, level: $level}})
            OPTIONAL MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
            WITH c, count(DISTINCT e) AS entity_count
            OPTIONAL MATCH (c)<-[:PARENT_COMMUNITY]-(child:Community)
            WITH c,
                 entity_count,
                 count(DISTINCT child) AS child_community_count
            RETURN c.id AS id,
                   c.level AS level,
                   entity_count,
                   child_community_count
            ORDER BY entity_count DESC, id
            """
            params = {"graph_name": graph_name, "level": level}
        else:
            query = f"""
            MATCH (c:{community_label} {{graph_name: $graph_name}})
            OPTIONAL MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
            WITH c, count(DISTINCT e) AS entity_count
            OPTIONAL MATCH (c)<-[:PARENT_COMMUNITY]-(child:Community)
            WITH c,
                 entity_count,
                 count(DISTINCT child) AS child_community_count
            RETURN c.id AS id,
                   c.level AS level,
                   entity_count,
                   child_community_count
            ORDER BY c.level, entity_count DESC, id
            """
            params = {"graph_name": graph_name}
        return self.execute_query(query, params)

    def store_community_report(
        self,
        report: CommunityReport,
        graph_name: str,
    ) -> None:
        """Store one report and keep the old vector only for identical text."""
        self._store_community_report(
            report,
            graph_name,
            expected_generation=None,
            guard_generation=False,
        )

    def _store_community_report_guarded(
        self,
        report: CommunityReport,
        graph_name: str,
        *,
        expected_generation: str | None,
    ) -> bool:
        """Store a report only while the Community keeps the expected generation."""
        rows = self._store_community_report(
            report,
            graph_name,
            expected_generation=_bound_generation(expected_generation),
            guard_generation=True,
        )
        return _affected_count(rows, "written") > 0

    def _store_community_report(
        self,
        report: CommunityReport,
        graph_name: str,
        *,
        expected_generation: str | None,
        guard_generation: bool,
    ) -> list[dict]:
        report_text = report_to_text(report)
        report_text_fingerprint = report_embedding_fingerprint(
            report.title, report_text
        )
        query = """
        MATCH (c:Community {
            graph_name: $graph_name,
            id: $cid,
            level: $level
        })
        // Acquire a write lock before the guard reads generation and text.
        SET c.`__recon_community_write_lock` = $lock_token
        WITH c,
             c.report_embedding AS previous_vector,
             c.report_embedding_content_fingerprint AS previous_content,
             c.report_embedding_model_fingerprint AS previous_model,
             c.report_embedding_error AS previous_error,
             coalesce(
                 c.report_embedding_input_fingerprint = $report_text_fingerprint
                 OR (c.report_embedding_input_fingerprint IS NULL
                     AND c.report_text = $report_text
                     AND coalesce(c.title, '') = coalesce($title, '')),
                 false
             ) AS same_text,
             coalesce(
                 (NOT $guard_generation)
                 OR (c.generation = $generation)
                 OR ($generation IS NULL AND c.generation IS NULL),
                 false
             ) AS accepted
        // Remove the temporary lock before the statement ends. A rejected row
        // keeps every report field unchanged and leaves no lock property.
        SET c.`__recon_community_write_lock` = NULL
        FOREACH (_ IN CASE WHEN accepted THEN [1] ELSE [] END |
            SET c.report_json = $report_json,
                c.report_text = $report_text,
                c.title = $title,
                c.rating = $rating,
                c.rating_explanation = $rating_explanation,
                c.report_status = 'success',
                c.report_error = NULL,
                c.schema_version = $schema_version,
                c.prompt_version = $prompt_version,
                c.context_tokens_used = $context_tokens_used,
                c.context_truncated = $context_truncated,
                c.input_fingerprint = $input_fingerprint,
                c.report_embedding_input_fingerprint = $report_text_fingerprint,
                c.report_embedding = CASE
                    WHEN same_text THEN previous_vector ELSE NULL END,
                c.report_embedding_content_fingerprint = CASE
                    WHEN same_text THEN previous_content ELSE NULL END,
                c.report_embedding_model_fingerprint = CASE
                    WHEN same_text THEN previous_model ELSE NULL END,
                c.report_embedding_error = CASE
                    WHEN same_text THEN previous_error ELSE NULL END,
                c.updated = timestamp()
        )
        RETURN sum(CASE WHEN accepted THEN 1 ELSE 0 END) AS written
        """
        return self.execute_query(
            query,
            {
                "graph_name": graph_name,
                "cid": report.community_id,
                "level": report.level,
                "report_json": report_to_json(report),
                "report_text": report_text,
                "report_text_fingerprint": report_text_fingerprint,
                "title": report.title,
                "rating": report.rating,
                "rating_explanation": report.rating_explanation,
                "schema_version": report.version.schema_version,
                "prompt_version": report.version.prompt_version,
                "input_fingerprint": report.version.input_fingerprint,
                "context_tokens_used": report.context_tokens_used,
                "context_truncated": report.context_truncated,
                "guard_generation": guard_generation,
                "generation": expected_generation,
                "lock_token": uuid4().hex,
            },
        )

    def mark_community_report_failed(
        self,
        graph_name: str,
        community_id: str,
        level: int,
        error: str,
    ) -> None:
        query = """
        MATCH (c:Community {
            graph_name: $graph_name,
            id: $cid,
            level: $level
        })
        SET c.report_status = 'failed',
            c.report_error = $error,
            c.updated = timestamp()
        """
        self.execute_query(
            query,
            {
                "graph_name": graph_name,
                "cid": community_id,
                "level": level,
                "error": error,
            },
        )

    def _mark_community_report_failed_guarded(
        self,
        graph_name: str,
        community_id: str,
        level: int,
        error: str,
        *,
        expected_generation: str | None,
    ) -> bool:
        """Mark a report failed only while the Community keeps the expected generation."""
        expected_generation = _bound_generation(expected_generation)
        query = """
        MATCH (c:Community {
            graph_name: $graph_name,
            id: $cid,
            level: $level
        })
        // Acquire a write lock before the guard reads generation.
        SET c.`__recon_community_write_lock` = $lock_token
        WITH c,
             coalesce(
                 (c.generation = $generation)
                 OR ($generation IS NULL AND c.generation IS NULL),
                 false
             ) AS accepted
        // Remove the temporary lock before the statement ends. A rejected row
        // keeps its report status and leaves no lock property.
        SET c.`__recon_community_write_lock` = NULL
        FOREACH (_ IN CASE WHEN accepted THEN [1] ELSE [] END |
            SET c.report_status = 'failed',
                c.report_error = $error,
                c.updated = timestamp()
        )
        RETURN sum(CASE WHEN accepted THEN 1 ELSE 0 END) AS written
        """
        rows = self.execute_query(
            query,
            {
                "graph_name": graph_name,
                "cid": community_id,
                "level": level,
                "error": error,
                "generation": expected_generation,
                "lock_token": uuid4().hex,
            },
        )
        return _affected_count(rows, "written") > 0

    def get_claims_for_entities(
        self,
        graph_name: str,
        entity_ids: list[str],
    ) -> list[dict]:
        if not entity_ids:
            return []
        query = """
        UNWIND $entity_ids AS eid
        MATCH (c:Claim {graph_name: $graph_name})-[:SUBJECT_OF]->
              (e:__Entity__ {graph_name: $graph_name})
        WHERE e.id = eid OR e.canonical_key = eid OR e.human_readable_id = eid
        OPTIONAL MATCH (c)-[:SOURCED_FROM]->(ch:Chunk {graph_name: $graph_name})
        RETURN c.id AS claim_id,
               coalesce(e.human_readable_id, e.canonical_key, e.id) AS entity_id,
               c.claim_type AS claim_type,
               c.description AS description,
               c.status AS status,
               c.start_date AS start_date,
               c.end_date AS end_date,
               c.object_entity_id AS object_entity_id,
               c.source_text AS source_text,
               c.text_unit_id AS text_unit_id,
               ch.id AS chunk_id
        ORDER BY c.claim_type, c.id
        """
        return self.execute_query(
            query,
            {"graph_name": graph_name, "entity_ids": entity_ids},
        )

    def resolve_chunk_citations(
        self,
        graph_name: str,
        chunk_ids: list[str],
    ) -> list[dict]:
        if not chunk_ids:
            return []
        query = """
        UNWIND $chunk_ids AS cid
        MATCH (c:Chunk {id: cid, graph_name: $graph_name})
        OPTIONAL MATCH (c)-[:PART_OF]->(d:Document {graph_name: $graph_name})
        RETURN c.id AS chunk_id,
               d.id AS document_id,
               coalesce(d.title, d.source, d.filename) AS document_name,
               c.page_start AS page_start,
               c.page_end AS page_end,
               properties(c) AS chunk_metadata,
               properties(d) AS document_metadata,
               substring(c.text, 0, 200) AS excerpt
        """
        return self.execute_query(
            query,
            {"graph_name": graph_name, "chunk_ids": chunk_ids},
        )

    def count_communities_with_reports(self, graph_name: str) -> int:
        query = """
        MATCH (c:Community {graph_name: $graph_name})
        WHERE coalesce(c.report_text, '') <> ''
        RETURN count(c) AS cnt
        """
        return self._count(query, {"graph_name": graph_name})

    def get_unembedded_community_reports(
        self,
        graph_name: str,
        limit: int = 500,
    ) -> list[dict]:
        """Return reports with no vector and no recorded embedding failure.

        This is the legacy public selector. Its paired public writer sets the
        vector and clears the failure marker, so a repeated select, embed, and
        ``upsert_community_report_vectors`` loop ends. The private managed
        selection refreshes missing certificates, changed text, and changed
        models.
        """
        query = """
        MATCH (c:Community {graph_name: $graph_name})
        WHERE c.report_embedding IS NULL
          AND c.report_embedding_error IS NULL
          AND coalesce(c.report_text, '') <> ''
        RETURN c.id AS id,
               c.level AS level,
               c.report_text AS report_text,
               coalesce(c.title, '') AS title
        ORDER BY c.level, c.id
        LIMIT $limit
        """
        return self.execute_query(
            query, {"graph_name": graph_name, "limit": limit}
        )

    def _get_community_generations(
        self,
        graph_name: str,
        level: int,
    ) -> list[str | None]:
        """Return distinct generation values observed at one community level."""
        query = """
        MATCH (c:Community {graph_name: $graph_name, level: $level})
        RETURN DISTINCT c.generation AS generation
        ORDER BY generation
        """
        rows = self.execute_query(
            query, {"graph_name": graph_name, "level": level}
        )
        return [row.get("generation") for row in rows]

    def _get_unembedded_community_reports_for_embedding(
        self,
        graph_name: str,
        limit: int = 500,
        *,
        model_fingerprint: str,
        exclude_keys: list[dict] | None,
        expected_generation: object = _GENERATION_UNGUARDED,
    ) -> list[dict]:
        """Select reports for a later embedding pass.

        ``expected_generation`` is ``_GENERATION_UNGUARDED`` for all rows, or
        ``None`` to require a legacy NULL generation. The exclusion applies
        before ``LIMIT``, and the stable order makes the window fair. Private
        rows add ``generation`` and ``embedding_revision`` to the public keys.
        """
        if expected_generation is _GENERATION_UNGUARDED:
            filter_generation = False
            generation = None
        else:
            filter_generation = True
            generation = cast(str | None, expected_generation)
        exclude_key_strings = [
            f"{key['level']}:{key['id']}" for key in (exclude_keys or [])
        ]
        query = """
        MATCH (c:Community {graph_name: $graph_name})
        WHERE coalesce(c.report_text, '') <> ''
          AND (NOT $filter_generation
               OR (c.generation = $generation)
               OR ($generation IS NULL AND c.generation IS NULL))
          AND NOT ((toString(c.level) + ':' + c.id) IN $exclude_key_strings)
          AND (c.report_embedding IS NULL
               OR c.report_embedding_input_fingerprint IS NULL
               OR c.report_embedding_content_fingerprint IS NULL
               OR c.report_embedding_content_fingerprint
                  <> c.report_embedding_input_fingerprint
               OR ($model_fingerprint IS NOT NULL
                   AND coalesce(c.report_embedding_model_fingerprint, '')
                       <> $model_fingerprint))
        RETURN c.id AS id,
               c.level AS level,
               coalesce(c.title, '') AS title,
               c.report_text AS report_text,
               c.generation AS generation,
               c.report_embedding_revision AS embedding_revision
        ORDER BY c.level, c.id
        LIMIT $limit
        """
        return self.execute_query(
            query,
            {
                "graph_name": graph_name,
                "limit": limit,
                "model_fingerprint": model_fingerprint,
                "exclude_key_strings": exclude_key_strings,
                "filter_generation": filter_generation,
                "generation": generation,
            },
        )

    def upsert_community_report_vectors(
        self,
        node_ids: list[str],
        vectors: list[list[float]],
        graph_name: str,
        levels: list[int],
    ) -> None:
        """Write vectors without certifying their report text or model.

        The public path cannot prove which text the caller embedded. It sets
        the vector and clears the failure marker, so the paired public
        selector no longer returns the row. It clears the certification and
        advances the revision, so a managed pass refreshes the vector and an
        older guarded snapshot cannot replace a newer one.
        """
        if not node_ids:
            return
        if len(node_ids) != len(vectors) or len(node_ids) != len(levels):
            raise ValueError("node_ids, vectors, and levels must have the same length")
        query = """
        UNWIND $pairs AS pair
        MATCH (c:Community {
            graph_name: $graph_name,
            id: pair.node_id,
            level: pair.level
        })
        SET c.report_embedding = pair.vector,
            c.report_embedding_content_fingerprint = NULL,
            c.report_embedding_model_fingerprint = NULL,
            c.report_embedding_revision =
                coalesce(c.report_embedding_revision, 0) + 1,
            c.report_embedding_error = NULL
        """
        pairs = [
            {"node_id": nid, "level": level, "vector": vec}
            for nid, level, vec in zip(node_ids, levels, vectors)
        ]
        self.execute_query(query, {"graph_name": graph_name, "pairs": pairs})

    def _upsert_community_report_vectors_guarded(
        self,
        graph_name: str,
        pairs: list[dict],
        *,
        model_fingerprint: str,
    ) -> int:
        """Write vectors that certify the snapshot selected for embedding.

        Every pair carries the generation, revision, title, report text, and
        content fingerprint read before the provider call. The write advances
        ``report_embedding_revision`` and clears the failure marker only when
        all snapshot checks still hold. A legacy row with no input certificate
        gets one from the verified content fingerprint. Missing or stale
        targets are not counted.
        """
        if not pairs:
            return 0
        for pair in pairs:
            expected = report_embedding_fingerprint(
                pair.get("title"), pair.get("report_text")
            )
            if pair.get("content_fingerprint") != expected:
                raise ValueError(
                    "content_fingerprint must match the pair title and report_text"
                )
        query = """
        UNWIND $pairs AS pair
        MATCH (c:Community {
            graph_name: $graph_name,
            id: pair.id,
            level: pair.level
        })
        // Acquire a write lock before the snapshot checks.
        SET c.`__recon_community_write_lock` = $lock_token
        WITH c, pair,
             coalesce(
                 (pair.generation IS NULL AND c.generation IS NULL)
                 OR (pair.generation IS NOT NULL
                     AND c.generation = pair.generation),
                 false
             ) AS generation_matches,
             coalesce(
                 (pair.embedding_revision IS NULL
                  AND c.report_embedding_revision IS NULL)
                 OR (pair.embedding_revision IS NOT NULL
                     AND c.report_embedding_revision = pair.embedding_revision),
                 false
             ) AS revision_matches,
             coalesce(
                 coalesce(c.title, '') = coalesce(pair.title, '')
                 AND coalesce(c.report_text, '') = coalesce(pair.report_text, ''),
                 false
             ) AS text_matches,
             coalesce(
                 c.report_embedding_input_fingerprint IS NULL
                 OR c.report_embedding_input_fingerprint = pair.content_fingerprint,
                 false
             ) AS input_matches,
             pair.vector IS NOT NULL AS vector_present,
             pair.content_fingerprint IS NOT NULL AS content_certified,
             coalesce(
                 pair.new_embedding_revision IS NOT NULL
                 AND (pair.embedding_revision IS NULL
                      OR pair.new_embedding_revision
                         <> pair.embedding_revision),
                 false
             ) AS revision_advanced
        // Remove the temporary lock before the statement ends. A rejected
        // row keeps its vector and leaves no lock property.
        SET c.`__recon_community_write_lock` = NULL
        FOREACH (_ IN CASE WHEN generation_matches AND revision_matches
                           AND text_matches AND input_matches AND vector_present
                           AND content_certified AND revision_advanced
                      THEN [1] ELSE [] END |
            SET c.report_embedding = pair.vector,
                c.report_embedding_content_fingerprint = pair.content_fingerprint,
                c.report_embedding_input_fingerprint = coalesce(
                    c.report_embedding_input_fingerprint, pair.content_fingerprint
                ),
                c.report_embedding_model_fingerprint = $model_fingerprint,
                c.report_embedding_revision = pair.new_embedding_revision,
                c.report_embedding_error = NULL
        )
        RETURN sum(CASE WHEN generation_matches AND revision_matches
                        AND text_matches AND input_matches AND vector_present
                        AND content_certified AND revision_advanced
                   THEN 1 ELSE 0 END) AS updated_count
        """
        rows = self.execute_query(
            query,
            {
                "graph_name": graph_name,
                "pairs": pairs,
                "model_fingerprint": model_fingerprint,
                "lock_token": uuid4().hex,
            },
        )
        return _affected_count(rows, "updated_count")

    def _mark_community_report_embedding_failed_guarded(
        self,
        graph_name: str,
        pair: dict,
        error: str,
    ) -> bool:
        """Mark an embedding failure only for the selected snapshot.

        The mark is diagnostic. It never advances the revision, so it cannot
        replace a newer successful certification.
        """
        query = """
        UNWIND $pairs AS pair
        MATCH (c:Community {
            graph_name: $graph_name,
            id: pair.id,
            level: pair.level
        })
        // Acquire a write lock before the snapshot checks.
        SET c.`__recon_community_write_lock` = $lock_token
        WITH c,
             coalesce(
                 (pair.generation IS NULL AND c.generation IS NULL)
                 OR (pair.generation IS NOT NULL
                     AND c.generation = pair.generation),
                 false
             ) AS generation_matches,
             coalesce(
                 (pair.embedding_revision IS NULL
                  AND c.report_embedding_revision IS NULL)
                 OR (pair.embedding_revision IS NOT NULL
                     AND c.report_embedding_revision = pair.embedding_revision),
                 false
             ) AS revision_matches,
             coalesce(
                 coalesce(c.title, '') = coalesce(pair.title, '')
                 AND coalesce(c.report_text, '') = coalesce(pair.report_text, ''),
                 false
             ) AS text_matches,
             coalesce(
                 c.report_embedding_input_fingerprint IS NULL
                 OR c.report_embedding_input_fingerprint = pair.content_fingerprint,
                 false
             ) AS input_matches
        // Remove the temporary lock before the statement ends. A rejected
        // row keeps its diagnostics and leaves no lock property.
        SET c.`__recon_community_write_lock` = NULL
        FOREACH (_ IN CASE WHEN generation_matches AND revision_matches
                           AND text_matches AND input_matches
                      THEN [1] ELSE [] END |
            SET c.report_embedding_error = $error
        )
        RETURN sum(CASE WHEN generation_matches AND revision_matches
                        AND text_matches AND input_matches
                   THEN 1 ELSE 0 END) AS written
        """
        rows = self.execute_query(
            query,
            {
                "graph_name": graph_name,
                "pairs": [pair],
                "error": error,
                "lock_token": uuid4().hex,
            },
        )
        return _affected_count(rows, "written") > 0
