"""Community generation publication helpers.

One ``CommunityGeneration`` node holds the current token for each graph.
Detection replaces the community hierarchy and advances that token in one
transaction. The unique constraint on ``graph_name`` makes the state write the
graph-scoped publication lock. The lock lasts for the replacement transaction
only; no lease or build-wide lock exists.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from recon_graphrag.graphdb.store_base import (
    WRITE_RETRY_ATTEMPTS,
    WRITE_RETRY_DELAY_SECONDS,
    QueryExecutor,
    is_retryable_write_error,
    replacement_transaction,
)

logger = logging.getLogger(__name__)

COMMUNITY_GENERATION_LABEL = "CommunityGeneration"
COMMUNITY_GENERATION_PROPERTY = "graph_name"
COMMUNITY_GENERATION_CONSTRAINT_NAME = "community_generation_unique"

HierarchyWriter = Callable[[QueryExecutor, str], None]
StatsReader = Callable[[QueryExecutor], list[dict[str, Any]]]
ConstraintVerifier = Callable[[Any], None]


def new_community_generation_token() -> str:
    """Return a token for one detection invocation."""
    return uuid4().hex


def _property_list(value: Any) -> list[str]:
    """Normalize a constraint property value to a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _missing_constraint_error(backend: str) -> RuntimeError:
    return RuntimeError(
        f"The {COMMUNITY_GENERATION_LABEL} uniqueness constraint on "
        f"{COMMUNITY_GENERATION_PROPERTY!r} is missing for {backend}. "
        "Run store.create_indexes() before community detection; existing "
        "communities were left untouched."
    )


def verify_neo4j_community_generation_constraint(store: Any) -> None:
    """Fail closed unless the exact named graph_name uniqueness constraint exists."""
    try:
        rows = store.execute_query(
            """
            SHOW CONSTRAINTS
            YIELD name, type, labelsOrTypes, properties
            RETURN name, type, labelsOrTypes, properties
            """
        )
    except Exception as error:
        raise RuntimeError(
            "Cannot verify the CommunityGeneration uniqueness constraint. "
            "Grant constraint-inspection access, run store.create_indexes(), "
            "and retry; existing communities were left untouched."
        ) from error

    matching = [
        row
        for row in rows or []
        if row.get("name") == COMMUNITY_GENERATION_CONSTRAINT_NAME
    ]
    if not matching:
        raise _missing_constraint_error("Neo4j")
    if len(matching) > 1:
        raise RuntimeError(
            f"Neo4j reports {len(matching)} constraints named "
            f"{COMMUNITY_GENERATION_CONSTRAINT_NAME!r}; community publication "
            "requires one exact constraint. Existing communities were left "
            "untouched."
        )

    row = matching[0]
    if (
        row.get("type") != "UNIQUENESS"
        or _property_list(row.get("labelsOrTypes")) != [COMMUNITY_GENERATION_LABEL]
        or _property_list(row.get("properties")) != [COMMUNITY_GENERATION_PROPERTY]
    ):
        raise RuntimeError(
            f"Constraint {COMMUNITY_GENERATION_CONSTRAINT_NAME!r} exists with an "
            f"incompatible definition (type={row.get('type')!r}, "
            f"labels={row.get('labelsOrTypes')!r}, "
            f"properties={row.get('properties')!r}). Recreate it as "
            f"UNIQUENESS FOR (g:{COMMUNITY_GENERATION_LABEL}) REQUIRE "
            f"g.{COMMUNITY_GENERATION_PROPERTY} IS UNIQUE; existing communities "
            "were left untouched."
        )


def verify_memgraph_community_generation_constraint(store: Any) -> None:
    """Fail closed unless the exact graph_name uniqueness constraint exists."""
    try:
        rows = store.execute_query("SHOW CONSTRAINT INFO")
    except Exception as error:
        raise RuntimeError(
            "Cannot verify the CommunityGeneration uniqueness constraint. "
            "Grant constraint-inspection access, run store.create_indexes(), "
            "and retry; existing communities were left untouched."
        ) from error

    found = False
    for row in rows or []:
        label = str(row.get("label") or "").lstrip(":")
        properties = _property_list(row.get("properties"))
        if label != COMMUNITY_GENERATION_LABEL or properties != [
            COMMUNITY_GENERATION_PROPERTY
        ]:
            continue
        found = True
        constraint_type = str(
            row.get("constraint type") or row.get("type") or ""
        ).lower()
        if constraint_type != "unique":
            raise RuntimeError(
                "The CommunityGeneration constraint on 'graph_name' exists with "
                f"an incompatible type ({constraint_type!r}, expected 'unique'). "
                "Recreate it as a unique constraint; existing communities were "
                "left untouched."
            )

    if not found:
        raise _missing_constraint_error("Memgraph")


def replace_community_hierarchy(
    store: Any,
    *,
    graph_name: str,
    token: str,
    verify_constraint: ConstraintVerifier,
    write_hierarchy: HierarchyWriter,
    read_stats: StatsReader,
) -> list[dict[str, Any]]:
    """Publish one hierarchy with one generation token in one transaction.

    The state-node write is the first statement of the transaction. It advances
    the graph token before the transaction deletes the old communities. The
    unique constraint serializes concurrent publication for the same graph.
    A retry runs only the replacement transaction with the same token; detection
    and provider work never repeat.
    """
    verify_constraint(store)
    for attempt in range(1, WRITE_RETRY_ATTEMPTS + 1):
        try:
            with replacement_transaction(store) as executor:
                executor.execute_query(
                    """
                    MERGE (g:CommunityGeneration {graph_name: $graph_name})
                    ON CREATE SET g.created = timestamp()
                    SET g.token = $token, g.updated = timestamp()
                    """,
                    {"graph_name": graph_name, "token": token},
                )
                write_hierarchy(executor, token)
                return read_stats(executor)
        except Exception as error:
            if (
                not is_retryable_write_error(error)
                or attempt == WRITE_RETRY_ATTEMPTS
            ):
                raise
            logger.warning(
                "community publication rolled back: graph_name=%s "
                "attempt=%s/%s error=%s: %s",
                graph_name,
                attempt,
                WRITE_RETRY_ATTEMPTS,
                type(error).__name__,
                error,
            )
            time.sleep(WRITE_RETRY_DELAY_SECONDS * attempt)
    raise AssertionError("unreachable: retry loop exhausted without raising")
