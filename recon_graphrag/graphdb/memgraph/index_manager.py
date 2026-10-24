"""Vector and fulltext index creation for Memgraph.

Memgraph uses native vector and text indexes:

- Vector: CREATE VECTOR INDEX name ON :Label(prop) WITH CONFIG {...}
- Text:   CREATE TEXT INDEX name ON :Label(prop1, prop2)

These are not MAGE modules; they are built into the Memgraph engine.
"""

from __future__ import annotations

import logging
from typing import Optional

from recon_graphrag.embeddings.base import BaseEmbedder
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.graphdb.cypher import (
    escape_cypher_identifier,
)
from recon_graphrag.models.types import IndexConfig

logger = logging.getLogger(__name__)


class IndexManager:
    """Create indexes for the Memgraph backend."""

    def __init__(
        self,
        graph_store: GraphStore,
        embedder: Optional[BaseEmbedder] = None,
        embedding_dim: Optional[int] = None,
        index_config: Optional[IndexConfig] = None,
    ):
        self.graph_store = graph_store
        self.embedder = embedder
        self.config = index_config or IndexConfig()

        if embedding_dim is not None:
            self.embedding_dim = embedding_dim
        elif embedder is not None:
            from recon_graphrag.embeddings.base import detect_embedding_dim

            self.embedding_dim = detect_embedding_dim(embedder) or 1536
        else:
            self.embedding_dim = 1536

    def create_indexes(self):
        """Create all required vector and text indexes."""
        self._drop_indexes()
        self.graph_store.create_vector_index(
            name=self.config.entity_vector_index,
            label=self.config.entity_label,
            embedding_property="embedding",
            dimensions=self.embedding_dim,
        )
        self.graph_store.create_fulltext_index(
            name=self.config.entity_fulltext_index,
            label=self.config.entity_label,
            node_properties=["name"],
        )
        self.graph_store.create_vector_index(
            name=self.config.community_vector_index,
            label=self.config.community_label,
            embedding_property=self.config.community_report_embedding_property,
            dimensions=self.embedding_dim,
        )
        self._create_constraints()

    def _drop_indexes(self):
        """Drop existing indexes so they can be recreated with updated settings."""
        for name in [
            self.config.entity_vector_index,
            self.config.entity_fulltext_index,
            self.config.community_vector_index,
        ]:
            try:
                self.graph_store.execute_query(
                    f"DROP VECTOR INDEX {escape_cypher_identifier(name)}"
                )
            except Exception:
                pass
            try:
                self.graph_store.execute_query(
                    f"DROP TEXT INDEX {escape_cypher_identifier(name)}"
                )
            except Exception:
                pass

    def _create_constraints(self):
        """Create constraints required by the SDK.

        Memgraph supports only single-property unique constraints, so the
        composite (graph_name, level, id) community key and the per-label
        (graph_name, id) writer key are each encoded as one `uid` property
        with the same length-prefixed encoding the writer uses (F8).
        Unexpected failures now propagate instead of being muted so a
        silently missing constraint cannot corrupt later ingestion.
        """
        self._create_community_constraint()
        self._create_uid_constraints()

    def _create_community_constraint(self):
        try:
            self.graph_store.execute_query(
                """
                CREATE CONSTRAINT ON (c:Community)
                ASSERT c.uid IS UNIQUE
                """
            )
        except Exception as e:
            if "already exists" in str(e).lower():
                return
            raise RuntimeError(
                "Failed to create the Community uniqueness constraint; "
                "duplicate communities already in the graph must be resolved "
                "before ingestion can continue safely. Original error: "
                f"{e}"
            ) from e

    # Labels whose per-graph (graph_name, id) pair must be unique.
    _UID_LABELS = ("Document", "Chunk", "__Entity__")

    def _create_uid_constraints(self):
        for label in self._UID_LABELS:
            self._preflight_unique_uids(label)
            self._backfill_uids(label)
            try:
                self.graph_store.execute_query(
                    f"""
                    CREATE CONSTRAINT ON (n:{label})
                    ASSERT n.uid IS UNIQUE
                    """
                )
            except Exception as e:
                if "already exists" in str(e).lower():
                    continue
                raise RuntimeError(
                    f"Failed to create the uid uniqueness constraint for label "
                    f"'{label}': {e}"
                ) from e

    def _preflight_unique_uids(self, label: str) -> None:
        rows = self.graph_store.execute_query(
            f"""
            MATCH (n:{label})
            WITH n.graph_name AS graph_name, n.id AS id, count(*) AS duplicates
            WHERE duplicates > 1
            RETURN graph_name, id, duplicates
            ORDER BY duplicates DESC
            LIMIT 50
            """
        )
        if rows:
            dup = rows[0]
            raise ValueError(
                f"Cannot create the uid uniqueness constraint for label "
                f"'{label}': the graph already contains duplicate (graph_name, id) "
                f"pairs, e.g. graph_name={dup['graph_name']!r}, id={dup['id']!r} "
                f"with {dup['duplicates']} duplicates. Resolve existing "
                "duplicates (for example by re-ingesting the affected documents "
                "after upgrading) before re-running index setup."
            )

    def _backfill_uids(self, label: str) -> None:
        self.graph_store.execute_query(
            f"""
            MATCH (n:{label})
            WHERE n.uid IS NULL
            SET n.uid = toString(size(n.graph_name)) + ':' + n.graph_name + ':' + n.id
            RETURN count(n) AS backfilled
            """
        )

    def verify(self):
        """Print current graph schema info: indexes, node counts, relationship counts."""
        print("\n=== Indexes ===")
        try:
            indexes = self.graph_store.execute_query("SHOW INDEX INFO")
            for idx in indexes:
                print(f"  {idx}")
        except Exception as exc:
            print(f"  Could not list indexes: {exc}")

        print("\n=== Node Counts ===")
        counts = self.graph_store.execute_query(
            "MATCH (n) RETURN labels(n) AS labels, count(*) AS count "
            "ORDER BY count DESC LIMIT 20"
        )
        for rec in counts:
            print(f"  {rec['labels']}: {rec['count']}")

        print("\n=== Relationship Counts ===")
        rels = self.graph_store.execute_query(
            "MATCH ()-[r]-() RETURN type(r) AS type, count(*) AS count "
            "ORDER BY count DESC LIMIT 20"
        )
        for rec in rels:
            print(f"  {rec['type']}: {rec['count']}")
