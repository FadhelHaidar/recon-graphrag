"""Vector and fulltext index creation for Memgraph.

Memgraph uses native vector and text indexes:

- Vector: CREATE VECTOR INDEX name ON :Label(prop) WITH CONFIG {...}
- Text:   CREATE TEXT INDEX name ON :Label(prop1, prop2)

These are not MAGE modules; they are built into the Memgraph engine.
"""

from __future__ import annotations

import logging
from typing import NoReturn, Optional

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
        """Create the required indexes, reusing compatible existing ones.

        Existing indexes are never dropped implicitly: a dropped vector index
        makes retrieval fail or degrade until it is rebuilt, so replacing an
        index stays an explicit administrative action (``drop_indexes()``).
        """
        existing = self._inspect_indexes()
        vector_indexes = self._inspect_vector_indexes()
        self._ensure_vector_index(
            existing,
            name=self.config.entity_vector_index,
            label=self.config.entity_label,
            embedding_property="embedding",
            vector_indexes=vector_indexes,
        )
        self._ensure_text_index(
            existing,
            name=self.config.entity_fulltext_index,
            label=self.config.entity_label,
            properties=["name"],
        )
        self._ensure_vector_index(
            existing,
            name=self.config.community_vector_index,
            label=self.config.community_label,
            embedding_property=self.config.community_report_embedding_property,
            vector_indexes=vector_indexes,
        )
        self._create_constraints()

    def _inspect_indexes(self) -> list[dict]:
        """Inspect indexes without treating an inspection failure as absence."""
        try:
            return self.graph_store.execute_query("SHOW INDEX INFO")
        except Exception:
            raise RuntimeError(
                "Cannot verify existing Memgraph indexes. Grant index-inspection "
                "access and retry; no indexes were changed."
            ) from None

    def _inspect_vector_indexes(self) -> list[dict] | None:
        """Return vector index details, or None when this Memgraph cannot report them."""
        try:
            return self.graph_store.execute_query("SHOW VECTOR INDEX INFO")
        except Exception:
            # Older Memgraph releases lack the command; the caller then fails
            # closed because dimensions cannot be verified.
            return None

    @staticmethod
    def _index_signature(row: dict) -> tuple[str | None, str, str | None, list[str]]:
        """Return (name, type, label, properties) from a SHOW INDEX INFO row.

        Memgraph reports index types as ``label+property_vector`` and
        ``label_text (name: <index>)``, labels with a leading colon, and omits
        the name for vector indexes. Normalising here keeps compatibility
        checks operating on one shape.
        """
        raw_kind = str(row.get("index type") or "")
        lowered = raw_kind.lower()
        name = row.get("index name") or row.get("name")
        if lowered.startswith("label_text"):
            kind = "text"
            if "name:" in raw_kind:
                name = raw_kind.split("name:", 1)[1].split(")", 1)[0].strip()
        elif "vector" in lowered:
            kind = "vector"
        elif lowered.startswith("label+property"):
            kind = "range"
        else:
            kind = lowered
        label = row.get("label")
        if isinstance(label, str):
            label = label.rsplit(":", 1)[-1] or None
        properties = row.get("property")
        if isinstance(properties, str):
            properties = [properties]
        elif properties is None:
            properties = []
        else:
            properties = list(properties)
        return name, kind, label, properties

    def _ensure_vector_index(
        self,
        existing: list[dict],
        *,
        name: str,
        label: str,
        embedding_property: str,
        vector_indexes: list[dict] | None = None,
    ) -> None:
        for row in existing:
            row_name, kind, row_label, properties = self._index_signature(row)
            if kind != "vector":
                continue

            schema_matches = row_label == label and properties == [embedding_property]
            # SHOW INDEX INFO omits the name of a vector index; SHOW VECTOR
            # INDEX INFO reports it, so resolve which index this row describes.
            info = self._match_vector_info(
                vector_indexes,
                name=row_name,
                label=row_label or "",
                embedding_property=properties[0] if properties else "",
            )
            resolved_name = row_name or (info or {}).get("index_name")

            if resolved_name is None:
                if schema_matches:
                    # The index exists on our schema but cannot be identified.
                    self._reject_unverifiable(name)
                continue
            if resolved_name != name:
                if schema_matches:
                    self._raise_if_incompatible(
                        name, [f"schema is indexed under another name {resolved_name!r}"]
                    )
                continue

            # Same name: definition and vector configuration must both match.
            self._raise_if_incompatible(
                name,
                self._definition_problems(
                    kind,
                    row_label,
                    properties,
                    expected_kind="vector",
                    expected_label=label,
                    expected_properties=[embedding_property],
                ),
            )
            if info is None:
                self._reject_unverifiable(name)
            self._raise_if_incompatible(name, self._vector_config_problems(info))
            return

        self.graph_store.create_vector_index(
            name=name,
            label=label,
            embedding_property=embedding_property,
            dimensions=self.embedding_dim,
        )

    @staticmethod
    def _match_vector_info(
        vector_indexes: list[dict] | None,
        *,
        name: str | None,
        label: str,
        embedding_property: str,
    ) -> dict | None:
        """Return the SHOW VECTOR INDEX INFO row describing an existing index."""
        for row in vector_indexes or []:
            if name is not None:
                if row.get("index_name") == name:
                    return row
                continue
            row_label = str(row.get("label") or "").rsplit(":", 1)[-1]
            if row_label == label and row.get("property") == embedding_property:
                return row
        return None

    def _vector_config_problems(self, info: dict) -> list[str]:
        problems = []
        dimension = info.get("dimension")
        if dimension != self.embedding_dim:
            problems.append(f"dimensions={dimension}, expected {self.embedding_dim}")
        metric = info.get("metric")
        if str(metric).lower() != "cos":
            problems.append(f"similarity={metric!r}, expected 'cos'")
        return problems

    def _ensure_text_index(
        self, existing: list[dict], *, name: str, label: str, properties: list[str]
    ) -> None:
        for row in existing:
            row_name, kind, row_label, row_properties = self._index_signature(row)
            if row_name == name:
                self._raise_if_incompatible(
                    name,
                    self._definition_problems(
                        kind,
                        row_label,
                        row_properties,
                        expected_kind="text",
                        expected_label=label,
                        expected_properties=properties,
                    ),
                )
                return
            if kind == "text" and row_label == label and row_properties == properties:
                if row_name is None:
                    # An unnamed text index is fully described by type, label and
                    # properties, so a schema match is a usable existing index.
                    return
                self._raise_if_incompatible(
                    name, [f"schema is indexed under another name {row_name!r}"]
                )

        self.graph_store.create_fulltext_index(
            name=name, label=label, node_properties=properties
        )

    @staticmethod
    def _definition_problems(
        kind: str,
        label: str | None,
        properties: list[str],
        *,
        expected_kind: str,
        expected_label: str,
        expected_properties: list[str],
    ) -> list[str]:
        problems = []
        if kind != expected_kind:
            problems.append(f"type={kind!r}, expected {expected_kind!r}")
        if label != expected_label:
            problems.append(f"label={label!r}, expected {expected_label!r}")
        if properties != expected_properties:
            problems.append(
                f"properties={properties!r}, expected {expected_properties!r}"
            )
        return problems

    @staticmethod
    def _reject_unverifiable(name: str) -> NoReturn:
        raise ValueError(
            f"Cannot verify dimensions and similarity for Memgraph index {name!r}: "
            "SHOW INDEX INFO does not expose its vector configuration. "
            "Call store.drop_indexes() and re-run index setup; existing indexes "
            "are never dropped automatically."
        )

    @staticmethod
    def _raise_if_incompatible(name: str, problems: list[str]) -> None:
        if not problems:
            return
        raise ValueError(
            f"Index {name!r} already exists with an incompatible definition "
            f"({'; '.join(problems)}). Call store.drop_indexes() and re-run index "
            "setup; existing indexes are never dropped automatically."
        )

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
