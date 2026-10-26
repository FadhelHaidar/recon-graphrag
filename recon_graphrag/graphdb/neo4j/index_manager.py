"""Vector and fulltext index creation + entity resolution.

Creates all required Neo4j indexes for semantic retrieval and runs
entity resolution to merge duplicate entities.
"""

from __future__ import annotations

import logging
from typing import Optional

from recon_graphrag.embeddings.base import BaseEmbedder
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.graphdb.cypher import escape_cypher_identifier
from recon_graphrag.graphdb.neo4j.entity_resolution import _Neo4jEntityResolver
from recon_graphrag.models.types import IndexConfig

logger = logging.getLogger(__name__)


class IndexManager:
    """Create indexes and resolve duplicate entities in the graph."""

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
        self._ensure_vector_index(
            existing,
            name=self.config.entity_vector_index,
            label=self.config.entity_label,
            embedding_property="embedding",
        )
        self._ensure_fulltext_index(
            existing,
            name=self.config.entity_fulltext_index,
            label=self.config.entity_label,
            node_properties=["name"],
        )
        self._ensure_vector_index(
            existing,
            name=self.config.community_vector_index,
            label=self.config.community_label,
            embedding_property=self.config.community_report_embedding_property,
        )
        self._create_constraints()

    def _inspect_indexes(self) -> dict[str, dict]:
        """Inspect extended columns, including the vector configuration."""
        try:
            rows = self.graph_store.execute_query("SHOW INDEXES YIELD *")
        except Exception:
            raise RuntimeError(
                "Cannot verify existing Neo4j indexes. Grant index-inspection "
                "access and retry; no indexes were changed."
            ) from None
        return {row["name"]: row for row in rows if row.get("name")}

    def _ensure_vector_index(
        self, existing: dict[str, dict], *, name: str, label: str, embedding_property: str
    ) -> None:
        current = existing.get(name)
        if current is None:
            self.graph_store.create_vector_index(
                name=name,
                label=label,
                embedding_property=embedding_property,
                dimensions=self.embedding_dim,
            )
            return

        problems = self._definition_problems(
            current, kind="VECTOR", label=label, properties=[embedding_property]
        )
        config = (current.get("options") or {}).get("indexConfig") or {}
        if "vector.dimensions" in config:
            if int(config["vector.dimensions"]) != int(self.embedding_dim):
                problems.append(
                    f"dimensions={config['vector.dimensions']}, "
                    f"expected {self.embedding_dim}"
                )
            similarity = config.get("vector.similarity_function")
            if str(similarity).lower() != "cosine":
                problems.append(
                    f"similarity_function={similarity!r}, expected 'cosine'"
                )
        else:
            problems.append(
                "vector configuration is unavailable; verify dimensions and "
                "similarity or explicitly rebuild the index"
            )
        self._raise_if_incompatible(name, problems)

    def _ensure_fulltext_index(
        self, existing: dict[str, dict], *, name: str, label: str, node_properties: list[str]
    ) -> None:
        current = existing.get(name)
        if current is None:
            self.graph_store.create_fulltext_index(
                name=name, label=label, node_properties=node_properties
            )
            return

        problems = self._definition_problems(
            current, kind="FULLTEXT", label=label, properties=node_properties
        )
        self._raise_if_incompatible(name, problems)

    @staticmethod
    def _definition_problems(
        current: dict, *, kind: str, label: str, properties: list[str]
    ) -> list[str]:
        problems = []
        if current.get("type") != kind:
            problems.append(f"type={current.get('type')!r}, expected {kind!r}")
        if list(current.get("labelsOrTypes") or []) != [label]:
            problems.append(
                f"labels={current.get('labelsOrTypes')!r}, expected {[label]!r}"
            )
        if list(current.get("properties") or []) != list(properties):
            problems.append(
                f"properties={current.get('properties')!r}, expected {properties!r}"
            )
        return problems

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
                    f"DROP INDEX {escape_cypher_identifier(name)} IF EXISTS"
                )
            except Exception:
                pass

    def _create_constraints(self):
        """Create constraints required by the SDK.

        Community hierarchy needs a composite uniqueness constraint; the
        writer needs per-label uid uniqueness for safe concurrent upserts
        (F8). Unexpected failures now propagate instead of being muted so
        a silently missing constraint cannot corrupt later ingestion.
        """
        self._create_community_constraint()
        self._create_uid_constraints()

    def _create_community_constraint(self):
        try:
            self.graph_store.execute_query(
                """
                CREATE CONSTRAINT community_unique IF NOT EXISTS
                FOR (c:Community)
                REQUIRE (c.graph_name, c.level, c.id) IS UNIQUE
                """
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to create the Community uniqueness constraint; "
                "duplicate communities already in the graph must be resolved "
                "before ingestion can continue safely."
            ) from e

    # Labels whose per-graph (graph_name, id) pair must be unique.
    _UID_LABELS = ("Document", "Chunk", "__Entity__")

    def _create_uid_constraints(self):
        for label in self._UID_LABELS:
            self._preflight_unique_uids(label)
            self._backfill_uids(label)
            self.graph_store.execute_query(
                f"""
                CREATE CONSTRAINT uid_{label}_unique IF NOT EXISTS
                FOR (n:{label})
                REQUIRE n.uid IS UNIQUE
                """
            )

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
        # Same length-prefixed encoding as the writer, so backfilled uids
        # match newly written ones for graphs ingested before F8.
        self.graph_store.execute_query(
            f"""
            MATCH (n:{label})
            WHERE n.uid IS NULL
            SET n.uid = toString(size(n.graph_name)) + ':' + n.graph_name + ':' + n.id
            RETURN count(n) AS backfilled
            """
        )

    async def resolve_entities_exact(
        self,
        graph_name: str = "entity-graph",
        resolve_property: str = "name",
        dry_run: bool = False,
    ) -> dict:
        resolver = _Neo4jEntityResolver(self.graph_store)
        return await resolver.resolve_exact(
            graph_name=graph_name,
            resolve_property=resolve_property,
            dry_run=dry_run,
        )

    async def resolve_entities_normalized(
        self,
        graph_name: str = "entity-graph",
        resolve_property: str = "name",
        dry_run: bool = False,
    ) -> dict:
        resolver = _Neo4jEntityResolver(self.graph_store)
        return await resolver.resolve_normalized(
            graph_name=graph_name,
            resolve_property=resolve_property,
            dry_run=dry_run,
        )

    async def resolve_entities_fuzzy(
        self,
        graph_name: str = "entity-graph",
        resolve_property: str = "name",
        dry_run: bool = False,
        merge_threshold: float = 95.0,
        review_threshold: float = 85.0,
        max_candidates_per_entity: int = 20,
    ) -> dict:
        resolver = _Neo4jEntityResolver(self.graph_store)
        return await resolver.resolve_fuzzy(
            graph_name=graph_name,
            resolve_property=resolve_property,
            dry_run=dry_run,
            merge_threshold=merge_threshold,
            review_threshold=review_threshold,
            max_candidates_per_entity=max_candidates_per_entity,
        )

    async def resolve_entities_hybrid(
        self,
        graph_name: str = "entity-graph",
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
        resolver = _Neo4jEntityResolver(self.graph_store)
        return await resolver.resolve_hybrid(
            graph_name=graph_name,
            resolve_property=resolve_property,
            dry_run=dry_run,
            merge_threshold=merge_threshold,
            review_threshold=review_threshold,
            max_candidates_per_entity=max_candidates_per_entity,
            aliases=aliases,
            embedder=embedder or self.embedder,
            llm=llm,
            llm_guidance=llm_guidance,
            allow_ai_auto_merge=allow_ai_auto_merge,
            context_properties=context_properties,
            conflict_properties=conflict_properties,
            context_mode=context_mode,
        )

    def verify(self):
        """Print current graph schema info: indexes, node counts, relationship counts."""
        print("\n=== Indexes ===")
        indexes = self.graph_store.execute_query("SHOW INDEXES")
        for idx in indexes:
            print(
                f"  {idx.get('name', '?')}: {idx.get('type', '?')} "
                f"on {idx.get('labelsOrTypes', '?')}.{idx.get('properties', '?')}"
            )

        print("\n=== Node Counts ===")
        counts = self.graph_store.execute_query(
            "MATCH (n) RETURN labels(n) AS labels, count(*) AS count "
            "ORDER BY count DESC LIMIT 20"
        )
        for rec in counts:
            print(f"  {rec['labels']}: {rec['count']}")

        print("\n=== Relationship Counts ===")
        rels = self.graph_store.execute_query(
            "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count "
            "ORDER BY count DESC LIMIT 20"
        )
        for rec in rels:
            print(f"  {rec['type']}: {rec['count']}")
