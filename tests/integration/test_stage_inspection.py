"""Stage-by-stage inspection scenario: real database, real providers.

Runs (``RUN_E2E_INTEGRATION_TESTS=1``):

    NEO4J_URL=bolt://localhost:7688 MEMGRAPH_URL=bolt://localhost:7689 \
    RUN_E2E_INTEGRATION_TESTS=1 uv run pytest -s \
        tests/integration/test_stage_inspection.py

Each stage asserts one contract and prints what it observed, so a failing
stage names the exact pipeline layer that broke:

1. extraction      -> entities and relationships written to the graph
2. resolution      -> which surface forms merge and which stay separate
3. embeddings      -> which nodes carry vectors and what similarity returns
4. communities     -> how entities group and which reports are produced
5. retrieval       -> which evidence a query pulls and which chunks it cites

Stage 2 seeds structured rows instead of text, so the merge outcome is exact
and does not depend on what the model happens to extract. Everything else uses
the real LLM and embedder, so only structural invariants are asserted: the
model's wording and entity coverage vary between runs.
"""

from __future__ import annotations

from typing import Any

import pytest

from recon_graphrag import CommunityPipeline, GraphBuilderPipeline, IndexManager
from recon_graphrag.graphdb.memgraph.store import MemgraphGraphStore
from recon_graphrag.retrieval.search_global import GlobalSearchRetriever
from recon_graphrag.retrieval.search_local import LocalSearchRetriever
from tests.integration.factories import (
    get_embedder,
    get_llm,
    get_memgraph_store,
    get_neo4j_store,
)
from tests.integration.stage_scenario_support import (
    STAGE_COMMUNITY_RELATIONSHIP_TYPES,
    STAGE_PAGES,
    STAGE_RESOLUTION_MAPPING,
    STAGE_RESOLUTION_ROWS,
    STAGE_SCHEMA,
    chunk_ids,
    chunk_rows,
    community_members,
    community_reports,
    entity_rows,
    normalized_name,
    relationship_rows,
)
from tests.integration.support import (
    cleanup_graph,
    require_integration_env,
    require_selected_provider_env,
)

BACKENDS = {
    "neo4j": (get_neo4j_store, ["NEO4J_URL", "NEO4J_USERNAME", "NEO4J_PASSWORD"]),
    "memgraph": (get_memgraph_store, ["MEMGRAPH_URL"]),
}

pytestmark = [pytest.mark.integration, pytest.mark.e2e]


class Scenario:
    """Runs the expensive stages once per backend, on first use."""

    def __init__(self, store: Any, graph_name: str) -> None:
        self.store = store
        self.graph_name = graph_name
        self.llm: Any = None
        self.embedder: Any = None
        self._build_result: dict | None = None
        self._community_result: dict | None = None

    def providers(self) -> tuple[Any, Any]:
        if self.llm is None:
            self.llm = get_llm()
            self.embedder = get_embedder()
        return self.llm, self.embedder

    async def built(self) -> dict:
        """Create indexes, then ingest the corpus and finalize the graph."""
        if self._build_result is None:
            llm, embedder = self.providers()
            dimension_probe = await embedder.async_embed_query("dimension probe")
            try:
                IndexManager(
                    self.store, embedding_dim=len(dimension_probe)
                ).create_indexes()
            except ValueError as error:
                # Memgraph's SHOW INDEX INFO omits vector configuration, so setup
                # never re-verifies a vector index it already created. The live
                # container is disposable, the index exists, and stage 3 proves
                # the index actually answers a query. Any other failure is real.
                if not (
                    isinstance(self.store, MemgraphGraphStore)
                    and "does not expose its vector configuration" in str(error)
                ):
                    raise
                print(f"[stage 1] reusing existing Memgraph index: {error}")
            builder = GraphBuilderPipeline(
                graph_store=self.store,
                llm=llm,
                embedder=embedder,
                graph_name=self.graph_name,
                entity_resolution_strategy="normalized",
                embed_chunks=True,
            )
            results = await builder.build_from_documents(
                [
                    {
                        "pages": STAGE_PAGES,
                        "metadata": {
                            "source": f"{self.graph_name}-source",
                            "collection": "stage-scenario",
                            "external_id": f"{self.graph_name}-external",
                        },
                    }
                ],
                schema=STAGE_SCHEMA,
                max_gleanings=0,
            )
            self._build_result = results[0]
        return self._build_result

    async def with_communities(self) -> dict:
        """Detect communities and generate reports over the built graph."""
        if self._community_result is None:
            await self.built()
            llm, embedder = self.providers()
            pipeline = CommunityPipeline(
                graph_store=self.store,
                llm=llm,
                embedder=embedder,
                relationship_types=STAGE_COMMUNITY_RELATIONSHIP_TYPES,
                graph_name=self.graph_name,
            )
            self._community_result = await pipeline.build()
        return self._community_result


@pytest.fixture(params=sorted(BACKENDS))
def scenario(request):
    backend = request.param
    store_factory, required_env = BACKENDS[backend]
    require_integration_env(
        "RUN_E2E_INTEGRATION_TESTS",
        required_env,
        f"stage-by-stage scenario ({backend})",
        fail_on_missing=True,
    )
    require_selected_provider_env(f"stage-by-stage scenario ({backend})")

    store = store_factory()
    graph_name = f"stage-scenario-{backend}"
    cleanup_graph(store, graph_name)
    try:
        yield Scenario(store, graph_name)
    finally:
        try:
            cleanup_graph(store, graph_name)
        finally:
            store.driver.close()


def _print_rows(title: str, rows: list[dict]) -> None:
    print(f"\n  {title} ({len(rows)}):")
    for row in rows:
        print(f"    {row}")


async def test_stage_1_extraction_writes_entities_and_relationships(scenario):
    """The LLM extraction pass turns pages into typed nodes and edges."""
    result = await scenario.built()

    entities = entity_rows(scenario.store, scenario.graph_name)
    relationships = relationship_rows(scenario.store, scenario.graph_name)
    chunks = chunk_rows(scenario.store, scenario.graph_name)

    print(f"\n[stage 1] build validation: {result.get('validation')}")
    print(f"[stage 1] write stats: {result.get('extraction', {}).get('write_stats')}")
    _print_rows("entities", entities)
    _print_rows("relationships", relationships)
    _print_rows("chunks", [{"id": c["id"], "chars": len(c["text"] or "")} for c in chunks])

    assert entities, "extraction produced no entities"
    assert relationships, "extraction produced no entity relationships"
    assert chunks, "ingestion stored no chunks"

    labels = {label for row in entities for label in row["labels"] if label != "__Entity__"}
    assert len(labels) >= 2, f"expected several entity types, got {sorted(labels)}"

    for row in entities:
        assert row["name"], f"entity {row['id']} has no name"
        assert row["descriptions"], f"entity {row['name']} has no descriptions"

    # Every relationship endpoint must be a stored entity, not a dangling name.
    names = {row["name"] for row in entities}
    for row in relationships:
        assert row["source"] in names, f"dangling relationship source {row['source']}"
        assert row["target"] in names, f"dangling relationship target {row['target']}"


async def test_stage_2_resolution_merges_normalized_variants(scenario):
    """Same rows, three strategies: what merges, what stays separate, and why."""
    llm, embedder = scenario.providers()
    graphs = {
        "unresolved": f"{scenario.graph_name}-unresolved",
        "normalized": f"{scenario.graph_name}-normalized",
        "fuzzy": f"{scenario.graph_name}-fuzzy",
    }

    async def ingest(graph_name: str, *, resolve: bool, strategy: str) -> None:
        builder = GraphBuilderPipeline(
            graph_store=scenario.store,
            llm=llm,
            embedder=embedder,
            graph_name=graph_name,
            perform_entity_resolution=resolve,
            entity_resolution_strategy=strategy,
            embed_entities=False,
            summarize_descriptions=False,
        )
        await builder.build_from_rows(
            STAGE_RESOLUTION_ROWS,
            STAGE_RESOLUTION_MAPPING,
            metadata={"source": f"{graph_name}-source", "collection": "stage-scenario"},
        )

    counts: dict[str, dict[str, int]] = {}
    try:
        await ingest(graphs["unresolved"], resolve=False, strategy="normalized")
        await ingest(graphs["normalized"], resolve=True, strategy="normalized")
        await ingest(graphs["fuzzy"], resolve=True, strategy="fuzzy")

        for stage, graph_name in graphs.items():
            entities = entity_rows(scenario.store, graph_name)
            counts[stage] = {}
            for row in entities:
                key = normalized_name(row["name"])
                counts[stage][key] = counts[stage].get(key, 0) + 1
            _print_rows(
                f"{stage}: name -> node count",
                [{"name": row["name"], "id": row["id"]} for row in entities],
            )
            print(f"[stage 2] {stage}: {counts[stage]}")

        # Before resolution every surface form is its own node.
        assert sum(counts["unresolved"].values()) == len(STAGE_RESOLUTION_ROWS)
        assert counts["unresolved"] == {"alicechen": 3, "achen": 1, "bobortiz": 2}

        # Normalization folds case, spacing, and punctuation, so the three
        # "Alice Chen" forms become one node and " bob ortiz " folds too.
        for stage in ("normalized", "fuzzy"):
            assert counts[stage] == {"alicechen": 1, "achen": 1, "bobortiz": 1}, (
                f"{stage} resolution produced {counts[stage]}"
            )

        merged = next(
            row
            for row in entity_rows(scenario.store, graphs["normalized"])
            if normalized_name(row["name"]) == "alicechen"
        )
        assert len(merged["descriptions"]) >= 2, (
            f"merged node kept {len(merged['descriptions'])} descriptions"
        )
        print(f"[stage 2] merged node keeps descriptions: {merged['descriptions']}")
        print(
            "[stage 2] 'A. Chen' stays separate: it normalizes to 'achen', and both "
            "strategies only merge entities whose normalized names are equal."
        )
    finally:
        for graph_name in graphs.values():
            cleanup_graph(scenario.store, graph_name)


async def test_stage_3_embeddings_are_stored_and_searchable(scenario):
    """Vectors are written for entities and chunks, then used for similarity."""
    await scenario.built()
    _, embedder = scenario.providers()

    entities = entity_rows(scenario.store, scenario.graph_name)
    chunks = chunk_rows(scenario.store, scenario.graph_name)

    print(f"\n[stage 3] entities with vectors: "
          f"{sum(1 for row in entities if row['has_vector'])}/{len(entities)}")
    print(f"[stage 3] chunks with vectors: "
          f"{sum(1 for row in chunks if row['has_vector'])}/{len(chunks)}")

    assert all(row["has_vector"] for row in entities), (
        "some entities were left unembedded: "
        f"{[row['name'] for row in entities if not row['has_vector']]}"
    )
    assert all(row["has_vector"] for row in chunks), (
        "chunk vectors missing although embed_chunks=True: "
        f"{[row['id'] for row in chunks if not row['has_vector']]}"
    )

    query = "Who investigated the ransomware incident at Northwind Health?"
    query_vector = await embedder.async_embed_query(query)
    matches = scenario.store.vector_search(
        "entity-embeddings",
        query_vector,
        k=3,
        filters={"graph_name": scenario.graph_name},
    )
    print(f"[stage 3] similarity search for {query!r} -> {matches}")

    assert matches, "entity vector search returned nothing"
    scores = [row["score"] for row in matches]
    assert scores == sorted(scores, reverse=True), f"similarity scores not descending: {scores}"
    assert len(matches) <= 3


async def test_stage_4_communities_group_entities_into_reports(scenario):
    """Leiden groups the entities and every group gets an LLM report."""
    result = await scenario.with_communities()

    members = community_members(scenario.store, scenario.graph_name)
    reports = community_reports(scenario.store, scenario.graph_name)
    stats = scenario.store.get_community_stats(scenario.graph_name)

    print(f"\n[stage 4] build result: {result}")
    _print_rows("community stats", stats)
    _print_rows("community membership", members)
    _print_rows(
        "reports",
        [{"id": row["id"], "level": row["level"],
          "chars": len(row["report_text"] or ""), "has_vector": row["has_vector"]}
         for row in reports],
    )

    assert result.get("communities", 0) > 0, "community detection produced nothing"
    assert result.get("reports", 0) > 0, "no community reports were generated"
    assert result.get("embedded_reports", 0) > 0, "no report vectors were stored"

    assert members, "no entity was assigned to a community"
    for row in reports:
        assert row["report_text"], f"report {row['id']} is empty"
        assert row["has_vector"], f"report {row['id']} has no vector"


async def test_stage_5_retrieval_builds_context_with_citations(scenario):
    """Local and global search turn the graph into cited context."""
    await scenario.with_communities()
    llm, embedder = scenario.providers()

    stored_chunk_ids = chunk_ids(scenario.store, scenario.graph_name)

    local = LocalSearchRetriever(
        scenario.store, llm, embedder, graph_name=scenario.graph_name
    )
    local_result = await local.search(
        "Who investigated the ransomware incident, and what mitigated it?"
    )

    global_search = GlobalSearchRetriever(
        scenario.store, llm, graph_name=scenario.graph_name
    )
    global_result = await global_search.search(
        "What systems and people are involved in incident response?",
        community_level="coarsest",
    )

    for label, result in (("local", local_result), ("global", global_result)):
        print(f"\n[stage 5] {label} answer: {result.answer[:400]!r}")
        print(f"[stage 5] {label} citations: {[(c.chunk_id, c.document_name) for c in result.citations]}")
        print(f"[stage 5] {label} metadata: {result.metadata}")
        if label == "local":
            print(f"[stage 5] local context ({len(result.context)} chars): "
                  f"{result.context[:600]!r}")

    assert local_result.answer.strip(), "local search returned an empty answer"
    assert local_result.context.strip(), "local search built no context"
    assert local_result.citations, "local search returned no citations"
    for citation in local_result.citations:
        assert citation.chunk_id in stored_chunk_ids, (
            f"citation {citation.chunk_id} is not a stored chunk"
        )

    assert global_result.answer.strip(), "global search returned an empty answer"
    assert global_result.metadata.get("reports_used", 0) >= 1, (
        f"global search used no reports: {global_result.metadata}"
    )
    assert global_result.metadata.get("map_batches", 0) >= 1, (
        f"global search ran no map batches: {global_result.metadata}"
    )
