"""Synthetic end-to-end smoke test.

Uses a compact security/incident corpus with structural-only assertions.
Controlled by ``RUN_E2E_INTEGRATION_TESTS=1``.

Marked ``@pytest.mark.integration`` + ``@pytest.mark.e2e``.
"""

from __future__ import annotations

import pytest

from tests.integration.factories import get_embedder, get_llm, get_memgraph_store, get_neo4j_store
from recon_graphrag import CommunityPipeline, GraphBuilderPipeline, GraphRAG, IndexManager
from tests.integration.support import (
    cleanup_graph,
    require_integration_env,
    require_selected_provider_env,
)
from tests.integration.synthetic_e2e_support import (
    SYNTHETIC_COMMUNITY_RELATIONSHIP_TYPES,
    SYNTHETIC_DRIFT_ANSWER_PROMPT,
    SYNTHETIC_GLOBAL_MAP_PROMPT,
    SYNTHETIC_GLOBAL_REDUCE_PROMPT,
    SYNTHETIC_LOCAL_ANSWER_PROMPT,
    SYNTHETIC_PAGES,
    SYNTHETIC_SCHEMA,
    assert_build_validation_positive,
    assert_graph_persisted,
)


def _make_store_fixture(store_factory, run_flag, required_env, graph_name):
    @pytest.fixture
    def fixture():
        require_integration_env(run_flag, required_env, "Synthetic E2E test", fail_on_missing=True)
        store = store_factory()
        cleanup_graph(store, graph_name)
        try:
            yield store, graph_name
        finally:
            try:
                cleanup_graph(store, graph_name)
            finally:
                store.driver.close()

    return fixture


neo4j_e2e = _make_store_fixture(
    get_neo4j_store,
    "RUN_E2E_INTEGRATION_TESTS",
    ["NEO4J_URL", "NEO4J_USERNAME", "NEO4J_PASSWORD"],
    "synthetic-e2e-neo4j",
)

memgraph_e2e = _make_store_fixture(
    get_memgraph_store,
    "RUN_E2E_INTEGRATION_TESTS",
    ["MEMGRAPH_URL"],
    "synthetic-e2e-memgraph",
)


async def _run_synthetic_e2e(store, graph_name: str):
    require_selected_provider_env("Synthetic E2E test")
    llm = get_llm()
    embedder = get_embedder()

    try:
        dimension_probe = await embedder.async_embed_query("probe")
        IndexManager(store, embedding_dim=len(dimension_probe)).create_indexes()

        builder = GraphBuilderPipeline(
            graph_store=store,
            llm=llm,
            embedder=embedder,
            schema=SYNTHETIC_SCHEMA,
            graph_name=graph_name,
            extract_claims=True,
            entity_resolution_strategy="hybrid",
            allow_ai_auto_merge=False,
        )
        build_result = (await builder.build_from_documents(
            [{
                "pages": SYNTHETIC_PAGES,
                "metadata": {
                    "source": f"{graph_name}-source",
                    "collection": "e2e-synthetic",
                    "external_id": f"{graph_name}-external",
                },
            }],
            window_size=2,
            window_overlap=1,
        ))[0]

        validation = build_result.get("validation", {})
        assert_build_validation_positive(validation)

        assert_graph_persisted(store, graph_name)

        writer_doc = build_result.get("graph_document")
        if writer_doc:
            for rel in writer_doc.relationships:
                assert rel.observation_count >= 1

        write_stats = build_result.get("extraction", {}).get("write_stats", {})
        claims_written = write_stats.get("claims", 0)
        print(f"Claims written: {claims_written}")

        community = CommunityPipeline(
            graph_store=store,
            llm=llm,
            embedder=embedder,
            relationship_types=SYNTHETIC_COMMUNITY_RELATIONSHIP_TYPES,
            graph_name=graph_name,
            summarize_concurrency=5,
        )
        community_result = await community.build()
        assert community_result.get("communities", 0) > 0
        assert community_result.get("reports", 0) > 0
        assert community_result.get("embedded_reports", 0) > 0

        graph_rag = GraphRAG(
            store,
            llm,
            embedder,
            graph_name=graph_name,
            use_mixed_context=True,
        )
        graph_rag.local.answer_prompt = SYNTHETIC_LOCAL_ANSWER_PROMPT
        graph_rag.global_.map_prompt = SYNTHETIC_GLOBAL_MAP_PROMPT
        graph_rag.global_.reduce_prompt = SYNTHETIC_GLOBAL_REDUCE_PROMPT
        graph_rag.drift.reduce_prompt = SYNTHETIC_DRIFT_ANSWER_PROMPT

        results = [
            await graph_rag.search(
                "Who investigated the suspicious login incident?",
                mode="local",
                synthesize_citation_metadata=True,
                synthesis_metadata_keys=["record_ids", "collections", "external_id"],
            ),
            await graph_rag.search(
                "What systems and organizations are involved in incident response?",
                mode="global",
                community_level="coarsest",
            ),
            await graph_rag.search(
                "How does the Identity Gateway connect to the Payments API?",
                mode="drift",
                community_level="finest",
                synthesize_citation_metadata=True,
                synthesis_metadata_keys=["record_ids", "collections", "external_id"],
            ),
            await graph_rag.search(
                "What are the main systems and their dependencies?",
                mode="global",
                community_level="coarsest",
                random_seed=42,
            ),
        ]

        for result in results:
            assert result.answer.strip(), f"Empty answer for {result.mode} search"

        drift_result = results[2]
        assert "drift_fallback_reason" not in drift_result.metadata
        assert drift_result.metadata["drift_trace"]["primer_report_ids"]

        for result in (results[0], results[2]):
            assert result.citations, f"Expected citations for {result.mode} search"
            citation = result.citations[0]
            assert citation.chunk_id
            assert citation.document_id

        global_result = results[3]
        assert global_result.metadata.get("reports_available", 0) > 0
        assert global_result.metadata.get("map_batches", 0) > 0
        assert global_result.metadata.get("elapsed_ms", 0) > 0

        import json

        def safe(text: str, limit: int = 0) -> str:
            text = text.encode("ascii", "replace").decode("ascii")
            if limit and len(text) > limit:
                text = text[:limit] + f"... ({len(text) - limit} chars truncated)"
            return text

        separator = "=" * 70
        section = "-" * 70

        report: dict = {
            "backend": graph_name,
            "build": validation,
            "communities": community_result,
            "searches": [],
        }

        for r in results:
            entry: dict = {
                "mode": r.mode,
                "answer": r.answer,
                "context_chars": len(r.context) if r.context else 0,
                "citations_count": len(r.citations),
                "citations": [
                    {
                        "document_id": c.document_id,
                        "chunk_id": c.chunk_id,
                        "document_name": c.document_name,
                        "excerpt": c.excerpt,
                        "metadata": c.metadata,
                    }
                    for c in r.citations
                ],
                "metadata": r.metadata,
            }
            report["searches"].append(entry)

        print(f"\n{separator}")
        print(f"  E2E SEARCH REPORT: {graph_name}")
        print(separator)

        print(f"\n{section}")
        print("  BUILD")
        print(section)
        print(json.dumps(report["build"], indent=2, default=str))

        print(f"\n{section}")
        print("  COMMUNITIES")
        print(section)
        print(json.dumps(report["communities"], indent=2, default=str))

        for entry in report["searches"]:
            mode = entry["mode"].upper()
            print(f"\n{separator}")
            print(f"  SEARCH MODE: {mode}")
            print(separator)

            print(f"\n  ANSWER:")
            print(f"  {safe(entry['answer'])}")

            print(f"\n  CITATIONS: {entry['citations_count']}")
            for i, c in enumerate(entry["citations"]):
                print(f"    [{i}] {c['document_id']}  |  {c['chunk_id']}")
                if c["document_name"]:
                    print(f"        name:     {c['document_name']}")
                if c["excerpt"]:
                    print(f"        excerpt:  {safe(c['excerpt'], 200)}")
                if c["metadata"]:
                    filtered = {
                        k: v for k, v in c["metadata"].items()
                        if k in ("record_ids", "collections", "external_id",
                                 "source", "page_start", "page_end")
                    }
                    if filtered:
                        print(f"        metadata: {json.dumps(filtered, default=str)}")

            print(f"\n  METADATA:")
            md = entry["metadata"]
            for k, v in md.items():
                if k == "drift_trace":
                    print(f"    drift_trace:")
                    if isinstance(v, dict):
                        for tk, tv in v.items():
                            if tk == "actions" and isinstance(tv, list):
                                print(f"      actions: {len(tv)}")
                                for a in tv:
                                    if isinstance(a, dict):
                                        print(f"        - id={a.get('id')}  depth={a.get('depth')}  "
                                              f"score={a.get('score')}  status={a.get('status')}")
                                        print(f"          query:  {safe(a.get('query', ''), 120)}")
                                        print(f"          answer: {safe(a.get('answer', ''), 120)}")
                            else:
                                val = safe(str(tv), 200) if isinstance(tv, str) else tv
                                print(f"      {tk}: {val}")
                    else:
                        print(f"      {safe(str(v), 200)}")
                elif k in ("context", "answer_synthesis_context"):
                    val_str = safe(str(v), 500)
                    print(f"    {k}: ({len(str(v))} chars) {val_str}")
                else:
                    print(f"    {k}: {v}")

            if entry["context_chars"]:
                print(f"\n  CONTEXT: {entry['context_chars']} chars")

    finally:
        for resource in (llm, embedder):
            close = getattr(resource, "aclose", None)
            if callable(close):
                await close()


@pytest.mark.integration
@pytest.mark.e2e
async def test_synthetic_e2e_neo4j(neo4j_e2e):
    store, graph_name = neo4j_e2e
    await _run_synthetic_e2e(store, graph_name)


@pytest.mark.integration
@pytest.mark.e2e
async def test_synthetic_e2e_memgraph(memgraph_e2e):
    store, graph_name = memgraph_e2e
    await _run_synthetic_e2e(store, graph_name)
