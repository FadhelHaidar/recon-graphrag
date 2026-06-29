"""Search-layer smoke test for aligned retrieval defaults.

Runs local, global, and DRIFT searches with the Microsoft-GraphRAG-aligned
defaults (use_mixed_context, top_k_relationships, DriftSearchConfig) against
a live Neo4j database that already contains graph data.

This is a lightweight diagnostic script — not a pytest test — intended for
quick developer feedback when tweaking retrieval defaults.  The unit-level
alignment behaviour (top_k_relationships capping, allow_general_knowledge,
HyDE, primer folds, action_use_mixed_context) is exercised with assertions
in ``tests/retrieval/test_alignment.py``.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


async def main():
    from tests.integration.factories import get_embedder, get_llm, get_neo4j_store
    from recon_graphrag.retrieval.search import GraphRAG
    from recon_graphrag.retrieval.drift_types import DriftSearchConfig

    store = get_neo4j_store()
    llm = get_llm()
    embedder = get_embedder()

    # Find available graph names
    rows = store.execute_query(
        "MATCH (n) WHERE n.graph_name IS NOT NULL "
        "RETURN DISTINCT n.graph_name AS gn LIMIT 10"
    )
    graph_names = [r["gn"] for r in rows]
    print(f"Available graph names: {graph_names}")

    if not graph_names:
        print("ERROR: No graphs found in database.")
        sys.exit(1)

    graph_name = graph_names[0]
    print(f"Using graph: {graph_name}")

    # Check community levels
    levels = store.execute_query(
        "MATCH (c:Community {graph_name: $gn}) "
        "RETURN DISTINCT c.level AS level ORDER BY level",
        {"gn": graph_name},
    )
    print(f"Community levels: {[r['level'] for r in levels]}")

    # Count entities
    entity_count = store.execute_query(
        "MATCH (e:__Entity__ {graph_name: $gn}) RETURN count(e) AS cnt",
        {"gn": graph_name},
    )
    print(f"Entities: {entity_count[0]['cnt']}")

    # Create GraphRAG with aligned defaults
    drift_config = DriftSearchConfig(
        use_hyde=False,  # skip HyDE for faster smoke test
        action_use_mixed_context=True,
    )
    grag = GraphRAG(
        store, llm, embedder,
        graph_name=graph_name,
        use_mixed_context=True,
        top_k_relationships=10,
        drift_config=drift_config,
    )

    query = "What are the main entities and their relationships?"
    failures: list[str] = []

    # --- Local search ---
    print("\n=== Local Search ===")
    result = await grag.search(query, mode="local", top_k=3)
    print(f"  answer: {result.answer[:200]}...")
    print(f"  citations: {len(result.citations)}")
    print(f"  metadata keys: {list(result.metadata.keys())}")
    print(f"  mixed_context: {result.metadata.get('mixed_context')}")
    if not result.answer.strip():
        failures.append("Local search returned empty answer")
    if result.metadata.get("mixed_context") is not True:
        failures.append("Local search missing mixed_context=True in metadata")

    # --- Global search ---
    if levels:
        print("\n=== Global Search ===")
        result = await grag.search(
            query, mode="global", community_level="coarsest",
        )
        print(f"  answer: {result.answer[:200]}...")
        print(f"  citations: {len(result.citations)}")
        print(f"  metadata: { {k: v for k, v in result.metadata.items() if k != 'source_resolution_errors'} }")
        if not result.answer.strip():
            failures.append("Global search returned empty answer")

    # --- DRIFT search ---
    print("\n=== DRIFT Search ===")
    result = await grag.search(query, mode="drift", top_k=3)
    print(f"  answer: {result.answer[:200]}...")
    print(f"  citations: {len(result.citations)}")
    trace = result.metadata.get("drift_trace", {})
    print(f"  actions: {len(trace.get('actions', []))}")
    print(f"  total_llm_calls: {trace.get('total_llm_calls')}")
    print(f"  stopping_reason: {trace.get('stopping_reason')}")
    if not result.answer.strip():
        failures.append("DRIFT search returned empty answer")
    if "drift_fallback_reason" in result.metadata:
        failures.append(
            f"DRIFT search fell back: {result.metadata['drift_fallback_reason']}"
        )

    # --- Summary ---
    print("\n=== Smoke test complete ===")
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All alignment smoke checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
