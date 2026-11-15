"""Focused data-quality checks on disposable Neo4j and Memgraph databases."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from recon_graphrag.extraction.description_summarizer import DescriptionSummarizer
from recon_graphrag.extraction.types import ExtractedClaim
from recon_graphrag.utils.tokens import ApproximateTokenCounter
from tests.extraction.test_claim_identity import assemble_claims
from tests.extraction.test_description_coverage import entity_document
from tests.extraction.test_relationship_descriptions import relationship_document
from tests.integration.factories import get_memgraph_store, get_neo4j_store
from tests.integration.support import cleanup_graph, require_integration_env
from tests.pipelines.test_writer_replacement import build_document

pytestmark = [pytest.mark.integration, pytest.mark.database]


@pytest.fixture(params=["neo4j", "memgraph"])
def quality_store(request):
    backend = request.param
    required = ["NEO4J_URL", "NEO4J_USERNAME", "NEO4J_PASSWORD"] if backend == "neo4j" else ["MEMGRAPH_URL"]
    require_integration_env("RUN_DATABASE_INTEGRATION_TESTS", required, "data-quality database tests", fail_on_missing=True)
    store = get_neo4j_store() if backend == "neo4j" else get_memgraph_store()
    graph_name = f"quality-{uuid4().hex}"
    try:
        yield store, graph_name
    finally:
        try:
            cleanup_graph(store, graph_name)
        finally:
            store.driver.close()


def test_distinct_claim_assertions_survive_replacement(quality_store):
    store, graph_name = quality_store
    first = ExtractedClaim("alice", "role", "Alice leads the company.", object_entity_id="acme")
    second = ExtractedClaim("alice", "role", "Alice leads the company.", object_entity_id="beta")
    legacy = assemble_claims([first], graph_name=graph_name)
    legacy.claims[0].id = "legacy-generated-claim"
    store.write_graph_document(legacy)
    current = assemble_claims([first, second], graph_name=graph_name)
    store.write_graph_document(current)
    store.write_graph_document(current)
    rows = store.execute_query(
        "MATCH (c:Claim {graph_name: $graph_name}) RETURN c.id AS id, c.object_entity_id AS object_id",
        {"graph_name": graph_name},
    )
    assert len(rows) == 2
    assert {row["id"] for row in rows} == {claim.id for claim in current.claims}
    assert len({row["object_id"] for row in rows}) == 2


@pytest.mark.asyncio
async def test_property_changes_regenerate_summary_and_omissions_do_not(quality_store):
    store, graph_name = quality_store
    document = build_document(chunks=["chunk"], entities=[("alice", "Alice", "Source fact")], graph_name=graph_name)
    document.entities[0].properties["role"] = "Engineer"
    store.write_graph_document(document)
    llm = MagicMock(ainvoke=AsyncMock(return_value=MagicMock(content="Current summary")))
    summarizer = DescriptionSummarizer(llm, store, graph_name)
    assert (await summarizer.summarize_entities())["summarized"] == 1
    store.write_graph_document(document)
    assert not store.get_entities_needing_summary(graph_name)
    del document.entities[0].properties["role"]
    store.write_graph_document(document)
    assert not store.get_entities_needing_summary(graph_name)
    document.entities[0].properties["role"] = "CEO"
    store.write_graph_document(document)
    assert (await summarizer.summarize_entities())["summarized"] == 1
    assert "CEO" in llm.ainvoke.call_args.args[0]
    document.entities[0].properties["role"] = None
    store.write_graph_document(document)
    assert (await summarizer.summarize_entities())["summarized"] == 1
    assert "CEO" not in llm.ainvoke.call_args.args[0]
    assert llm.ainvoke.await_count == 3


@pytest.mark.asyncio
async def test_resolution_invalidates_previous_entity_summaries(quality_store):
    store, graph_name = quality_store
    document = build_document(chunks=["chunk"], entities=[("one", "Alice", "First fact"), ("two", "ALICE", "Second fact")], graph_name=graph_name)
    store.write_graph_document(document)
    llm = MagicMock(ainvoke=AsyncMock(return_value=MagicMock(content="Summary")))
    summarizer = DescriptionSummarizer(llm, store, graph_name)
    assert (await summarizer.summarize_entities())["summarized"] == 2
    result = await store.resolve_entities_normalized(graph_name=graph_name)
    assert result["merged_groups"] == 1
    assert result["merged_nodes"] == 2
    assert (await summarizer.summarize_entities())["summarized"] == 1


@pytest.mark.asyncio
async def test_relationship_evidence_survives_summary_and_document_replacement(quality_store):
    store, graph_name = quality_store
    store.write_graph_document(relationship_document(["First fact", "Second fact"], graph_name=graph_name))
    store.write_graph_document(relationship_document(["Other document fact"], document_id="other", graph_name=graph_name))
    llm = MagicMock(ainvoke=AsyncMock(return_value=MagicMock(content="Summary")))
    summarizer = DescriptionSummarizer(llm, store, graph_name)
    assert (await summarizer.summarize_relationships())["summarized"] == 1
    assert "First fact" in llm.ainvoke.call_args.args[0]
    assert "Second fact" in llm.ainvoke.call_args.args[0]
    store.write_graph_document(relationship_document(["First fact"], graph_name=graph_name))
    pending = store.get_relationships_needing_summary(graph_name)
    assert len(pending) == 1
    assert set(pending[0]["descriptions"]) == {"First fact", "Other document fact"}
    assert (await summarizer.summarize_relationships())["summarized"] == 1
    assert "Second fact" not in llm.ainvoke.call_args.args[0]


@pytest.mark.asyncio
async def test_complete_entity_evidence_reaches_bounded_summary_requests(quality_store):
    store, graph_name = quality_store
    descriptions = ["Evidence " * 500, "MARKER_AFTER_2000"]
    document = entity_document(descriptions, graph_name=graph_name)
    store.write_graph_document(document)
    candidates = store.get_entities_needing_summary(graph_name)
    assert len(candidates) == 1
    assert candidates[0]["descriptions"] == [text.strip() for text in descriptions]
    llm = MagicMock(ainvoke=AsyncMock(return_value=MagicMock(content="Bounded summary")))
    counter = ApproximateTokenCounter(ratio=1)
    summarizer = DescriptionSummarizer(llm, store, graph_name, token_counter=counter,
                                      summary_token_budget=900, summary_output_tokens=64)
    assert (await summarizer.summarize_entities())["summarized"] == 1
    prompts = [call.args[0] for call in llm.ainvoke.await_args_list]
    assert len(prompts) > 1
    assert any("MARKER_AFTER_2000" in prompt for prompt in prompts)
    assert all(counter.count(prompt) + 64 <= 900 for prompt in prompts)
    store.write_graph_document(entity_document(["Other owned fact"], document_id="other", graph_name=graph_name))
    store.write_graph_document(entity_document(["Replacement fact"], graph_name=graph_name))
    pending = store.get_entities_needing_summary(graph_name)
    assert set(pending[0]["descriptions"]) == {"Other owned fact", "Replacement fact"}


@pytest.mark.asyncio
async def test_embedding_failed_first_batch_does_not_starve_backlog(quality_store):
    from recon_graphrag.embeddings import EntityEmbedder

    store, graph_name = quality_store
    store.execute_query(
        "UNWIND range(0, 3) AS i CREATE (e:__Entity__:Person "
        "{graph_name: $graph_name, id: toString(i), type: 'Person', "
        "name: 'node' + toString(i), description: 'fact'})", {"graph_name": graph_name},
    )
    first = store.get_unembedded_entities(graph_name, limit=2)
    failed_names = {item["name"] for item in first}
    llm = MagicMock()

    async def embed(text):
        if any(name in text for name in failed_names):
            raise RuntimeError("failed provider")
        return [0.1] * 1536

    llm.async_embed_query = AsyncMock(side_effect=embed)
    worker = EntityEmbedder(store, llm, graph_name=graph_name)
    await worker.embed_entities(batch_size=2)
    pending = store.get_unembedded_entities(graph_name, model_fingerprint=worker.model_fingerprint)
    assert {item["name"] for item in pending} == failed_names
    assert llm.async_embed_query.await_count == 4
    assert not store.get_unembedded_entities(graph_name, exclude_ids=[item["id"] for item in pending])
    llm.async_embed_query = AsyncMock(return_value=[0.1] * 1536)
    await worker.embed_entities(batch_size=2)
    assert not store.get_unembedded_entities(graph_name, model_fingerprint=worker.model_fingerprint)
