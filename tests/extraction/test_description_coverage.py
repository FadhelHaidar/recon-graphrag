"""Complete raw evidence and bounded summary request coverage."""

from dataclasses import replace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from recon_graphrag.extraction.assembler import GraphDocumentAssembler
from recon_graphrag.extraction.chunking import TextChunk
from recon_graphrag.extraction.description_summarizer import (
    DescriptionSummarizationError,
    DescriptionSummarizer,
)
from recon_graphrag.extraction.types import ExtractedNode, GraphExtraction
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.utils.tokens import ApproximateTokenCounter
from tests.extraction.test_description_summarizer import SummaryStore
from tests.pipelines.test_writer_replacement import make_writer
from tests.pipelines.writer_scenarios import StatefulFakeGraphStore


def entity_document(descriptions, *, document_id="document", graph_name="quality"):
    chunks = [TextChunk(id=f"{document_id}:{i}", text="source", index=i, metadata={}) for i in range(len(descriptions))]
    return GraphDocumentAssembler().assemble(
        document_id=document_id, text_hash="hash", chunks=chunks, metadata={}, graph_name=graph_name,
        chunk_extractions={chunk.id: GraphExtraction(nodes=[
            ExtractedNode("alice", "Person", {"canonical_key": "alice", "description": text}),
        ]) for chunk, text in zip(chunks, descriptions, strict=True)},
    )


def test_writer_persists_full_observations_not_the_capped_preview():
    descriptions = ["First " * 500, "MARKER_AFTER_2000"]
    document = entity_document(descriptions)
    assert "MARKER_AFTER_2000" not in document.entities[0].properties["description"]
    store = StatefulFakeGraphStore()
    make_writer(store).write_graph_document(document)
    node = store.get_node(document.entities[0].id)
    assert node is not None
    assert node["descriptions"] == [text.strip() for text in descriptions]


def test_observation_ownership_is_validated_before_database_queries():
    document = entity_document(["Source fact"])
    observation = document.entities[0].description_observations[0]
    observation.source = replace(observation.source, document_id="someone-else")
    store = StatefulFakeGraphStore()
    with pytest.raises(ValueError, match="observation"):
        make_writer(store).write_graph_document(document)
    assert not store.queries


@pytest.mark.asyncio
async def test_summary_partitions_cover_oversized_observation_and_bound_every_request():
    store = SummaryStore()
    original = "0123456789" * 300 + "FINAL_MARKER"
    store.items[0]["descriptions"] = [original]
    counter = ApproximateTokenCounter(ratio=1)
    llm = MagicMock(ainvoke=AsyncMock(return_value=MagicMock(content="Concise")))
    summarizer = DescriptionSummarizer(llm, cast(GraphStore, store), "g", token_counter=counter,
                                      summary_token_budget=700, summary_output_tokens=64)
    batches = summarizer._partition_descriptions(store.items[0], [original], summarizer._build_entity_summary_prompt)
    assert "".join(text for batch in batches for text in batch) == original
    assert (await summarizer.summarize_entities())["summarized"] == 1
    prompts = [call.args[0] for call in llm.ainvoke.await_args_list]
    assert len(prompts) > 1
    assert any("FINAL_MARKER" in prompt for prompt in prompts)
    assert all(counter.count(prompt) + 64 <= 700 for prompt in prompts)
    assert store.items[0]["descriptions"] == [original]


@pytest.mark.asyncio
async def test_failed_evidence_batch_retains_raw_text_and_marks_item_failed():
    store = SummaryStore()
    original = "Evidence " * 600
    store.items[0]["descriptions"] = [original]
    llm = MagicMock(ainvoke=AsyncMock(side_effect=[MagicMock(content="First summary"), RuntimeError("private")]))
    summarizer = DescriptionSummarizer(llm, cast(GraphStore, store), "g", token_counter=ApproximateTokenCounter(ratio=1),
                                      summary_token_budget=700, summary_output_tokens=64)
    with pytest.raises(DescriptionSummarizationError):
        await summarizer.summarize_entities()
    assert store.items[0]["description_summary_status"] == "failed"
    assert store.items[0]["descriptions"] == [original.strip()]
    assert store.items[0]["description"] == "previous"
    summarizer.llm = MagicMock(ainvoke=AsyncMock(return_value=MagicMock(content="Recovered")))
    assert (await summarizer.summarize_entities())["summarized"] == 1
    assert store.items[0]["descriptions"] == [original.strip()]


@pytest.mark.asyncio
async def test_fixed_context_that_cannot_fit_fails_without_provider_calls():
    store = SummaryStore()
    store.items[0]["props"] = {"context": "x" * 1000}
    llm = MagicMock(ainvoke=AsyncMock())
    summarizer = DescriptionSummarizer(llm, cast(GraphStore, store), "g", token_counter=ApproximateTokenCounter(ratio=1),
                                      summary_token_budget=700, summary_output_tokens=64)
    with pytest.raises(DescriptionSummarizationError):
        await summarizer.summarize_entities()
    llm.ainvoke.assert_not_awaited()


@pytest.mark.parametrize("original", ["αβ🙂 " * 350, "plain " * 500])
def test_partition_keeps_every_unicode_character(original):
    store = SummaryStore()
    summarizer = DescriptionSummarizer(MagicMock(), cast(GraphStore, store), "g", token_counter=ApproximateTokenCounter(ratio=1),
                                      summary_token_budget=700, summary_output_tokens=64)
    batches = summarizer._partition_descriptions(store.items[0], [original], summarizer._build_entity_summary_prompt)
    assert "".join(text for batch in batches for text in batch) == original
    assert all(summarizer.token_counter.count(summarizer._budgeted_prompt(
        store.items[0], batch, summarizer._build_entity_summary_prompt)) + 64 <= 700 for batch in batches)


@pytest.mark.asyncio
@pytest.mark.parametrize(("content", "finish_reason", "message"), [
    ("x" * 65, "stop", "oversized output"),
    ("short", "length", "incomplete output"),
])
async def test_invalid_batch_output_is_not_persisted_as_success(content, finish_reason, message, monkeypatch):
    store = SummaryStore()
    llm = MagicMock(ainvoke=AsyncMock(return_value=MagicMock(content=content, finish_reason=finish_reason)))
    monkeypatch.setattr("recon_graphrag.extraction.description_summarizer.asyncio.sleep", AsyncMock())
    summarizer = DescriptionSummarizer(llm, cast(GraphStore, store), "g", token_counter=ApproximateTokenCounter(ratio=1),
                                      summary_token_budget=700, summary_output_tokens=64)
    with pytest.raises(DescriptionSummarizationError, match=message):
        await summarizer.summarize_entities()
    assert llm.ainvoke.await_count == 3
    assert store.items[0]["description_summary_status"] == "failed"
    assert store.items[0]["description"] == "previous"


@pytest.mark.parametrize(("budget", "reserve"), [(0, 0), (64, 64), (64, 65), (100, -1)])
def test_invalid_summary_budget_is_rejected(budget, reserve):
    with pytest.raises(ValueError, match="budget"):
        DescriptionSummarizer(MagicMock(), cast(GraphStore, SummaryStore()), "g", summary_token_budget=budget,
                              summary_output_tokens=reserve)


@pytest.mark.asyncio
async def test_nonshrinking_summary_consolidation_fails_instead_of_looping():
    store = SummaryStore()
    store.items[0]["descriptions"] = [letter * 100 for letter in "ABC"]
    llm = MagicMock(ainvoke=AsyncMock(return_value=MagicMock(content="x" * 180)))
    summarizer = DescriptionSummarizer(llm, cast(GraphStore, store), "g",
                                      token_counter=ApproximateTokenCounter(ratio=1),
                                      summary_token_budget=400, summary_output_tokens=180)
    summarizer._build_entity_summary_prompt = lambda entity: "\n".join(entity["descriptions"])
    with pytest.raises(DescriptionSummarizationError):
        await summarizer.summarize_entities()
    assert llm.ainvoke.await_count <= 6
    assert store.items[0]["description_summary_status"] == "failed"
    assert store.items[0]["descriptions"] == [letter * 100 for letter in "ABC"]
