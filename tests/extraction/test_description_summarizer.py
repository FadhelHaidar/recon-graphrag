"""Description summarization retries, partial persistence, and task ownership."""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.llm.base import BaseLLM
from recon_graphrag.extraction import description_summarizer as module
from recon_graphrag.extraction.description_summarizer import (
    DescriptionSummarizationError,
    DescriptionSummarizer,
)


class SummaryStore:
    def __init__(self, count=1):
        self.items = [
            {"id": str(i), "name": str(i), "descriptions": [" original "],
             "description": "previous", "description_summary_status": None}
            for i in range(count)
        ]
        self.writes = []
        self.fetches = 0
        self.write_error = None

    def fetch(self, graph_name, limit):
        self.fetches += 1
        return [dict(item) for item in self.items
                if item["description_summary_status"] != "success"][:limit]

    def persist(self, graph_name, rows):
        self.writes.append(rows)
        if self.write_error:
            raise self.write_error
        for row in rows:
            item = next(item for item in self.items if item["id"] == row["id"])
            item.update({key: value for key, value in row.items()
                         if key != "description" or value is not None})

    get_entities_needing_summary = fetch
    get_relationships_needing_summary = fetch
    persist_entity_summaries = persist
    persist_relationship_summaries = persist


@pytest.fixture
def setup(monkeypatch):
    store = SummaryStore()
    llm = SimpleNamespace(ainvoke=AsyncMock())
    summarizer = DescriptionSummarizer(cast(BaseLLM, llm), cast(GraphStore, store), "g")
    sleep = AsyncMock()
    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    return summarizer, store, llm, sleep


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["entities", "relationships"])
@pytest.mark.parametrize("invalid_count", [0, 1, 2])
async def test_description_summarizer_success_and_parser_retry(setup, monkeypatch, kind, invalid_count):
    summarizer, store, llm, sleep = setup
    llm.ainvoke.side_effect = [SimpleNamespace(content="  ")] * invalid_count + [
        SimpleNamespace(content=" concise ")]
    stages = []

    @contextmanager
    def stage(name):
        stages.append(name)
        yield

    monkeypatch.setattr(module, "token_stage", stage)
    result = await getattr(summarizer, f"summarize_{kind}")()
    assert result == {"summarized": 1, "batches": 1}
    assert llm.ainvoke.await_count == invalid_count + 1
    assert len({call.args[0] for call in llm.ainvoke.await_args_list}) == 1
    assert [call.args[0] for call in sleep.await_args_list] == list(range(1, invalid_count + 1))
    assert stages == [f"construction.describe_{kind}"] * (invalid_count + 1)
    assert len(store.writes) == 1
    row = store.writes[0][0]
    assert row["description"] == "concise"
    assert row["description_summary_status"] == "success"
    assert row["description_summary_error"] is None
    assert row["descriptions"] == ["original"]
    fingerprint = (summarizer._entity_fingerprint if kind == "entities"
                   else summarizer._relationship_fingerprint)
    assert row["description_input_fingerprint"] == fingerprint(store.items[0])
    assert isinstance(row["description_summary_updated"], int)


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["", "  ", "```text\n\n```", "```markdown\n ```"])
async def test_description_summarizer_exhausts_invalid_output(setup, content):
    summarizer, store, llm, sleep = setup
    llm.ainvoke.return_value = SimpleNamespace(content=content)
    with pytest.raises(DescriptionSummarizationError, match="1 failed, 0 succeeded") as caught:
        await summarizer.summarize_entities()
    assert isinstance(caught.value.__cause__, ValueError)
    assert "after 3 attempts (empty output)" in str(caught.value)
    assert llm.ainvoke.await_count == 3
    assert [call.args[0] for call in sleep.await_args_list] == [1, 2]
    assert len(store.writes) == 1
    assert store.fetches == 1
    row = store.writes[0][0]
    assert row["description"] is None
    assert row["description_summary_status"] == "failed"
    assert "after 3 attempts (empty output)" in row["description_summary_error"]
    assert store.items[0]["description"] == "previous"


@pytest.mark.asyncio
async def test_description_summarizer_persists_mixed_outcomes_and_recovery_skips_success(setup):
    summarizer, _, llm, _ = setup
    store = SummaryStore(2)
    summarizer.graph_store = cast(GraphStore, store)
    provider_error = RuntimeError("PRIVATE RESPONSE " * 1000)

    async def invoke(prompt):
        if 'entity_name: 1' in prompt:
            raise provider_error
        return SimpleNamespace(content="valid")

    # Use an explicit deterministic prompt to identify each item.
    summarizer._build_entity_summary_prompt = lambda item: f"entity_name: {item['id']}"
    llm.ainvoke.side_effect = invoke
    with pytest.raises(DescriptionSummarizationError) as caught:
        await summarizer.summarize_entities()
    assert caught.value.__cause__ is provider_error
    assert "1 failed, 1 succeeded" in str(caught.value)
    assert "first failed ID='1'" in str(caught.value)
    assert "PRIVATE" not in str(caught.value)
    assert len(str(caught.value)) < 400
    assert store.fetches == 1
    assert len(store.writes) == 1
    assert [row["description_summary_status"] for row in store.writes[0]] == ["success", "failed"]
    assert "PRIVATE" not in store.writes[0][1]["description_summary_error"]
    assert llm.ainvoke.await_count == 2
    llm.ainvoke.reset_mock(side_effect=True)
    llm.ainvoke.return_value = SimpleNamespace(content="recovered")
    assert await summarizer.summarize_entities() == {"summarized": 1, "batches": 1}
    llm.ainvoke.assert_awaited_once_with("entity_name: 1")


@pytest.mark.asyncio
@pytest.mark.parametrize("error_source", ["provider", "prompt"])
async def test_description_summarizer_does_not_retry_non_parser_value_errors(setup, error_source):
    summarizer, store, llm, sleep = setup
    error = ValueError("private prompt/provider detail")
    if error_source == "provider":
        llm.ainvoke.side_effect = error
    else:
        summarizer._build_entity_summary_prompt = MagicMock(side_effect=error)
    with pytest.raises(DescriptionSummarizationError) as caught:
        await summarizer.summarize_entities()
    assert caught.value.__cause__ is error
    assert llm.ainvoke.await_count == (1 if error_source == "provider" else 0)
    sleep.assert_not_awaited()
    assert len(store.writes) == 1
    assert store.writes[0][0]["description_summary_status"] == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("output", ["valid", ""])
async def test_description_summarizer_write_failure_never_writes_again(setup, output):
    summarizer, store, llm, _ = setup
    llm.ainvoke.return_value = SimpleNamespace(content=output)
    store.write_error = OSError("private database detail")
    with pytest.raises(DescriptionSummarizationError, match="Could not persist") as caught:
        await summarizer.summarize_entities()
    assert caught.value.__cause__ is store.write_error
    assert "persisted" not in str(caught.value)
    assert "private" not in str(caught.value)
    assert len(store.writes) == 1
    assert store.writes[0][0]["description_summary_status"] == ("success" if output else "failed")
    assert store.fetches == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("child_cancel", [False, True])
async def test_description_summarizer_cancellation_drains_running_and_queued_tasks(child_cancel):
    store = SummaryStore(6)
    started = asyncio.Event()
    release = asyncio.Event()
    active = set()
    calls = []

    async def invoke(prompt):
        task = asyncio.current_task()
        active.add(task)
        calls.append(task)
        try:
            if len(active) == 2:
                started.set()
            await release.wait()
        finally:
            active.remove(task)
        return SimpleNamespace(content="valid")

    summarizer = DescriptionSummarizer(
        cast(BaseLLM, SimpleNamespace(ainvoke=invoke)), cast(GraphStore, store), "g", concurrency=2
    )
    parent = asyncio.create_task(summarizer.summarize_entities())
    await asyncio.wait_for(started.wait(), 2)
    if child_cancel:
        calls[0].cancel()
    else:
        parent.cancel()
    with pytest.raises(asyncio.CancelledError):
        await parent
    assert not active
    assert all(task.done() for task in calls)
    assert not store.writes
    assert not [task for task in asyncio.all_tasks()
                if task is not asyncio.current_task() and "_summarize" in getattr(task.get_coro(), "__qualname__", "")]


@pytest.mark.asyncio
async def test_description_summarizer_retry_calls_obey_concurrency():
    store = SummaryStore(8)
    active = 0
    peak = 0
    attempts = {}

    async def invoke(prompt):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0)
            attempts[prompt] = attempts.get(prompt, 0) + 1
            return SimpleNamespace(content="" if attempts[prompt] == 1 else "valid")
        finally:
            active -= 1

    summarizer = DescriptionSummarizer(
        cast(BaseLLM, SimpleNamespace(ainvoke=invoke)), cast(GraphStore, store), "g", concurrency=2
    )
    assert await summarizer.summarize_entities() == {"summarized": 8, "batches": 1}
    assert peak == 2
    assert list(attempts.values()) == [2] * 8
