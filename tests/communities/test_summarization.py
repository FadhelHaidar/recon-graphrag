"""Tests for community summarization context formatting."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from recon_graphrag.llm.base import LLMResponse
from recon_graphrag.communities.summarization import BuildStats, CommunitySummarizer
from recon_graphrag.communities.reports import ReportRubric, ReportValidationError


class FakeReportGraphStore:
    def __init__(self):
        self.stored_reports = []
        self.failed_reports = []
        self.existing_fingerprint = None

    def get_communities(self, graph_name, level=None):
        return [{"id": "community:1", "level": 0, "entity_count": 2}]

    def get_community_ranked_context(self, graph_name, community_id, level=0):
        return [
            {
                "e_id": "person:alice",
                "e_name": "Alice",
                "e_description": "CEO",
                "e_labels": ["Person"],
                "e_degree": 1,
                "rel_type": "WORKS_AT",
                "rel_description": "Alice works at Acme",
                "observation_count": 1,
                "combined_degree": 2,
                "other_id": "org:acme",
                "other_name": "Acme",
                "other_description": "Company",
                "other_labels": ["Organization"],
                "other_degree": 1,
            }
        ]

    def get_claims_for_entities(self, graph_name, entity_ids):
        return []

    def get_community_reports_by_keys(self, graph_name, keys, top_k):
        if self.existing_fingerprint is None:
            return []
        return [
            {
                "id": keys[0]["id"],
                "level": keys[0]["level"],
                "report_text": "Existing report",
                "input_fingerprint": self.existing_fingerprint,
            }
        ]

    def store_community_report(self, report, graph_name):
        self.stored_reports.append((graph_name, report))

    def mark_community_report_failed(self, graph_name, community_id, level, error):
        self.failed_reports.append((graph_name, community_id, level, error))

    def get_child_community_reports(
        self, graph_name, community_id, level, child_level
    ):
        return []


class FakeReportLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, prompt):
        self.calls += 1
        return LLMResponse(
            content=json.dumps({"title": "Bad", "summary": "No findings", "findings": []})
        )


class FakeValidReportLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, prompt):
        self.calls += 1
        return LLMResponse(
            content=json.dumps(
                {
                    "title": "Alice and Acme",
                    "summary": "Alice works with Acme.",
                    "rating": 7.0,
                    "rating_explanation": "Important relationship.",
                    "findings": [
                        {
                            "description": "Alice is connected to Acme.",
                            "references": [
                                {"target_id": "person:alice", "target_type": "entity"}
                            ],
                        }
                    ],
                }
            )
        )


class ScriptedReportLLM:
    def __init__(self, responses: list[str | Exception]):
        self.responses = responses
        self.prompts: list[str] = []

    @property
    def calls(self):
        return len(self.prompts)

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        response = self.responses[self.calls - 1]
        if isinstance(response, Exception):
            raise response
        return LLMResponse(content=response)


class CapturingValidReportLLM(FakeValidReportLLM):
    def __init__(self):
        super().__init__()
        self.prompts = []

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return await super().ainvoke(prompt)


@pytest.mark.asyncio
@pytest.mark.parametrize("include_error_details", [False, True])
async def test_report_generation_failure_is_marked_not_stored(include_error_details):
    store = FakeReportGraphStore()
    llm = FakeReportLLM()
    summarizer = CommunitySummarizer(
        cast(Any, store), llm=cast(Any, llm), include_error_details=include_error_details
    )

    results, stats = await summarizer.generate_all(level=0)

    assert results == []
    assert stats.failed == 1
    assert stats.succeeded == 0
    assert store.stored_reports == []
    assert len(store.failed_reports) == 1
    assert store.failed_reports[0][:3] == ("entity-graph", "community:1", 0)
    assert stats.errors == [{
        "community_id": "community:1", "error_type": "ReportValidationError",
        "message": (
            store.failed_reports[0][3] if include_error_details
            else "Community report generation or storage failed."
        ),
    }]


@pytest.mark.asyncio
async def test_report_generation_stores_input_fingerprint():
    store = FakeReportGraphStore()
    llm = FakeValidReportLLM()
    summarizer = CommunitySummarizer(store, llm=llm)

    results, stats = await summarizer.generate_all(level=0)

    assert stats.succeeded == 1
    assert stats.errors == []
    assert results
    _, report = store.stored_reports[0]
    assert report.version.input_fingerprint


@pytest.mark.asyncio
async def test_report_skip_existing_requires_matching_fingerprint():
    store = FakeReportGraphStore()
    llm = FakeValidReportLLM()
    summarizer = CommunitySummarizer(store, llm=llm)
    context = summarizer._fetch_community_context_obj("community:1", 0)
    store.existing_fingerprint = summarizer._context_fingerprint(context)

    results, stats = await summarizer.generate_all(level=0, skip_existing=True)

    assert results == []
    assert stats.skipped == 1
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_parent_report_substitutes_child_reports_when_direct_context_truncates():
    class ParentStore(FakeReportGraphStore):
        def get_community_ranked_context(self, graph_name, community_id, level=0):
            rows = super().get_community_ranked_context(graph_name, community_id, level)
            rows[0]["e_description"] = "A" * 1000
            rows[0]["other_description"] = "B" * 1000
            return rows

        def get_child_community_reports(
            self, graph_name, community_id, level, child_level
        ):
            return [{
                "id": "child:1",
                "level": child_level,
                "report_text": "Important child report.",
                "input_fingerprint": "child-fingerprint",
                "context_tokens_used": 500,
            }]

    llm = CapturingValidReportLLM()
    summarizer = CommunitySummarizer(ParentStore(), llm=llm, max_context_tokens=200)

    report = await summarizer.generate_report("community:1", level=0)

    assert "Important child report." in llm.prompts[0]
    assert report.context_truncated is True


@pytest.mark.asyncio
async def test_custom_report_prompt_appears_in_llm_prompt():
    store = FakeReportGraphStore()
    llm = CapturingValidReportLLM()
    summarizer = CommunitySummarizer(
        store,
        llm=llm,
        report_prompt="You are a security analyst. Identify risks in this community.",
    )

    await summarizer.generate_all(level=0)

    assert llm.prompts
    assert "You are a security analyst. Identify risks in this community." in llm.prompts[0]



class MultiCommunityStore(FakeReportGraphStore):
    """Store with many communities, used for bounded-work checks."""

    def __init__(self, count: int):
        super().__init__()
        self.count = count

    def get_communities(self, graph_name, level=None):
        return [
            {"id": f"community:{i}", "level": 0, "entity_count": 2}
            for i in range(self.count)
        ]


class TrackedValidReportLLM(FakeValidReportLLM):
    """Records peak concurrent calls and peak live tasks."""

    def __init__(self):
        super().__init__()
        self.active = 0
        self.max_active = 0
        self.max_tasks = 0

    async def ainvoke(self, prompt):
        import asyncio

        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.max_tasks = max(self.max_tasks, len(asyncio.all_tasks()))
        try:
            await asyncio.sleep(0)
            return await super().ainvoke(prompt)
        finally:
            self.active -= 1


@pytest.mark.asyncio
async def test_communities_bounded_workers_bound_pending_tasks():
    count = 40
    store = MultiCommunityStore(count)
    llm = TrackedValidReportLLM()
    summarizer = CommunitySummarizer(cast(Any, store), llm=cast(Any, llm), concurrency=2)

    results, stats = await summarizer.generate_all(level=0)

    assert stats.succeeded == count
    assert llm.calls == count
    # Concurrency bounds active report work, not the queued task count.
    assert llm.max_active <= 2
    # One task per community would push live tasks to ~count.
    assert llm.max_tasks <= 8
    # Report order stays the deterministic community order.
    assert [r["id"] for r in results] == [f"community:{i}" for i in range(count)]


def test_build_stats_error_lists_are_isolated():
    first = BuildStats()
    second = BuildStats()
    first.errors.append({"community_id": "c0", "error_type": "ValueError", "message": "bad"})
    assert second.errors == []


@pytest.mark.asyncio
@pytest.mark.parametrize("include_error_details", [False, True])
@pytest.mark.parametrize("phase", ["generation", "storage", "resume"])
async def test_report_errors_preserve_original_only_when_opted_in(
    monkeypatch, include_error_details, phase
):
    store = FakeReportGraphStore()
    summarizer = CommunitySummarizer(
        cast(Any, store), cast(Any, FakeValidReportLLM()),
        include_error_details=include_error_details,
    )
    # Even a custom exception class name can contain sensitive data.
    error = type("SensitiveProviderName", (Exception,), {})("private provider detail")

    def fail(*args, **kwargs):
        raise error

    async def fail_async(*args, **kwargs):
        raise error

    if phase == "generation":
        monkeypatch.setattr(summarizer, "generate_report", fail_async)
    elif phase == "storage":
        monkeypatch.setattr(store, "store_community_report", fail)
        # Secondary persistence failures must not replace the original exception.
        monkeypatch.setattr(store, "mark_community_report_failed", fail)
    else:
        monkeypatch.setattr(store, "get_community_ranked_context", fail)

    results, stats = await summarizer.generate_all(skip_existing=phase == "resume")

    assert results == []
    assert stats.attempted == stats.failed == 1
    assert stats.succeeded == 0
    safe_message = (
        "Community report preparation failed." if phase == "resume"
        else "Community report generation or storage failed."
    )
    assert stats.errors == [{
        "community_id": "community:1",
        "error_type": "SensitiveProviderName" if include_error_details else "Exception",
        "message": str(error) if include_error_details else safe_message,
    }]


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed", [
    '{"title" "Missing colon"}',
    '{title: "Unquoted key"}',
    'Provider preamble\n```json\n{"title" "Missing colon"}\n```\nTrailing text',
])
async def test_generate_report_repairs_malformed_json(malformed):
    valid = (await FakeValidReportLLM().ainvoke("")).content
    llm = ScriptedReportLLM([malformed, valid])
    summarizer = CommunitySummarizer(
        cast(Any, FakeReportGraphStore()), cast(Any, llm),
        report_rubric=ReportRubric(min_rating=2.0, max_rating=8.0),
        max_context_tokens=1000,
    )
    with pytest.raises(json.JSONDecodeError) as parse_error:
        summarizer._report_parser.parse(malformed, "community:1", 0)

    report = await summarizer.generate_report("community:1", input_fingerprint="input-v1")

    assert llm.calls == 2
    prompt = llm.prompts[1]
    assert malformed in prompt
    assert parse_error.value.msg in prompt
    assert f"line {parse_error.value.lineno}" in prompt
    assert f"column {parse_error.value.colno}" in prompt
    assert "person:alice" in prompt
    assert "org:acme" in prompt
    assert '"rating": 2.0-8.0 or null' in prompt
    assert report.title == "Alice and Acme"
    assert report.community_id == "community:1"
    assert report.level == 0
    assert report.version.input_fingerprint == "input-v1"
    assert report.context_tokens_used is not None
    assert report.context_truncated is False
    assert report.findings[0].references[0].target_id == "person:alice"


@pytest.mark.asyncio
async def test_generate_report_valid_json_needs_no_repair():
    valid = (await FakeValidReportLLM().ainvoke("")).content
    llm = ScriptedReportLLM([valid])
    summarizer = CommunitySummarizer(cast(Any, FakeReportGraphStore()), cast(Any, llm))

    report = await summarizer.generate_report("community:1")

    assert report.version.input_fingerprint
    assert llm.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("repaired, error_type", [
    ('{"title" "Still malformed"}', json.JSONDecodeError),
    ('{"title": "No findings"}', ReportValidationError),
])
async def test_generate_report_stops_after_one_json_repair(repaired, error_type):
    llm = ScriptedReportLLM(['{title: "Malformed"}', repaired])
    summarizer = CommunitySummarizer(cast(Any, FakeReportGraphStore()), cast(Any, llm))

    with pytest.raises(error_type) as error:
        await summarizer.generate_report("community:1")

    assert llm.calls == 2
    if isinstance(error.value, json.JSONDecodeError):
        assert error.value.doc == repaired
    else:
        assert error.value.raw_content == repaired


@pytest.mark.asyncio
async def test_generate_report_still_repairs_validation_errors():
    valid = (await FakeValidReportLLM().ainvoke("")).content
    invalid = '{"title": "No findings"}'
    llm = ScriptedReportLLM([invalid, valid])
    summarizer = CommunitySummarizer(cast(Any, FakeReportGraphStore()), cast(Any, llm))

    report = await summarizer.generate_report("community:1")

    assert report.title == "Alice and Acme"
    assert llm.calls == 2
    assert invalid in llm.prompts[1]
    assert "Missing or empty 'summary'." in llm.prompts[1]


@pytest.mark.asyncio
async def test_generate_report_does_not_repair_provider_errors():
    error = ConnectionError("Provider unavailable")
    llm = ScriptedReportLLM([error])
    summarizer = CommunitySummarizer(cast(Any, FakeReportGraphStore()), cast(Any, llm))

    with pytest.raises(ConnectionError) as caught:
        await summarizer.generate_report("community:1")

    assert caught.value is error
    assert llm.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("include_error_details", [False, True])
async def test_generate_all_continues_after_json_repair_failure(include_error_details):
    valid = (await FakeValidReportLLM().ainvoke("")).content
    malformed = '{title: "Malformed"}'
    final_malformed = '{"title" "Still malformed"}'
    llm = ScriptedReportLLM([malformed, valid, malformed, final_malformed, valid])
    store = MultiCommunityStore(3)
    summarizer = CommunitySummarizer(
        cast(Any, store), cast(Any, llm), concurrency=1,
        include_error_details=include_error_details,
    )

    results, stats = await summarizer.generate_all()

    assert llm.calls == 5
    assert [result["id"] for result in results] == ["community:0", "community:2"]
    assert stats.attempted == 3
    assert stats.succeeded == 2
    assert stats.failed == 1
    assert len(store.stored_reports) == 2
    assert len(store.failed_reports) == 1
    assert store.failed_reports[0][:3] == ("entity-graph", "community:1", 0)
    with pytest.raises(json.JSONDecodeError) as final_error:
        json.loads(final_malformed)
    assert store.failed_reports[0][3] == str(final_error.value)
    assert stats.errors == [{
        "community_id": "community:1",
        "error_type": "JSONDecodeError",
        "message": (
            str(final_error.value) if include_error_details
            else "Community report generation or storage failed."
        ),
    }]


class StructuredReportLLM:
    supports_structured_output = True

    def __init__(self, responses):
        self.responses = iter(responses)
        self.kwargs = []

    async def ainvoke(self, prompt, **kwargs):
        self.kwargs.append(kwargs)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.mark.asyncio
@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
async def test_generate_report_request_format_on_initial_and_repair(structured, wrapped):
    from recon_graphrag.communities.reports import COMMUNITY_REPORT_RESPONSE_FORMAT
    from recon_graphrag.observability import track_usage

    valid = await FakeValidReportLLM().ainvoke("")
    llm = StructuredReportLLM([LLMResponse("not json"), valid])
    summarizer = CommunitySummarizer(
        cast(Any, FakeReportGraphStore()),
        track_usage(cast(Any, llm)) if wrapped else cast(Any, llm),
        max_context_tokens=200,
        report_structured_output=structured,
    )
    report = await summarizer.generate_report("community:1", input_fingerprint="input-v1")
    expected = {"response_format": COMMUNITY_REPORT_RESPONSE_FORMAT} if structured else {}
    assert llm.kwargs == [expected, expected]
    assert report.title == "Alice and Acme"
    assert report.version.input_fingerprint == "input-v1"
    assert report.context_tokens_used is not None
    assert report.findings[0].references[0].target_id == "person:alice"


@pytest.mark.parametrize("wrapped", [False, True])
def test_report_structured_output_rejects_unsupported_adapter(wrapped):
    from recon_graphrag.llm.factory import AnthropicLLM, OllamaLLM
    from recon_graphrag.observability import track_usage

    for adapter in (AnthropicLLM, OllamaLLM):
        llm = object.__new__(adapter)
        with pytest.raises(ValueError, match="does not support report structured output"):
            CommunitySummarizer(
                cast(Any, FakeReportGraphStore()),
                track_usage(cast(Any, llm)) if wrapped else cast(Any, llm),
                report_structured_output=True,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["length", "max_tokens", "refusal", "content_filter"])
@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("valid_json", [False, True])
async def test_generate_report_rejects_incomplete_output_before_parsing(reason, repair, valid_json):
    from recon_graphrag.observability import track_usage

    valid = (await FakeValidReportLLM().ainvoke("")).content
    response = LLMResponse(valid if valid_json else "sensitive provider text", finish_reason=reason)
    responses = [LLMResponse("not json")] if repair else []
    llm = StructuredReportLLM(responses + [response])
    summarizer = CommunitySummarizer(
        cast(Any, FakeReportGraphStore()), track_usage(cast(Any, llm)), report_structured_output=True,
    )
    with pytest.raises(ValueError, match="Community report") as error:
        await summarizer.generate_report("community:1")
    assert "sensitive" not in str(error.value)
    assert "No further JSON repair will be attempted" in str(error.value)
    assert len(llm.kwargs) == (2 if repair else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", [None, "stop", "end_turn", "unknown"])
async def test_generate_report_finished_or_unknown_status_preserves_legacy(reason):
    valid = (await FakeValidReportLLM().ainvoke("")).content
    llm = StructuredReportLLM([LLMResponse(valid, finish_reason=reason)])
    summarizer = CommunitySummarizer(cast(Any, FakeReportGraphStore()), cast(Any, llm))
    assert (await summarizer.generate_report("community:1")).title == "Alice and Acme"
    assert llm.kwargs == [{}]


@pytest.mark.asyncio
@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("message", ["HTTP 400 schema unsupported", "HTTP 401 authentication failed", "network failed"])
async def test_structured_report_provider_errors_are_not_retried(repair, message):
    error = RuntimeError(message)
    responses = [LLMResponse("not json")] if repair else []
    llm = StructuredReportLLM(responses + [error])
    summarizer = CommunitySummarizer(
        cast(Any, FakeReportGraphStore()), cast(Any, llm), report_structured_output=True,
    )
    with pytest.raises(RuntimeError) as caught:
        await summarizer.generate_report("community:1")
    assert caught.value is error
    assert len(llm.kwargs) == (2 if repair else 1)


@pytest.mark.asyncio
async def test_structured_report_repair_remains_bounded():
    llm = StructuredReportLLM([LLMResponse("not json"), LLMResponse("still not json")])
    summarizer = CommunitySummarizer(
        cast(Any, FakeReportGraphStore()), cast(Any, llm), report_structured_output=True,
    )
    with pytest.raises(json.JSONDecodeError):
        await summarizer.generate_report("community:1")
    assert len(llm.kwargs) == 2
