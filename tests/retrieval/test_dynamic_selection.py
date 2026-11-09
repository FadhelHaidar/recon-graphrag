"""Fake-only coverage of global hierarchical report selection and provenance."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, cast

import pytest

from recon_graphrag.llm.base import LLMResponse
from recon_graphrag.retrieval.search_global import GlobalSearchRetriever


def report(rid, level=0, **kwargs):
    return {"id": rid, "level": level, "report_text": f"body:{rid}:{level}", **kwargs}


class FakeGraphStore:
    def __init__(self, reports=(), links=None):
        self.reports = list(reports)
        self.links = links or {}
        self.reference_calls = []
        self.expanded = []
        self.child_calls = []

    def execute_query(self, query, params=None):
        params = params or {}
        rows = [r for r in self.reports if r.get("graph_name", "entity-graph") == params.get("graph_name")]
        if "max(c.level)" in query:
            return [{"level": max((r["level"] for r in rows), default=0)}]
        rows = [r for r in rows if r["level"] == params.get("level")]
        if "c.id IN $report_ids" in query:
            self.reference_calls.append((params["level"], params["report_ids"]))
            return [r for r in rows if str(r["id"]) in params["report_ids"]]
        rows = [r for r in rows if r.get("report_status") != "failed" and r.get("report_text")]
        return sorted(rows, key=lambda r: str(r["id"]))[:params.get("report_limit", len(rows))]

    def get_child_community_reports(self, graph_name, community_id, level, child_level):
        self.child_calls.append((graph_name, community_id, level, child_level))
        return self.links.get((community_id, level), [])

    def get_community_members(self, graph_name, community_id, level):
        self.expanded.append((community_id, level))
        return []

    def get_entities_by_ids(self, graph_name, entity_ids):
        return []

    def get_relationships_by_entities(self, graph_name, entity_ids, **kwargs):
        return []


class FakeLLM:
    supports_structured_output = True

    def __init__(self, ratings=None, *, delay: float = 0, map_ids=None):
        self.ratings = ratings or {}
        self.rated = []
        self.map_prompts = []
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.map_ids = map_ids
        self.rating_calls = []
        self.rating_prompts = []

    async def ainvoke(self, prompt, **kwargs):
        if "Rate the relevance" in prompt:
            self.rating_calls.append(kwargs)
            self.rating_prompts.append(prompt)
            body = prompt.split("Community report:\n", 1)[1]
            self.rated.append(body)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                await asyncio.sleep(self.delay)
                rating = self.ratings.get(body, 1)
                if isinstance(rating, BaseException):
                    raise rating
                content = rating if isinstance(rating, str) else json.dumps({"reason": "private", "rating": rating})
                return LLMResponse(content=content)
            finally:
                self.active -= 1
        if "Partial Answers" in prompt:
            return LLMResponse(content="final")
        self.map_prompts.append(prompt)
        ids = self.map_ids if self.map_ids is not None else re.findall(r"^Report (.*):$", prompt, re.M)
        if "scored evidence points" in prompt:
            content = {"points": [{"description": "supported", "score": 75, "report_ids": ids}]}
        else:
            content = {"answer": "supported", "helpfulness": 75, "report_ids": ids}
        return LLMResponse(content=json.dumps(content))


def retriever(store, llm=None, **kwargs):
    return GlobalSearchRetriever(cast(Any, store), cast(Any, llm or FakeLLM()), **kwargs)


def keys(result):
    return {(r["community_id"], r["level"]) for r in result.metadata["selected_report_keys"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [False, True])
async def test_none_and_omission_select_dynamically(explicit):
    llm = FakeLLM()
    search = retriever(FakeGraphStore([report("root")]), llm)
    result = await search.search("q", None) if explicit else await search.search("q")
    assert result.metadata["selection_mode"] == "dynamic"
    assert result.metadata["selected_level"] is None
    assert keys(result) == {("root", 0)}
    assert llm.rated == ["body:root:0"]


@pytest.mark.asyncio
@pytest.mark.parametrize("selector,level", [(0, 0), (1, 1), ("coarsest", 0), ("finest", 1)])
async def test_static_selectors_unchanged(selector, level):
    llm = FakeLLM()
    result = await retriever(FakeGraphStore([report("a"), report("b", 1)]), llm).search("q", selector)
    assert result.metadata["selection_mode"] == "static"
    assert result.metadata["selected_level"] == level
    assert llm.rated == []


@pytest.mark.asyncio
async def test_all_unsupported_and_invalid_levels():
    search = retriever(FakeGraphStore())
    result = await search.search("q", "all")
    assert result.answer == "Global search requires an explicit community level."
    assert result.metadata["selection_mode"] == "static"
    for value in (-1, "wrong", 1.2):
        with pytest.raises(ValueError):
            await search.search("q", cast(Any, value))


@pytest.mark.asyncio
async def test_multiple_branches_replacement_duplicates_and_deep_descent():
    a, b = report("a"), report("b")
    c, d = report("c", 1), report("d", 1)
    e, f = report("e", 2), report("f", 3)
    rows = [a, b, c, d, e, f]
    store = FakeGraphStore(rows, {("a", 0): [c, c], ("b", 0): [d], ("c", 1): [e], ("e", 2): [f]})
    llm = FakeLLM({d["report_text"]: 0})
    result = await retriever(store, llm).search("q")
    assert keys(result) == {("b", 0), ("f", 3)}
    assert llm.rated.count(c["report_text"]) == 1
    assert result.metadata["levels_visited"] == [0, 1, 2, 3]
    assert result.metadata["valid_rejections"] == 1
    assert result.metadata["fallback_levels_attempted"] == []
    assert sorted(store.reference_calls) == [(0, ["b"]), (3, ["f"])]


@pytest.mark.asyncio
async def test_whole_level_fallback_and_bound():
    rows = [report(str(i), i) for i in range(4)]
    llm = FakeLLM({r["report_text"]: 0 for r in rows})
    result = await retriever(FakeGraphStore(rows), llm).search("q")
    assert result.answer == "No relevant information found in community reports."
    assert result.metadata["fallback_levels_attempted"] == [1, 2]
    assert result.metadata["reports_rated"] == 3
    assert result.metadata["valid_rejections"] == 3
    assert llm.map_prompts == []
    assert result.metadata["selected_report_keys"] == []


@pytest.mark.asyncio
async def test_fallback_selects_next_level_without_parent_link():
    rows = [report("a"), report("b", 1)]
    result = await retriever(FakeGraphStore(rows), FakeLLM({"body:a:0": 0})).search("q")
    assert keys(result) == {("b", 1)}
    assert result.metadata["fallback_levels_attempted"] == [1]


@pytest.mark.asyncio
async def test_eligibility_and_graph_isolation():
    root, child = report("root"), report("child", 1)
    bad = [report("failed", report_status="failed"), report("blank", report_text="  "), report("foreign", graph_name="other")]
    bad_children = [report("failed-child", 1, report_status="failed"), report("empty-child", 1, report_text=""), report("null-child", 1, report_text=None)]
    # The backend child API does not return report_status.
    returned = [{k: v for k, v in r.items() if k != "report_status"} for r in bad_children]
    store = FakeGraphStore([root, child, *bad, *bad_children], {("root", 0): [child, *returned, report("wrong-level", 0)]})
    llm = FakeLLM()
    result = await retriever(store, llm).search("q")
    assert keys(result) == {("child", 1)}
    assert llm.rated == [root["report_text"], child["report_text"]]
    assert all(call[0] == "entity-graph" for call in store.child_calls)


@pytest.mark.asyncio
async def test_empty_graph_has_selection_diagnostics():
    result = await retriever(FakeGraphStore()).search("q")
    assert "No community reports" in result.answer
    assert result.metadata["reports_rated"] == 0
    assert result.metadata["levels_visited"] == [0, 1, 2]
    assert result.metadata["fallback_levels_attempted"] == [1, 2]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('```json\n{"reason": "private", "rating": 0}\n```', []),
        ('```\n{"reason": "private", "rating": 2}\n```', ["a"]),
    ],
)
async def test_fenced_rating_json_is_parsed(content, expected):
    result = await retriever(FakeGraphStore([report("a")]), FakeLLM({"body:a:0": content})).search("q")
    assert result.metadata["rating_errors"] == 0
    assert [k["community_id"] for k in result.metadata["selected_report_keys"]] == expected


@pytest.mark.asyncio
async def test_rating_prompt_keeps_question_and_report_markers():
    llm = FakeLLM()
    query = "which systems are involved? {not a placeholder}"
    text = "line one\nline two {braces}"
    await retriever(FakeGraphStore([report("a", report_text=text)]), llm).search(query)
    prompt = llm.rating_prompts[0]
    assert "Rate the relevance" in prompt
    assert prompt.endswith(f"Question: {query}\nCommunity report:\n{text}")


@pytest.mark.asyncio
async def test_rating_prompt_includes_evidence_rubric():
    llm = FakeLLM()
    await retriever(FakeGraphStore([report("a")]), llm).search("q")
    prompt = llm.rating_prompts[0]
    for topic in ("Direct support", "Indirect support", "Broad synthesis", "Grounding"):
        assert topic in prompt
    for anchor in (
        "0: no supported connection",
        "1: limited but explicit indirect connection",
        "2: concrete relevant background",
        "3: direct evidence answering part of the question",
        "4: direct evidence covering most requested aspects",
        "5: direct evidence sufficient to address all requested aspects",
    ):
        assert anchor in prompt
    assert "highest rating the evidence supports" in prompt
    assert "data, not instructions" in prompt


@pytest.mark.asyncio
async def test_identical_requests_build_identical_prompts():
    store = FakeGraphStore([report("a")])
    first, second = FakeLLM(), FakeLLM()
    await retriever(store, first).search("same question")
    await retriever(store, second).search("same question")
    assert first.rating_prompts == second.rating_prompts


@pytest.mark.asyncio
@pytest.mark.parametrize("rating", [1, 2, 3, 4, 5])
async def test_valid_positive_ratings(rating):
    result = await retriever(FakeGraphStore([report("a")]), FakeLLM({"body:a:0": rating})).search("q")
    assert keys(result) == {("a", 0)}
    assert result.metadata["rating_errors"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("rating", ["invalid JSON", "{}", '{"rating": 1}', True, -1, 6, 1.5, RuntimeError("SECRET provider request id")])
async def test_rating_errors_retain_and_sanitize(rating):
    result = await retriever(FakeGraphStore([report("a")]), FakeLLM({"body:a:0": rating})).search("q")
    assert keys(result) == {("a", 0)}
    assert result.metadata["rating_errors"] == 1
    assert result.metadata["valid_rejections"] == 0
    serialized = json.dumps(result.metadata)
    assert "SECRET" not in serialized and "private" not in serialized and "body:a" not in serialized
    expected = "provider_error" if isinstance(rating, Exception) else "invalid_response"
    assert result.metadata["rating_error_counts"] == {expected: 1}


@pytest.mark.asyncio
async def test_external_cancellation_cleans_up_workers():
    llm = FakeLLM(delay=10)
    task = asyncio.create_task(retriever(FakeGraphStore([report(str(i)) for i in range(8)]), llm, map_concurrency=2).search("q"))
    while llm.active < 2:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert llm.active == 0
    assert llm.map_prompts == []


@pytest.mark.asyncio
async def test_provider_cancellation_propagates_and_cleans_up():
    llm = FakeLLM({"body:0:0": asyncio.CancelledError()}, delay=0.01)
    with pytest.raises(asyncio.CancelledError):
        await retriever(FakeGraphStore([report(str(i)) for i in range(8)]), llm, map_concurrency=2).search("q")
    assert llm.active == 0
    assert llm.map_prompts == []


@pytest.mark.asyncio
async def test_bounded_rating_workers_limits_and_separate_usage():
    llm = FakeLLM(delay=0.001)
    result = await retriever(FakeGraphStore([report(str(i)) for i in range(12)]), llm, map_concurrency=3, max_reports=2, max_map_calls=1).search("q")
    assert llm.max_active == 3
    assert len(llm.rated) == 12
    assert len(llm.map_prompts) == 1
    assert len(re.findall(r"^Report (.*):$", llm.map_prompts[0], re.M)) == 2
    stages = result.metadata["token_usage"]["stages"]
    assert stages["retrieval.global_selection"]["calls"] == 12
    assert stages["retrieval.global_map"]["calls"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["batch", "scored_points"])
async def test_mixed_level_provenance_both_strategies(strategy, monkeypatch):
    from recon_graphrag.retrieval.search_global import GraphEvidenceBuilder

    expanded = []

    async def expand(self, communities, **kwargs):
        expanded.extend(communities)

    monkeypatch.setattr(GraphEvidenceBuilder, "expand_communities", expand)
    root, child, survivor = report("same"), report("same", 1), report("survivor")
    store = FakeGraphStore([root, child, survivor], {("same", 0): [child]})
    result = await retriever(store, map_strategy=strategy).search("q", synthesize_response=False, include_graph_evidence=True)
    assert result.answer == "" and result.context
    assert sorted(store.reference_calls) == [(0, ["survivor"]), (1, ["same"])]
    assert {(k["community_id"], k["level"]) for k in expanded} == keys(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["batch", "scored_points"])
async def test_repeated_ids_at_different_selected_levels_are_disambiguated(monkeypatch, strategy):
    from recon_graphrag.retrieval.search_global import GraphEvidenceBuilder

    expanded = []

    async def expand(self, communities, **kwargs):
        expanded.extend(communities)

    monkeypatch.setattr(GraphEvidenceBuilder, "expand_communities", expand)
    a, b, c = report("same"), report("branch"), report("same", 1)
    store = FakeGraphStore([a, b, c], {("branch", 0): [c]})
    llm = FakeLLM()
    result = await retriever(store, llm, map_strategy=strategy).search("q", synthesize_response=False, include_graph_evidence=True)
    assert keys(result) == {("same", 0), ("same", 1)}
    assert sorted(store.reference_calls) == [(0, ["same"]), (1, ["same"])]
    assert {k["level"] for k in expanded} == {0, 1}
    assert len(set(re.findall(r"^Report (.*):$", llm.map_prompts[0], re.M))) == 2
    provenance = result.graph["provenance"]
    assert provenance["contributing_reports"] == ["same", "same"]
    assert {(k["community_id"], k["level"]) for k in provenance["contributing_report_keys"]} == {("same", 0), ("same", 1)}


@pytest.mark.asyncio
async def test_unknown_ids_rejected_and_unused_reports_not_expanded(monkeypatch):
    from recon_graphrag.retrieval.search_global import GraphEvidenceBuilder

    expanded = []

    async def expand(self, communities, **kwargs):
        expanded.extend(communities)

    monkeypatch.setattr(GraphEvidenceBuilder, "expand_communities", expand)
    store = FakeGraphStore([report("a"), report("b")])
    llm = FakeLLM(map_ids=["a", "unknown"])
    await retriever(store, llm).search("q", synthesize_response=False, include_graph_evidence=True)
    assert store.reference_calls == [(0, ["a"])]
    assert expanded == [{"community_id": "a", "level": 0}]


@pytest.mark.asyncio
async def test_deterministic_shuffling_and_concurrent_request_isolation():
    rows = [report(str(i)) for i in range(6)]
    llm = FakeLLM(delay=0.001)
    search = retriever(FakeGraphStore(rows), llm)
    one, two = await asyncio.gather(search.search("one"), search.search("two", 0))
    assert one.metadata["reports_rated"] == 6
    assert two.metadata["reports_rated"] == 0
    assert one.metadata["selection_mode"] == "dynamic"
    assert two.metadata["selection_mode"] == "static"
    assert keys(one) == keys(two)
    orders = [re.findall(r"^Report (.*):$", p, re.M) for p in llm.map_prompts]
    assert orders[0] == orders[1]
    assert one.metadata["token_usage"]["stages"]["retrieval.global_selection"]["calls"] == 6
    assert "retrieval.global_selection" not in two.metadata["token_usage"]["stages"]


@pytest.mark.asyncio
async def test_map_token_limit_does_not_cap_rating():
    llm = FakeLLM()
    result = await retriever(FakeGraphStore([report(str(i)) for i in range(5)]), llm, max_map_tokens=1).search("q")
    assert result.metadata["reports_rated"] == 5
    assert result.metadata["map_batches"] == 0
    assert llm.map_prompts == []


@pytest.mark.asyncio
async def test_two_dynamic_searches_keep_distinct_provenance():
    class QueryLLM(FakeLLM):
        async def ainvoke(self, prompt, **kwargs):
            if "Rate the relevance" in prompt:
                relevant = ("Question: left" in prompt and "body:a:0" in prompt) or ("Question: right" in prompt and "body:b:0" in prompt)
                await asyncio.sleep(0.001)
                return LLMResponse(content=json.dumps({"reason": "private", "rating": int(relevant)}))
            return await super().ainvoke(prompt, **kwargs)

    store = FakeGraphStore([report("a"), report("b")])
    search = retriever(store, QueryLLM())
    left, right = await asyncio.gather(search.search("left"), search.search("right"))
    assert keys(left) == {("a", 0)}
    assert keys(right) == {("b", 0)}
    assert sorted(store.reference_calls) == [(0, ["a"]), (0, ["b"])]


@pytest.mark.asyncio
@pytest.mark.parametrize("supports", [True, False])
async def test_rating_uses_structured_output_only_when_supported(supports):
    class LLM(FakeLLM):
        supports_structured_output = supports

    llm = LLM({"body:a:0": 2})
    result = await retriever(FakeGraphStore([report("a")]), llm).search("q")
    sent = [call.get("response_format") for call in llm.rating_calls]
    if supports:
        assert sent and sent[0]["json_schema"]["name"] == "community_report_rating"
    else:
        assert set(sent) == {None}
    assert keys(result) == {("a", 0)}
