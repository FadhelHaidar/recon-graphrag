"""Tests for community summarization context formatting."""

from __future__ import annotations

import json
from typing import cast

import pytest

from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.llm.base import BaseLLM, LLMResponse
from recon_graphrag.communities.context import ClaimContext, CommunityContext, EntityContext
from recon_graphrag.communities.reports import build_report_prompt
from recon_graphrag.communities.summarization import CommunitySummarizer


class FakeReportGraphStore:
    def __init__(self):
        self.stored_reports = []
        self.failed_reports = []
        self.existing_fingerprint: str | None = None

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


class CapturingValidReportLLM(FakeValidReportLLM):
    def __init__(self):
        super().__init__()
        self.prompts = []

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return await super().ainvoke(prompt)


@pytest.mark.asyncio
async def test_report_generation_failure_is_marked_not_stored():
    store = FakeReportGraphStore()
    llm = FakeReportLLM()
    summarizer = CommunitySummarizer(store, llm=llm)

    results, stats = await summarizer.generate_all(level=0)

    assert results == []
    assert stats.failed == 1
    assert stats.succeeded == 0
    assert store.stored_reports == []
    assert len(store.failed_reports) == 1
    assert store.failed_reports[0][:3] == ("entity-graph", "community:1", 0)


@pytest.mark.asyncio
async def test_report_generation_stores_input_fingerprint():
    store = FakeReportGraphStore()
    llm = FakeValidReportLLM()
    summarizer = CommunitySummarizer(store, llm=llm)

    results, stats = await summarizer.generate_all(level=0)

    assert stats.succeeded == 1
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



class _CharacterCounter:
    def count(self, text: str) -> int:
        return len(text)

    def truncate(self, text: str, max_tokens: int) -> str:
        return text[:max_tokens]


class _LegacyClaimReportLLM(CapturingValidReportLLM):
    async def ainvoke(self, prompt):
        response = await super().ainvoke(prompt)
        payload = json.loads(response.content)
        payload["findings"][0]["references"] = [
            {"target_id": claim_id, "target_type": "claim"}
            for claim_id in ("claim:alice", "claim:charlie", "claim:child")
        ]
        return LLMResponse(content=json.dumps(payload))


_LEGACY_DIRECT_EDGE = (
    "- [Person] Alice: CEO\n- [Organization] Acme: Company\n"
    "  Alice --[WORKS_AT]--> Acme: Alice works at Acme"
)
_LEGACY_DIRECT_ISOLATE = "- [Entity] Charlie: Intern"
_LEGACY_DIRECT_TEXT = _LEGACY_DIRECT_EDGE + "\n" + _LEGACY_DIRECT_ISOLATE
_LEGACY_CHILD_ONE = (
    "--- Sub-community child:1 (level 1) ---\nChild narrative cites claim:child."
)
_LEGACY_CHILD_TWO = "--- Sub-community child:2 (level 1) ---\nDiscardable child narrative."
_DIRECT_CLAIMS = (
    "\n\nClaims:\n"
    "  [claim:alice] (role; status=active, dates=2020..?) Alice is CEO Source: Reviewed source\n"
    "  [claim:charlie] (role; status=active) Charlie is intern"
)
_LEGACY_PACKED_IDS = [
    "person:alice", "org:acme", "person:alice:WORKS_AT:org:acme", "person:charlie",
    "claim:alice", "claim:charlie",
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "budget,parent,text,used,truncated,refs",
    [
        (None, False, _LEGACY_DIRECT_TEXT + (
            "\n\nClaims:\n"
            "  [claim:alice] (role; status=active, dates=2020..?) Alice is CEO Source: Reviewed source\n"
            "  [claim:charlie] (role; status=active) Charlie is intern"
        ), None, False, [
            "person:alice", "org:acme", "person:charlie", "person:alice:WORKS_AT:org:acme",
            "claim:alice", "claim:charlie",
        ]),
        (1000, False, _LEGACY_DIRECT_TEXT + _DIRECT_CLAIMS,
         len(_LEGACY_DIRECT_TEXT + _DIRECT_CLAIMS), False, _LEGACY_PACKED_IDS),
        (0, False, "", 0, False, []),
        (1000, True, f"{_LEGACY_CHILD_ONE}\n\n{_LEGACY_CHILD_TWO}\n\n{_LEGACY_DIRECT_TEXT}{_DIRECT_CLAIMS}",
         len(f"{_LEGACY_CHILD_ONE}\n\n{_LEGACY_CHILD_TWO}\n\n{_LEGACY_DIRECT_TEXT}{_DIRECT_CLAIMS}"),
         False, _LEGACY_PACKED_IDS),
        (180, True, _LEGACY_CHILD_ONE + (
            "\n\n--- Sub-community child:2 (level\n\n" + _LEGACY_DIRECT_ISOLATE
        ), 136, True, ["person:charlie"]),
        (40, True, "--- Sub-community child:", 24, True, []),
        (0, True, "", 0, False, []),
        (-1, True, "", 0, False, []),
    ],
)
async def test_selected_evidence_controls_report_prompt_telemetry_and_claims(
    budget, parent, text, used, truncated, refs,
):
    class ParentStore(FakeReportGraphStore):
        def get_child_community_reports(self, graph_name, community_id, level, child_level):
            if not parent:
                return []
            return [
                {"id": "child:1", "level": 1, "report_text": "Child narrative cites claim:child.",
                 "context_tokens_used": 100},
                {"id": "child:2", "level": 1, "report_text": "Discardable child narrative.",
                 "context_tokens_used": 50},
            ]

    llm = _LegacyClaimReportLLM()
    summarizer = CommunitySummarizer(
        cast(GraphStore, ParentStore()), llm=cast(BaseLLM, llm),
        max_context_tokens=budget, token_counter=_CharacterCounter(),
    )
    context = summarizer._fetch_community_context_obj("community:1", 0)
    context.entities.append(EntityContext("person:charlie", "Charlie", "Intern"))
    context.claims.extend([
        ClaimContext("claim:alice", "person:alice", "role", "Alice is CEO",
                     source_text="Reviewed source", start_date="2020"),
        ClaimContext("claim:charlie", "person:charlie", "role", "Charlie is intern"),
    ])
    report = await summarizer.generate_report(
        "community:1", context=context, input_fingerprint="unchanged-fingerprint",
    )
    assert llm.prompts == [build_report_prompt("community:1", 0, text, refs)]
    assert llm.calls == 1
    assert report.context_tokens_used == used
    assert report.context_truncated is truncated
    assert report.version.input_fingerprint == "unchanged-fingerprint"
    # Only fully rendered direct claims grant IDs, never child narratives.
    assert [ref.target_id for finding in report.findings for ref in finding.references] == [
        claim_id for claim_id in ("claim:alice", "claim:charlie") if claim_id in refs
    ]


class _ClaimStore(FakeReportGraphStore):
    def get_claims_for_entities(self, graph_name, entity_ids):
        return [
            {"claim_id": "claim:shown", "entity_id": "person:alice", "claim_type": "role",
             "description": "The claim-only fact."},
            {"claim_id": "claim:hidden", "entity_id": "person:alice", "claim_type": "role",
             "description": "H" * 1000},
        ]


class _RepairClaimLLM(CapturingValidReportLLM):
    def __init__(self, fail_repair=False):
        super().__init__()
        self.fail_repair = fail_repair

    async def ainvoke(self, prompt):
        response = await super().ainvoke(prompt)
        payload = json.loads(response.content)
        if self.calls == 1 or self.fail_repair:
            payload["findings"] = []  # An actual structural failure triggers repair.
        else:
            payload["findings"][0]["references"] = [
                {"target_id": claim_id, "target_type": "claim"}
                for claim_id in ("claim:shown", "claim:hidden", "claim:child")
            ]
        return LLMResponse(content=json.dumps(payload))


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_repair", [False, True])
async def test_claim_selection_shared_by_prompt_repair_and_failure_storage(fail_repair):
    store = _ClaimStore()
    llm = _RepairClaimLLM(fail_repair)
    summarizer = CommunitySummarizer(
        cast(GraphStore, store), llm=cast(BaseLLM, llm),
        max_context_tokens=250, token_counter=_CharacterCounter(),
    )
    results, stats = await summarizer.generate_all(level=0)
    expected_text = _LEGACY_DIRECT_EDGE + (
        "\n\nClaims:\n  [claim:shown] (role; status=active) The claim-only fact."
    )
    refs = ["person:alice", "org:acme", "person:alice:WORKS_AT:org:acme", "claim:shown"]
    assert llm.calls == 2
    assert llm.prompts[0] == build_report_prompt("community:1", 0, expected_text, refs)
    repair_ids = llm.prompts[1].split("## Valid reference IDs (use ONLY these)\n")[1].split(
        "\n\n## Required JSON format",
    )[0]
    assert repair_ids == "\n".join(f"  - {ref}" for ref in refs)
    if fail_repair:
        assert results == []
        assert stats.failed == 1
        assert stats.succeeded == 0
        assert store.stored_reports == []
        assert len(store.failed_reports) == 1
        assert store.failed_reports[0][:3] == ("entity-graph", "community:1", 0)
    else:
        assert stats.succeeded == 1
        assert stats.failed == 0
        assert len(store.stored_reports) == 1
        assert store.failed_reports == []
        report = store.stored_reports[0][1]
        assert report.context_tokens_used == len(expected_text)
        assert report.context_truncated is True
        assert report.version.input_fingerprint
        assert [ref.target_id for ref in report.findings[0].references] == ["claim:shown"]


class _ParentBoundaryCounter(_CharacterCounter):
    def __init__(self, boundary_adjustment):
        self.boundary_adjustment = boundary_adjustment

    def count(self, text: str) -> int:
        if "--- Sub-community" in text and "- [Entity]" in text:
            return max(0, len(text) + self.boundary_adjustment)
        return len(text)


@pytest.mark.parametrize("adjustment,expected_refs", [
    (-70, ["subject", "claim:shown"]), (200, []),
])
def test_parent_direct_admission_counts_full_nonadditive_composition(adjustment, expected_refs):
    counter = _ParentBoundaryCounter(adjustment)
    summarizer = CommunitySummarizer(
        cast(GraphStore, FakeReportGraphStore()), llm=cast(BaseLLM, FakeValidReportLLM()),
        token_counter=counter,
    )
    context = CommunityContext(
        "c1", 0, entities=[EntityContext("subject", "Alice", "CEO")],
        claims=[ClaimContext("claim:shown", "subject", "role", "Fact")],
    )
    children = [{"id": "child:1", "level": 1, "report_text": "Child report"}]
    text, used, truncated, refs = summarizer._build_parent_context(context, children, 100)
    child = "--- Sub-community child:1 (level 1) ---\nChild report"
    assert used == counter.count(text) <= 100
    assert refs == expected_refs
    if adjustment < 0:
        assert text == child + (
            "\n\n- [Entity] Alice: CEO\n\nClaims:\n  [claim:shown] (role; status=active) Fact"
        )
        assert truncated is False
        assert len(text) > 100  # It fits the supplied counter, not a character remainder.
    else:
        assert text == child
        assert truncated is True


@pytest.mark.parametrize("bad_truncation", ["oversized", "fabricated", "loses-prefix"])
def test_parent_rejects_invalid_custom_truncation_and_retains_only_rendered_evidence(bad_truncation):
    class Counter(_CharacterCounter):
        def truncate(self, text: str, max_tokens: int) -> str:
            if bad_truncation == "oversized":
                return text
            if bad_truncation == "fabricated":
                return "Fabricated narrative"
            return text[:5]  # Fits, but loses the already admitted first child.

    counter = Counter()
    summarizer = CommunitySummarizer(
        cast(GraphStore, FakeReportGraphStore()), llm=cast(BaseLLM, FakeValidReportLLM()),
        token_counter=counter,
    )
    context = CommunityContext(
        "c1", 0, entities=[EntityContext("subject", "Alice", "CEO")],
        claims=[ClaimContext("claim:shown", "subject", "role", "Fact")],
    )
    children = [
        {"id": "child:1", "level": 1, "report_text": "First"},
        {"id": "child:2", "level": 1, "report_text": "H" * 1000},
    ]
    text, used, truncated, refs = summarizer._build_parent_context(context, children, 250)
    assert text.startswith("--- Sub-community child:1 (level 1) ---\nFirst\n\n- [Entity]")
    assert "child:2" not in text
    assert "Fabricated" not in text
    assert used == counter.count(text) <= 250
    assert truncated is True
    assert refs == ["subject", "claim:shown"]


@pytest.mark.asyncio
async def test_changed_packing_requires_explicit_regeneration_of_matching_stored_reports():
    store = _ClaimStore()
    llm = CapturingValidReportLLM()
    summarizer = CommunitySummarizer(
        cast(GraphStore, store), llm=cast(BaseLLM, llm), max_context_tokens=250,
        token_counter=_CharacterCounter(),
    )
    context = summarizer._fetch_community_context_obj("community:1", 0)
    fingerprint = summarizer._context_fingerprint(context)
    store.existing_fingerprint = fingerprint  # A pre-C06b report with unchanged inputs.
    results, stats = await summarizer.generate_all(level=0, skip_existing=True)
    assert results == []
    assert stats.skipped == 1
    assert llm.calls == 0
    # Packing policy/budget is intentionally not salted into the input fingerprint.
    summarizer.max_context_tokens = 300
    assert summarizer._report_input_fingerprint(context, "community:1", 0) == fingerprint
    results, stats = await summarizer.generate_all(level=0, skip_existing=False)
    assert len(results) == 1
    assert stats.succeeded == 1
    assert llm.calls == 1
    assert store.stored_reports[0][1].version.input_fingerprint == fingerprint
    assert "The claim-only fact." in llm.prompts[0]


@pytest.mark.asyncio
async def test_claims_without_any_entity_preserve_empty_community_contract():
    llm = CapturingValidReportLLM()
    summarizer = CommunitySummarizer(
        cast(GraphStore, FakeReportGraphStore()), llm=cast(BaseLLM, llm), max_context_tokens=100,
    )
    report = await summarizer.generate_report(
        "c1", context=CommunityContext("c1", 0, claims=[
            ClaimContext("claim:orphan", "missing", "role", "No admitted subject"),
        ]),
    )
    assert report.title == "Empty community"
    assert llm.calls == 0
