"""Tests for community summarization context formatting."""

from __future__ import annotations

import json
from typing import cast

import pytest

from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.llm.base import BaseLLM, LLMResponse
from recon_graphrag.communities.context import ClaimContext, EntityContext
from recon_graphrag.communities.reports import build_report_prompt
from recon_graphrag.communities.summarization import CommunitySummarizer


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
        (1000, False, _LEGACY_DIRECT_TEXT, 128, False, _LEGACY_PACKED_IDS),
        (0, False, "", 0, False, []),
        (1000, True, f"{_LEGACY_CHILD_ONE}\n\n{_LEGACY_CHILD_TWO}\n\n{_LEGACY_DIRECT_TEXT}",
         270, False, _LEGACY_PACKED_IDS),
        (180, True, _LEGACY_CHILD_ONE + (
            "\n\n--- Sub-community child:2 (level 1\n\n" + _LEGACY_DIRECT_ISOLATE
        ), 134, True, ["person:charlie", "claim:charlie"]),
        (40, True, "--- Sub-community child:", 24, True, []),
        (0, True, "", 0, False, []),
    ],
)
async def test_private_selection_preserves_report_prompt_telemetry_and_legacy_claims(
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
    # Legacy direct claims stay authorized even when absent from the context.
    # A child's narrative never authorizes its claim IDs, even when retained.
    assert [ref.target_id for finding in report.findings for ref in finding.references] == [
        claim_id for claim_id in ("claim:alice", "claim:charlie") if claim_id in refs
    ]
