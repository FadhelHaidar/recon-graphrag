"""Characterization tests for community report inputs and outputs.

These tests pin order, fingerprints, prompt bytes, and store calls that the
report path must preserve while it becomes cheaper.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, cast

import pytest

from recon_graphrag.communities.context import build_reference_ids
from recon_graphrag.communities.summarization import CommunitySummarizer
from recon_graphrag.llm.base import LLMResponse


def _edge_row(
    source: str,
    target: str,
    *,
    rel_type: str = "WORKS_AT",
    combined_degree: int = 7,
    observation_count: int = 2,
) -> dict[str, Any]:
    return {
        "e_id": source,
        "e_name": source,
        "e_description": f"{source} description",
        "e_labels": ["Person"],
        "e_degree": 3,
        "rel_type": rel_type,
        "rel_description": f"{source} {rel_type} {target}",
        "observation_count": observation_count,
        "combined_degree": combined_degree,
        "other_id": target,
        "other_name": target,
        "other_description": f"{target} description",
        "other_labels": ["Organization"],
        "other_degree": 2,
    }


def _isolated_row(entity_id: str, *, name: str = "Solo") -> dict[str, Any]:
    return {
        "e_id": entity_id,
        "e_name": name,
        "e_description": "Isolated",
        "e_labels": ["Thing"],
        "e_degree": 0,
        "rel_type": None,
        "rel_description": None,
        "observation_count": 0,
        "combined_degree": 0,
        "other_id": None,
        "other_name": None,
        "other_description": None,
        "other_labels": [],
        "other_degree": 0,
    }


class RecordingStore:
    """Store double that records claim lookups and child-report reads."""

    def __init__(self, rows: list[dict[str, Any]], children: list[dict[str, Any]] | None = None):
        self.rows = rows
        self.children = children or []
        self.claim_calls: list[list[str]] = []
        self.child_reads = 0
        self.existing_fingerprint: str | None = None

    def get_communities(self, graph_name, level=None):
        return [{"id": "community:1", "level": 0, "entity_count": len(self.rows)}]

    def get_community_ranked_context(self, graph_name, community_id, level=0):
        return self.rows

    def get_claims_for_entities(self, graph_name, entity_ids):
        self.claim_calls.append(list(entity_ids))
        return [
            {
                "claim_id": "claim:1",
                "entity_id": "person:alice",
                "claim_type": "fact",
                "description": "Alice leads Acme.",
                "status": "active",
                "chunk_id": "doc:1:c0",
            }
        ]

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

    def get_child_community_reports(self, graph_name, community_id, level, child_level):
        self.child_reads += 1
        return [
            {
                "id": "child:2",
                "level": child_level,
                "report_text": "Second child report.",
                "input_fingerprint": "fp-2",
                "context_tokens_used": 300,
            },
            {
                "id": "child:1",
                "level": child_level,
                "report_text": "First child report.",
                "input_fingerprint": "fp-1",
                "context_tokens_used": 120,
            },
        ]

    def store_community_report(self, report, graph_name):
        self.stored = (graph_name, report)

    def mark_community_report_failed(self, *args):
        pass


class CapturingLLM:
    def __init__(self):
        self.prompts: list[str] = []

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return LLMResponse(
            content=json.dumps(
                {
                    "title": "Alice and Acme",
                    "summary": "Alice leads Acme.",
                    "rating": 6.0,
                    "rating_explanation": "Core relationship.",
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


def _summarizer(store: RecordingStore, llm: CapturingLLM | None = None, **kwargs):
    return CommunitySummarizer(
        cast(Any, store),
        llm=cast(Any, llm or CapturingLLM()),
        **kwargs,
    )


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (
            [
                _edge_row("person:alice", "org:acme"),
                _isolated_row("solo:ü"),
                _edge_row("org:acme", "person:bob", rel_type="EMPLOYS", combined_degree=2),
            ],
            ["solo:ü", "person:alice", "org:acme", "person:bob"],
        ),
        ([_edge_row("person:alice", "person:alice")], ["person:alice"]),
        (
            [
                _edge_row("person:alice", "org:acme"),
                _edge_row("org:acme", "person:bob", rel_type="EMPLOYS", combined_degree=2),
                _edge_row("person:bob", "person:alice", rel_type="KNOWS", combined_degree=1),
            ],
            ["person:alice", "org:acme", "person:bob"],
        ),
        ([], []),
    ],
)
def test_claim_entity_ids_keep_first_seen_order(rows, expected):
    store = RecordingStore(rows)
    summarizer = _summarizer(store)

    summarizer._fetch_community_context_obj("community:1", 0)

    if not expected:
        assert store.claim_calls == []
    else:
        assert store.claim_calls == [expected]


@pytest.mark.asyncio
async def test_report_outputs_keep_golden_bytes_and_metadata():
    store = RecordingStore(
        [_edge_row("person:alice", "org:acme"), _isolated_row("solo:ü", name="Solo")],
        children=[
            {
                "id": "child:1",
                "level": 1,
                "report_text": "First child report.",
                "input_fingerprint": "fp-1",
                "context_tokens_used": 120,
            },
            {
                "id": "child:2",
                "level": 1,
                "report_text": "Second child report.",
                "input_fingerprint": "fp-2",
                "context_tokens_used": 300,
            },
        ],
    )
    llm = CapturingLLM()
    summarizer = _summarizer(store, llm, max_context_tokens=200)
    context = summarizer._fetch_community_context_obj("community:1", 0)

    assert store.claim_calls == [["solo:ü", "person:alice", "org:acme"]]
    assert summarizer._context_fingerprint(context) == (
        "c84cff15ae1e342b1736500b39b1330e04de6986b1dd8cbe8e3b98ad578ddab2"
    )
    assert summarizer._report_input_fingerprint(context, "community:1", 0) == (
        "4a5c0f3071ee5589e5e41e2618531dcbe38ebea191639899c5327150110b11eb"
    )
    assert build_reference_ids(context) == [
        "person:alice",
        "org:acme",
        "solo:ü",
        "person:alice:WORKS_AT:org:acme",
        "claim:1",
    ]

    report = await summarizer.generate_report("community:1", level=0)

    assert hashlib.sha256(llm.prompts[0].encode("utf-8")).hexdigest() == (
        "2afd07e6b68972502083fc6f05d7ae921c30817bfb271cb370d72ec424ddc5c7"
    )
    assert report.context_tokens_used == 79
    assert report.context_truncated is False
    assert report.version.input_fingerprint == (
        "4a5c0f3071ee5589e5e41e2618531dcbe38ebea191639899c5327150110b11eb"
    )
