"""Tests for evaluation artifact schemas."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from evaluation.schemas import (
    PerQuestionResult,
    PipelineConfigSnapshot,
    RetrievedContext,
    RunManifest,
    SearchConfigSnapshot,
    TokenUsage,
)


def test_pipeline_config_defaults():
    cfg = PipelineConfigSnapshot()
    assert cfg.chunk_size == 1000
    assert cfg.graph_name == "entity-graph"


def test_search_config_level_optional():
    cfg = SearchConfigSnapshot()
    assert cfg.level is None


def test_token_usage_allows_null_counts():
    usage = TokenUsage()
    assert usage.request_tokens is None
    assert usage.response_tokens is None
    assert usage.total_tokens is None


def test_retrieved_context_requires_fields():
    with pytest.raises(ValidationError):
        RetrievedContext()

    ctx = RetrievedContext(community_id="c1", level=0, summary="s", score=0.5)
    assert ctx.community_id == "c1"


def test_per_question_result_round_trip():
    result = PerQuestionResult(
        question_id="q1",
        question="What is GraphRAG?",
        answer="A graph-based RAG approach.",
        retrieved_contexts=[
            RetrievedContext(community_id="c1", level=0, summary="s", score=0.9)
        ],
        elapsed_seconds=1.23,
    )
    raw = json.loads(json.dumps(result.model_dump(mode="json"), sort_keys=True))
    assert raw["question_id"] == "q1"
    assert raw["answer"] == "A graph-based RAG approach."
    assert raw["token_usage"] is None
    assert raw["retry_count"] == 0


def test_manifest_serialization_sorted_keys():
    manifest = RunManifest(
        run_id="abc",
        created_at=datetime(2026, 6, 22, 12, 0, 0, tzinfo=timezone.utc),
        git_sha="def",
        package_version="0.3.0",
        corpus_id="corpus",
        corpus_hash="hash",
        question_set_id="questions",
        pipeline_config=PipelineConfigSnapshot(),
        search_config=SearchConfigSnapshot(),
        model_identifiers={},
        prompt_versions={},
    )
    text = json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True)
    keys = list(json.loads(text).keys())
    assert keys == sorted(keys)
    assert "created_at" in text
    assert "package_version" in text


def test_manifest_missing_required_field_raises():
    with pytest.raises(ValidationError):
        RunManifest()
