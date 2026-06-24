"""Evaluation run artifact schemas.

Run artifacts are split into a JSON manifest (run-level metadata) and a JSON
Lines file (one record per evaluated question). Stable sorted keys make diffs
predictable.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class PipelineConfigSnapshot(BaseModel):
    """Pipeline configuration captured at evaluation time."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    extraction_concurrency: int = 5
    graph_name: str = "entity-graph"
    perform_entity_resolution: bool = False
    embed_entities: bool = False


class SearchConfigSnapshot(BaseModel):
    """Search configuration captured at evaluation time."""

    mode: str = "global"
    top_k: int = 5
    level: Optional[int] = None


class TokenUsage(BaseModel):
    """Provider-reported token usage; null when unavailable."""

    request_tokens: Optional[int] = None
    response_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class RetrievedContext(BaseModel):
    """One retrieved community context."""

    community_id: str
    level: int
    summary: str
    score: float


class PerQuestionResult(BaseModel):
    """Result for a single evaluation question."""

    question_id: str
    question: str
    answer: str
    retrieved_contexts: list[RetrievedContext]
    token_usage: Optional[TokenUsage] = None
    elapsed_seconds: float
    errors: list[str] = Field(default_factory=list)
    retry_count: int = 0


class RunManifest(BaseModel):
    """Run-level metadata for an evaluation run."""

    run_id: str
    created_at: datetime
    git_sha: str
    package_version: str
    corpus_id: str
    corpus_hash: str
    question_set_id: str
    pipeline_config: PipelineConfigSnapshot
    search_config: SearchConfigSnapshot
    model_identifiers: dict[str, Any]
    prompt_versions: dict[str, str]
    schema_version: str = "1.0"
