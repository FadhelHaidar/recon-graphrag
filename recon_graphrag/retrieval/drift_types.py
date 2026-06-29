"""Data types for iterative DRIFT search."""

from __future__ import annotations

from dataclasses import dataclass, field

from recon_graphrag.retrieval.community_levels import CommunityLevelSelector


@dataclass
class DriftAction:
    """One step in the DRIFT traversal tree."""

    id: str
    parent_id: str | None = None
    depth: int = 0
    query: str = ""
    answer: str = ""
    score: float = 0.0
    status: str = "pending"
    context: str = ""
    citations: list = field(default_factory=list)
    follow_ups: list[str] = field(default_factory=list)
    references: list[dict] = field(default_factory=list)
    report_ids: list[str] = field(default_factory=list)


@dataclass
class DriftQueryState:
    """Accumulated state of a DRIFT search."""

    query: str = ""
    primer_reports: list[dict] = field(default_factory=list)
    actions: list[DriftAction] = field(default_factory=list)
    stopping_reason: str = ""
    total_llm_calls: int = 0
    phase_tokens: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)


@dataclass
class DriftSearchConfig:
    """Configuration for iterative DRIFT search."""

    primer_top_k: int = 3
    max_followups: int = 3
    max_depth: int = 3
    min_expand_score: float = 20.0
    max_llm_calls: int = 20
    action_concurrency: int = 3
    community_level: CommunityLevelSelector = "coarsest"
    reduce_budget_tokens: int = 12000

    def __post_init__(self) -> None:
        if self.primer_top_k < 1:
            raise ValueError("primer_top_k must be positive")
        if self.max_followups < 0 or self.max_depth < 0:
            raise ValueError("max_followups and max_depth must be non-negative")
        if not 0 <= self.min_expand_score <= 100:
            raise ValueError("min_expand_score must be between 0 and 100")
        if self.max_llm_calls < 1 or self.action_concurrency < 1:
            raise ValueError("max_llm_calls and action_concurrency must be positive")
        if self.reduce_budget_tokens < 0:
            raise ValueError("reduce_budget_tokens must be non-negative")
