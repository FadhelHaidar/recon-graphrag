"""Pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass

from recon_graphrag.utils.tokens import TokenCounter


@dataclass
class PipelineConfig:
    """Configuration for GraphBuilderPipeline.

    Attributes:
        chunk_size: Character-level chunk size for text splitting.
        chunk_overlap: Overlap between consecutive chunks.
        embedding_dim: Vector embedding dimension. Auto-detected for
            sentence-transformers; defaults to 1536 (OpenAI) if not specified.
        extraction_concurrency: Maximum number of chunks to extract concurrently.
            Set to 1 to process chunks sequentially.
        token_counter: Optional token counter for budget-aware operations.
            When absent, callers fall back to ``ApproximateTokenCounter``.
    """

    chunk_size: int = 1200
    chunk_overlap: int = 100
    embedding_dim: int | None = None
    extraction_concurrency: int = 5
    max_gleanings: int = 1
    extract_claims: bool = False
    use_mixed_context: bool = False
    token_counter: TokenCounter | None = None

    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        if self.max_gleanings < 0:
            raise ValueError("max_gleanings must be >= 0")
