"""Pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass

from recon_graphrag.utils.tokens import TokenCounter


@dataclass
class PipelineConfig:
    """Standalone pipeline options record.

    GraphBuilderPipeline does not accept or automatically apply this record.
    Pass the corresponding values to pipeline constructors or build methods.

    Attributes:
        chunk_size: Text chunk size in ``chunk_unit`` units.
        chunk_overlap: Overlap between consecutive chunks.
        chunk_unit: ``"tokens"`` (default) or ``"characters"``.
        embedding_dim: Optional vector dimension. This record does not detect it.
        extraction_concurrency: Maximum concurrent chunk extraction calls.
            Set to 1 for sequential extraction.
        max_gleanings: Maximum follow-up extraction loops after the first pass.
        extract_claims: Whether to extract claims about known entities.
        use_mixed_context: Whether to combine retrieval context types.
        token_counter: Optional counter for token budgets.
    """

    chunk_size: int = 1200
    chunk_overlap: int = 100
    chunk_unit: str = "tokens"
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
