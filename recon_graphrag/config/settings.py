"""Pipeline configuration.

Dataclass-based config objects to avoid massive constructor signatures.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Configuration for GraphBuilderPipeline.

    Attributes:
        chunk_size: Character-level chunk size for text splitting.
        chunk_overlap: Overlap between consecutive chunks.
        embedding_dim: Vector embedding dimension. Auto-detected for
            sentence-transformers; defaults to 1536 (OpenAI) if not specified.
    """

    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_dim: int | None = None
