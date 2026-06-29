"""Shared exception types for Recon-GraphRAG."""


class ReindexRequiredError(Exception):
    """Raised when a write operation detects a legacy graph without a v2 schema marker.

    The graph must be rebuilt from scratch using the v2 pipeline.
    """

    def __init__(self, message: str = ""):
        super().__init__(
            message
            or "This graph uses the legacy schema (v1). "
            "A full reindex is required. Rebuild the graph using GraphBuilderPipeline."
        )
