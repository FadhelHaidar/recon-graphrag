"""Community detection, summarization, and embedding."""

from recon_graphrag.communities.summarization import CommunitySummarizer
from recon_graphrag.communities.embeddings import CommunityEmbedder
from recon_graphrag.communities.pipeline import CommunityPipeline

__all__ = [
    "CommunityDetector",
    "CommunitySummarizer",
    "CommunityEmbedder",
    "CommunityPipeline",
]


def __getattr__(name: str):
    if name == "CommunityDetector":
        from recon_graphrag.communities.neo4j.detection import CommunityDetector

        return CommunityDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
