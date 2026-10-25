"""Community detection and summarization."""

from typing import TYPE_CHECKING

from recon_graphrag.communities.summarization import CommunitySummarizer
from recon_graphrag.communities.pipeline import CommunityPipeline

if TYPE_CHECKING:
    from recon_graphrag.communities.neo4j.detection import CommunityDetector

__all__ = [
    "CommunityDetector",
    "CommunitySummarizer",
    "CommunityPipeline",
]


def __getattr__(name: str):
    if name == "CommunityDetector":
        from recon_graphrag.communities.neo4j.detection import CommunityDetector

        return CommunityDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
