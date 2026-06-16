"""FalkorDB graph database backend."""

__all__ = [
    "FalkorDBGraphStore",
    "FalkorDBGraphWriter",
    "IndexManager",
    "CommunityDetector",
]


def __getattr__(name: str):
    if name == "FalkorDBGraphStore":
        from recon_graphrag.graphdb.falkordb.store import FalkorDBGraphStore

        return FalkorDBGraphStore
    if name == "FalkorDBGraphWriter":
        from recon_graphrag.pipelines.falkordb.writer import FalkorDBGraphWriter

        return FalkorDBGraphWriter
    if name == "IndexManager":
        from recon_graphrag.graphdb.falkordb.index_manager import IndexManager

        return IndexManager
    if name == "CommunityDetector":
        from recon_graphrag.communities.falkordb.detection import CommunityDetector

        return CommunityDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
