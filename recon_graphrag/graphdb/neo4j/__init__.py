"""Neo4j graph database backend."""

__all__ = [
    "Neo4jGraphStore",
    "Neo4jGraphWriter",
    "IndexManager",
    "CommunityDetector",
]


def __getattr__(name: str):
    if name == "Neo4jGraphStore":
        from recon_graphrag.graphdb.neo4j.store import Neo4jGraphStore

        return Neo4jGraphStore
    if name == "Neo4jGraphWriter":
        from recon_graphrag.pipelines.neo4j.writer import Neo4jGraphWriter

        return Neo4jGraphWriter
    if name == "IndexManager":
        from recon_graphrag.graphdb.neo4j.index_manager import IndexManager

        return IndexManager
    if name == "CommunityDetector":
        from recon_graphrag.communities.neo4j.detection import CommunityDetector

        return CommunityDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
