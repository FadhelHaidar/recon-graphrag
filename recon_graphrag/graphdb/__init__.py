"""Graph database package."""

from recon_graphrag.graphdb.base import GraphStore, GraphWriter

__all__ = [
    "GraphStore",
    "GraphWriter",
    "Neo4jGraphStore",
    "FalkorDBGraphStore",
    "IndexManager",
    "Neo4jGraphWriter",
    "FalkorDBGraphWriter",
]


def __getattr__(name: str):
    if name == "Neo4jGraphStore":
        from recon_graphrag.graphdb.neo4j.store import Neo4jGraphStore

        return Neo4jGraphStore
    if name == "FalkorDBGraphStore":
        from recon_graphrag.graphdb.falkordb.store import FalkorDBGraphStore

        return FalkorDBGraphStore
    if name == "IndexManager":
        from recon_graphrag.graphdb.neo4j.index_manager import IndexManager

        return IndexManager
    if name == "Neo4jGraphWriter":
        from recon_graphrag.pipelines.neo4j.writer import Neo4jGraphWriter

        return Neo4jGraphWriter
    if name == "FalkorDBGraphWriter":
        from recon_graphrag.pipelines.falkordb.writer import FalkorDBGraphWriter

        return FalkorDBGraphWriter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
