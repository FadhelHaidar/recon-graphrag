"""Graph database package."""

from recon_graphrag.graphdb.base import GraphStore, GraphWriter

__all__ = [
    "GraphStore",
    "GraphWriter",
    "Neo4jGraphStore",
    "IndexManager",
    "Neo4jGraphWriter",
]


def __getattr__(name: str):
    if name == "Neo4jGraphStore":
        from recon_graphrag.graphdb.neo4j.store import Neo4jGraphStore

        return Neo4jGraphStore
    if name == "IndexManager":
        from recon_graphrag.graphdb.neo4j.index_manager import IndexManager

        return IndexManager
    if name == "Neo4jGraphWriter":
        from recon_graphrag.pipelines.neo4j.writer import Neo4jGraphWriter

        return Neo4jGraphWriter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
