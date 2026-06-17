"""Graph database package."""

from recon_graphrag.graphdb.base import GraphStore, GraphWriter

__all__ = [
    "GraphStore",
    "GraphWriter",
    "Neo4jGraphStore",
    "MemgraphGraphStore",
    "IndexManager",
    "Neo4jGraphWriter",
    "MemgraphGraphWriter",
]


def __getattr__(name: str):
    if name == "Neo4jGraphStore":
        from recon_graphrag.graphdb.neo4j.store import Neo4jGraphStore

        return Neo4jGraphStore
    if name == "IndexManager":
        from recon_graphrag.graphdb.neo4j.index_manager import IndexManager

        return IndexManager
    if name == "MemgraphGraphStore":
        from recon_graphrag.graphdb.memgraph.store import MemgraphGraphStore

        return MemgraphGraphStore
    if name == "Neo4jGraphWriter":
        from recon_graphrag.pipelines.neo4j.writer import Neo4jGraphWriter

        return Neo4jGraphWriter
    if name == "MemgraphGraphWriter":
        from recon_graphrag.pipelines.memgraph.writer import MemgraphGraphWriter

        return MemgraphGraphWriter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
