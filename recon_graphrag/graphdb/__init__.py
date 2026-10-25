"""Graph database package."""

from typing import TYPE_CHECKING

from recon_graphrag.graphdb.base import GraphStore, GraphWriter

if TYPE_CHECKING:
    from recon_graphrag.graphdb.memgraph.store import MemgraphGraphStore
    from recon_graphrag.graphdb.neo4j.index_manager import IndexManager
    from recon_graphrag.graphdb.neo4j.store import Neo4jGraphStore

__all__ = [
    "GraphStore",
    "GraphWriter",
    "Neo4jGraphStore",
    "MemgraphGraphStore",
    "IndexManager",
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
