"""Memgraph graph database backend package."""

from __future__ import annotations

__all__ = [
    "MemgraphGraphStore",
    "MemgraphGraphWriter",
    "IndexManager",
]


def __getattr__(name: str):
    if name == "MemgraphGraphStore":
        from recon_graphrag.graphdb.memgraph.store import MemgraphGraphStore

        return MemgraphGraphStore
    if name == "MemgraphGraphWriter":
        from recon_graphrag.pipelines.memgraph.writer import MemgraphGraphWriter

        return MemgraphGraphWriter
    if name == "IndexManager":
        from recon_graphrag.graphdb.memgraph.index_manager import IndexManager

        return IndexManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
