"""Memgraph graph database backend package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from recon_graphrag.graphdb.memgraph.index_manager import IndexManager
    from recon_graphrag.graphdb.memgraph.store import MemgraphGraphStore

__all__ = [
    "MemgraphGraphStore",
    "IndexManager",
]


def __getattr__(name: str):
    if name == "MemgraphGraphStore":
        from recon_graphrag.graphdb.memgraph.store import MemgraphGraphStore

        return MemgraphGraphStore
    if name == "IndexManager":
        from recon_graphrag.graphdb.memgraph.index_manager import IndexManager

        return IndexManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
