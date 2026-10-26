"""Shared constants for the movie example scripts.

Provides backend selection and search-option constants used by
extract.py, ingest.py, communities.py, and search.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from recon_graphrag import IndexManager as Neo4jIndexManager
from recon_graphrag.graphdb.memgraph.index_manager import IndexManager as MemgraphIndexManager

from config import get_memgraph_store, get_neo4j_store


DEFAULT_ARTIFACT_PATH = Path(__file__).with_name("artifacts") / "movie_graph.json"

ALL_BACKENDS = "all"

SEARCH_OPTIONS = {
    "local": {},
    "global": {
        "community_level": "coarsest",
    },
    "drift": {
        "top_k": 10,
        "community_level": "finest",
    },
}

BACKEND_REGISTRY = {
    "neo4j": (get_neo4j_store, Neo4jIndexManager),
    "memgraph": (get_memgraph_store, MemgraphIndexManager),
}
BACKEND_CHOICES = (*BACKEND_REGISTRY, ALL_BACKENDS)


def ensure_indexes(index_manager_cls: type, store: Any, embedding_dim: int) -> None:
    """Create retrieval indexes, tolerating an already-present unverifiable index.

    Memgraph does not report vector index dimensions, so a second setup run
    fails closed. The example scripts are re-runnable, so an index this script
    already created is treated as present; change the embedding dimension with
    ``store.drop_indexes()`` first.
    """
    try:
        index_manager_cls(store, embedding_dim=embedding_dim).create_indexes()
    except ValueError as exc:
        if "Cannot verify dimensions" not in str(exc):
            raise
        print(f"Indexes already present ({exc})")


def get_backend_targets(backend: str) -> list[tuple[str, Any, type]]:
    """Return (name, store, index_manager_cls) for the selected backend(s)."""
    if backend == ALL_BACKENDS:
        selected = BACKEND_REGISTRY.items()
    elif backend in BACKEND_REGISTRY:
        selected = [(backend, BACKEND_REGISTRY[backend])]
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return [
        (name, store_factory(), index_manager_cls)
        for name, (store_factory, index_manager_cls) in selected
    ]
