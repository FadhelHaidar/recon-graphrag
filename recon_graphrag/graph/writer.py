"""Database-agnostic graph writer protocol."""

from typing import Protocol

from recon_graphrag.extraction.types import GraphDocument


class GraphWriter(Protocol):
    def write_graph_document(self, graph_document: GraphDocument) -> dict:
        ...
