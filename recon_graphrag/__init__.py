"""Recon-GraphRAG: Domain-agnostic GraphRAG SDK built on Neo4j."""

__version__ = "0.1.2"

# Pipelines
from recon_graphrag.pipelines.graphrag_pipeline import GraphBuilderPipeline

# Retrieval
from recon_graphrag.retrieval.search import GraphRAG
from recon_graphrag.retrieval.base import BaseRetriever
from recon_graphrag.retrieval.local import LocalSearchRetriever
from recon_graphrag.retrieval.global_search import GlobalSearchRetriever
from recon_graphrag.retrieval.drift import DriftSearchRetriever

# Providers
from recon_graphrag.llm import create_llm, BaseLLM, LLMResponse, LLMUsage
from recon_graphrag.embeddings import create_embedder, BaseEmbedder, ModelParamsEmbedder

# Graph store
from recon_graphrag.graphdb import GraphStore

# Schema
from recon_graphrag.extraction.schema import (
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
    build_schema,
)

# Models
from recon_graphrag.models.types import SearchResult, IndexConfig
from recon_graphrag.extraction.artifacts import (
    graph_document_from_dict,
    graph_document_to_dict,
    load_graph_document_json,
    save_graph_document_json,
)

# Communities
from recon_graphrag.communities import (
    CommunitySummarizer,
    CommunityEmbedder,
    CommunityPipeline,
)

# Config
from recon_graphrag.config.settings import PipelineConfig

__all__ = [
    # Pipelines
    "GraphBuilderPipeline",
    # Retrieval
    "GraphRAG",
    "BaseRetriever",
    "LocalSearchRetriever",
    "GlobalSearchRetriever",
    "DriftSearchRetriever",
    # Providers
    "create_llm",
    "create_embedder",
    "BaseLLM",
    "LLMResponse",
    "LLMUsage",
    "BaseEmbedder",
    "ModelParamsEmbedder",
    # Graph store
    "GraphStore",
    "Neo4jGraphStore",
    "MemgraphGraphStore",
    "IndexManager",
    # Schema
    "GraphSchema",
    "NodeType",
    "PropertyType",
    "RelationshipType",
    "build_schema",
    # Models
    "SearchResult",
    "IndexConfig",
    "graph_document_to_dict",
    "graph_document_from_dict",
    "save_graph_document_json",
    "load_graph_document_json",
    # Communities
    "CommunityDetector",
    "CommunitySummarizer",
    "CommunityEmbedder",
    "CommunityPipeline",
    # Config
    "PipelineConfig",
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
    if name == "CommunityDetector":
        from recon_graphrag.communities.neo4j.detection import CommunityDetector

        return CommunityDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
