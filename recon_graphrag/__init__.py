"""Recon-GraphRAG: Domain-agnostic GraphRAG SDK built on Neo4j."""

__version__ = "0.1.0"

# Types
from recon_graphrag.types import SearchResult, IndexConfig

# Graph store
from recon_graphrag.graph_store import GraphStore, Neo4jGraphStore

# Providers
from recon_graphrag.providers import create_llm, create_embedder

# Schema
from recon_graphrag.schema import GraphSchema, NodeType, PropertyType, RelationshipType, build_schema

# Pipeline
from recon_graphrag.pipeline import GraphBuilderPipeline

# Indexes
from recon_graphrag.indexes import IndexManager

# Communities
from recon_graphrag.communities import (
    CommunityDetector,
    CommunitySummarizer,
    CommunityEmbedder,
    CommunityPipeline,
)

# Retrieval
from recon_graphrag.retrieval import (
    GraphRAG,
    LocalSearchRetriever,
    GlobalSearchRetriever,
    DriftSearchRetriever,
)

__all__ = [
    # Types
    "SearchResult",
    "IndexConfig",
    # Graph store
    "GraphStore",
    "Neo4jGraphStore",
    # Providers
    "create_llm",
    "create_embedder",
    # Schema
    "GraphSchema",
    "NodeType",
    "PropertyType",
    "RelationshipType",
    "build_schema",
    # Pipeline
    "GraphBuilderPipeline",
    # Indexes
    "IndexManager",
    # Communities
    "CommunityDetector",
    "CommunitySummarizer",
    "CommunityEmbedder",
    "CommunityPipeline",
    # Retrieval
    "GraphRAG",
    "LocalSearchRetriever",
    "GlobalSearchRetriever",
    "DriftSearchRetriever",
]
