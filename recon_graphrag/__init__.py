"""Recon-GraphRAG: Domain-agnostic GraphRAG SDK built on Neo4j."""

__version__ = "0.1.0"

# Pipelines
from recon_graphrag.pipelines.graphrag_pipeline import GraphBuilderPipeline

# Retrieval
from recon_graphrag.retrieval.search import GraphRAG
from recon_graphrag.retrieval.base import BaseRetriever
from recon_graphrag.retrieval.local import LocalSearchRetriever
from recon_graphrag.retrieval.global_search import GlobalSearchRetriever
from recon_graphrag.retrieval.drift import DriftSearchRetriever

# Providers
from recon_graphrag.llm import create_llm, BaseLLM
from recon_graphrag.embeddings import create_embedder, BaseEmbedder, ModelParamsEmbedder

# Graph store
from recon_graphrag.graph import GraphStore, Neo4jGraphStore, IndexManager

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

# Communities
from recon_graphrag.communities import (
    CommunityDetector,
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
    "BaseEmbedder",
    "ModelParamsEmbedder",
    # Graph store
    "GraphStore",
    "Neo4jGraphStore",
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
    # Communities
    "CommunityDetector",
    "CommunitySummarizer",
    "CommunityEmbedder",
    "CommunityPipeline",
    # Config
    "PipelineConfig",
]
