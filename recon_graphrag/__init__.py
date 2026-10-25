"""Recon-GraphRAG: a GraphRAG SDK for Neo4j and Memgraph."""

from typing import TYPE_CHECKING

from recon_graphrag._version import __version__, get_git_sha

# Communities
from recon_graphrag.communities import (
    CommunityPipeline,
    CommunitySummarizer,
)

# Config
from recon_graphrag.config.settings import PipelineConfig
from recon_graphrag.embeddings import (
    BaseEmbedder,
    CommunityReportEmbedder,
    EntityEmbedder,
    ModelParamsEmbedder,
    create_embedder,
)
from recon_graphrag.extraction.artifacts import (
    graph_document_from_dict,
    graph_document_to_dict,
    load_graph_document_json,
    save_graph_document_json,
)
from recon_graphrag.extraction.prompts import SchemaPromptBuilder

# Schema
from recon_graphrag.extraction.schema import (
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
    build_schema,
    load_schema_json,
    save_schema_json,
    schema_to_dict,
)
from recon_graphrag.extraction.schema_analyzer import aanalyze_schema, analyze_schema

# Structured data
from recon_graphrag.extraction.structured import (
    ColumnEntity,
    RowMapping,
    RowRelationship,
    TextColumn,
    iter_csv,
    iter_excel,
    iter_sql,
)

# Graph store
from recon_graphrag.graphdb import GraphStore

# Providers
from recon_graphrag.llm import BaseLLM, LLMResponse, LLMUsage, create_llm

# Models
from recon_graphrag.models.types import IndexConfig, SearchResult
from recon_graphrag.observability import (
    JsonlUsageSink,
    TokenUsageEvent,
    TokenUsageLedger,
    UsageTrackingLLM,
    load_usage_events,
    render_usage_table,
    run_scope,
    summarize_events,
    token_stage,
    track_usage,
    usage_delta,
    usage_snapshot,
)

# Pipelines
from recon_graphrag.pipelines.graphrag_pipeline import GraphBuilderPipeline

# Retrieval
from recon_graphrag.retrieval.base import BaseRetriever
from recon_graphrag.retrieval.drift_types import (
    DriftAction,
    DriftQueryState,
    DriftSearchConfig,
)
from recon_graphrag.retrieval.search_drift import DriftSearchRetriever
from recon_graphrag.retrieval.search_global import GlobalSearchRetriever
from recon_graphrag.retrieval.search_local import LocalSearchRetriever

# The names below are served lazily by __getattr__ to keep driver imports off
# the import path; declare them here so type checkers see the real classes.
if TYPE_CHECKING:
    from recon_graphrag.communities.neo4j.detection import CommunityDetector
    from recon_graphrag.graphdb.index_manager import IndexManager
    from recon_graphrag.graphdb.memgraph.store import MemgraphGraphStore
    from recon_graphrag.graphdb.neo4j.store import Neo4jGraphStore

__all__ = [
    # Version
    "__version__",
    "get_git_sha",
    # Pipelines
    "GraphBuilderPipeline",
    # Retrieval
    "BaseRetriever",
    "LocalSearchRetriever",
    "GlobalSearchRetriever",
    "DriftSearchRetriever",
    "DriftAction",
    "DriftQueryState",
    "DriftSearchConfig",
    # Providers
    "create_llm",
    "create_embedder",
    "BaseLLM",
    "LLMResponse",
    "LLMUsage",
    "BaseEmbedder",
    "CommunityReportEmbedder",
    "EntityEmbedder",
    "ModelParamsEmbedder",
    # Observability
    "UsageTrackingLLM",
    "TokenUsageEvent",
    "TokenUsageLedger",
    "token_stage",
    "run_scope",
    "track_usage",
    "usage_snapshot",
    "usage_delta",
    "render_usage_table",
    "JsonlUsageSink",
    "load_usage_events",
    "summarize_events",
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
    "schema_to_dict",
    "save_schema_json",
    "load_schema_json",
    "analyze_schema",
    "aanalyze_schema",
    "SchemaPromptBuilder",
    # Structured data
    "ColumnEntity",
    "RowMapping",
    "RowRelationship",
    "TextColumn",
    "iter_csv",
    "iter_excel",
    "iter_sql",
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
    "CommunityPipeline",
    # Config
    "PipelineConfig",
]


def __getattr__(name: str):
    if name == "Neo4jGraphStore":
        from recon_graphrag.graphdb.neo4j.store import Neo4jGraphStore

        return Neo4jGraphStore
    if name == "IndexManager":
        from recon_graphrag.graphdb.index_manager import IndexManager

        return IndexManager
    if name == "MemgraphGraphStore":
        from recon_graphrag.graphdb.memgraph.store import MemgraphGraphStore

        return MemgraphGraphStore
    if name == "CommunityDetector":
        from recon_graphrag.communities.neo4j.detection import CommunityDetector

        return CommunityDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
