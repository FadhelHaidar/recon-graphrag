"""Shared config for Neo4j, FalkorDB, LLM, and embedder setup."""

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

from recon_graphrag import FalkorDBGraphStore, Neo4jGraphStore, create_embedder, create_llm

load_dotenv()


def get_neo4j_store() -> Neo4jGraphStore:
    url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    driver = GraphDatabase.driver(url, auth=(user, password))
    return Neo4jGraphStore(driver, database=database)


def get_falkordb_store() -> FalkorDBGraphStore:
    from falkordb import FalkorDB

    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "6379"))
    graph_name = os.getenv("FALKORDB_GRAPH_NAME", "movie-graph")
    client = FalkorDB(host=host, port=port)
    return FalkorDBGraphStore(client, graph_name=graph_name)


EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))


def get_llm(provider: str | None = None):
    """Return an LLM instance for the requested provider.

    Supported providers: "openrouter", "azure_openai", "openai".
    Falls back to the LLM_PROVIDER env var, then "azure_openai".
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "azure_openai")).lower()

    if provider == "openrouter":
        return create_llm(
            "openrouter",
            model_name=os.getenv("OPENROUTER_LLM_MODEL", "qwen/qwen3.6-flash"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model_params={"extra_body": {"reasoning": {"enabled": False}}},
        )

    if provider == "openai":
        return create_llm(
            "openai",
            model_name=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    if provider == "azure_openai":
        return create_llm(
            "azure_openai",
            model_name=os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT_NAME", "gpt-4o"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
        )

    raise ValueError(
        f"Unknown LLM provider: {provider}. "
        "Use one of: openrouter, azure_openai, openai"
    )


def get_embedder(provider: str | None = None):
    """Return an embedder instance for the requested provider.

    Supported providers: "openrouter", "azure_openai", "openai",
    "sentence-transformer".
    Falls back to the EMBEDDER_PROVIDER env var, then "azure_openai".
    """
    provider = (provider or os.getenv("EMBEDDER_PROVIDER", "azure_openai")).lower()

    if provider == "openrouter":
        return create_embedder(
            "openrouter",
            model=os.getenv("OPENROUTER_EMBED_MODEL", "qwen/qwen3-embedding-4b"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model_params={"encoding_format": "float", "dimensions": EMBEDDING_DIM},
        )

    if provider == "openai":
        return create_embedder(
            "openai",
            model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    if provider == "sentence-transformer":
        return create_embedder(
            "sentence-transformer",
            model=os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"),
        )

    if provider == "azure_openai":
        return create_embedder(
            "azure_openai",
            model=os.getenv("AZURE_OPENAI_EMBED_MODEL_DEPLOYMENT_NAME", "text-embedding-3-small"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
        )

    raise ValueError(
        f"Unknown embedder provider: {provider}. "
        "Use one of: openrouter, azure_openai, openai, sentence-transformer"
    )
