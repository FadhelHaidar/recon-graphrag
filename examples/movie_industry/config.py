"""Shared config for Neo4j, LLM, and embedder setup."""

import os

from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase

from recon_graphrag import Neo4jGraphStore, create_llm, create_embedder


def get_neo4j_store() -> Neo4jGraphStore:
    url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    driver = GraphDatabase.driver(url, auth=(user, password))
    return Neo4jGraphStore(driver, database=database)


def get_llm():
    # Option 1: OpenRouter
    return create_llm(
        "openrouter",
        model_name="qwen/qwen3.6-flash",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model_params={"extra_body": {"reasoning": {"enabled": True}}},
    )
    # Option 2: Azure OpenAI
    # return create_llm(
    #     "azure_openai",
    #     model_name=os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT_NAME", "gpt-4o"),
    #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    #     api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
    #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    #     model_params={"temperature": 0},
    # )
    # Option 3: Custom OpenAI-compatible
    # return create_llm(
    #     "openai",
    #     model_name="custom-model",
    #     base_url="http://localhost:8000/v1",
    #     api_key="dummy",
    # )


def get_embedder():
    # Option 1: OpenRouter
    return create_embedder(
        "openrouter",
        model="qwen/qwen3-embedding-8b",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model_params={"encoding_format": "float", "dimensions": 1536},
    )
    # Option 2: Sentence-Transformers (local, no API key)
    # return create_embedder("sentence-transformer", model="all-MiniLM-L6-v2")
    # Option 3: Azure OpenAI
    # return create_embedder(
    #     "azure_openai",
    #     model=os.getenv("AZURE_OPENAI_EMBED_MODEL_DEPLOYMENT_NAME", "text-embedding-3-small"),
    #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    #     api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
    #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    # )
