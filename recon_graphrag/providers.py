"""LLM and Embedder factory functions.

Convenience module to create LLM and Embedder instances without
needing to know the exact import paths. Phase 1 supports OpenAI,
Azure OpenAI, and Ollama. Phase 2 will add more providers.

Users can also pass their own LLMInterface/Embedder implementations
directly — the factory is just convenience.
"""

from __future__ import annotations

from typing import Any

from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.embeddings import Embedder


def create_llm(provider: str, **kwargs: Any) -> LLMInterface:
    """Create an LLM instance for a supported provider.

    Args:
        provider: One of "openai", "azure_openai", "ollama".
        **kwargs: Passed to the underlying LLM constructor.

    Returns:
        An LLMInterface instance.

    Example:
        llm = create_llm("openai", model_name="gpt-4o", api_key="sk-...")
        llm = create_llm("azure_openai", model_name="my-deployment",
                          api_key="...", api_version="2025-03-01-preview",
                          azure_endpoint="https://my-resource.openai.azure.com/")
        llm = create_llm("ollama", model_name="llama3")
    """
    providers = {
        "openai": _create_openai_llm,
        "azure_openai": _create_azure_openai_llm,
        "ollama": _create_ollama_llm,
    }
    if provider not in providers:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            f"Phase 1 supports: {', '.join(providers.keys())}"
        )
    return providers[provider](**kwargs)


def create_embedder(provider: str, **kwargs: Any) -> Embedder:
    """Create an Embedder instance for a supported provider.

    Args:
        provider: One of "openai", "azure_openai", "ollama", "sentence-transformer".
        **kwargs: Passed to the underlying Embedder constructor.

    Returns:
        An Embedder instance.

    Example:
        embedder = create_embedder("openai", model="text-embedding-3-small", api_key="sk-...")
        embedder = create_embedder("sentence-transformer", model="all-MiniLM-L6-v2")
    """
    providers = {
        "openai": _create_openai_embedder,
        "azure_openai": _create_azure_openai_embedder,
        "ollama": _create_ollama_embedder,
        "sentence-transformer": _create_sentence_transformer_embedder,
    }
    if provider not in providers:
        raise ValueError(
            f"Unknown embedder provider: '{provider}'. "
            f"Phase 1 supports: {', '.join(providers.keys())}"
        )
    return providers[provider](**kwargs)


# --- LLM constructors ---

def _create_openai_llm(**kwargs) -> LLMInterface:
    from neo4j_graphrag.llm import OpenAILLM
    return OpenAILLM(**kwargs)


def _create_azure_openai_llm(**kwargs) -> LLMInterface:
    from neo4j_graphrag.llm import AzureOpenAILLM
    return AzureOpenAILLM(**kwargs)


def _create_ollama_llm(**kwargs) -> LLMInterface:
    from neo4j_graphrag.llm import OllamaLLM
    return OllamaLLM(**kwargs)


# --- Embedder constructors ---

def _create_openai_embedder(**kwargs) -> Embedder:
    from neo4j_graphrag.embeddings import OpenAIEmbeddings
    return OpenAIEmbeddings(**kwargs)


def _create_azure_openai_embedder(**kwargs) -> Embedder:
    from neo4j_graphrag.embeddings import AzureOpenAIEmbeddings
    return AzureOpenAIEmbeddings(**kwargs)


def _create_ollama_embedder(**kwargs) -> Embedder:
    from neo4j_graphrag.embeddings import OllamaEmbeddings
    return OllamaEmbeddings(**kwargs)


def _create_sentence_transformer_embedder(**kwargs) -> Embedder:
    from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
    return SentenceTransformerEmbeddings(**kwargs)
