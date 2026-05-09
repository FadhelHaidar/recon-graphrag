"""LLM base interface.

Re-exports neo4j-graphrag's LLMInterface so users can depend on
recon_graphrag.llm.BaseLLM without importing internal neo4j-graphrag paths.
"""

from neo4j_graphrag.llm import LLMInterface as BaseLLM

__all__ = ["BaseLLM"]
