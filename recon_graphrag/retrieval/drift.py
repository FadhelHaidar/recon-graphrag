"""DRIFT search: hybrid entity-centric + community-context retrieval.

Combines the specificity of local search (entity subgraph traversal) with
the broader context of global search (community summaries). The pipeline:

1. Vector search on entities → find seed entities matching the query
2. Expand to communities via IN_COMMUNITY relationships
3. Fetch community summaries for high-level context
4. Fetch other entities in those communities for bridging connections
5. Build enriched context combining all signals
6. LLM generates answer from the combined context
"""

from __future__ import annotations

from typing import Optional

from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.embeddings import Embedder
from neo4j_graphrag.retrievers import HybridCypherRetriever

from recon_graphrag.graph_store import GraphStore
from recon_graphrag.types import SearchResult


DEFAULT_DRIFT_QUERY = """
OPTIONAL MATCH (node)-[r]-(neighbor)
WHERE NOT neighbor:Chunk AND NOT neighbor:Document AND NOT neighbor:Community
WITH node, score, collect(DISTINCT {
    entity: labels(node)[-1] + ': ' + coalesce(node.name, node.description, ''),
    rel: type(r),
    neighbor: labels(neighbor)[-1] + ': ' + coalesce(neighbor.name, neighbor.description, '')
}) AS connections
OPTIONAL MATCH (node)<-[:FROM_CHUNK]-(chunk:Chunk)
WITH node, score, connections, collect(DISTINCT chunk.text) AS source_texts
OPTIONAL MATCH (node)-[:IN_COMMUNITY]->(c:Community)
WITH node, score, connections, source_texts,
     collect(DISTINCT {id: c.id, summary: c.summary}) AS communities
RETURN node.name + ' (' + labels(node)[-1] + ')' AS title,
       [c IN connections WHERE c.rel IS NOT NULL |
           c.entity + ' -[' + c.rel + ']-> ' + c.neighbor] AS relationships,
       source_texts AS source_text,
       communities,
       score
"""

DEFAULT_ANSWER_PROMPT = """You have access to detailed findings and broader context.

Query: {query}

=== Specific Findings ===
{entity_context}

=== Broader Context ===
{community_context}

=== Related Entities ===
{bridging_context}

Synthesize all the above information to answer the query. Use specific details
from the findings, high-level insights from the context, and relevant
connections from related entities.

Answer:"""


class DriftSearchRetriever:
    """DRIFT search: entity-centric with community expansion."""

    def __init__(
        self,
        graph_store: GraphStore,
        llm: LLMInterface,
        embedder: Embedder,
        retrieval_query: Optional[str] = None,
        answer_prompt: Optional[str] = None,
        vector_index_name: str = "entity-embeddings",
        fulltext_index_name: str = "entity-names",
    ):
        self.graph_store = graph_store
        self.llm = llm
        self.embedder = embedder
        self.retrieval_query = retrieval_query or DEFAULT_DRIFT_QUERY
        self.answer_prompt = answer_prompt or DEFAULT_ANSWER_PROMPT
        self.vector_index_name = vector_index_name
        self.fulltext_index_name = fulltext_index_name
        self._retriever = self._build_retriever()

    def _build_retriever(self) -> HybridCypherRetriever:
        neo4j_database = getattr(self.graph_store, "_database", None)
        return HybridCypherRetriever(
            driver=self.graph_store.driver,
            vector_index_name=self.vector_index_name,
            fulltext_index_name=self.fulltext_index_name,
            retrieval_query=self.retrieval_query,
            embedder=self.embedder,
            neo4j_database=neo4j_database,
        )

    async def search(
        self,
        query: str,
        top_k: int = 10,
        community_top_k: int = 3,
    ) -> SearchResult:
        """Run DRIFT search.

        1. Vector search on entities → seed entities
        2. Extract community IDs from seed entities
        3. Fetch community summaries
        4. Fetch other entities in those communities (bridging)
        5. Combine all context → LLM answer
        """
        retriever_result = self._retriever.search(query_text=query, top_k=top_k)

        entity_context = self._format_entity_context(retriever_result)
        community_ids = self._extract_community_ids(retriever_result)

        community_context = ""
        bridging_context = ""
        if community_ids:
            communities = self._fetch_community_summaries(community_ids, community_top_k)
            community_context = self._format_communities(communities)

            bridging_entities = self._fetch_community_entities(community_ids)
            bridging_context = self._format_bridging_entities(bridging_entities)

        answer = await self._generate_answer(
            query, entity_context, community_context, bridging_context
        )

        full_context = f"{entity_context}\n\n{community_context}\n\n{bridging_context}"
        return SearchResult(query=query, mode="drift", answer=answer, context=full_context)

    def _format_entity_context(self, retriever_result) -> str:
        parts = []
        for item in retriever_result.items:
            content = item.content
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, dict):
                title = content.get("title", "Unknown")
                rels = content.get("relationships", [])
                sources = content.get("source_text", [])
                section = f"Finding: {title}"
                if rels:
                    section += "\n  Connections:\n    " + "\n    ".join(rels)
                if sources:
                    section += "\n  Evidence:\n    " + "\n    ".join(sources[:2])
                parts.append(section)
        return "\n\n".join(parts)

    def _extract_community_ids(self, retriever_result) -> list[str]:
        """Extract unique community IDs from retriever results."""
        ids = set()
        for item in retriever_result.items:
            content = item.content
            if isinstance(content, dict):
                for comm in content.get("communities", []):
                    if comm.get("id"):
                        ids.add(comm["id"])
        return list(ids)

    def _fetch_community_summaries(
        self, community_ids: list[str], top_k: int
    ) -> list[dict]:
        query = """
        MATCH (c:Community)
        WHERE c.id IN $ids AND c.summary IS NOT NULL
        RETURN c.id AS id, c.summary AS summary, c.level AS level
        ORDER BY c.level ASC
        LIMIT $top_k
        """
        return self.graph_store.execute_query(
            query, {"ids": community_ids, "top_k": top_k}
        )

    def _fetch_community_entities(self, community_ids: list[str]) -> list[dict]:
        """Fetch entities in specified communities (excluding already-seen seed entities)."""
        query = """
        MATCH (c:Community)<-[:IN_COMMUNITY]-(e:__Entity__)
        WHERE c.id IN $ids
        OPTIONAL MATCH (e)-[r]-(other:__Entity__)
        WHERE (other)-[:IN_COMMUNITY]->(c) AND id(e) < id(other)
        RETURN DISTINCT e.name AS name, labels(e) AS labels,
               collect(DISTINCT type(r) + ' -> ' + coalesce(other.name, other.description)) AS rels
        LIMIT 50
        """
        return self.graph_store.execute_query(query, {"ids": community_ids})

    @staticmethod
    def _format_communities(communities: list[dict]) -> str:
        if not communities:
            return "No community context available."
        parts = []
        for comm in communities:
            parts.append(f"Segment {comm['id']}:\n{comm['summary']}")
        return "\n\n".join(parts)

    @staticmethod
    def _format_bridging_entities(entities: list[dict]) -> str:
        if not entities:
            return "No bridging entities found."
        parts = []
        for ent in entities:
            label = [lbl for lbl in ent.get("labels", []) if lbl != "__Entity__"]
            label = label[0] if label else "Entity"
            name = ent.get("name", "")
            rels = ent.get("rels", [])
            line = f"Related: [{label}] {name}"
            if rels:
                line += "\n    Connected to: " + "\n    Connected to: ".join(rels[:5])
            parts.append(line)
        return "\n".join(parts)

    async def _generate_answer(
        self,
        query: str,
        entity_context: str,
        community_context: str,
        bridging_context: str,
    ) -> str:
        prompt = self.answer_prompt.format(
            query=query,
            entity_context=entity_context,
            community_context=community_context,
            bridging_context=bridging_context,
        )
        response = await self.llm.ainvoke(prompt)
        return response.content
