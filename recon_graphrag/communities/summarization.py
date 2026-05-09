"""LLM-based community summarization.

For each community, collect its entities and relationships, format as structured
text, and generate a summary via LLM. This enables global-level retrieval over
high-level community insights instead of individual nodes.
"""

from __future__ import annotations

from typing import Optional

from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.llm.types import LLMResponse

from recon_graphrag.graph.base import GraphStore


DEFAULT_SUMMARY_PROMPT = """Summarize the following cluster of related entities and their connections.

Entities and relationships:
{context}

Generate a concise but comprehensive summary (2-4 paragraphs) that:
1. Identifies the main theme or area covered
2. Describes the key entities involved
3. Highlights important patterns and connections
4. Notes any notable insights or implications

Write in plain, clear language. Do not mention communities, graphs, nodes, or edges.

Summary:"""


class CommunitySummarizer:
    """Generate LLM summaries for each community in the knowledge graph."""

    def __init__(
        self,
        graph_store: GraphStore,
        llm: LLMInterface,
        prompt_template: Optional[str] = None,
    ):
        self.graph_store = graph_store
        self.llm = llm
        self.prompt_template = prompt_template or DEFAULT_SUMMARY_PROMPT

    async def summarize_all(self, level: int = 0) -> list[dict]:
        """Summarize all communities at a given hierarchy level."""
        communities = self._get_communities(level)
        if not communities:
            print(f"No communities found at level {level}")
            return []

        results = []
        for comm in communities:
            cid = comm["id"]
            print(f"  Summarizing community {cid} ({comm['entity_count']} entities)...")
            try:
                summary = await self.summarize_community(cid, level)
                self._store_summary(cid, summary)
                results.append({"id": cid, "summary": summary})
            except Exception as e:
                print(f"  Error summarizing community {cid}: {e}")
        return results

    async def summarize_community(self, community_id: str, level: int = 0) -> str:
        """Summarize a single community by collecting its entity/relationship context."""
        context = self._fetch_community_context(community_id, level)
        if not context.strip():
            return ""

        prompt = self.prompt_template.format(context=context)
        response: LLMResponse = await self.llm.ainvoke(prompt)
        return response.content

    def _get_communities(self, level: int) -> list[dict]:
        query = """
        MATCH (c:Community {level: $level})
        OPTIONAL MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
        WITH c, count(e) AS entity_count
        RETURN c.id AS id, entity_count
        ORDER BY entity_count DESC
        """
        return self.graph_store.execute_query(query, {"level": level})

    def _fetch_community_context(self, community_id: str, level: int = 0) -> str:
        """Fetch context for a community.

        Level 0: entities and intra-community relationships.
        Level > 0: child community summaries (bottom-up aggregation).
        """
        if level == 0:
            return self._fetch_entity_context(community_id)
        return self._fetch_child_summary_context(community_id)

    def _fetch_entity_context(self, community_id: str) -> str:
        """Fetch all entities and intra-community relationships as text."""
        query = """
        MATCH (c:Community {id: $cid})<-[:IN_COMMUNITY]-(e:__Entity__)
        OPTIONAL MATCH (e)-[r]-(other:__Entity__)
        WHERE (other)-[:IN_COMMUNITY]->(c)
          AND id(e) < id(other)
        RETURN e, type(r) AS rel_type, other
        """
        lines = []
        seen_entities = set()
        results = self.graph_store.execute_query(query, {"cid": community_id})
        for record in results:
            entity = record["e"]
            label = list(entity.labels - {"__Entity__"})[0] if entity.labels - {"__Entity__"} else "Entity"
            name = entity.get("name", "") or entity.get("description", "")
            key = f"{label}:{name}"
            if key not in seen_entities:
                lines.append(f"- [{label}] {name}")
                seen_entities.add(key)

            other = record["other"]
            if other and record["rel_type"]:
                other_name = other.get("name", "") or other.get("description", "")
                lines.append(f"  {name} --[{record['rel_type']}]--> {other_name}")

        return "\n".join(lines)

    def _fetch_child_summary_context(self, community_id: str) -> str:
        """Fetch child community summaries for higher-level communities."""
        query = """
        MATCH (child:Community)-[:PARENT_COMMUNITY]->(c:Community {id: $cid})
        WHERE child.summary IS NOT NULL
        RETURN child.id AS id, child.summary AS summary, child.level AS level
        ORDER BY child.level, child.id
        """
        results = self.graph_store.execute_query(query, {"cid": community_id})
        if not results:
            return ""

        lines = []
        for record in results:
            lines.append(f"--- Sub-community {record['id']} (level {record['level']}) ---")
            lines.append(record["summary"])
            lines.append("")
        return "\n".join(lines)

    def _store_summary(self, community_id: str, summary: str):
        query = """
        MATCH (c:Community {id: $cid})
        SET c.summary = $summary
        """
        self.graph_store.execute_query(query, {"cid": community_id, "summary": summary})
