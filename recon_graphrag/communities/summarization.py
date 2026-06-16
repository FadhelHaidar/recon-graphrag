"""LLM-based community summarization.

For each community, collect its entities and relationships, format as structured
text, and generate a summary via LLM. This enables global-level retrieval over
high-level community insights instead of individual nodes.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.llm.base import BaseLLM, LLMResponse


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
        llm: BaseLLM,
        prompt_template: Optional[str] = None,
        graph_name: str = "entity-graph",
    ):
        self.graph_store = graph_store
        self.llm = llm
        self.prompt_template = prompt_template or DEFAULT_SUMMARY_PROMPT
        self.graph_name = graph_name

    async def summarize_all(self, level: int = 0) -> list[dict]:
        """Summarize all communities at a given hierarchy level."""
        communities = self.graph_store.get_communities(self.graph_name, level=level)
        if not communities:
            print(f"  No communities found at level {level}")
            return []

        results = []
        for comm in communities:
            cid = comm["id"]
            print(f"  Summarizing community {cid} ({comm.get('entity_count', 0)} entities)...")
            try:
                summary = await self.summarize_community(cid, level)
                if not summary.strip():
                    continue
                self.graph_store.store_community_summary(cid, level, summary, self.graph_name)
                results.append({"id": cid, "level": level, "summary": summary})
            except Exception as e:
                print(f"  Error summarizing community {cid}: {e}")
        return results

    async def summarize_community(self, community_id: str, level: int = 0) -> str:
        """Summarize a single community by collecting its context."""
        context = self._fetch_community_context(community_id, level)
        if not context.strip():
            return ""

        prompt = self.prompt_template.format(context=context)
        response: LLMResponse = await self.llm.ainvoke(prompt)
        return response.content

    def _fetch_community_context(self, community_id: str, level: int = 0) -> str:
        """Fetch context for a community.

        Level 0: entities and intra-community relationships.
        Level > 0: child community summaries first, then entity context fallback.
        """
        if level == 0:
            return self._fetch_entity_context(community_id, level)

        child_context = self._fetch_child_summary_context(community_id, level)
        if child_context.strip():
            return child_context

        return self._fetch_entity_context(community_id, level)

    def _fetch_entity_context(self, community_id: str, level: int = 0) -> str:
        """Fetch all entities and intra-community relationships as text."""
        results = self.graph_store.get_community_entity_context(
            graph_name=self.graph_name,
            community_id=community_id,
            level=level,
        )
        lines = []
        seen_entities = set()
        for record in results:
            entity = record["e"]
            non_entity_labels = self._domain_labels(entity)
            label = list(non_entity_labels)[0] if non_entity_labels else "Entity"
            name = self._node_property(entity, "name") or self._node_property(
                entity, "description"
            )
            key = f"{label}:{name}"
            if key not in seen_entities:
                lines.append(f"- [{label}] {name}")
                seen_entities.add(key)

            other = record["other"]
            if other and record["rel_type"]:
                other_name = self._node_property(other, "name") or self._node_property(
                    other, "description"
                )
                lines.append(f"  {name} --[{record['rel_type']}]--> {other_name}")

        return "\n".join(lines)

    @staticmethod
    def _domain_labels(entity) -> list[str]:
        """Return labels excluding the internal entity marker."""
        labels = getattr(entity, "labels", [])
        if isinstance(labels, str):
            labels = [labels]
        elif not isinstance(labels, Iterable):
            labels = []

        return [label for label in labels if label != "__Entity__"]

    @staticmethod
    def _node_property(entity, key: str, default: str = ""):
        """Read a node property from dict-like or backend node objects."""
        if entity is None:
            return default

        if hasattr(entity, "get"):
            return entity.get(key, default)

        properties = getattr(entity, "properties", None)
        if isinstance(properties, dict):
            return properties.get(key, default)

        try:
            return entity[key]
        except (KeyError, TypeError, AttributeError):
            return default

    def _fetch_child_summary_context(self, community_id: str, level: int) -> str:
        """Fetch child community summaries for higher-level communities."""
        results = self.graph_store.get_community_child_summary_context(
            graph_name=self.graph_name,
            community_id=community_id,
            level=level,
            child_level=level - 1,
        )
        if not results:
            return ""

        lines = []
        for record in results:
            lines.append(f"--- Sub-community {record['id']} (level {record['level']}) ---")
            lines.append(record["summary"])
            lines.append("")
        return "\n".join(lines)
