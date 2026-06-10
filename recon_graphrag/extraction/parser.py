"""JSON parser for LLM graph extraction output."""

import json
import re

from recon_graphrag.extraction.types import (
    ExtractedNode,
    ExtractedRelationship,
    GraphExtraction,
)


class GraphExtractionParser:
    def parse(self, content: str) -> GraphExtraction:
        payload = self._extract_json(content)
        data = json.loads(payload)

        nodes = [
            ExtractedNode(
                id=str(node["id"]),
                label=str(node["label"]),
                properties=dict(node.get("properties") or {}),
            )
            for node in (data.get("nodes") or [])
            if node.get("id") and node.get("label")
        ]

        relationships = [
            ExtractedRelationship(
                source_id=str(rel["source_id"]),
                target_id=str(rel["target_id"]),
                type=str(rel["type"]),
                properties=dict(rel.get("properties") or {}),
            )
            for rel in (data.get("relationships") or [])
            if rel.get("source_id") and rel.get("target_id") and rel.get("type")
        ]

        return GraphExtraction(nodes=nodes, relationships=relationships)

    def _extract_json(self, content: str) -> str:
        content = content.strip()

        fenced = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
        if fenced:
            return fenced.group(1).strip()

        return content
