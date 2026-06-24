"""JSON parser for LLM graph extraction output."""

import json
import re

from recon_graphrag.extraction.types import (
    ExtractedClaim,
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


class AssessmentParser:
    """Parse a binary yes/no assessment response from the LLM."""

    def parse(self, content: str) -> bool:
        """Return True if the LLM indicates entities were missed."""
        normalized = content.strip().lower()
        # Check the first meaningful token
        tokens = normalized.split()
        if tokens and tokens[0] == "yes":
            return True
        if "yes" in normalized[:30]:
            return True
        return False


class ClaimParser:
    """Parse claim extraction JSON from the LLM.

    Validates that each claim has required fields and that subject entity IDs
    reference nodes present in the provided entity set.
    """

    def __init__(self) -> None:
        self._json_parser = GraphExtractionParser()

    def parse(
        self,
        content: str,
        valid_entity_ids: set[str] | None = None,
    ) -> list[ExtractedClaim]:
        """Parse claims from LLM output.

        Args:
            content: Raw LLM response content.
            valid_entity_ids: Optional set of entity IDs extracted from the same
                text. Claims referencing unknown entities are skipped.

        Returns:
            List of validated ExtractedClaim instances.
        """
        payload = self._json_parser._extract_json(content)
        data = json.loads(payload)

        raw_claims = data if isinstance(data, list) else data.get("claims", [])

        claims: list[ExtractedClaim] = []
        for item in raw_claims:
            if not isinstance(item, dict):
                continue

            subject = str(item.get("subject_entity_id", "")).strip()
            claim_type = str(item.get("claim_type", "")).strip()
            description = str(item.get("description", "")).strip()

            if not subject or not description:
                continue

            # Skip claims referencing unknown entities
            if valid_entity_ids is not None and subject not in valid_entity_ids:
                continue

            claims.append(
                ExtractedClaim(
                    subject_entity_id=subject,
                    claim_type=claim_type or "general",
                    description=description,
                    status=str(item.get("status", "active")).strip() or "active",
                    start_date=item.get("start_date"),
                    end_date=item.get("end_date"),
                )
            )

        return claims
