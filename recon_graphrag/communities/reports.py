"""Structured, evidence-referenced community report generation and validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from recon_graphrag.models.artifacts import (
    CommunityFinding,
    CommunityReport,
    FindingReference,
)


# ---------------------------------------------------------------------------
# Report prompt builder
# ---------------------------------------------------------------------------

DEFAULT_REPORT_PROMPT = """\
You are generating a structured community report for a knowledge graph.

## Community context

The following entities, relationships, and claims belong to community
{community_id} (level {level}):

{context}

## Reference ID allowlist

Every finding MUST reference at least one ID from this list:
{reference_ids}

## Instructions

1. Write a title (one line) summarizing the community's main theme.
2. Write a summary (2-4 paragraphs) describing the community.
3. List key findings. Each finding must:
   - Have a short description.
   - Reference at least one entity, relationship, or claim ID from the allowlist.
   - Use "entity" for entity IDs, "relationship" for relationship keys
     (source:type:target), or "claim" for claim IDs.
4. Rate the community's importance from {min_rating} to {max_rating}
   ({rating_name}: {rating_description}).
5. Return valid JSON only. Do not include markdown fences.

JSON format:
{{
  "title": "Community theme title",
  "summary": "Multi-paragraph summary...",
  "rating": 7.5,
  "rating_explanation": "One sentence explaining the rating.",
  "findings": [
    {{
      "description": "Finding description.",
      "references": [
        {{"target_id": "person:alice", "target_type": "entity"}},
        {{"target_id": "person:alice:WORKS_AT:org:acme", "target_type": "relationship"}}
      ]
    }}
  ]
}}

If there are no valid findings, return an empty findings array.
Do NOT reference IDs not present in the allowlist.
"""


@dataclass
class ReportRubric:
    """Rating rubric for community reports."""

    rating_name: str = "impact"
    rating_description: str = "How impactful is this community to the overall knowledge graph"
    min_rating: float = 0.0
    max_rating: float = 10.0


def build_report_prompt(
    community_id: str,
    level: int,
    context: str,
    reference_ids: list[str],
    rubric: ReportRubric | None = None,
) -> str:
    """Build a structured report prompt for a community.

    Args:
        community_id: Community identifier.
        level: Community hierarchy level.
        context: Rendered community context text.
        reference_ids: Valid reference IDs that findings may cite.
        rubric: Rating rubric. Uses default if None.

    Returns:
        Prompt string requesting structured JSON report.
    """
    rubric = rubric or ReportRubric()
    ref_list = "\n".join(f"  - {rid}" for rid in reference_ids)
    return DEFAULT_REPORT_PROMPT.format(
        community_id=community_id,
        level=level,
        context=context,
        reference_ids=ref_list,
        min_rating=rubric.min_rating,
        max_rating=rubric.max_rating,
        rating_name=rubric.rating_name,
        rating_description=rubric.rating_description,
    )


# ---------------------------------------------------------------------------
# Reference ID extraction from context
# ---------------------------------------------------------------------------


def extract_reference_ids(
    entity_ids: list[str],
    relationship_keys: list[str],
    claim_ids: list[str],
) -> list[str]:
    """Build the reference ID allowlist from context components.

    Args:
        entity_ids: Entity IDs in the community.
        relationship_keys: Relationship keys (source:type:target format).
        claim_ids: Claim IDs linked to community entities.

    Returns:
        Flat list of reference IDs for the prompt allowlist.
    """
    ids: list[str] = []
    ids.extend(entity_ids)
    ids.extend(relationship_keys)
    ids.extend(claim_ids)
    return ids


# ---------------------------------------------------------------------------
# Report parser
# ---------------------------------------------------------------------------


class ReportParser:
    """Parse structured report JSON from LLM output."""

    def parse(
        self,
        content: str,
        community_id: str,
        level: int,
        valid_ids: set[str] | None = None,
    ) -> CommunityReport:
        """Parse LLM output into a CommunityReport.

        Args:
            content: Raw LLM response.
            community_id: Community identifier.
            level: Community hierarchy level.
            valid_ids: Optional set of valid reference IDs. If provided,
                findings with invalid references are rejected.

        Returns:
            Parsed CommunityReport.

        Raises:
            ReportValidationError: If the report fails validation.
        """
        data = self._extract_json(content)
        errors = self._validate(data, valid_ids)
        if errors:
            raise ReportValidationError(errors=errors, raw_content=content)

        findings = self._parse_findings(data.get("findings", []), valid_ids)

        return CommunityReport(
            id=f"report:{community_id}:{level}",
            community_id=community_id,
            level=level,
            title=str(data.get("title", "")),
            summary=str(data.get("summary", "")),
            rating=data.get("rating"),
            rating_explanation=data.get("rating_explanation"),
            findings=findings,
        )

    def _extract_json(self, content: str) -> dict:
        """Extract JSON from LLM response, stripping fences."""
        content = content.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
        if fenced:
            content = fenced.group(1).strip()
        return json.loads(content)

    def _validate(
        self,
        data: dict,
        valid_ids: set[str] | None,
    ) -> list[str]:
        """Validate top-level report structure."""
        errors: list[str] = []

        if not isinstance(data, dict):
            return ["Response is not a JSON object."]

        if not data.get("title"):
            errors.append("Missing or empty 'title'.")

        if not data.get("summary"):
            errors.append("Missing or empty 'summary'.")

        findings = data.get("findings", [])
        if not isinstance(findings, list):
            errors.append("'findings' must be an array.")
        elif len(findings) == 0:
            errors.append("'findings' array is empty.")

        rating = data.get("rating")
        if rating is not None:
            if not isinstance(rating, (int, float)):
                errors.append("'rating' must be a number or null.")

        return errors

    def _parse_findings(
        self,
        raw_findings: list[dict],
        valid_ids: set[str] | None,
    ) -> list[CommunityFinding]:
        """Parse and validate individual findings."""
        findings: list[CommunityFinding] = []
        for i, raw in enumerate(raw_findings):
            if not isinstance(raw, dict):
                continue

            description = str(raw.get("description", "")).strip()
            if not description:
                continue

            references = self._parse_references(
                raw.get("references", []), valid_ids
            )

            # Skip findings with no valid references when validation is active
            if valid_ids is not None and not references:
                continue

            findings.append(
                CommunityFinding(
                    id=f"finding:{i}",
                    description=description,
                    references=references,
                    rank=float(raw.get("rank", 0.0)),
                )
            )
        return findings

    def _parse_references(
        self,
        raw_refs: list[dict],
        valid_ids: set[str] | None,
    ) -> list[FindingReference]:
        """Parse and validate finding references."""
        refs: list[FindingReference] = []
        seen: set[str] = set()
        for raw in raw_refs:
            if not isinstance(raw, dict):
                continue
            target_id = str(raw.get("target_id", "")).strip()
            target_type = str(raw.get("target_type", "")).strip()
            if not target_id or target_type not in (
                "entity",
                "relationship",
                "claim",
            ):
                continue
            if valid_ids is not None and target_id not in valid_ids:
                continue
            key = f"{target_type}:{target_id}"
            if key not in seen:
                seen.add(key)
                refs.append(
                    FindingReference(target_id=target_id, target_type=target_type)
                )
        return refs


# ---------------------------------------------------------------------------
# Validation error and repair
# ---------------------------------------------------------------------------


class ReportValidationError(Exception):
    """Raised when a report fails validation."""

    def __init__(self, errors: list[str], raw_content: str = ""):
        self.errors = errors
        self.raw_content = raw_content
        super().__init__(f"Report validation failed: {'; '.join(errors)}")


def build_repair_prompt(
    raw_content: str,
    errors: list[str],
    valid_ids: list[str],
    rubric: ReportRubric | None = None,
) -> str:
    """Build a repair prompt for a failed report.

    Args:
        raw_content: The invalid LLM response.
        errors: Validation error messages.
        valid_ids: Valid reference ID allowlist.
        rubric: Rating rubric for schema reference.

    Returns:
        Prompt asking the LLM to fix the report.
    """
    rubric = rubric or ReportRubric()
    error_list = "\n".join(f"  - {e}" for e in errors)
    ref_list = "\n".join(f"  - {rid}" for rid in valid_ids)
    return f"""\
Your previous report had validation errors. Fix them and return valid JSON.

## Errors
{error_list}

## Valid reference IDs (use ONLY these)
{ref_list}

## Required JSON format
{{
  "title": "...",
  "summary": "...",
  "rating": {rubric.min_rating}-{rubric.max_rating} or null,
  "rating_explanation": "..." or null,
  "findings": [
    {{
      "description": "...",
      "references": [
        {{"target_id": "...", "target_type": "entity|relationship|claim"}}
      ]
    }}
  ]
}}

## Your previous response
{raw_content}

Return ONLY valid JSON. Do not include markdown fences.
"""
