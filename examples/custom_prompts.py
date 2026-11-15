"""Pipeline-level prompt configuration.

``GraphBuilderPipeline`` and ``CommunityPipeline`` accept plain string
prompts. The backend wraps each string with the standard schema, rules,
JSON format, and rubric sections. Claim-enabled builds require strict JSON-schema
support from the selected LLM deployment. Optional placeholders (e.g. ``{text}``,
``{context}``) let a string control where the backend inserts content.

Run this file to print the prompts the backend actually sends to the LLM:

    python examples/custom_prompts.py
"""

from __future__ import annotations

from recon_graphrag import (
    GraphSchema,
    NodeType,
    RelationshipType,
    SchemaPromptBuilder,
)
from recon_graphrag.communities.reports import build_report_prompt


def make_schema() -> GraphSchema:
    return GraphSchema(
        node_types=[
            NodeType(label="Person", description="An individual person"),
            NodeType(label="Organization", description="A company or group"),
        ],
        relationship_types=[
            RelationshipType(
                label="WORKS_AT",
                description="Person works at an organization",
            ),
        ],
        patterns=[
            ("Person", "WORKS_AT", "Organization"),
        ],
    )


# In a real pipeline, pass the same strings directly to the constructors:
#
#     GraphBuilderPipeline(
#         graph_store=store, llm=llm, embedder=embedder,
#         extraction_prompt="You are a legal analyst. Extract parties and obligations from the text.",
#         claim_prompt="Extract claims about the known entities.",
#         entity_summary_prompt="Summarize this legal entity in one sentence.",
#     )
#     CommunityPipeline(
#         graph_store=store, llm=llm,
#         report_prompt="You are a security analyst. Identify risks in this community.",
#     )


def main() -> None:
    schema = make_schema()

    builder = SchemaPromptBuilder(
        extraction_prompt="You are a legal analyst. Extract parties and obligations from the text.",
        claim_prompt="Extract claims about the known entities.",
    )
    print("=== Extraction prompt ===")
    print(builder.build_prompt("Alice signed a contract with Acme Corp.", schema))

    print()
    print("=== Community report prompt ===")
    print(
        build_report_prompt(
            community_id="community-1",
            level=0,
            context="(entities, relationships, and claims for the community)",
            reference_ids=["person:alice", "org:acme"],
            prompt_template="You are a security analyst. Identify risks in this community.",
        )
    )


if __name__ == "__main__":
    main()
