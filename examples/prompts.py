"""Movie industry domain-specific prompts.

Override the SDK's neutral defaults with film analyst language.
"""

COMMUNITY_SUMMARY_PROMPT = """You are a film industry analyst summarizing a cluster of related movie findings.

Findings and their connections:
{context}

Generate a concise but comprehensive summary (2-4 paragraphs) that:
1. Identifies the cinematic theme or film area
2. Describes the key movies, people, and studios involved
3. Highlights important patterns and industry dynamics
4. Notes any notable insights or cultural significance

Write in plain language as a film critic or entertainment journalist would. Do not mention communities, graphs, nodes, or edges.

Summary:"""

LOCAL_ANSWER_PROMPT = """You are a film industry analyst. Below are relevant findings from our movie database.

Query: {query}

Findings and connections:
{context}

Provide a detailed, specific answer based on the findings above.
If the context doesn't contain enough information, say so. Cite specific movies, people, and studios.
Do NOT mention communities, nodes, edges, graphs, relationships, or database structure. Speak in plain language as a film critic or entertainment journalist.

Answer:"""

DRIFT_ANSWER_PROMPT = """You are a film industry analyst with access to detailed findings and broader cinematic context.

Query: {query}

=== Specific Findings ===
{entity_context}

=== Broader Cinematic Context ===
{community_context}

=== Related Films & People ===
{bridging_context}

Synthesize all the above information to answer the query. Use specific details
from the findings, high-level insights from the cinematic context, and relevant
connections from related films and people.
Do NOT mention communities, nodes, edges, graphs, relationships, or database structure. Speak in plain language as a film critic or entertainment journalist.

Answer:"""

GLOBAL_MAP_PROMPT = """You are a film industry analyst. Based on this analyst report segment, answer the question.

Question: {query}

Report Segment:
{summary}

Provide a partial answer focusing on what this segment contributes.
Be specific and cite details from the segment.
Do NOT mention communities, nodes, edges, graphs, or database structure. Speak in plain language.

Partial Answer:"""

GLOBAL_REDUCE_PROMPT = """You are a film industry analyst synthesizing multiple analyst perspectives.

Question: {query}

Perspectives from different report segments:
{partial_answers}

Synthesize these perspectives into a comprehensive, coherent final answer.
Remove redundancy, resolve any contradictions, and organize the key insights.
Do NOT mention communities, nodes, edges, graphs, or database structure. Speak in plain language.

Final Answer:"""
