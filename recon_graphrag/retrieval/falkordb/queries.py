"""FalkorDB-specific OpenCypher query templates.

FalkorDB/OpenCypher does not support Neo4j list-comprehension/map-literal
syntax, so these queries return flat rows that
FalkorDBGraphStore.fetch_entity_context aggregates into the structure
expected by retrievers.
"""

from __future__ import annotations

# ------------------------------------------------------------------
# Local search
# ------------------------------------------------------------------
DEFAULT_LOCAL_RETRIEVAL_QUERY = """
OPTIONAL MATCH (node)-[r]-(neighbor)
WHERE NOT neighbor:Chunk AND NOT neighbor:Document AND NOT neighbor:Community
WITH node, score, neighbor, type(r) AS rel_type
OPTIONAL MATCH (node)<-[:FROM_CHUNK]-(chunk:Chunk)
WITH node, score, neighbor, rel_type, collect(DISTINCT chunk.text) AS source_texts
RETURN node.name + ' (' + head(labels(node)) + ')' AS title,
       head(labels(node)) + ': ' + coalesce(node.name, node.description, '')
         + ' -[' + rel_type + ']-> '
         + head(labels(neighbor)) + ': ' + coalesce(neighbor.name, neighbor.description, '') AS relationship,
       source_texts AS source_text,
       score
"""

# ------------------------------------------------------------------
# DRIFT search
# ------------------------------------------------------------------
DEFAULT_DRIFT_RETRIEVAL_QUERY = """
OPTIONAL MATCH (node)-[r]-(neighbor)
WHERE NOT neighbor:Chunk AND NOT neighbor:Document AND NOT neighbor:Community
WITH node, score, neighbor, type(r) AS rel_type
OPTIONAL MATCH (node)<-[:FROM_CHUNK]-(chunk:Chunk)
WITH node, score, neighbor, rel_type, collect(DISTINCT chunk.text) AS source_texts
OPTIONAL MATCH (node)-[:IN_COMMUNITY]->(c:Community)
WITH node, score, neighbor, rel_type, source_texts,
     c.id AS community_id,
     c.level AS community_level,
     c.graph_name AS community_graph_name,
     c.summary AS community_summary
RETURN node.name + ' (' + head(labels(node)) + ')' AS title,
       head(labels(node)) + ': ' + coalesce(node.name, node.description, '')
         + ' -[' + rel_type + ']-> '
         + head(labels(neighbor)) + ': ' + coalesce(neighbor.name, neighbor.description, '') AS relationship,
       source_texts AS source_text,
       community_id,
       community_level,
       community_graph_name,
       community_summary,
       score
"""

# ------------------------------------------------------------------
# Community summarization — entity context (level 0)
# ------------------------------------------------------------------
COMMUNITY_ENTITY_CONTEXT_QUERY = """
MATCH (c:Community {
    graph_name: $graph_name,
    id: $cid,
    level: $level
})<-[:IN_COMMUNITY]-(e:__Entity__)
OPTIONAL MATCH (e)-[r]-(other:__Entity__)
WHERE (other)-[:IN_COMMUNITY]->(c)
  AND id(e) < id(other)
RETURN e, type(r) AS rel_type, other
"""

# ------------------------------------------------------------------
# Community summarization — child summary context (level > 0)
# ------------------------------------------------------------------
COMMUNITY_CHILD_SUMMARY_QUERY = """
MATCH (child:Community)-[:PARENT_COMMUNITY]->(c:Community {
    graph_name: $graph_name,
    id: $cid,
    level: $level
})
WHERE child.graph_name = $graph_name
  AND child.level = $child_level
  AND child.summary IS NOT NULL
RETURN child.id AS id, child.summary AS summary, child.level AS level
ORDER BY child.level, child.id
"""

# ------------------------------------------------------------------
# DRIFT — community summaries by key
# ------------------------------------------------------------------
DRIFT_COMMUNITY_SUMMARIES_QUERY = """
UNWIND $keys AS key
MATCH (c:Community {
    graph_name: $graph_name,
    id: key.id,
    level: key.level
})
WHERE c.summary IS NOT NULL
RETURN c.id AS id, c.summary AS summary, c.level AS level
ORDER BY c.level ASC
LIMIT $top_k
"""

# ------------------------------------------------------------------
# DRIFT — bridging entities in communities
# ------------------------------------------------------------------
DRIFT_COMMUNITY_ENTITIES_QUERY = """
UNWIND $keys AS key
MATCH (c:Community {
    graph_name: $graph_name,
    id: key.id,
    level: key.level
})<-[:IN_COMMUNITY]-(e:__Entity__)
OPTIONAL MATCH (e)-[r]-(other:__Entity__)
WHERE (other)-[:IN_COMMUNITY]->(c) AND id(e) < id(other)
RETURN DISTINCT e.name AS name, labels(e) AS labels,
       type(r) + ' -> ' + coalesce(other.name, other.description) AS rel
LIMIT 50
"""
