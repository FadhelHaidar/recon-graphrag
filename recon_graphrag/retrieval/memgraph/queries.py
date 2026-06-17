"""Memgraph-specific Cypher query templates used by the Memgraph graph store."""

from __future__ import annotations

# ------------------------------------------------------------------
# Local search
# ------------------------------------------------------------------
DEFAULT_LOCAL_RETRIEVAL_QUERY = """
OPTIONAL MATCH (node)-[r]-(neighbor)
WHERE NOT neighbor:Chunk AND NOT neighbor:Document AND NOT neighbor:Community
WITH node, score, collect(DISTINCT {
    entity: labels(node)[-1] + ': ' + coalesce(node.name, node.description, ''),
    rel: type(r),
    neighbor: labels(neighbor)[-1] + ': ' + coalesce(neighbor.name, neighbor.description, '')
}) AS connections
OPTIONAL MATCH (node)<-[:FROM_CHUNK]-(chunk:Chunk)
WITH node, score, connections, collect(DISTINCT chunk.text) AS source_texts
RETURN node.name + ' (' + labels(node)[-1] + ')' AS title,
       [c IN connections WHERE c.rel IS NOT NULL |
           c.entity + ' -[' + c.rel + ']-> ' + c.neighbor] AS relationships,
       source_texts AS source_text,
       score
ORDER BY score DESC, title
"""

# ------------------------------------------------------------------
# DRIFT search
# ------------------------------------------------------------------
DEFAULT_DRIFT_RETRIEVAL_QUERY = """
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
     collect(DISTINCT {
        id: c.id,
        level: c.level,
        graph_name: c.graph_name,
        summary: c.summary
     }) AS communities
RETURN node.name + ' (' + labels(node)[-1] + ')' AS title,
       [c IN connections WHERE c.rel IS NOT NULL |
           c.entity + ' -[' + c.rel + ']-> ' + c.neighbor] AS relationships,
       source_texts AS source_text,
       communities,
       score
ORDER BY score DESC, title
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
ORDER BY coalesce(e.name, e.description), rel_type, coalesce(other.name, other.description)
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
       collect(DISTINCT type(r) + ' -> ' + coalesce(other.name, other.description)) AS rels
ORDER BY name
LIMIT 50
"""
