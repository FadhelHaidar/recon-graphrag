"""Neo4j-specific Cypher query templates used by the Neo4j graph store.

When adding a new graph backend, create a parallel feature module (e.g.
`retrieval/<backend>/queries.py`) with equivalent query language templates.
Generic retrieval consumers should continue to call GraphStore methods instead
of importing backend query templates directly.
"""

from __future__ import annotations

# ------------------------------------------------------------------
# Local search
# ------------------------------------------------------------------
DEFAULT_LOCAL_RETRIEVAL_QUERY = """
CALL (node) {
    OPTIONAL MATCH (node)-[r]->(neighbor)
    WHERE NOT neighbor:Chunk AND NOT neighbor:Document AND NOT neighbor:Community
    WITH node, r, neighbor
    RETURN collect(DISTINCT {
        entity: labels(node)[-1] + ': ' + coalesce(node.name, node.description, ''),
        rel: type(r),
        neighbor: labels(neighbor)[-1] + ': ' + coalesce(neighbor.name, neighbor.description, ''),
        dir: 'out'
    }) AS out_connections
    UNION
    OPTIONAL MATCH (node)<-[r]-(neighbor)
    WHERE NOT neighbor:Chunk AND NOT neighbor:Document AND NOT neighbor:Community
    WITH node, r, neighbor
    RETURN collect(DISTINCT {
        entity: labels(node)[-1] + ': ' + coalesce(node.name, node.description, ''),
        rel: type(r),
        neighbor: labels(neighbor)[-1] + ': ' + coalesce(neighbor.name, neighbor.description, ''),
        dir: 'in'
    }) AS in_connections
}
WITH node, score, out_connections + in_connections AS connections
OPTIONAL MATCH (node)<-[:FROM_CHUNK]-(chunk:Chunk)
WITH node, score, connections,
     collect(DISTINCT chunk.text) AS source_texts,
     collect(DISTINCT chunk.id) AS source_chunk_ids
RETURN node.name + ' (' + labels(node)[-1] + ')' AS title,
       [c IN connections WHERE c.rel IS NOT NULL |
           CASE c.dir
             WHEN 'out' THEN c.entity + ' -[' + c.rel + ']-> ' + c.neighbor
             ELSE c.neighbor + ' -[' + c.rel + ']-> ' + c.entity
           END] AS relationships,
       source_texts AS source_text,
       source_chunk_ids AS source_chunk_ids,
       score
"""

# ------------------------------------------------------------------
# DRIFT search
# ------------------------------------------------------------------
DEFAULT_DRIFT_RETRIEVAL_QUERY = """
CALL (node) {
    OPTIONAL MATCH (node)-[r]->(neighbor)
    WHERE NOT neighbor:Chunk AND NOT neighbor:Document AND NOT neighbor:Community
    WITH node, r, neighbor
    RETURN collect(DISTINCT {
        entity: labels(node)[-1] + ': ' + coalesce(node.name, node.description, ''),
        rel: type(r),
        neighbor: labels(neighbor)[-1] + ': ' + coalesce(neighbor.name, neighbor.description, ''),
        dir: 'out'
    }) AS out_connections
    UNION
    OPTIONAL MATCH (node)<-[r]-(neighbor)
    WHERE NOT neighbor:Chunk AND NOT neighbor:Document AND NOT neighbor:Community
    WITH node, r, neighbor
    RETURN collect(DISTINCT {
        entity: labels(node)[-1] + ': ' + coalesce(node.name, node.description, ''),
        rel: type(r),
        neighbor: labels(neighbor)[-1] + ': ' + coalesce(neighbor.name, neighbor.description, ''),
        dir: 'in'
    }) AS in_connections
}
WITH node, score, out_connections + in_connections AS connections
OPTIONAL MATCH (node)<-[:FROM_CHUNK]-(chunk:Chunk)
WITH node, score, connections,
     collect(DISTINCT chunk.text) AS source_texts,
     collect(DISTINCT chunk.id) AS source_chunk_ids
OPTIONAL MATCH (node)-[:IN_COMMUNITY]->(c:Community)
WITH node, score, connections, source_texts, source_chunk_ids,
     collect(DISTINCT {
        id: c.id,
        level: c.level,
        graph_name: c.graph_name,
        summary: c.summary
     }) AS communities
RETURN node.name + ' (' + labels(node)[-1] + ')' AS title,
       [c IN connections WHERE c.rel IS NOT NULL |
           CASE c.dir
             WHEN 'out' THEN c.entity + ' -[' + c.rel + ']-> ' + c.neighbor
             ELSE c.neighbor + ' -[' + c.rel + ']-> ' + c.entity
           END] AS relationships,
       source_texts AS source_text,
       source_chunk_ids AS source_chunk_ids,
       communities,
       score
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
WHERE (other)-[:IN_COMMUNITY]->(c) AND elementId(e) < elementId(other)
RETURN DISTINCT e.name AS name, labels(e) AS labels,
       collect(DISTINCT type(r) + ' -> ' + coalesce(other.name, other.description)) AS rels
LIMIT 50
"""

# ------------------------------------------------------------------
# Community summarization — degree-ranked context (Phase 4A)
# ------------------------------------------------------------------
COMMUNITY_RANKED_CONTEXT_QUERY = """
MATCH (c:Community {
    graph_name: $graph_name,
    id: $cid,
    level: $level
})<-[:IN_COMMUNITY]-(e:__Entity__)
OPTIONAL MATCH (e)-[r]-(other:__Entity__)
WHERE (other)-[:IN_COMMUNITY]->(c)
  AND elementId(e) < elementId(other)
WITH e, r, other,
     SIZE([(e)-[r1]-()
       WHERE NOT type(r1) IN ['IN_COMMUNITY', 'FROM_CHUNK', 'SOURCED_FROM']
     | r1]) AS e_degree,
     SIZE([(other)-[r2]-()
       WHERE NOT type(r2) IN ['IN_COMMUNITY', 'FROM_CHUNK', 'SOURCED_FROM']
     | r2]) AS other_degree
RETURN coalesce(e.human_readable_id, e.canonical_key, e.id) AS e_id,
       coalesce(e.name, e.id) AS e_name,
       coalesce(e.description, '') AS e_description,
       labels(e) AS e_labels,
       e_degree,
       type(r) AS rel_type,
       coalesce(r.description, '') AS rel_description,
       coalesce(r.observation_count, 1) AS observation_count,
       e_degree + other_degree AS combined_degree,
       coalesce(other.human_readable_id, other.canonical_key, other.id) AS other_id,
       coalesce(other.name, other.id) AS other_name,
       coalesce(other.description, '') AS other_description,
       labels(other) AS other_labels,
       other_degree
ORDER BY combined_degree DESC, observation_count DESC, type(r) ASC
"""
