"""Memgraph-specific Cypher query templates used by the Memgraph graph store."""

from __future__ import annotations

# ------------------------------------------------------------------
# Local search
# ------------------------------------------------------------------
DEFAULT_LOCAL_RETRIEVAL_QUERY = """
CALL (node) {
    OPTIONAL MATCH (node)-[r]->(neighbor)
    WHERE NOT neighbor:Chunk AND NOT neighbor:Document AND NOT neighbor:Community
      AND neighbor.graph_name = node.graph_name
    WITH node, r, neighbor
    RETURN collect(DISTINCT {
        source_id: coalesce(node.human_readable_id, node.canonical_key, node.id),
        source_name: coalesce(node.name, node.description, node.id),
        target_id: coalesce(neighbor.human_readable_id, neighbor.canonical_key, neighbor.id),
        target_name: coalesce(neighbor.name, neighbor.description, neighbor.id),
        entity: labels(node)[-1] + ': ' + coalesce(node.name, node.description, ''),
        rel: type(r),
        neighbor: labels(neighbor)[-1] + ': ' + coalesce(neighbor.name, neighbor.description, ''),
        description: coalesce(r.description, ''),
        weight: coalesce(r.weight, r.observation_count, r.strength, 1.0),
        dir: 'out'
    }) AS connections
    UNION
    OPTIONAL MATCH (node)<-[r]-(neighbor)
    WHERE NOT neighbor:Chunk AND NOT neighbor:Document AND NOT neighbor:Community
      AND neighbor.graph_name = node.graph_name
    WITH node, r, neighbor
    RETURN collect(DISTINCT {
        source_id: coalesce(neighbor.human_readable_id, neighbor.canonical_key, neighbor.id),
        source_name: coalesce(neighbor.name, neighbor.description, neighbor.id),
        target_id: coalesce(node.human_readable_id, node.canonical_key, node.id),
        target_name: coalesce(node.name, node.description, node.id),
        entity: labels(node)[-1] + ': ' + coalesce(node.name, node.description, ''),
        rel: type(r),
        neighbor: labels(neighbor)[-1] + ': ' + coalesce(neighbor.name, neighbor.description, ''),
        description: coalesce(r.description, ''),
        weight: coalesce(r.weight, r.observation_count, r.strength, 1.0),
        dir: 'in'
    }) AS connections
}
WITH node, score, connections
OPTIONAL MATCH (node)<-[:FROM_CHUNK]-(chunk:Chunk)
WHERE chunk.graph_name = node.graph_name
WITH node, score, connections,
     collect(DISTINCT chunk.text) AS source_texts,
     collect(DISTINCT chunk.id) AS source_chunk_ids
RETURN node.name + ' (' + labels(node)[-1] + ')' AS title,
       coalesce(node.human_readable_id, node.canonical_key, node.id) AS entity_id,
       coalesce(node.name, node.id) AS entity_name,
       labels(node) AS entity_labels,
       coalesce(node.description, '') AS entity_description,
       [c IN connections WHERE c.rel IS NOT NULL | c] AS relationship_records,
       [c IN connections WHERE c.rel IS NOT NULL |
           CASE c.dir
             WHEN 'out' THEN c.entity + ' -[' + c.rel + ']-> ' + c.neighbor
             ELSE c.neighbor + ' -[' + c.rel + ']-> ' + c.entity
           END] AS relationships,
       source_texts AS source_text,
       source_chunk_ids AS source_chunk_ids,
       score
ORDER BY score DESC, title
"""

# ------------------------------------------------------------------
# DRIFT search
# ------------------------------------------------------------------
DEFAULT_DRIFT_RETRIEVAL_QUERY = """
CALL (node) {
    OPTIONAL MATCH (node)-[r]->(neighbor)
    WHERE NOT neighbor:Chunk AND NOT neighbor:Document AND NOT neighbor:Community
      AND neighbor.graph_name = node.graph_name
    WITH node, r, neighbor
    RETURN collect(DISTINCT {
        source_id: coalesce(node.human_readable_id, node.canonical_key, node.id),
        source_name: coalesce(node.name, node.description, node.id),
        target_id: coalesce(neighbor.human_readable_id, neighbor.canonical_key, neighbor.id),
        target_name: coalesce(neighbor.name, neighbor.description, neighbor.id),
        entity: labels(node)[-1] + ': ' + coalesce(node.name, node.description, ''),
        rel: type(r),
        neighbor: labels(neighbor)[-1] + ': ' + coalesce(neighbor.name, neighbor.description, ''),
        description: coalesce(r.description, ''),
        weight: coalesce(r.weight, r.observation_count, r.strength, 1.0),
        dir: 'out'
    }) AS connections
    UNION
    OPTIONAL MATCH (node)<-[r]-(neighbor)
    WHERE NOT neighbor:Chunk AND NOT neighbor:Document AND NOT neighbor:Community
      AND neighbor.graph_name = node.graph_name
    WITH node, r, neighbor
    RETURN collect(DISTINCT {
        source_id: coalesce(neighbor.human_readable_id, neighbor.canonical_key, neighbor.id),
        source_name: coalesce(neighbor.name, neighbor.description, neighbor.id),
        target_id: coalesce(node.human_readable_id, node.canonical_key, node.id),
        target_name: coalesce(node.name, node.description, node.id),
        entity: labels(node)[-1] + ': ' + coalesce(node.name, node.description, ''),
        rel: type(r),
        neighbor: labels(neighbor)[-1] + ': ' + coalesce(neighbor.name, neighbor.description, ''),
        description: coalesce(r.description, ''),
        weight: coalesce(r.weight, r.observation_count, r.strength, 1.0),
        dir: 'in'
    }) AS connections
}
WITH node, score, connections
OPTIONAL MATCH (node)<-[:FROM_CHUNK]-(chunk:Chunk)
WHERE chunk.graph_name = node.graph_name
WITH node, score, connections,
     collect(DISTINCT chunk.text) AS source_texts,
     collect(DISTINCT chunk.id) AS source_chunk_ids
OPTIONAL MATCH (node)-[:IN_COMMUNITY]->(c:Community)
WHERE c.graph_name = node.graph_name
WITH node, score, connections, source_texts, source_chunk_ids,
     collect(DISTINCT {
        id: c.id,
        level: c.level,
        graph_name: c.graph_name,
        report_text: c.report_text
     }) AS communities
RETURN node.name + ' (' + labels(node)[-1] + ')' AS title,
       coalesce(node.human_readable_id, node.canonical_key, node.id) AS entity_id,
       coalesce(node.name, node.id) AS entity_name,
       labels(node) AS entity_labels,
       coalesce(node.description, '') AS entity_description,
       [c IN connections WHERE c.rel IS NOT NULL | c] AS relationship_records,
       [c IN connections WHERE c.rel IS NOT NULL |
           CASE c.dir
             WHEN 'out' THEN c.entity + ' -[' + c.rel + ']-> ' + c.neighbor
             ELSE c.neighbor + ' -[' + c.rel + ']-> ' + c.entity
           END] AS relationships,
       source_texts AS source_text,
       source_chunk_ids AS source_chunk_ids,
       communities,
       score
ORDER BY score DESC, title
"""

# ------------------------------------------------------------------
# Community reporting — child report context (level > 0)
# ------------------------------------------------------------------
COMMUNITY_CHILD_REPORT_QUERY = """
MATCH (child:Community)-[:PARENT_COMMUNITY]->(c:Community {
    graph_name: $graph_name,
    id: $cid,
    level: $level
})
WHERE child.graph_name = $graph_name
  AND child.level = $child_level
  AND child.report_text IS NOT NULL
RETURN child.id AS id,
       child.report_text AS report_text,
       child.level AS level,
       child.input_fingerprint AS input_fingerprint,
       coalesce(child.context_tokens_used, 0) AS context_tokens_used
ORDER BY context_tokens_used DESC, child.id
"""

# ------------------------------------------------------------------
# Community reports by key
# ------------------------------------------------------------------
COMMUNITY_REPORTS_BY_KEY_QUERY = """
UNWIND $keys AS key
MATCH (c:Community {
    graph_name: $graph_name,
    id: key.id,
    level: key.level
})
WHERE c.report_text IS NOT NULL
RETURN c.id AS id,
       c.report_text AS report_text,
       c.level AS level,
       c.input_fingerprint AS input_fingerprint
ORDER BY c.level ASC
LIMIT $top_k
"""

# ------------------------------------------------------------------
# DRIFT — bridging entities in communities
# ------------------------------------------------------------------
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
  AND id(e) < id(other)
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
