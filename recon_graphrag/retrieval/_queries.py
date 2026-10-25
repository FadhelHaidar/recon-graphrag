"""Shared Cypher query templates for graph retrieval.

Neo4j and Memgraph run the same query language here; they diverge only in two
mechanical spots, both parameterized below so the backend modules stay
byte-identical to their hand-written predecessors:

- the internal-id function (``elementId`` on Neo4j, ``id`` on Memgraph)
- an optional ``ORDER BY score DESC, title`` tie-break appended to the
  local/DRIFT queries (Memgraph only, where vector-search order is not stable)

Generic retrieval consumers should call GraphStore methods, not import these.
"""

from __future__ import annotations

_ORDER_BY_SCORE = "ORDER BY score DESC, title\n"

# Effective "no limit" for the per-seed relationship/source caps. Retrievers pass
# real values through ``query_params``; the stores default to this so the exported
# ``DEFAULT_*_RETRIEVAL_QUERY`` constants and direct ``fetch_entity_context``
# callers keep their unlimited behavior.
DEFAULT_RETRIEVAL_LIMIT = 2**31 - 1

# Deterministic relationship ranking, mirrored in both expansion branches.
_REL_WEIGHT = "coalesce(r.weight, r.observation_count, r.strength, 1.0)"
_REL_ORDER = (
    f"{_REL_WEIGHT} DESC, type(r) ASC, "
    "coalesce(neighbor.human_readable_id, neighbor.canonical_key, neighbor.id) ASC"
)

# Source-chunk selection. Ranked mode scores each candidate against the query
# vector with a portable dot product; chunk vectors are stored L2-normalized at
# ingestion, so dot == cosine. Candidates with no vector, a dimension mismatch,
# or no query vector score 0.0, which falls back to id ordering because the
# tie-break stays ``chunk.id ASC``.
_SOURCE_SELECT_UNRANKED = "    WITH DISTINCT chunk\n    ORDER BY chunk.id ASC\n"
_SOURCE_SELECT_RANKED = """    WITH DISTINCT chunk
    WITH chunk,
         CASE WHEN $query_vector IS NULL
                   OR chunk.embedding IS NULL
                   OR size(chunk.embedding) <> size($query_vector)
              THEN 0.0
              ELSE reduce(d = 0.0, i IN range(0, size($query_vector) - 1)
                          | d + $query_vector[i] * chunk.embedding[i])
         END AS source_score
    ORDER BY source_score DESC, chunk.id ASC
"""

# Local search — core ends at the final RETURN column (no tie-break).
_LOCAL_CORE = """
CALL (node) {
    OPTIONAL MATCH (node)-[r]->(neighbor)
    WHERE NOT neighbor:Chunk AND NOT neighbor:Document AND NOT neighbor:Community
      AND neighbor.graph_name = node.graph_name
    WITH node, r, neighbor
    ORDER BY WEIGHT_PLACEHOLDER
    LIMIT $rel_limit
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
    ORDER BY WEIGHT_PLACEHOLDER
    LIMIT $rel_limit
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
CALL (node) {
    OPTIONAL MATCH (node)<-[:FROM_CHUNK]-(chunk:Chunk)
    WHERE chunk.graph_name = node.graph_name
    WITH DISTINCT chunk
    ORDER BY chunk.id ASC
    LIMIT $source_limit
    RETURN collect(chunk) AS selected_chunks
}
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
       [s IN selected_chunks | s.text] AS source_text,
       [s IN selected_chunks | s.id] AS source_chunk_ids,
       score
"""

# DRIFT search — as local, plus community context.
_DRIFT_CORE = """
CALL (node) {
    OPTIONAL MATCH (node)-[r]->(neighbor)
    WHERE NOT neighbor:Chunk AND NOT neighbor:Document AND NOT neighbor:Community
      AND neighbor.graph_name = node.graph_name
    WITH node, r, neighbor
    ORDER BY WEIGHT_PLACEHOLDER
    LIMIT $rel_limit
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
    ORDER BY WEIGHT_PLACEHOLDER
    LIMIT $rel_limit
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
CALL (node) {
    OPTIONAL MATCH (node)<-[:FROM_CHUNK]-(chunk:Chunk)
    WHERE chunk.graph_name = node.graph_name
    WITH DISTINCT chunk
    ORDER BY chunk.id ASC
    LIMIT $source_limit
    RETURN collect(chunk) AS selected_chunks
}
OPTIONAL MATCH (node)-[:IN_COMMUNITY]->(c:Community)
WHERE c.graph_name = node.graph_name
WITH node, score, connections, selected_chunks,
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
       [s IN selected_chunks | s.text] AS source_text,
       [s IN selected_chunks | s.id] AS source_chunk_ids,
       communities,
       score
"""

# Community reporting — child report context (level > 0). Identical per backend.
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

# Community reports by key. Identical per backend.
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

# Community summarization — degree-ranked context (Phase 4A).
# ``__IDFUNC__`` is replaced with the backend internal-id function.
_RANKED_CONTEXT_CORE = """
MATCH (c:Community {
    graph_name: $graph_name,
    id: $cid,
    level: $level
})<-[:IN_COMMUNITY]-(e:__Entity__)
OPTIONAL MATCH (e)-[r]-(other:__Entity__)
WHERE (other)-[:IN_COMMUNITY]->(c)
  AND __IDFUNC__(e) < __IDFUNC__(other)
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


# Community member graph — bounded, direction-preserving expansion of one
# community report into its members and their directed edges. Identical per
# backend: members are ranked by the same degree expression as
# ``_RANKED_CONTEXT_CORE`` but only edges between selected members are returned,
# so every returned edge has both endpoints in the member list.
COMMUNITY_MEMBER_GRAPH_QUERY = """
MATCH (c:Community {
    graph_name: $graph_name,
    id: $community_id,
    level: $level
})<-[:IN_COMMUNITY]-(e:__Entity__)
WHERE e.graph_name = $graph_name
WITH e, SIZE([(e)-[r1]-()
       WHERE NOT type(r1) IN ['IN_COMMUNITY', 'FROM_CHUNK', 'SOURCED_FROM']
     | r1]) AS degree
ORDER BY degree DESC,
         coalesce(e.human_readable_id, e.canonical_key, e.id) ASC
LIMIT $member_limit
WITH collect({node: e, degree: degree}) AS member_rows
WITH member_rows,
     [row IN member_rows |
        coalesce(row.node.human_readable_id, row.node.canonical_key, row.node.id)
     ] AS member_ids
WITH member_rows, member_ids,
     [row IN member_rows | {
        id: coalesce(row.node.human_readable_id, row.node.canonical_key, row.node.id),
        name: coalesce(row.node.name, row.node.description, row.node.id),
        labels: labels(row.node),
        description: coalesce(row.node.description, ''),
        degree: row.degree
     }] AS members
UNWIND member_rows AS member_row
WITH members, member_ids, member_row.node AS member
OPTIONAL MATCH (member)-[r]->(neighbor:__Entity__)
WHERE neighbor.graph_name = $graph_name
  AND coalesce(neighbor.human_readable_id, neighbor.canonical_key, neighbor.id) IN member_ids
  AND NOT neighbor:Community
  AND NOT neighbor:Chunk
  AND NOT neighbor:Document
RETURN members,
       size(members) AS member_count,
       collect(DISTINCT CASE WHEN r IS NULL THEN null ELSE {
           source: coalesce(member.human_readable_id, member.canonical_key, member.id),
           target: coalesce(neighbor.human_readable_id, neighbor.canonical_key, neighbor.id),
           type: type(r),
           description: coalesce(r.description, ''),
           weight: coalesce(r.weight, r.observation_count, r.strength, 1.0)
       } END) AS edges,
       count(r) AS edge_count
"""


def set_retrieval_limit_defaults(parameters: dict) -> None:
    """Fill the per-seed relationship/source caps when the caller did not."""
    parameters.setdefault("rel_limit", DEFAULT_RETRIEVAL_LIMIT)
    parameters.setdefault("source_limit", DEFAULT_RETRIEVAL_LIMIT)


def local_retrieval_query(*, order_by_score: bool, rank_sources: bool = False) -> str:
    """Local-search retrieval query, optionally appending the score tie-break."""
    core = _LOCAL_CORE.replace("WEIGHT_PLACEHOLDER", _REL_ORDER)
    if rank_sources:
        core = core.replace(_SOURCE_SELECT_UNRANKED, _SOURCE_SELECT_RANKED)
    return core + (_ORDER_BY_SCORE if order_by_score else "")


def drift_retrieval_query(*, order_by_score: bool, rank_sources: bool = False) -> str:
    """DRIFT-search retrieval query, optionally appending the score tie-break."""
    core = _DRIFT_CORE.replace("WEIGHT_PLACEHOLDER", _REL_ORDER)
    if rank_sources:
        core = core.replace(_SOURCE_SELECT_UNRANKED, _SOURCE_SELECT_RANKED)
    return core + (_ORDER_BY_SCORE if order_by_score else "")


def ranked_context_query(*, id_func: str) -> str:
    """Degree-ranked community context query for the given internal-id function."""
    return _RANKED_CONTEXT_CORE.replace("__IDFUNC__", id_func)
