"""Backend query contracts for directed, degree-ranked community facts."""

import pytest

from recon_graphrag.retrieval.memgraph.queries import (
    COMMUNITY_RANKED_CONTEXT_QUERY as MEMGRAPH_QUERY,
)
from recon_graphrag.retrieval.neo4j.queries import (
    COMMUNITY_RANKED_CONTEXT_QUERY as NEO4J_QUERY,
)


@pytest.mark.parametrize("query", [NEO4J_QUERY, MEMGRAPH_QUERY], ids=["neo4j", "memgraph"])
def test_ranked_context_query_preserves_directed_endpoints(query):
    assert "})<-[:IN_COMMUNITY]-(e:__Entity__)" in query
    assert "OPTIONAL MATCH (e)-[r]->(other:__Entity__)" in query
    assert "WHERE (other)-[:IN_COMMUNITY]->(c)" in query
    assert "OPTIONAL MATCH (e)-[r]-(other:__Entity__)" not in query
    assert "elementId(" not in query
    assert "id(" not in query
    assert "__IDFUNC__" not in query
    assert "DISTINCT" not in query

    # Endpoint columns and undirected degree/ranking semantics stay unchanged.
    for endpoint in ("e", "other"):
        assert f"coalesce({endpoint}.human_readable_id, {endpoint}.canonical_key, {endpoint}.id) AS {endpoint}_id" in query
        assert f"coalesce({endpoint}.name, {endpoint}.id) AS {endpoint}_name" in query
        assert f"coalesce({endpoint}.description, '') AS {endpoint}_description" in query
        assert f"labels({endpoint}) AS {endpoint}_labels" in query
    assert "SIZE([(e)-[r1]-()" in query
    assert "SIZE([(other)-[r2]-()" in query
    assert query.count("WHERE NOT type(") == 2
    assert query.count("IN ['IN_COMMUNITY', 'FROM_CHUNK', 'SOURCED_FROM']") == 2
    assert "type(r) AS rel_type" in query
    assert "coalesce(r.description, '') AS rel_description" in query
    assert "coalesce(r.observation_count, 1) AS observation_count" in query
    assert "e_degree + other_degree AS combined_degree" in query
    assert "ORDER BY combined_degree DESC, observation_count DESC, type(r) ASC" in query
