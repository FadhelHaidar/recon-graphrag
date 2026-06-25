"""Shared Cypher escaping contract tests."""

import pytest

from recon_graphrag.graphdb.memgraph.cypher import (
    escape_cypher_identifier as escape_memgraph_identifier,
)
from recon_graphrag.graphdb.neo4j.cypher import (
    escape_cypher_identifier as escape_neo4j_identifier,
)
from recon_graphrag.graphdb.cypher import (
    cypher_string_literal,
    escape_cypher_identifier,
)


@pytest.mark.parametrize(
    "escape_identifier",
    [escape_neo4j_identifier, escape_memgraph_identifier],
)
def test_escape_cypher_identifier(escape_identifier):
    assert escape_identifier("Movie") == "`Movie`"
    assert escape_identifier("M`ovie") == "`M``ovie`"


def test_backend_cypher_helpers_reexport_shared_helpers():
    assert escape_neo4j_identifier is escape_cypher_identifier
    assert escape_memgraph_identifier is escape_cypher_identifier


def test_cypher_string_literal_escapes_quotes_and_backslashes():
    assert cypher_string_literal("Bob's \\ Movie") == "'Bob\\'s \\\\ Movie'"
