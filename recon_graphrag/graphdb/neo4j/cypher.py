"""Neo4j-specific Cypher construction helpers.

These are re-exported from the backend-neutral ``graphdb.cypher`` module.
Backends that need vendor-specific escaping can override here.
"""

from __future__ import annotations

from recon_graphrag.graphdb.cypher import (
    cypher_string_literal,
    escape_cypher_identifier,
)

__all__ = ["escape_cypher_identifier", "cypher_string_literal"]
