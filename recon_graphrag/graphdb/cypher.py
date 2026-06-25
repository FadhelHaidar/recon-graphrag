"""Backend-neutral Cypher construction helpers.

These utilities are safe to share across Neo4j, Memgraph, and any future
Cypher-compatible backend because they only deal with identifier/string
escaping, not vendor-specific syntax.
"""

from __future__ import annotations


def escape_cypher_identifier(identifier: str) -> str:
    """Escape an index, label, relationship, or property identifier."""
    return "`" + identifier.replace("`", "``") + "`"


def cypher_string_literal(value: str) -> str:
    """Render a Cypher string literal for DDL clauses."""
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
