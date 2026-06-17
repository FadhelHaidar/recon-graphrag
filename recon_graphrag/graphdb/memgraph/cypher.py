"""Cypher escaping helpers for Memgraph.

Memgraph uses the same identifier/string literal rules as Neo4j for
labels, relationship types, property names, and string values, so these
helpers mirror the Neo4j equivalents.
"""

from __future__ import annotations


def escape_cypher_identifier(identifier: str) -> str:
    """Escape a label, relationship type, or property name for Cypher."""
    return "`" + identifier.replace("`", "``") + "`"


def cypher_string_literal(value: str) -> str:
    """Return a safely-quoted Cypher string literal."""
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
