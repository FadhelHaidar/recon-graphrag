"""Small Cypher construction helpers."""

from __future__ import annotations


def escape_cypher_identifier(identifier: str) -> str:
    """Escape an index, label, relationship, or property identifier."""
    return "`" + identifier.replace("`", "``") + "`"


def cypher_string_literal(value: str) -> str:
    """Render a Cypher string literal for DDL clauses."""
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
