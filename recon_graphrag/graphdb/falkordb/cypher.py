"""Cypher escaping helpers for FalkorDB.

Mirrors the interface in recon_graphrag.graphdb.neo4j.cypher so that
backend-specific code can share the same helper names.
"""

from __future__ import annotations


def escape_cypher_identifier(identifier: str) -> str:
    """Escape a label, relationship type, or property name for Cypher."""
    return "`" + identifier.replace("`", "``") + "`"


def cypher_string_literal(value: str) -> str:
    """Return a safely-quoted Cypher string literal."""
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
