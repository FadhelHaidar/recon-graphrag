"""Inspect summary query contracts; live backend tests own Cypher semantics."""

from unittest.mock import MagicMock

import pytest

from recon_graphrag.graphdb.memgraph.store import MemgraphGraphStore
from recon_graphrag.graphdb.neo4j.store import Neo4jGraphStore


@pytest.fixture(params=[Neo4jGraphStore, MemgraphGraphStore], ids=["neo4j", "memgraph"])
def store(request):
    store = request.param(None)
    store.execute_query = MagicMock(return_value=[])
    return store


@pytest.mark.parametrize("kind,alias", [("entities", "e"), ("relationships", "r")])
def test_summary_selection_skips_successes_and_requires_raw_descriptions(store, kind, alias):
    getattr(store, f"get_{kind}_needing_summary")("g", limit=7)
    query, params = store.execute_query.call_args.args
    assert f"coalesce({alias}.description_summary_status, '') <> 'success'" in query
    assert f"size(coalesce({alias}.descriptions, [])) > 0" in query
    assert params == {"graph_name": "g", "limit": 7}


@pytest.mark.parametrize("kind,alias", [("entity", "e"), ("relationship", "r")])
def test_failed_summary_preserves_existing_description_and_persists_outcomes(store, kind, alias):
    rows = [
        {"id": "bad", "description": None, "descriptions": ["original"],
         "description_summary_status": "failed", "description_input_fingerprint": "fp",
         "description_summary_updated": 1, "description_summary_error": "failed"},
        {"id": "good", "description": "summary", "descriptions": ["source"],
         "description_summary_status": "success", "description_input_fingerprint": "fp2",
         "description_summary_updated": 1, "description_summary_error": None},
    ]
    getattr(store, f"persist_{kind}_summaries")("g", rows)
    store.execute_query.assert_called_once()
    query, params = store.execute_query.call_args.args
    assert params == {"graph_name": "g", "summaries": rows}
    assert f"WHEN row.description IS NULL THEN {alias}.description" in query
    assert "ELSE row.description" in query
    for field in ("descriptions", "description_summary_status", "description_input_fingerprint",
                  "description_summary_updated", "description_summary_error"):
        assert f"{alias}.{field} = row.{field}" in query


@pytest.mark.parametrize("kind", ["entity", "relationship"])
def test_empty_summary_batch_does_not_write(store, kind):
    getattr(store, f"persist_{kind}_summaries")("g", [])
    store.execute_query.assert_not_called()
