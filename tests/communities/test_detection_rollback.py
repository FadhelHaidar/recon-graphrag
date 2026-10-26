"""Exercise replacement against seeded state, not only recorded Cypher."""

from contextlib import contextmanager
from copy import deepcopy
from typing import Any, cast

import pytest

from recon_graphrag.communities.memgraph.detection import (
    CommunityDetector as MemgraphDetector,
)
from recon_graphrag.communities.neo4j.detection import (
    CommunityDetector as Neo4jDetector,
)
from recon_graphrag.graphdb.base import GraphStore
from tests.communities.memgraph.test_memgraph_detection import (
    FakeGraphStore as MemgraphFake,
)
from tests.communities.neo4j.test_neo4j_detection import FakeGraphStore as Neo4jFake


def old_artifacts():
    return {
        "communities": [{"id": "old", "report_text": "usable report", "embedding": [0.5, 0.7]}],
        "memberships": [{"entity_id": "entity:old", "community_id": "old"}],
        "parents": [{"child_id": "old", "parent_id": "old-parent"}],
    }


class StatefulStore:
    def __init__(self, backend, fail_on=None):
        self.delegate: Any = Neo4jFake() if backend == "neo4j" else MemgraphFake()
        self.fail_on = fail_on
        self.state = {"target": old_artifacts(), "other": old_artifacts()}
        self.transactions = 0
        self.writes = []

    def execute_query(self, query, parameters=None):
        if "DETACH DELETE" in query or "MERGE" in query:
            raise AssertionError("replacement write escaped its transaction")
        if "RETURN count(e) AS cnt" in query and self.fail_on == "no_entities":
            return [{"cnt": 0}]
        if "RETURN DISTINCT type(r) AS t" in query and self.fail_on == "relationships":
            return []
        if "gds.graph.project" in query and self.fail_on == "projection":
            raise RuntimeError("injected projection failure")
        if "gds.leiden.stream" in query or "leiden_community_detection.get" in query:
            if self.fail_on == "procedure":
                raise RuntimeError("injected procedure failure")
            if self.fail_on == "empty":
                return []
            if self.fail_on == "malformed":
                return [{"communityId": 1, "communities": [1]}]
        return self.delegate.execute_query(query, parameters)

    @contextmanager
    def _transaction(self):
        self.transactions += 1
        pending = deepcopy(self.state)
        store = self

        class Executor:
            def execute_query(self, query, parameters):
                graph_name = parameters.get("graph_name") or parameters["rows"][0]["graph_name"]
                graph = pending[graph_name]
                if "DETACH DELETE" in query:
                    stage = "delete"
                    graph.clear()
                    graph.update(communities=[], memberships=[], parents=[])
                elif "PARENT_COMMUNITY" in query:
                    stage = "parents"
                    graph["parents"].extend(deepcopy(parameters["rows"]))
                elif "IN_COMMUNITY" in query:
                    stage = "memberships"
                    graph["memberships"].extend(deepcopy(parameters["rows"]))
                    # Neo4j creates community nodes in the membership query.
                    if "MERGE (c:" in query:
                        graph["communities"].extend(deepcopy(parameters["rows"]))
                else:
                    stage = "communities"
                    graph["communities"].extend(deepcopy(parameters["rows"]))
                store.writes.append(stage)
                if store.fail_on == stage:
                    raise RuntimeError(f"injected {stage} failure")
                return []

        yield Executor()
        self.state = pending


@pytest.mark.parametrize("backend,detector_type", [
    ("neo4j", Neo4jDetector), ("memgraph", MemgraphDetector),
])
@pytest.mark.parametrize("failure", [
    "no_entities", "relationships", "procedure", "empty", "malformed",
    "delete", "memberships", "parents",
])
def test_failed_detection_preserves_complete_snapshot(backend, detector_type, failure):
    store = StatefulStore(backend, failure)
    before = deepcopy(store.state)
    detector = detector_type(cast(GraphStore, store), graph_name="target")

    with pytest.raises((RuntimeError, KeyError)):
        detector.detect()

    assert store.state == before
    if failure in {"delete", "memberships", "parents"}:
        assert store.transactions == 1
        assert failure in store.writes
    else:
        assert store.transactions == 0


def test_failed_projection_preserves_complete_snapshot():
    store = StatefulStore("neo4j", "projection")
    before = deepcopy(store.state)
    with pytest.raises(RuntimeError, match="projection"):
        Neo4jDetector(cast(GraphStore, store), graph_name="target").detect()
    assert store.state == before
    assert store.transactions == 0


@pytest.mark.parametrize("backend,detector_type", [
    ("neo4j", Neo4jDetector), ("memgraph", MemgraphDetector),
])
def test_success_replaces_only_target_graph(backend, detector_type):
    store = StatefulStore(backend)
    other = deepcopy(store.state["other"])
    detector = detector_type(cast(GraphStore, store), graph_name="target")

    assert detector.detect()
    assert store.transactions == 1
    assert store.state["other"] == other
    assert store.state["target"]["communities"]
    assert store.state["target"]["memberships"]
    assert store.state["target"]["parents"]
    assert all("report_text" not in row for row in store.state["target"]["communities"])


def test_store_without_transactions_is_rejected_before_deletion():
    store = StatefulStore("neo4j")
    store._transaction = None  # type: ignore[assignment]
    before = deepcopy(store.state)
    with pytest.raises(RuntimeError, match="requires a store with _transaction"):
        Neo4jDetector(cast(GraphStore, store), graph_name="target").detect()
    assert store.state == before
    assert store.writes == []
