"""Packaging boundaries for local embeddings and the Neo4j driver range."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import tomllib

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pyproject() -> dict:
    with open(os.path.join(REPOSITORY_ROOT, "pyproject.toml"), "rb") as handle:
        return tomllib.load(handle)


def test_core_dependencies_exclude_local_embedding_stack():
    dependencies = _pyproject()["project"]["dependencies"]

    assert all(not item.startswith("sentence-transformers") for item in dependencies)


def test_local_embeddings_extra_is_available_and_in_all():
    extras = _pyproject()["project"]["optional-dependencies"]

    assert "sentence-transformers>=5.4.1" in extras["local-embeddings"]
    assert "sentence-transformers>=5.4.1" in extras["all"]


def test_neo4j_dependency_supports_driver_5_and_6():
    dependencies = _pyproject()["project"]["dependencies"]

    assert "neo4j>=5.28.8,<7" in dependencies


def test_core_import_does_not_import_local_embedding_stack():
    script = textwrap.dedent(
        """
        import sys

        class LocalDependencyBlocker:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.partition(".")[0] in {
                    "numpy",
                    "sentence_transformers",
                    "torch",
                    "transformers",
                }:
                    raise ImportError(f"blocked local dependency: {fullname}")
                return None

        sys.meta_path.insert(0, LocalDependencyBlocker())
        import recon_graphrag

        assert "sentence_transformers" not in sys.modules
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPOSITORY_ROOT)

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        env=env,
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
