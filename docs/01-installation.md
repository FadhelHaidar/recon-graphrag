# Installation

Recon-GraphRAG is a Python library for building GraphRAG pipelines on Neo4j or Memgraph. It is not yet published to PyPI, so install it directly from GitHub.

We recommend using [`uv`](https://docs.astral.sh/uv/) for fast, reliable Python package management. The instructions below use `uv` by default and show `pip` alternatives where applicable.

## Requirements

- **Python** >= 3.11
- One supported graph database:
  - **Neo4j** with **APOC** for merge helpers and **GDS** for Leiden community detection
  - **Memgraph** with **MAGE** for Leiden community detection

If you do not already have a configured database, use the Docker Compose setup below.

---

## Graph databases with Docker Compose

The repository includes Neo4j with APOC/GDS and Memgraph with MAGE. Memgraph Lab is also available for browser-based inspection.

```bash
# Clone the repo
git clone https://github.com/FadhelHaidar/Recon-GraphRAG.git
cd Recon-GraphRAG

# Copy and customize the environment file (optional)
cp .env.example .env

# Start one backend
docker compose up -d neo4j
docker compose up -d memgraph lab

# Or start both backends and Memgraph Lab
docker compose up -d

# Wait for it to be ready (usually takes ~30 seconds)
docker compose logs -f neo4j
docker compose logs -f memgraph
```

The services will be available at:

- **Neo4j Browser**: <http://localhost:7475> (login: `neo4j` / `password`)
- **Neo4j Bolt**: `bolt://localhost:7688`
- **Memgraph Bolt**: `bolt://localhost:7689`
- **Memgraph Lab**: <http://localhost:3000>

To stop:

```bash
docker compose down
# Or to also remove volumes (data will be lost):
docker compose down -v
```

### Manual Neo4j setup

If you prefer to use your own Neo4j instance:

1. Install Neo4j 5.x (Community or Enterprise).
2. Install the APOC and GDS plugins in `$NEO4J_HOME/plugins`.
3. Add the following to `neo4j.conf` so APOC procedures are allowed:

   ```properties
   dbms.security.procedures.unrestricted=apoc.*,gds.*
   ```

4. Restart Neo4j and confirm the plugins load by running `CALL apoc.meta.schema()` and `CALL gds.version()` in the Neo4j Browser.

### Manual Memgraph setup

Install a current Memgraph release with MAGE and expose its Bolt endpoint. Set `MEMGRAPH_URL`, plus `MEMGRAPH_USERNAME`, `MEMGRAPH_PASSWORD`, and `MEMGRAPH_DATABASE` when your deployment requires them. The Python Neo4j driver is used for both backends because Memgraph supports the Bolt protocol.

---

## Install with uv (recommended)

If you are using `uv` to manage your project, add Recon-GraphRAG from GitHub and let `uv` resolve and lock the dependencies:

```bash
uv add git+https://github.com/FadhelHaidar/Recon-GraphRAG.git
uv sync
```

Then run your scripts with:

```bash
uv run python your_script.py
```

`uv run` automatically creates or updates the virtual environment and installs dependencies from `uv.lock` before running your script.

---

## Optional extras

The core package includes the minimum dependencies needed to run the SDK, including the OpenAI client, `sentence-transformers`, and the Neo4j driver. The only remaining optional extra is for the Ollama provider.

| Extra | Installs | Enables |
| --------- | --------- | --------- |
| `[ollama]` | `ollama` | Ollama local LLM/embedder provider |
| `[all]` | All optional extras | All supported providers |
| `[dev]` | `pytest`, `pytest-asyncio`, `pydantic` | Running the test suite |
| `[evaluation]` | `pydantic` | Evaluation helpers |

With `uv`:

```bash
# All optional providers
uv add "recon-graphrag[all] @ git+https://github.com/FadhelHaidar/Recon-GraphRAG.git"

# Ollama provider only (OpenAI, sentence-transformers, and Neo4j are already included)
uv add "recon-graphrag[ollama] @ git+https://github.com/FadhelHaidar/Recon-GraphRAG.git"

uv sync
```

### Pinning a version

Pin to a specific release tag so your build stays reproducible:

```bash
uv add git+https://github.com/FadhelHaidar/Recon-GraphRAG.git@v0.4.0
uv sync
```

With extras:

```bash
uv add "recon-graphrag[all] @ git+https://github.com/FadhelHaidar/Recon-GraphRAG.git@v0.4.0"
uv sync
```

With `pip`:

```bash
pip install "recon-graphrag[all] @ git+https://github.com/FadhelHaidar/Recon-GraphRAG.git@v0.4.0"
```

---

## Install with pip

If you do not use `uv`:

```bash
pip install git+https://github.com/FadhelHaidar/Recon-GraphRAG.git
```

---

## Editable install for development

If you plan to change the Recon-GraphRAG library code itself, install it in editable mode from a local clone.

With `uv`:

```bash
git clone https://github.com/FadhelHaidar/Recon-GraphRAG.git
cd Recon-GraphRAG

# With all optional provider extras and dev dependencies
uv pip install -e ".[all,dev]"
```

Or use the uv project workflow:

```bash
git clone https://github.com/FadhelHaidar/Recon-GraphRAG.git
cd Recon-GraphRAG
uv sync --extra dev --group dev
uv run pytest -m "not integration"
```

With `pip`:

```bash
git clone https://github.com/FadhelHaidar/Recon-GraphRAG.git
cd Recon-GraphRAG
pip install -e ".[all,dev]"
# The `dotenv` package is managed through uv's dependency group; install it
# manually when running tests or examples that load environment files:
pip install python-dotenv
```

---

## Clone without installing

You can also clone the repository and import it directly without running `pip install`, as long as your script runs from the parent directory of the cloned folder:

```bash
# From your project root (above the Recon-GraphRAG folder)
python your_script.py
```

```python
# your_script.py
from recon_graphrag import GraphRAG, GraphBuilderPipeline
```

Python automatically adds the current working directory to `sys.path`, so the package is importable without installation.

---

## Verify the installation

```python
import recon_graphrag

print(recon_graphrag.__version__)
```

You should see the installed version, for example:

```text
0.4.0
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'ollama'`

You are using the Ollama provider, which requires the `[ollama]` extra. Re-install with the appropriate extra:

```bash
uv add "recon-graphrag[ollama] @ git+https://github.com/FadhelHaidar/Recon-GraphRAG.git"
uv sync
```

Or with `pip`:

```bash
pip install "recon-graphrag[ollama] @ git+https://github.com/FadhelHaidar/Recon-GraphRAG.git"
```

### Neo4j connection errors

- Ensure Neo4j is running and the Bolt URL and credentials are correct.
- If you are using the Docker Compose setup, the default credentials are `neo4j` / `password`.
- Check that the configured Bolt port is not blocked by a firewall.

### Memgraph connection errors

- Ensure Memgraph is running and `MEMGRAPH_URL` points to its Bolt endpoint.
- Docker Compose exposes Memgraph at `bolt://localhost:7689` without authentication by default.
- Community detection requires the MAGE image or a Memgraph installation with `leiden_community_detection` available.

### GDS or APOC not found

- Confirm the plugins are in the Neo4j `plugins` directory.
- For manual installs, make sure `dbms.security.procedures.unrestricted=apoc.*,gds.*` is set in `neo4j.conf`.
- With Docker Compose, both plugins are pre-installed. If they are missing, try `docker compose down -v` followed by `docker compose up -d`.

### Conflicting Python version

Recon-GraphRAG requires Python >= 3.11. Check your version with:

```bash
python --version
```

If you need to manage multiple Python versions, consider using [`uv`](https://docs.astral.sh/uv/) or a virtual environment.

---

## Next steps

- Follow the [Quick Start](02-quickstart.md) to build and query your first graph.
- Read about the pipeline architecture in [Pipelines](05-pipelines.md).
- Configure providers in [Providers](08-providers.md).
