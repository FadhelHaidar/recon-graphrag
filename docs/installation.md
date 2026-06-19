# Installation

Recon-GraphRAG is a Python library for building GraphRAG pipelines on Neo4j. It is not yet published to PyPI, so install it directly from GitHub.

We recommend using [`uv`](https://docs.astral.sh/uv/) for fast, reliable Python package management. The instructions below use `uv` by default and show `pip` alternatives where applicable.

## Requirements

- **Python** >= 3.11
- **Neo4j** (Community or Enterprise edition) with:
  - **APOC** — optional, used for some internal duplicate entity resolution helpers
  - **GDS** (Graph Data Science) — required for community detection (Leiden algorithm)

If you do not already have a Neo4j instance with these plugins, use the Docker Compose setup below.

## Neo4j with Docker Compose

The repository includes a ready-to-use Neo4j Enterprise container with APOC and GDS pre-installed.

```bash
# Clone the repo
git clone https://github.com/FadhelHaidar/Recon-GraphRAG.git
cd Recon-GraphRAG

# Copy and customize the environment file (optional)
cp .env.example .env

# Start Neo4j
docker-compose up -d

# Wait for it to be ready (usually takes ~30 seconds)
docker-compose logs -f neo4j
```

Neo4j will be available at:

- **Browser**: <http://localhost:7474> (login: `neo4j` / password)
- **Bolt**: `bolt://localhost:7687`

To stop:

```bash
docker-compose down
# Or to also remove volumes (data will be lost):
docker-compose down -v
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

## Optional extras

The core package includes the minimum dependencies needed to run the SDK. Provider-specific clients and heavier dependencies are available as extras.

| Extra | Installs | Enables |
| --------- | --------- | --------- |
| `[openai]` | `openai>=1.0` | OpenAI, Azure OpenAI, and OpenRouter providers |
| `[ollama]` | `ollama` | Ollama local LLM/embedder providers |
| `[sentence-transformers]` | `sentence-transformers>=2.0` | Local sentence-transformer embedders |
| `[all]` | All of the above | All supported providers |
| `[dev]` | `pytest`, `pytest-asyncio` | Running the test suite |

With `uv`:

```bash
# All providers
uv add "recon-graphrag[all] @ git+https://github.com/FadhelHaidar/Recon-GraphRAG.git"

# Individual providers
uv add "recon-graphrag[openai] @ git+https://github.com/FadhelHaidar/Recon-GraphRAG.git"
uv add "recon-graphrag[ollama] @ git+https://github.com/FadhelHaidar/Recon-GraphRAG.git"
uv add "recon-graphrag[sentence-transformers] @ git+https://github.com/FadhelHaidar/Recon-GraphRAG.git"

uv sync
```

### Pinning a version

Pin to a specific release tag so your build stays reproducible:

```bash
uv add git+https://github.com/FadhelHaidar/Recon-GraphRAG.git@v0.1.1
uv sync
```

With extras:

```bash
uv add "recon-graphrag[all] @ git+https://github.com/FadhelHaidar/Recon-GraphRAG.git@v0.1.1"
uv sync
```

With `pip`:

```bash
pip install "recon-graphrag[all] @ git+https://github.com/FadhelHaidar/Recon-GraphRAG.git@v0.1.1"
```

## Install with pip

If you do not use `uv`:

```bash
pip install git+https://github.com/FadhelHaidar/Recon-GraphRAG.git
```

## Editable install for development

If you plan to change the Recon-GraphRAG library code itself, install it in editable mode from a local clone.

With `uv`:

```bash
git clone https://github.com/FadhelHaidar/Recon-GraphRAG.git
cd Recon-GraphRAG

# With all provider extras and dev dependencies
uv pip install -e ".[all,dev]"
```

Or use the uv project workflow:

```bash
git clone https://github.com/FadhelHaidar/Recon-GraphRAG.git
cd Recon-GraphRAG
uv sync --group dev
uv run pytest -m "not integration"
```

With `pip`:

```bash
git clone https://github.com/FadhelHaidar/Recon-GraphRAG.git
cd Recon-GraphRAG
pip install -e ".[all,dev]"
```

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

## Verify the installation

```python
import recon_graphrag

print(recon_graphrag.__version__)
```

You should see the installed version, for example:

```text
0.1.1
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'openai'` (or `ollama`, etc.)

You are using a provider that requires an optional extra. Re-install with the appropriate extra:

```bash
uv add "recon-graphrag[openai] @ git+https://github.com/FadhelHaidar/Recon-GraphRAG.git"
uv sync
```

Or with `pip`:

```bash
pip install "recon-graphrag[openai] @ git+https://github.com/FadhelHaidar/Recon-GraphRAG.git"
```

### Neo4j connection errors

- Ensure Neo4j is running and the Bolt URL and credentials are correct.
- If you are using the Docker Compose setup, the default credentials are `neo4j` / `password`.
- Check that the Bolt port (`7687` by default) is not blocked by a firewall.

### GDS or APOC not found

- Confirm the plugins are in the Neo4j `plugins` directory.
- For manual installs, make sure `dbms.security.procedures.unrestricted=apoc.*,gds.*` is set in `neo4j.conf`.
- With Docker Compose, both plugins are pre-installed. If they are missing, try `docker-compose down -v` followed by `docker-compose up -d`.

### Conflicting Python version

Recon-GraphRAG requires Python >= 3.11. Check your version with:

```bash
python --version
```

If you need to manage multiple Python versions, consider using [`uv`](https://docs.astral.sh/uv/) or a virtual environment.

## Next steps

- Follow the [Quick Start](quickstart.md) to build and query your first graph.
- Read about the pipeline architecture in [Pipelines](pipelines.md).
- Configure providers in [Providers](providers.md).
