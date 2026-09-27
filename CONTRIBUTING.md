# Contributing to Recon-GraphRAG

Thank you for your interest in Recon-GraphRAG! This project is open source and welcomes contributions from the community.

## Project values

- **Practical:** We prioritize working code and clear examples.
- **Collaborative:** Discuss big changes before investing heavy implementation time.
- **Respectful:** Be kind, constructive, and inclusive in all interactions.

## How to contribute

### Report bugs

If you find a bug, please open a GitHub issue and include:

- A clear description of the problem.
- Steps to reproduce it.
- The expected behavior and the actual behavior.
- Your Python, Recon-GraphRAG, and graph database versions (Neo4j or Memgraph).
- Relevant code snippets or error messages.

### Request features

Feature requests are welcome. Open a GitHub issue and describe:

- The use case you are trying to solve.
- Why the current API does not handle it well.
- Any proposed API or behavior.

### Ask questions

For questions, troubleshooting help, or general discussion, use GitHub Discussions instead of issues.

## Development setup

1. Fork and clone the repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/recon-graphrag.git
   cd recon-graphrag
   ```

2. Install dependencies. We recommend using `uv`:

   ```bash
   uv sync --extra dev --group dev
   ```

   This creates the virtual environment and installs all dependencies from `uv.lock`, including the dev extra and dependency group.

   If you prefer `pip`, install in editable mode:

   ```bash
   pip install -e ".[all,dev]"
   # `dotenv` is managed through uv's dependency group; install it manually
   # when running tests or examples that load environment files:
   pip install python-dotenv
   ```

3. Start the database needed for your change (Neo4j with APOC/GDS, Memgraph with MAGE, or both):

   ```bash
   docker compose up -d neo4j
   docker compose up -d memgraph lab
   ```

4. Copy the environment file and fill in any values needed for tests:

   ```bash
   cp .env.example .env
   ```

5. Run the mandatory test suite:

   ```bash
   uv run pytest -m "not integration"
   ```

   or, if you installed dependencies with `pip` into an activated virtual environment:

   ```bash
   pytest -m "not integration"
   ```

See [Testing](docs/10-testing.md) for more test commands and integration test setup.

## Branch naming

Use descriptive branch names with a prefix that matches the change type:

| Prefix | Use for |
| --------- | --------- |
| `feat/` | New features |
| `fix/` | Bug fixes |
| `docs/` | Documentation changes |
| `refactor/` | Code refactoring |
| `test/` | Test-only changes |
| `ci/` | CI/CD changes |
| `build/` | Build system or dependency changes |

Examples:

```text
docs/readme-reorganization
feat/local-search-reranking
fix/neo4j-index-race
```

## Commit style

Use [Conventional Commits](https://www.conventionalcommits.org/) so that automated release tooling can classify changes correctly.

### Common types

| Type | Triggers release? | Use for |
| --------- | --------- | --------- |
| `feat:` | Yes — minor bump | New features |
| `fix:` | Yes — patch bump | Bug fixes |
| `docs:` | No | Documentation changes |
| `chore:` | No | Routine maintenance |
| `refactor:` | No | Code refactoring |
| `test:` | No | Test changes |
| `ci:` | No | CI/CD changes |
| `build:` | No | Build/dependency changes |

Example commit messages:

```text
docs: add installation troubleshooting section
feat: add Ollama embedder provider
fix: handle empty entity resolution candidates
```

Documentation-only PRs should use `docs:` so they do not trigger a version bump.

### Breaking changes and major version bumps

To trigger a **major** version bump, indicate a breaking change using one of these patterns:

1. Add `!` after the type or scope:

   ```text
   feat!: redesign the search API
   fix(retrieval)!: change the default top_k behavior
   ```

2. Include a `BREAKING CHANGE:` footer in the commit body:

   ```text
   feat: redesign the search API
   
   BREAKING CHANGE: the `search()` method now requires `mode` as a keyword argument.
   ```

Release Please uses these signals to determine that the next release should be a major version.

## Pull request process

1. Open your pull request against the `main` branch.
2. Fill in the PR description with:
   - What changed and why.
   - Any breaking changes.
   - How you verified the change.
3. Ensure the mandatory test suite passes:
   ```bash
   pytest -m "not integration"
   ```
4. Respond to review feedback promptly and respectfully.
5. Do not bump the version in `pyproject.toml` or `recon_graphrag/__init__.py` — versioning is handled automatically by Release Please.

## Interface conventions: ABC vs Protocol

The project uses both `ABC` and `Protocol` intentionally. Follow this contract:

### Decision rule

| Question | Answer | Use |
|---|---|---|
| Does the base class provide shared implementation? | Yes | `ABC` |
| Is it only a shape/contract with no shared code? | Yes | `Protocol` |
| Is this a plugin interface for external users? | Yes | `Protocol` |
| Is this an internal base class with a template method pattern? | Yes | `ABC` |

### Examples

**`Protocol`** — the user brings their own implementation, no inheritance needed:

```python
@runtime_checkable
class BaseLLM(Protocol):
    def invoke(self, prompt: str, **kwargs) -> LLMResponse: ...
    async def ainvoke(self, prompt: str, **kwargs) -> LLMResponse: ...
```

**`ABC`** — subclasses inherit shared logic and override abstract methods:

```python
class BaseEntityResolver(ABC):
    async def resolve(self, ...):  # shared logic
        entities = self._load_entities(...)
        ...

    @abstractmethod
    def _load_entities(self, ...):  # backend-specific
        ...
```

### Current inventory

| Class | Style | Reason |
|---|---|---|
| `GraphStore` | `Protocol` | Plugin interface, ~30 methods, no shared code |
| `GraphWriter` | `Protocol` | Narrow writer-only contract |
| `BaseLLM` | `Protocol` | Plugin interface, no shared code |
| `BaseEmbedder` | `Protocol` | Plugin interface, no shared code |
| `TokenCounter` | `Protocol` | Utility contract, no shared code |
| `CypherGraphWriter` | Concrete | Single Cypher writer shared by all Cypher backends; no subclassing |
| `BaseEntityResolver` | `ABC` | Template method with shared resolution/fuzzy/LLM logic |
| `BaseRetriever` | `Protocol` | Single abstract method, no shared code |

### Rules

1. **Do not mix styles for the same role.** If a class is `Protocol`, keep it `Protocol`. If `ABC`, keep it `ABC`.
2. **Do not duplicate protocol definitions.** Import from the canonical location. (e.g., `TokenCounter` lives in `utils/tokens.py`, not in `extraction/chunking.py`.)
3. **New external-facing plugin interfaces** → `Protocol`.
4. **New internal base classes with shared logic** → `ABC`.
5. **When in doubt**, start with `Protocol`. Only switch to `ABC` if you need shared implementation.

## Code of conduct

Be respectful, inclusive, and constructive. Harassment or discriminatory behavior will not be tolerated.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](../LICENSE).
