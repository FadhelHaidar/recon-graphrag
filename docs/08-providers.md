# Providers

Recon-GraphRAG uses factory functions to create LLM and embedder clients. The factories normalize the interface so you can swap providers without changing pipeline code.

## LLM providers

Create an LLM with `create_llm()`:

```python
from recon_graphrag import create_llm

llm = create_llm("openai", model_name="gpt-4o", api_key="sk-...")
```

### Supported LLM providers

| Provider | Key | Requires |
| --------- | --------- | --------- |
| OpenAI | `"openai"` | Included in core package |
| Azure OpenAI | `"azure_openai"` | Included in core package |
| Ollama | `"ollama"` | `[ollama]` extra |
| OpenRouter | `"openrouter"` | Included in core package (OpenAI-compatible API) |

### Examples

#### OpenAI

```python
llm = create_llm("openai", model_name="gpt-4o", api_key="sk-...")
```

#### Azure OpenAI

```python
llm = create_llm(
    "azure_openai",
    model_name="gpt-4o",
    api_key="...",
    azure_endpoint="https://your-resource.openai.azure.com/",
    azure_deployment="your-deployment-name",
)
```

#### Ollama

```python
llm = create_llm("ollama", model_name="llama3")
```

#### OpenRouter

```python
llm = create_llm(
    "openrouter",
    model_name="anthropic/claude-sonnet",
    api_key="sk-or-...",
)
```

#### Custom OpenAI-compatible endpoint

```python
llm = create_llm(
    "openai",
    model_name="custom-model",
    base_url="http://localhost:8000/v1",
    api_key="dummy",
)
```

---

## Embedder providers

Create an embedder with `create_embedder()`:

```python
from recon_graphrag import create_embedder

embedder = create_embedder("openai", model="text-embedding-3-small", api_key="sk-...")
```

### Supported embedder providers

| Provider | Key | Default Dim | Requires |
| --------- | --------- | --------- | --------- |
| OpenAI | `"openai"` | 1536 | Included in core package |
| Azure OpenAI | `"azure_openai"` | 1536 | Included in core package |
| Ollama | `"ollama"` | Varies | `[ollama]` extra |
| OpenRouter | `"openrouter"` | Varies | Included in core package |
| Sentence-Transformers | `"sentence-transformer"` | Auto-detected | Included in core package |

### Embedder examples

#### OpenAI (embedder)

```python
embedder = create_embedder("openai", model="text-embedding-3-small", api_key="sk-...")
```

#### Azure OpenAI (embedder)

```python
embedder = create_embedder(
    "azure_openai",
    model="text-embedding-3-small",
    api_key="...",
    azure_endpoint="https://your-resource.openai.azure.com/",
    azure_deployment="your-embedding-deployment",
)
```

#### Ollama (embedder)

```python
embedder = create_embedder("ollama", model="nomic-embed-text")
```

#### OpenRouter (embedder)

```python
embedder = create_embedder(
    "openrouter",
    model="openai/text-embedding-3-small",
    api_key="sk-or-...",
)
```

#### Custom OpenAI-compatible endpoint (embedder)

```python
embedder = create_embedder(
    "openai",
    model="nomic-embed-text",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)
```

#### Sentence-Transformers

```python
embedder = create_embedder("sentence-transformer", model="all-MiniLM-L6-v2")
```

Sentence-Transformers runs locally and does not require an API key.

---

## Passing `model_params`

Use `model_params` to forward extra arguments on every LLM call:

```python
llm = create_llm(
    "openai",
    model_name="gpt-4o",
    api_key="sk-...",
    model_params={"temperature": 0.1, "max_tokens": 2048},
)
```

Use `model_params` to forward extra arguments on every embedding call:

```python
embedder = create_embedder(
    "openai",
    model="text-embedding-3-small",
    api_key="sk-...",
    model_params={"dimensions": 512},
)
```

This is useful for OpenAI's `dimensions` or `encoding_format` parameters, and for LLM parameters such as `temperature`.

---

## Embedding dimensions

Most providers have a fixed output dimension. Make sure the `embedding_dim` passed to `store.create_indexes()` matches your embedder.

| Provider | How to set dimension |
| --------- | --------- |
| OpenAI / Azure OpenAI / OpenRouter | Pass `embedding_dim` to `store.create_indexes()`. Use `model_params={"dimensions": ...}` to reduce it. |
| Sentence-Transformers | Detect the dimension with `detect_embedding_dim(embedder)` and pass it to `store.create_indexes()`. |
| Ollama | Pass `embedding_dim` explicitly. |

See [Indexing](04-indexing.md) for more details.

---

## Provider integration tests

The repository includes integration tests that call real provider endpoints. They are disabled by default so routine test runs do not incur cost.

See [Testing](10-testing.md) for run commands and required environment variables.

---

## Next steps

- Follow the full setup in [Quick Start](02-quickstart.md).
- Configure indexes in [Indexing](04-indexing.md).
- Learn about search modes in [Search](06-search.md).
