# Structured Data Ingestion

Ingest CSV files, Excel worksheets, SQL query results, or any list of dicts
into the knowledge graph. Two modes work together on the same table:

1. **Direct mapping** — columns map deterministically to entities, properties,
   and relationships via a `RowMapping`. No LLM call per row: fast, exact,
   free.
2. **LLM extraction on free-text columns** — columns holding prose (a
   `notes` or `description` column) get an LLM extraction pass, and the
   extracted entities are linked back to the row's entities.

```python
from recon_graphrag import (
    GraphBuilderPipeline, RowMapping, ColumnEntity, RowRelationship, TextColumn,
)

mapping = RowMapping(
    entities=[
        ColumnEntity(label="Company", identity_column="supplier_name"),
        ColumnEntity(
            label="Product",
            identity_column="product_name",
            properties={"unit_price": "unit_price", "sku": "sku"},
            types={"unit_price": "FLOAT"},
            description_template="{product_name}, supplied by {supplier_name} at {unit_price} USD",
        ),
    ],
    relationships=[RowRelationship("Company", "SUPPLIES", "Product")],
    text_columns=[TextColumn("notes", anchor_label="Product")],
    record_id_column="order_id",
    metadata_columns=["region"],
)

pipeline = GraphBuilderPipeline(graph_store=store, llm=llm, embedder=embedder)
await pipeline.build_from_csv("orders.csv", mapping)
```

A runnable end-to-end example lives in
[`examples/structured_ingest.py`](../examples/structured_ingest.py).

---

## Sources

All four methods share the same core; the wrappers only produce rows.

### `build_from_csv(path, mapping, *, encoding="utf-8-sig", ...)`

Reads with stdlib `csv.DictReader` — the header row becomes the dict keys, and
every cell arrives as a string (declare `types` for anything numeric). The
default `utf-8-sig` encoding transparently strips the BOM that Excel adds to
CSV exports. Extra `csv.DictReader` format params can be passed through
`iter_csv` if you feed `build_from_rows` yourself.

### `build_from_excel(path, mapping, *, sheet=None, ...)`

Requires the optional dependency: `pip install recon-graphrag[excel]`
(openpyxl). The first row of the worksheet is the header; `sheet` selects a
worksheet by name (default: the active one). Cells keep their Excel types —
numbers arrive as `int`/`float` and dates as `datetime`, which the type
coercion passes through unchanged.

### `build_from_sql(connection, query, mapping, *, params=None, ...)`

Works with **any DB-API 2.0 connection** — recon-graphrag adds no database
dependency; you bring the driver you already use:

| Database | Driver | Placeholder style |
| --- | --- | --- |
| SQLite | `sqlite3` (stdlib) | `?` |
| PostgreSQL | `psycopg` / `psycopg2` | `%s` |
| MySQL / MariaDB | `PyMySQL` / `mysqlclient` | `%s` |
| SQL Server | `pyodbc` | `?` |
| Oracle | `oracledb` | `:name` |
| DuckDB, Snowflake, BigQuery, … | official connectors | driver-specific |

Column names come from `cursor.description`; `params` is passed straight to
`cursor.execute`, so the query placeholder style follows **your driver**:

```python
import psycopg
conn = psycopg.connect("dbname=shop")
await pipeline.build_from_sql(
    conn,
    "SELECT * FROM orders WHERE year = %s",
    mapping,
    params=(2026,),
    metadata={"source": "orders-2026"},   # recommended: stable document id
)
```

### `build_from_rows(rows, mapping, ...)`

The core method — takes any iterable of dicts. Use it for JSON records, API
responses, ORM results, pandas `df.to_dict("records")`, or anything else:

```python
await pipeline.build_from_rows(list_of_dicts, mapping, metadata={"source": "tickets"})
```

**Document identity:** the CSV/Excel wrappers set `metadata["source"]` to the
file name, which produces a stable, collision-resistant source hash as the
document ID. Re-runs replace that document's contribution, not just its Document
node. For `build_from_sql`/`build_from_rows`, pass `metadata={"source": ...}` or
an explicit `metadata["document_id"]` yourself — otherwise the ID falls back to
a content hash and changes whenever the data does.

---

## The RowMapping reference

### `ColumnEntity` — one entity per row

```python
ColumnEntity(
    label="Product",                      # node label
    identity_column="product_name",       # cell value = entity name
    properties={"unit_price": "unit_price"},  # {column: property_name}
    types={"unit_price": "FLOAT"},        # {property_name: type}, default STRING
    description_template="{product_name} at {unit_price} USD",  # optional
)
```

- **`identity_column`** — the cell value becomes the entity's `name`. Rows
  where it is empty/None simply don't produce this entity (and skip any
  relationship touching it); no error.
- **`identity_property`** — defaults to `"name"` (a candidate, not proof of identity).
  Set a non-name property such as `"record_key"` when `identity_column` contains
  trusted IDs. The identity cell is also stored under this property; the same
  trusted ID merges observations and different trusted IDs stay separate in memory.
  `types` applies to this property even when it is not explicitly column-mapped.
  The property name must be non-empty and, if explicitly column-mapped, must map
  from `identity_column` rather than a different column.
- **`properties`** — maps *columns* to *property names* (rename freely). Empty
  cells omit the property rather than writing nulls.
- **`types`** — per property name: `STRING` (default), `INTEGER`, `FLOAT`,
  `BOOLEAN`, `DATE`, `DATETIME`, `LIST`. See coercion below.
- **`description_template`** — a `str.format` template over the **full row**
  (any column, not just mapped ones). When omitted, a compact summary of the
  mapped properties is generated: `"Product Widget — unit_price: 9.99"`.
  Either way, direct entities are never description-less — the description
  feeds entity embeddings (local/DRIFT search), description summarization, and
  the hybrid-resolution LLM judge.

### `RowRelationship` — an edge between two row entities

```python
RowRelationship("Company", "SUPPLIES", "Product", properties={"order_id": "order_id"})
```

Created for every row where **both** endpoint entities are present. The same
(source, type, target) appearing in many rows dedups into one edge whose
`observation_count`/`weight` grows — so community detection and retrieval see
frequency for free. `properties` maps columns onto the edge (strings, taken
as-is).

### `TextColumn` — free-text columns for LLM extraction

```python
TextColumn(column="notes", anchor_label="Product", relationship="MENTIONS")
```

- Rows where `column` is non-empty get one LLM extraction pass (rows without
  it cost zero LLM calls).
- The pass runs on the **whole row text**, not just the column, so
  `FROM_CHUNK` evidence and citations point at exactly what the LLM saw.
- Every extracted entity is linked from the row's `anchor_label` entity via a
  `relationship` edge (default `MENTIONS`). Omit `anchor_label` to extract
  without anchoring — extracted entities are still tied to the row through
  the shared chunk.
- Multiple `TextColumn`s are allowed; a row is extracted once, each anchor
  adds its own edges.

### `record_id_column` and `metadata_columns`

```python
RowMapping(..., record_id_column="order_id", metadata_columns=["region", "order_date"])
```

Each row becomes one `Chunk` whose text is readable `"column: value"` lines.
`record_id_column` puts the row's business key into
`chunk.metadata["record_id"]` (falls back to the row index), and
`metadata_columns` copies additional cells into chunk metadata under their
column names. Both surface later in search results through
`citation.metadata` — so an answer can cite `order O-1, region EU` instead of
an opaque chunk id.

Column names in `metadata_columns` must avoid the writer's reserved chunk
persistence fields, including `id`, `text`, and `index`. A collision raises
`ValueError` before that artifact is written — rename the column or drop it from
`metadata_columns`. The separate `metadata=` argument to `build_from_rows`
follows document-metadata rules, not chunk-metadata rules. The full list,
per-record rules, and recovery steps are in
[Reserved persistence fields](09-workflows.md#reserved-persistence-fields).

---

## Type coercion

| Type | Accepted input | Stored as |
| --- | --- | --- |
| `STRING` | anything | `str(value)` |
| `INTEGER` | `"42"`, `42` | `int` |
| `FLOAT` | `"9.99"`, `9.99` | `float` |
| `BOOLEAN` | `true/false`, `yes/no`, `y/n`, `t/f`, `1/0` (any case), real bools | `bool` |
| `DATE` | ISO strings (`"2026-07-09"`), `date`/`datetime` objects | ISO date string |
| `DATETIME` | ISO strings, `datetime` objects | ISO datetime string |
| `LIST` | `"a, b, c"` (comma-split) or an actual list | `list` |

Already-typed values from Excel/database drivers pass through unchanged.
`None` and empty strings omit the property. A value that cannot be coerced
**fails loudly** with the column and row in the message — structured input is
a trust boundary, silent nulls hide data problems:

```text
ValueError: cannot coerce column 'price' value 'oops' to FLOAT (row 3)
```

---

## What the graph looks like

One row, both modes. Input:

```python
{"order_id": "O-1", "supplier_name": "Acme Corp", "product_name": "Widget",
 "unit_price": "9.99", "notes": "Widget uses the AX-9 chip made by ChipCo."}
```

```text
(:Document {id:"<stable source-derived ID>"})
   ▲ PART_OF
(:Chunk {id:"...:row:0", record_id:"O-1", text:"order_id: O-1\nsupplier_name: Acme Corp\n..."})
   │ FROM_CHUNK                          │ FROM_CHUNK (also to every LLM entity)
   ▼                                     ▼
(:Company {name:"Acme Corp"}) ─SUPPLIES→ (:Product {name:"Widget", unit_price:9.99})
                                             │ MENTIONS (anchor edge)      │ MENTIONS
                                             ▼                             ▼
                              (:Component {name:"AX-9 chip"}) ─MADE_BY→ (:Company {name:"ChipCo"})
```

The direct and LLM-extracted subgraphs are connected three ways:

1. **Shared row chunk** — everything from one row shares `FROM_CHUNK`
   provenance, which is what local search and citations traverse.
2. **Anchor edges** — explicit domain edges from the anchor entity.
3. **Resolution merge** — see below.

---

## Identity, merging, and entity resolution

Direct entities use **label-scoped canonical keys**: `"Product:Widget"`, not
`"Widget"`. Consequences:

- The same name under two labels stays two nodes — a `Company` named
  *Mercury* never collapses into a `Person` named *Mercury*.
- Direct entities and LLM-extracted entities are **merged by entity
  resolution during finalize**, not by id equality. Resolution is
  label-scoped in every strategy, so an LLM-extracted `Product` named
  *Widget* merges with the direct `Product:Widget` under the default
  `normalized` strategy; near-matches (*Widget Pro* vs *Widget-PRO*) go
  through `fuzzy`/`hybrid` review.
- With `perform_entity_resolution=False`, direct and LLM entities with the
  same name stay separate nodes (still connected via shared chunk and anchor
  edges).

### In-memory conflicts before writing

Repeated name-key rows with compatible/complementary properties still coalesce.
If populated scalar properties disagree, **all observations in that candidate
group become separate provisional entities** before writing. This is conservative
candidate blocking, not similarity scoring or a declaration that they are
certainly different people/items. Keys are deterministic for identical document,
chunks, and observation order; row-reorder invariance is not promised. Relationships,
claims, and chunk evidence use the selected variant endpoints. `None`, `""`, `[]`,
and `{}` are missing values; `0` and `False` are populated. Names, titles, and
consolidated descriptions do not by themselves block name candidates; explicitly
configured `entity_resolution_conflict_properties` can include names/titles.

For confirmed identity (a non-name `identity_property`, or a manually supplied
explicit canonical key), scalar properties retain the first populated value.
Conflicting observations and their sources are preserved separately, not converted
into scalar-to-list database properties. Only schema-declared `LIST` properties
are unioned, in first-seen order. Name-only LIST differences are **not** automatically
unioned or sufficient alone to split candidates: the first list is retained and
both observations are reported for review. For row builds with an
`extraction_schema` override, the mapping's property types and identity rule take
precedence for assembly of direct labels; LLM extraction validation still uses
the override unchanged.

`GraphDocument.assembly_review_groups` is available to writers **before** the write;
the same JSON-safe list is returned as
`result["extraction"]["assembly_review_groups"]` (also for text/page builds). Each
entry contains `domain_label`, `reason="property_conflict"`, `decision`,
`entity_ids`, `canonical_keys`, `conflicts`, and empty matching scores (`None`).
Each conflict contains a `property` and source-bearing `observations` with `value`
and `source` (document/chunk/page/character references). Decisions are `blocked`
for split candidates, `identity_retained` for confirmed identity scalar conflicts,
and `review_required` for name-only LIST differences. Reports round-trip in graph
JSON artifacts; older artifacts default to an empty report. They are **not persisted
as database node properties** or an automatic review queue; retain the artifact or
result if you need the conflicting values after the call. The report does not
stop the write.

**Downstream limitation:** default normalized database resolution during finalize
can still merge provisional same-name variants (and different trusted IDs).
Configure `entity_resolution_conflict_properties` to block conflicting IDs/scalars,
or use `finalize=False` on row/document builds to inspect first. Disabling
`perform_entity_resolution` prevents this later stage. This change does not alter
database resolver policy or repair earlier merges.

Two knobs worth setting for structured-heavy graphs:

```python
pipeline = GraphBuilderPipeline(
    graph_store=store, llm=llm, embedder=embedder,
    entity_resolution_strategy="hybrid",
    # Typed columns are strong disambiguation signal for the LLM judge:
    entity_resolution_context_properties=mapping.resolution_context(),
)
```

- **`mapping.resolution_context()`** returns `{label: [property names]}`
  ready for `entity_resolution_context_properties`, so hybrid LLM review sees
  `sku`/`unit_price`/etc. next to names and descriptions.
- **Code-like identities** (`"A-100"` vs `"A100"`) can over-merge under
  `normalized`/`fuzzy`. If your identity columns hold codes rather than
  names, use `entity_resolution_strategy="exact"`.

### The extraction schema for text columns

The LLM pass is validated against a schema. By default that is
`mapping.to_schema()` — the mapping's own labels, typed properties, and
relationship types (including each `TextColumn.relationship`), with no
patterns so extraction isn't over-constrained. Keeping the table's labels is
what makes resolution merge direct and extracted entities.

Pass `extraction_schema` when the prose mentions things the table doesn't
model:

```python
notes_schema = build_schema(
    node_types=[
        {"label": "Product"},                     # keep table labels so merging works
        {"label": "Company"},
        {"label": "Component", "properties": ["name"]},
    ],
    relationship_types=[{"label": "MENTIONS"}, {"label": "MADE_BY"}],
    patterns=[],
)
await pipeline.build_from_csv("orders.csv", mapping, extraction_schema=notes_schema)
```

---

## Zero-LLM ingestion

A structured ingest makes **no LLM calls at all** when:

- the mapping has no `text_columns` (no extraction pass),
- `entity_resolution_strategy` is `exact`, `normalized`, or `fuzzy` (all
  deterministic; only `hybrid` uses the LLM), and
- `summarize_descriptions=False` (otherwise finalize may summarize entities
  that accumulated multiple distinct descriptions).

```python
pipeline = GraphBuilderPipeline(
    graph_store=store, llm=llm, embedder=embedder,
    summarize_descriptions=False,
)
await pipeline.build_from_csv("orders.csv", mapping_without_text_columns)
```

Entity embedding still runs (it uses the embedder, not the LLM), so the graph
remains searchable with local/DRIFT search.

---

## Method parameters

| Parameter | Methods | Description |
| --- | --- | --- |
| `mapping` | all | The `RowMapping`. Validated up front; a bad mapping raises before any work. |
| `metadata` | all | Document-level metadata. Set `source` for a stable document id. CSV/Excel wrappers default it to the file name. |
| `extraction_schema` | all | Schema for the text-column LLM pass. Defaults to `mapping.to_schema()`. |
| `extraction_concurrency` | all | Parallel LLM passes over rows (semaphore + progress bar). Defaults to `5`. |
| `max_gleanings` | all | Follow-up extraction loops per row, same as the text pipeline. Defaults to `1`. |
| `finalize` | all | Run resolution/summarization/embedding after the write. Defaults to `True`; set `False` when batching several tables and finalize once (e.g. via a final `build_from_rows([], ...)` or the last table). |
| `encoding` | `build_from_csv` | File encoding. Defaults to `"utf-8-sig"`. |
| `sheet` | `build_from_excel` | Worksheet name. Defaults to the active sheet. |
| `params` | `build_from_sql` | Query parameters, passed to `cursor.execute` as-is. |

The result dict matches the text pipeline:
`{"extraction": {"document_id", "chunks", "llm_rows", "write_stats"}, "validation": {...}}`.

---

## Worked example: supplier orders

End to end, from a CSV to search results with row-level citations. The
runnable version is [`examples/structured_ingest.py`](../examples/structured_ingest.py).

### 1. The data — `orders.csv`

| order_id | supplier_name | product_name | unit_price | region | notes |
| --- | --- | --- | --- | --- | --- |
| O-1 | Acme Corp | Widget | 9.99 | EU | The Widget uses the AX-9 chip manufactured by ChipCo in Taiwan. |
| O-2 | Acme Corp | Gadget | 24.50 | US | |
| O-3 | Globex | Widget | 9.49 | EU | Competing Widget batch; ChipCo supplies the same AX-9 chip. |

Three rows, two suppliers, two products — and prose in `notes` that mentions
things the table has no columns for (a component, a manufacturer, a country).

### 2. Setup

```python
from neo4j import GraphDatabase
from recon_graphrag import (
    GraphBuilderPipeline, Neo4jGraphStore, IndexConfig,
    RowMapping, ColumnEntity, RowRelationship, TextColumn,
    build_schema, create_llm, create_embedder,
)

driver = GraphDatabase.driver("bolt://localhost:7688", auth=("neo4j", "password"))
store = Neo4jGraphStore(driver)
store.create_indexes(IndexConfig(), embedding_dim=1536)

llm = create_llm("openai", model_name="gpt-4o", api_key="sk-...")
embedder = create_embedder("openai", model="text-embedding-3-small", api_key="sk-...")
```

### 3. The mapping

```python
mapping = RowMapping(
    entities=[
        ColumnEntity(label="Company", identity_column="supplier_name"),
        ColumnEntity(
            label="Product",
            identity_column="product_name",
            properties={"unit_price": "unit_price"},
            types={"unit_price": "FLOAT"},
            description_template="{product_name}, supplied by {supplier_name} at {unit_price} USD",
        ),
    ],
    relationships=[RowRelationship("Company", "SUPPLIES", "Product")],
    text_columns=[TextColumn("notes", anchor_label="Product")],
    record_id_column="order_id",
    metadata_columns=["region"],
)
```

The `notes` prose mentions a **Component** ("AX-9 chip") — a label the table
doesn't model. The default extraction schema (`mapping.to_schema()`) only
knows `Company` and `Product`, so the validator would **drop** the AX-9 chip.
To capture it, pass a richer `extraction_schema` (keep the table's labels so
resolution can still merge direct and extracted entities):

```python
notes_schema = build_schema(
    node_types=[
        {"label": "Company"},                              # table label — keep
        {"label": "Product"},                              # table label — keep
        {"label": "Component", "description": "A part used in a product"},
    ],
    relationship_types=[
        {"label": "MENTIONS"},
        {"label": "MADE_BY", "description": "Component is manufactured by a company"},
    ],
    patterns=[],
)
```

### 4. Ingest

```python
pipeline = GraphBuilderPipeline(
    graph_store=store, llm=llm, embedder=embedder,
    entity_resolution_context_properties=mapping.resolution_context(),
)
result = await pipeline.build_from_csv("orders.csv", mapping, extraction_schema=notes_schema)
print(result["extraction"])
# {'document_id': '<stable source-derived ID>', 'chunks': 3, 'llm_rows': 2, 'write_stats': {...}}
```

`chunks: 3` — one chunk per row. `llm_rows: 2` — only O-1 and O-3 have notes;
O-2 cost zero LLM calls.

### 5. What lands in the graph

Direct mapping (deterministic, identical on every run):

- `(:Company {name: "Acme Corp"})`, `(:Company {name: "Globex"})`
- `(:Product {name: "Widget", unit_price: 9.99})`, `(:Product {name: "Gadget", unit_price: 24.5})`
  — Widget's description consolidates both rows' templates:
  `"Widget, supplied by Acme Corp at 9.99 USD; Widget, supplied by Globex at 9.49 USD"`
- `SUPPLIES` edges: Acme→Widget, Acme→Gadget, Globex→Widget
- 3 `(:Chunk)` nodes carrying `record_id` (`O-1`/`O-2`/`O-3`) and `region`,
  each `FROM_CHUNK`-linked to its row's entities

LLM pass over O-1 and O-3 (typical extraction):

- `(:Component {name: "AX-9 chip"})`, `(:Company {name: "ChipCo"})`
- `(:Component)-[:MADE_BY]->(:Company {name: "ChipCo"})`
- Anchor edges: `(:Product {name:"Widget"})-[:MENTIONS]->` AX-9 chip and ChipCo
- The LLM also re-extracts "Widget" from the prose → a name-keyed `Product`
  node that **entity resolution merges** into the direct `Product:Widget`
  during finalize (same label + same normalized name)

### 6. Inspect it

```cypher
// Row provenance: which order is an entity evidenced by?
MATCH (c:Chunk)-[:FROM_CHUNK]->(e:__Entity__ {name: "AX-9 chip"})
RETURN c.record_id, c.region, c.text;
// → "O-1", "EU", "order_id: O-1\nsupplier_name: Acme Corp\n..."

// The bridge between structured and extracted knowledge:
MATCH (co:Company)-[:SUPPLIES]->(p:Product)-[:MENTIONS]->(x)
RETURN co.name, p.name, labels(x), x.name;
// → Acme Corp | Widget | [__Entity__, Component] | AX-9 chip
//   Acme Corp | Widget | [__Entity__, Company]   | ChipCo
//   Globex    | Widget | ...

// Frequency for free: Widget appears in two orders
MATCH (:Company)-[r:SUPPLIES]->(:Product {name: "Widget"})
RETURN r.observation_count;   // 1 per supplier; chunk evidence lists both rows
```

### 7. Search with row citations

```python
from recon_graphrag import LocalSearchRetriever

search = LocalSearchRetriever(store, llm, embedder)
result = await search.search("Who makes the chip used in the Widget?")
print(result.answer)
# "The Widget uses the AX-9 chip, which is manufactured by ChipCo..."
for citation in result.citations:
    print(citation.metadata)
# {'record_id': 'O-1', 'region': 'EU', ...}  ← the answer cites the exact order row
```

This is the payoff of `record_id_column`/`metadata_columns`: answers cite
business keys, not opaque chunk ids.

### 8. Re-running

Run the same ingest again after the CSV changes: the exact file-name `source`
keeps `document_id` stable. The shared writer replaces that document's
contribution, including removed rows/chunks, evidence, claims, descriptions and
relationship observations; other documents' contributions survive. Replays do
not inflate observation counts. Properties still use the existing update rules;
this is not historical version storage or a document-wide transaction.

Direct column mappings use label-scoped candidate keys by default; declare a
non-name `ColumnEntity.identity_property` for trusted identifiers. By contrast,
LLM-extracted local IDs such as `p1` are chunk-local references, not domain
identities: name-only observations stay separate until conflict-aware resolution.
Configure conflict properties (for example, a birth date or registration number)
for normalized resolution; matching names alone cannot prove identity. Existing
incorrect merges and legacy descriptions/strength without ownership require a
source rebuild, not an ownership guess. See [replacement and migration](05-pipelines.md#document-replacement-and-legacy-graphs).

---

## Scale notes

- Each row becomes one `Chunk` node: a 1M-row table means 1M chunks, and — if
  every row has notes — up to 1M LLM calls. For large tables, ingest the
  direct mapping only, or filter to the rows whose text actually matters.
- Rows are materialized in memory (`list(rows)`) before processing. For large
  tables, each slice must have its own stable `source` or `document_id`; reusing
  one ID for successive slices replaces earlier slices. `finalize=False` delays
  resolution/summarization/embedding only — it does not turn replacement into
  append. Submit the complete current contents of each identified slice when
  updating it.
- Finish `store.create_indexes()` before concurrent ingestion. Uniqueness
  constraints prevent duplicate identities, but concurrent replacement/property
  updates are not serialized and a competing writer can receive a backend
  conflict. No automatic retry is provided.
