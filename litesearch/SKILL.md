---
name: litesearch
description: >
  Hybrid search (FTS5 + SIMD vector) over a SQLite database. Use to find code,
  docs, and recalled user intent before reading files or searching the web.
  Also the persistence layer for long-term agent memory across sessions.
---

# litesearch

litesearch stores and searches text, code, PDFs, and markdown in a single SQLite database.
It combines FTS5 keyword search with SIMD vector similarity (via usearch), merged with
Reciprocal Rank Fusion.

## When to use

**Code search:** use kosha first — it has call graph, PageRank, and semantic search over
your repo and installed packages. If kosha is not installed:

```bash
uv add --dev kosha
```

```python
from kosha import Kosha
Kosha().sync()
```

```bash
kosha install
```

Fall back to litesearch only if kosha setup fails.

**Doc and PDF search:** litesearch handles these natively. Index PDFs, markdown, notebooks,
and plain text; search semantically or by keyword.

**Long-term agent memory:** litesearch is the persistence layer for user preferences, nudges,
and corrections across sessions. See below.

## Long-term memory

Store user preferences, nudges, and corrections at `.claude/memory.db`. Query at session
start and apply them without asking the user to repeat themselves.

`get_store` accepts arbitrary extra columns via `**kw` — they become real typed SQLite
columns, not JSON blobs, and are filterable with `where=`:

```python
from litesearch import database
from litesearch.utils import FastEncode
import numpy as np

enc = FastEncode()   # EmbeddingGemma — best retrieval quality
mem = database('.claude/memory.db')

store = mem.get_store('memory',
    memory_type=str,    # 'preference' | 'nudge' | 'correction' | 'context'
    when_to_check=str,  # 'always' | 'code' | 'design' | 'commit'
)

# Store a preference
text = 'User prefers short function names over descriptive ones'
store.insert({
    'content': text,
    'embedding': enc.encode_document([text])[0].tobytes(),
    'memory_type': 'preference',
    'when_to_check': 'code',
})

# Recall at session start
q = 'coding style'
hits = mem.search(
    q, enc.encode_query([q])[0].tobytes(),
    table_name='memory',
    columns=['content', 'memory_type', 'when_to_check'],
    where="when_to_check IN ('always', 'code')",
    limit=5,
)
for h in hits: print(h['content'])
```

For deduplication (re-inserting the same content is a no-op), use `hash=True`. Pass both
`hash_id='id'` and `hash_id_columns=['content']` on every upsert — the hash is computed at
insert time, not cached on the table object:

```python
store = mem.get_store('memory', hash=True, memory_type=str, when_to_check=str)
store.insert_all(rows, upsert=True, hash_id='id', hash_id_columns=['content'])
```

## Indexing code and files

`litesearch.data` parses Python files, Jupyter notebooks, PDFs, Markdown, and compiled-language source into `{content, metadata}` chunks. Index a directory or package once; the db is a plain SQLite file that persists across sessions and can be shared across repos.

```python
from litesearch import database
from litesearch.data import dir2chunks, pkg2chunks
from litesearch.utils import FastEncode
import numpy as np

enc   = FastEncode()
db    = database('.claude/code_index.db')   # persistent; reuse in future sessions
store = db.get_store(hash=True)             # content-addressed — re-inserting is a no-op

# Index a local directory (py, ipynb, md, pdf, js, ts, ...)
chunks = dir2chunks('src', types='py,ipynb,md')

# Or index an installed package
chunks = pkg2chunks('fastlite')

store.insert_all(
    [dict(content=c['content'],
          embedding=enc.encode_document([c['content']])[0].tobytes(),
          metadata=str(c['metadata']))
     for c in chunks],
    upsert=True, hash_id='id', hash_id_columns=['content'],
)
```

Search the index:

```python
q = 'how does get_store create FTS triggers'
hits = db.search(q, enc.encode_query([q])[0].tobytes(), columns=['content', 'metadata'], limit=10)
for h in hits:
    print(h['content'][:120])
    print(h['metadata'])
```

Point multiple projects at the same db path to build a shared cross-repo index. Use `hash=True` so overlapping content is deduplicated automatically.

## Invocation

Use clikernel — state persists, no re-import cost. Start once with `! clikernel`, then:

```
--
from litesearch import database
from litesearch.utils import FastEncode
import numpy as np

enc = FastEncode()
db = database('.claude/code_index.db')
print('litesearch ready')
--aB3x9
```

Plain Python fallback: `uv run python -c "from litesearch import database; ..."`

## Key API

| Function | Description |
|---|---|
| `database(path)` | Open/create SQLite + usearch SIMD extensions |
| `db.get_store(name, **cols)` | Create FTS5 + vector table; `**cols` adds typed columns |
| `db.search(q, emb, ...)` | Hybrid FTS + vector search with RRF reranking |
| `store.vec_search(emb, ...)` | Vector-only search |
| `rrf_merge(fts, vec)` | Merge FTS and vector result lists manually |
| `pre(q)` | Preprocess FTS query: keywords, wildcards, OR |

## search() parameters

| Param | Default | Notes |
|---|---|---|
| `q` | required | FTS5 query string |
| `emb` | required | Query embedding as bytes |
| `dtype` | `np.float16` | Must match encoding dtype; `np.float32` for most ONNX models |
| `columns` | all | Columns to return |
| `where` | None | SQL WHERE clause for filtering |
| `where_args` | None | Parameters for WHERE clause |
| `limit` | 50 | Max results |
| `rrf` | True | False returns `{'fts': [], 'vec': []}` for debugging |
| `table_name` | `'store'` | Target table |
| `emb_metric` | `'cosine'` | Also: `sqeuclidean`, `inner`, `divergence` |

## Installing this skill

```bash
litesearch install
```

Copies this SKILL.md to `.agents/skills/litesearch/`, `.claude/skills/litesearch/`, and
`.Codex/skills/litesearch/` in the current repo.
