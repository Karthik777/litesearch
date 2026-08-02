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

## Knowledge graph (no LLM)

`litesearch.graph` builds an entity graph next to the store and adds it as a third RRF leg.
Code is parsed (AST gives exact `calls`/`imports` edges), prose is tagged (spaCy `noun_chunks`,
yake fallback), and edges between prose entities come from normalized-PMI co-occurrence.

```python
from litesearch import database, build_graph, resolve_entities, topic_nodes, spacy_pipe
from litesearch.data import dir2chunks
from litesearch.utils import FastEncode, doc_encoder

enc, db = FastEncode(), database('.claude/kg.db')
store = db.get_store(hash=True, ann=True)   # index chunks first, as above
chunks = dir2chunks('src', types='py')

build_graph(db, chunks, prose=False, emb_fn=doc_encoder(enc))  # prose=True + nlp= for docs
resolve_entities(db)     # merge surface variants (skips exact AST symbols)
topic_nodes(db)          # cluster the HNSW index into labelled topic nodes

hits = db.graph_search(q, enc.encode_query([q])[0].tobytes(), columns=['content'], limit=10)
```

Use `graph_search` when the answer is *related to* the query rather than worded like it — it
reaches chunks connected only by a shared entity. Use plain `db.search` for direct lookup.

On **code** corpora pass `graph_w=0`: call edges link different levels of abstraction, not
substitutable answers, so the graph leg adds little to ranking. Use the code graph for context
assembly instead — after retrieval, pull the callers/callees of the top hits as supporting context.

For prose, pass `nlp=spacy_pipe(terms=code_symbols)` — the `EntityRuler` then links prose
mentions of your functions to the code nodes. Needs `pip install litesearch[graph]` and
`python -m spacy download en_core_web_sm`; without it, extraction falls back to yake.

## Document structure (`litesearch.tree`)

For books, reports, papers and doc sites — anything where "which chapter" is a better answer than
"which 400 characters". Every document gets a node tree at ingest time (markdown headings →
chapter lines → page windows), every chunk is linked to a node, and search results roll up.

```python
db = database('library.db')
db.get_tree('store')                                  # docs + nodes tables beside the chunk store
db.add_file('books/report.pdf', emb_fn=doc_encoder(enc))
db.add_dir('books/', emb_fn=doc_encoder(enc))         # pdf, md, txt, rst, ipynb

db.doc_search(q, qv, limit=5)      # hybrid search; each hit carries a breadcrumb + node_id
db.sections(q, qv, limit=3)        # ranked *sections*, each with snippets and a read() handle
db.toc('report', max_depth=2)      # the tree — no embeddings computed
db.read('a1b2c3d4#12')             # one whole section + its children
```

Two rules of thumb:

- Use `sections()` when the question is about a topic ("how is X timed and interpreted") and
  `doc_search()` when it is about a fact. `sections` groups hits by node and sums their RRF mass,
  so five weak hits in one chapter outrank one strong hit in an unrelated appendix.
- Use `toc()` + `read()` and skip retrieval entirely when the agent already knows where to look
  ("summarise chapter 3", template filling). That path costs no embedding at all.

`add_doc(pages, title, ...)` is the lower-level entry point when you have text in hand rather than
a file. `summarize=` and `chunker=` are callables — an LLM summariser per node is the highest-value
place to spend tokens, because `toc()` is what an agent reads when deciding where to go.

**Source code does not belong here.** Its tree is module › class › function and comes from the
AST, not from headings — that is `kosha`.

## Neighbours, clusters and peers

Three questions about *shape* rather than about a query string. All three need `ann=True` on the
store and cost no model call — they reuse the vectors usearch already holds.

```python
store.ann_neighbors(rowid, limit=15, columns=['content'])   # what else looks like this row
res = store.clusters(min_size=3, columns=['content'])       # a map of the whole corpus
[(c.size, c.label) for c in res.clusters]                   # labels are c-TF-IDF, not an LLM
store.peers(rowid, limit=25, columns=['content'])           # the family this row belongs to
```

`clusters()` and `peers()` return `AttrDict(…, method, note)`. **Show `note` to the user.** It
says `usearch` (HNSW cut) or `knn` (greedy fallback, used when the index is too small to cut), and
on an empty result it says why — an index with no vectors, a store with no ANN registration, or a
row that landed in a group of one. A clustering that silently returns `[]` is indistinguishable
from a broken index.

Reach for `peers` over `ann_neighbors` when the question is "where else did we already do this?".
k-NN always returns `limit` rows whether or not they are related; a cluster returns a family and
can be honest about its size.

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
| `database(path)` | Open/create SQLite + usearch SIMD extensions + apsw FTS5 tokenizers |
| `db.get_store(name, **cols)` | Create FTS5 + vector table; `**cols` adds typed columns |
| `db.search(q, emb, ...)` | Hybrid FTS + vector search with RRF reranking |
| `store.vec_search(emb, ...)` | Vector-only search |
| `store.ann_neighbors(rowid, ...)` | Nearest rows to an already-indexed row (no re-embedding) |
| `store.clusters(...)` | Labelled clusters over the whole store (`.clusters`, `.method`, `.note`) |
| `store.peers(rowid, ...)` | The cluster a row belongs to; degrades to k-NN and says so |
| `db.get_tree(store)` | Add docs/nodes tables + node-aware columns to a chunk store |
| `db.add_file(path, ...)` / `db.add_dir(dir, ...)` | Ingest documents: tree → node-linked chunks → embed |
| `db.doc_search(q, emb, ...)` | Hybrid search with adaptive RRF, span merging, breadcrumbs |
| `db.sections(q, emb, ...)` | Ranked sections with snippets and `read()` handles |
| `db.toc(doc)` / `db.read(node_id)` | Navigate structure with no embeddings at all |
| `rrf_merge(fts, vec)` | Merge FTS and vector result lists manually |
| `pre(q)` | Preprocess FTS query: keywords, wildcards, OR |
| `build_graph(db, chunks, ...)` | Extract entities/mentions/edges into the graph tables |
| `db.graph_search(q, emb, ...)` | Hybrid search + personalized-PageRank graph leg |
| `resolve_entities(db)` | Merge duplicate entities (ANN + lexical guard) |
| `topic_nodes(db)` | Cluster the ANN index into labelled topic nodes |

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
