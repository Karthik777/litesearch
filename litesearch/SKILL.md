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

# EmbeddingGemma. Strongest of the encoders measured — by 0.007 over a 32M static embedder, at
# 1,700x the indexing cost (docs/rag_tiers.md). For memory, which is small and written once, that
# trade is fine; for a corpus you re-index, start with static_retrieval_embedder().
enc = FastEncode()
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

**Default to `graph_w=0`.** The graph leg was measured over 351 known-item queries on three prose
corpora (`docs/rag_tiers.md`) and it lost on all of them, monotonically in `graph_w`: 0.798 → 0.707 →
0.688 → 0.672 on papers as the weight rises through 0.25/0.5/1.0, and the same shape on legislation
and on books. It also costs 55–107ms against 8–13ms. The loss holds on the queries it was written for,
where the answer shares no vocabulary with the query at all.

That was already the standing advice for **code** corpora, where call edges link levels of
abstraction rather than substitutable answers. It turns out to hold for prose too: sharing an entity
is not evidence of answering a question.

Build the graph anyway if you want it for **context assembly** — after retrieval, pull the
callers/callees or co-mentioned entities of the top hits as supporting context. That use is not what
these numbers measure, and it is the one the graph is good for.

`topic_nodes()` adds nothing to *ranking*: deleting every topic entity from a built graph moved the
score by +0.003 to +0.006, i.e. very slightly better without them. Call it because `clusters()`-style
labels in the graph are useful to read, not to improve retrieval.

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
  `doc_search()` when it is about a fact. `sections` groups hits by node and scores each node by its
  *best* evidence (`score='max'`); summing the RRF mass of every hit in a node is a length prior in
  disguise and measured worse, so `sum` is opt-in.
- **The tree layer is a wash for retrieval.** Against flat chunks at the same encoder and chunk size
  it measured −0.05 to +0.01 across three genres — a small positive on legislation, negative on
  papers. Use it for `toc()`, `read()`, breadcrumbs and section-scoped answers, which is what it is
  for; do not expect ranking to improve. On papers, `with_heading=False` scored *better* than the
  default (0.817 vs 0.798): a heading path repeated across every chunk of a section dilutes more than
  it contextualises.
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

**Use `clusters()` for a map, and `ann_neighbors()` for neighbours.** Measured over three genres
(`docs/rag_tiers.md`), `clusters()` is good at what it claims: 0.99 coverage, 0.97 purity when asked
to separate three genres in one store, labels 120–400x more frequent inside their cluster than in the
corpus, all in under half a second.

`peers()` is the part to avoid. At matched group size its members share the anchor's section 2–3x
*less* often than the same number of nearest neighbours do — 0.046 against 0.124 on legislation,
0.273 against 0.453 on papers, 0.109 against 0.287 on books. The "a cluster returns a family, k-NN
returns a list" intuition is appealing and did not survive measurement, so prefer `ann_neighbors` for
"where else did we already do this?" and keep `peers` for when you specifically want a bounded group
rather than a ranked list.

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

## Two things that matter more than the model

Measured over 351 known-item queries on legislation, papers and 1820s–1920s books
(`docs/rag_tiers.md`). Both are free.

**Both of these are now defaults, as of 0.1.6.** Together they moved `db.search` on its own defaults
by +0.10 to +0.13 weighted section MRR. They are described here because the second one is a trade you
may want to turn off.

**1. Chunks are ~512 characters (`data.CHUNK_SIZE`).** They used to be 4096 — `FastChunker`'s own
default — and since `add_doc` calls the chunker once per node segment, nothing was ever split and a
chunk was a whole page. Page-sized against 512-character chunks costs **0.06 to 0.14** section MRR,
the largest single effect measured, and a 2,300-character chunk overflows a 512-token encoder so its
tail is embedded by nothing. `RecursiveChunker(chunk_size=512, tokenizer='character')` scores ~0.01
better again at ~5x the chunking cost: `db.add_doc(..., chunker=ck)`.

**2. The FTS leg goes through `pre()` (`search(fts_pre=True)`).** Quoting every token made FTS an
implicit AND over the whole query, so a reworded question matched nothing and the hybrid quietly
became vector-only. **This is a trade:**

| your users | setting | why |
|---|---|---|
| type questions and paraphrases | `fts_pre=True` (default) | worth +0.09 to +0.33 on reworded queries |
| paste exact phrases, cite terms of art | `fts_pre=False` | conjunctive quoting is worth +0.02 to +0.24 on verbatim queries |

The verbatim penalty is largest at coarse chunk sizes and small (≈0.016) at 512 characters, which is
why the two changes shipped together — at page size, `pre()` was a net *loss* on two genres of three.
Do not fuse deeper candidate lists hoping for more: 30 per leg instead of 10 measured *worse*
everywhere.

**And if you are going to spend latency, spend it on `reranking=True` rather than on a bigger
encoder.** A flashrank cross-encoder is worth +0.03 to +0.08 for ~40ms. Upgrading a 32M static
embedder to a 300M transformer was worth +0.007 for 1,700x the indexing compute, and lost on one
genre of three.

## search() parameters

| Param | Default | Notes |
|---|---|---|
| `q` | required | FTS5 query string |
| `emb` | required | Query embedding as bytes |
| `dtype` | `np.float16` | **Must match what you inserted.** `FastEncode` returns float16 (its own default), `model2vec`/`StaticModel.encode` returns float32. Get it wrong and every distance comes back exactly 0.0 with no error — the vector leg returns rowid order and hybrid search silently becomes FTS-only. Since 0.1.6 this warns; before it did not. |
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
