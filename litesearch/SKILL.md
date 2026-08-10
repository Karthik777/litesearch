---
name: litesearch
description: >
  Hybrid search (FTS5 + SIMD vector) over a SQLite database. Use to find code,
  docs and PDFs before reading files or searching the web.
---

# litesearch

Hybrid search — FTS5 keyword + SIMD vector (usearch), merged with Reciprocal Rank Fusion — over
text, code, PDFs and markdown in one SQLite file.

**Read `README.md` first.** It covers setup, the full API and the measured tradeoffs. This file is
the short version for getting work done.

## When to use

- **Code search** → use `kosha` first (call graph, PageRank, semantic search). litesearch is the
  fallback if kosha setup fails.
- **Docs and PDFs** → litesearch, natively.

## Quickstart

Default to a **static embedder**: ~1,700x cheaper to index than a transformer and within 0.007 MRR
of the best encoder measured.

```python
from litesearch import database
from litesearch.utils import static_retrieval_embedder, doc_encoder, query_encoder
from litesearch.data import dir2chunks

enc = static_retrieval_embedder()
emb, qemb = doc_encoder(enc), query_encoder(enc)
db = database('.claude/index.db')           # persistent, reusable across sessions
store = db.get_store(hash=True, ann=True)   # content-addressed: re-inserting is a no-op

chunks = dir2chunks('src', types='py,ipynb,md')
store.insert_all([dict(content=c['content'], metadata=str(c['metadata']),
                       embedding=e.tobytes())
                  for c, e in zip(chunks, emb([c['content'] for c in chunks]))],
                 upsert=True, hash_id='id', hash_id_columns=['content'])
store.rebuild_index()

q = 'how does get_store create FTS triggers'
hits = db.search(q, qemb([q])[0].tobytes(), columns=['content', 'metadata'], limit=10)
```

`dtype` must match what you inserted — `StaticModel.encode` gives float32, `FastEncode` float16.
Mismatch silently returns rowid order, not distances.

Point several projects at one db path for a shared cross-repo index.

## Documents with structure (`litesearch.tree`)

For books, reports, papers, doc sites — where "which chapter" beats "which 400 characters".

```python
db.get_tree('store')
db.add_dir('books/', emb_fn=emb)        # pdf, md, txt, rst, ipynb, xml, htm
db.doc_search(q, qv, limit=5)           # hits carry a breadcrumb + node_id
db.sections(q, qv, limit=3)             # ranked sections, each with a read() handle
db.toc('report'); db.read('a1b2c3d4#12')
```

`sections()` for topics, `doc_search()` for facts. `toc()` + `read()` costs no embedding at all —
use it when you already know where to look. The tree layer is a wash for *ranking*; use it for
navigation. Source code does not belong here — that is `kosha`.

## Knowledge graph (no LLM)

Entity graph beside the store, added as a third RRF leg. AST gives exact `calls`/`imports` for code;
prose entities come from an extractor plus normalized-PMI co-occurrence.

```python
from litesearch import build_graph, resolve_entities, topic_nodes
build_graph(db, chunks, prose=False, emb_fn=emb)   # prose=True for docs
resolve_entities(db); topic_nodes(db)
```

**Default to `graph_w=0`.** The graph leg lost on all 351 known-item queries across three prose
corpora, at 55–107ms against 8–13ms. Build it for **context assembly** — pulling callers/callees or
co-mentioned entities of the top hits — not for ranking.

Prose extraction is yake, which is a *keyphrase* extractor doing an *entity* extractor's job: it
fragments one concept across nodes (`member state` / `member states` / `member` / `states`). On a
novel it left only 27% of chunk pairs naming the same character sharing an entity, against 97% for
spaCy NER and 89% for a proper-noun regex costing 1/78th of spaCy. Swap it:

```python
build_graph(db, chunks, terms_fn=my_extractor)     # (text, topk) -> terms
```

Use `litesearch.sanskrit.sanskrit_terms()` for Devanagari. Extraction runs serially when `terms_fn`
is set (extractors usually close over unpicklable state).

## Sanskrit (`litesearch.sanskrit`)

Two `Profile`s register at import, so `add_file` needs no arguments — it picks the reader (GRETIL,
TEI, vedicreader XML, DCS), the `verse` tree mode, `VerseChunker` and metrical facets itself.

```python
db.add_file('gretil/manu.htm', emb_fn=emb)
db.by_meter(meter='mandākrāntā')          # filter over facets, not a ranking
```

Cross-script search is on for **every** store: the `sanskrit` FTS5 tokenizer colocates an ASCII fold
of each token, so `श्रीमाता`, `śrīmātā` and `srimata` reach the same row. Cost: a store built with
this chain cannot be opened by a connection that has not registered the tokenizer, including plain
`sqlite3`.

## Two settings that matter more than the model

Both are defaults; both are free.

- **~512-character chunks.** Page-sized chunks cost 0.06–0.14 section MRR, the largest single effect
  measured, and overflow a 512-token encoder so the tail is embedded by nothing.
- **`fts_pre=True`.** Worth +0.09 to +0.33 on reworded queries. Set `False` if your users paste exact
  phrases or terms of art (+0.02 to +0.24 on verbatim).

Spending latency? `reranking=True` (flashrank, ~40ms) is worth +0.03 to +0.08. A 300M transformer
over a 32M static embedder was worth +0.007 for 1,700x the indexing compute.

## Key API

| Function | Description |
|---|---|
| `database(path)` | Open/create SQLite + usearch + FTS5 tokenizers |
| `db.get_store(name, **cols)` | FTS5 + vector table; `**cols` adds typed, filterable columns |
| `db.search(q, emb, ...)` | Hybrid FTS + vector with RRF |
| `store.ann_neighbors(rowid)` | Nearest rows to an indexed row, no re-embedding |
| `store.clusters()` / `store.peers(rowid)` | Labelled clusters; the cluster a row belongs to |
| `db.get_tree(store)` | Add docs/nodes tables to a chunk store |
| `db.add_file(path)` / `db.add_dir(dir)` | Ingest: tree → node-linked chunks → embed |
| `db.doc_search(q, emb)` / `db.sections(q, emb)` | Search with breadcrumbs / ranked sections |
| `db.toc(doc)` / `db.read(node_id)` | Navigate structure, no embeddings |
| `build_graph(db, chunks, terms_fn=)` | Entities/mentions/edges; swap the prose extractor |
| `resolve_entities(db)` / `topic_nodes(db)` | Merge duplicates; labelled topic nodes |
| `db.graph_search(q, emb)` | Hybrid + personalized-PageRank leg |

**Graph leg: opt-in, and query-dependent.** `db.context()` defaults to `graph=False`.
Use hybrid (`db.search`) for **known-item** queries — the words you search for appear in the
passage you want; the graph leg costs 15-20 points of p_mrr there and 2-4x the latency, worse the
higher `graph_w` goes. Use `db.graph_search(..., graph_w=1.0)` for **bridge** queries — the answer
never uses your words and is reachable only through a shared entity; measured significantly better
in 7 of 9 paired comparisons, and better the higher `graph_w` goes. When unsure, start with hybrid.
`clusters` / `peers` / `topic_nodes` are for reading a corpus, not ranking it, and are unaffected.
| `pre(q)` | Preprocess an FTS query: keywords, wildcards, OR |

## Installing this skill

```bash
litesearch install
```

Copies this file to `.agents/`, `.claude/` and `.Codex/` skills dirs in the current repo.
