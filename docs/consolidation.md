# Consolidating search across litesearch, kosha, leela, lego and chitragupta

> Review + roadmap. The goal: **one generic retrieval core (litesearch), one code specialisation
> (kosha), and everything else a client.** LLM-assisted retrieval (Rishi) attaches at named seams
> rather than being woven through.

## 1. Where things stand

| repo | what it is | depends on |
|---|---|---|
| **litesearch** | SQLite + FTS5 + usearch. Stores, hybrid search, RRF, ONNX/model2vec encoders, PDF & code parsing, an entity-graph layer | — |
| **kosha** | Code intelligence: repo + installed-package index, AST call graph, PageRank, `where_to_add`, MCP server | litesearch |
| **leela** | An IDE (TUI + web) with an agent host, notebook kernels, a sandboxed FS, and a code-search pane | kosha |
| **lego** | FastHTML app framework (auth/blog/dash blocks) powering vedicreader.com; the `atlas` block is the embedding/cluster UI | litesearch |
| **chitragupta** | A document library: PageIndex-style trees, RAGLite-derived chunking, web/PDF acquisition | litesearch |

Two things are true of that list. First, **leela and chitragupta each built retrieval features
that are not about code or about astrology** — they are about retrieval, and they belong one layer
down. Second, **litesearch had most of the machinery already but not the API**: the clustering that
leela reimplemented was sitting inside `topic_nodes` as a private function; the late-chunking
encoder chitragupta wraps is `litesearch.utils.LateChunkFastEncode`.

## 2. What belongs where

Classification rule used throughout: *does this feature mention a function, a file path, or a
symbol?* If yes it is code-specific and goes to kosha. If it mentions a document, a page or a
heading, it is generic and goes to litesearch.

### Generic → litesearch

| capability | was in | status |
|---|---|---|
| k-NN from an already-indexed row (no re-embedding) | `leela/search.py: Search.similar` | **landed** — `store.ann_neighbors` |
| Clustering a corpus with usearch, kNN fallback | `leela/search.py: Search.clusters`, `litesearch/graph.py` (private) | **landed** — `store.clusters` |
| The cluster one row belongs to | `leela/search.py: Search.peers` | **landed** — `store.peers` |
| c-TF-IDF cluster labels | `lego/atlas/cluster.py` → already ported | in `litesearch.graph` |
| Reporting *which backend answered and why* | `leela/search.py: Search.backend/.note` | **landed** — `method`/`note` on every degradable call |
| Document trees (headings → chapters → page windows) | `chitragupta/tree.py` | **landed** — `litesearch.tree` |
| Chunks linked to tree nodes, contextual heading paths | `chitragupta/ingest.py` | **landed** — `db.add_doc` |
| `toc()` / `read()` vectorless navigation | `chitragupta/store.py` | **landed** — `db.toc` / `db.read` |
| Section-level rollup of evidence | `chitragupta: Library.research` | **landed** — `db.sections` |
| Adaptive RRF weighting | `chitragupta: Library._weights` | **landed** — `adaptive_weights` |
| Chunk span merging | `chitragupta: Library._spans` | **landed** — `merge_spans` |
| Cost-model chunking (sentences → chunklets → semantic) | `chitragupta/chunking.py` | **planned** (§4.1) |
| Agent write-back memory (`note()`) | `chitragupta/store.py` | **planned** (§4.2) |
| Evidence packs for templated docs | `chitragupta: Library.evidence` | **planned** (§4.2) |

### Code-specific → kosha

| capability | was in | status |
|---|---|---|
| `path:line` → chunk anchoring | `leela/search.py: Search._anchor` | **landed** — `k.anchor` |
| "What else looks like this function" | `leela/search.py: Search.similar` | **landed** — `k.similar` |
| "The family this function belongs to" | `leela/search.py: Search.peers` | **landed** — `k.peers` |
| A map of the codebase by shape | `leela/search.py: Search.clusters` | **landed** — `k.code_clusters` |
| Literal grep fallback over open folders | `leela/search.py: Search._scan` | **stays in leela** — it reads *unsaved buffers*, which no index can |
| File symbol outline | `leela/search.py: Search.symbols` | **stays in leela** — regex over the live buffer, not the index |
| Code trees (package › module › class › function) | — | **planned** (§4.3) |

### Stays put

- **leela's** `Search` class itself: after this refactor it is an adapter — pick a backend, format
  hits for the UI, and keep the grep path for unsaved edits. That is genuinely leela's job.
- **lego's** `dash` block: dashboard inference over arbitrary SQLite tables. Nothing to do with
  retrieval.
- **chitragupta's** acquisition layer (fossick: arXiv, YouTube, GitHub, OCR) and its astrology-
  shaped CLI. Acquisition is a different concern from indexing and has heavier dependencies.

### One caveat about `lego/atlas`

`lego/atlas/cluster.py` is referenced in `litesearch/graph.py` but **is not in the pushed lego
repo** — the `atlas` block appears to be local-only. This review covers it from that reference and
from leela's equivalent list-based UI. §5 is written against the API rather than against your
code, so it should hold either way, but the atlas plan is the one part here that has not been
checked against the source.

## 3. What landed in this pass

### litesearch

- `store.ann_vec(key)` / `store.ann_neighbors(key, limit, columns, ...)` — reuse the vector usearch
  already holds instead of re-embedding the row. One HNSW probe plus one `rowid IN (...)` fetch.
- `store.clusters(...)` → `AttrDict(clusters, method, note)`; each cluster is
  `AttrDict(centroid, size, label, member_keys, members)`, labelled by c-TF-IDF.
- `store.peers(key, ...)` → `AttrDict(hits, method, note)`, degrading to `ann_neighbors`.
- Clustering is cached per `(store, params, index size)` on the connection — `peers` used to mean
  re-clustering the entire index on every call.
- `litesearch.tree`: `db.get_tree`, `db.add_doc` / `add_file` / `add_dir`, `db.toc`, `db.read`,
  `db.breadcrumb`, `db.doc_search`, `db.sections`, `db.delete_doc`, plus `build_tree`,
  `heading_path`, `adaptive_weights`, `merge_spans`, `detect_mode`, `summarize_extractive`.
- Fix: `ann_search` returned up to `2 × limit` rows; it now trims to `limit` after filtering.

### kosha

- `k.anchor(path, line)` — resolves a cursor position to the covering chunk. Tries the absolute
  path, then a path *suffix*, so a caller holding a repo-relative path does not have to know how
  the index spelled it.
- `k.similar(path, line)`, `k.peers(path, line)`, `k.code_clusters()` — all with optional
  `graph=True` enrichment (callers, callees, pagerank).
- The same three exposed as MCP tools.

## 4. Roadmap

### 4.1 Chunking — port chitragupta's cost-model splitter

`litesearch.tree` currently chunks with `chunk_markdown` (chonkie `FastChunker`). chitragupta's
three-stage splitter is better on prose and is already written and tested:

1. markdown-aware sentence splitting (headings/lists/tables atomic, abbreviation guard),
2. DP grouping into ~3-statement chunklets that *start* on structural boundaries,
3. semantic merge, cutting where adjacent chunklet embeddings are least similar after projecting
   out the document's discourse vector.

**Plan.** Move `chitragupta/chunking.py` to `litesearch/nbs/07_chunking.ipynb` unchanged in
behaviour, expose `chunk_text(text, encode=None, max_size=1600)`, and make it the default
`chunker=` for `add_doc` when an `emb_fn` is present (stage 3 needs embeddings; without one it
degrades to stages 1–2). chitragupta then imports it rather than owning it.

### 4.2 Memory and evidence packs

`note()` (write-back memory searched alongside the corpus) and `evidence(questions)` (a cited
markdown pack for filling templates) are both generic and both small. They want one design
decision first: a separate `notes` store, or the same store with `kind='note'`? Same store is
simpler and makes `include_notes=False` a `where` clause; a separate store keeps corpus
statistics (and c-TF-IDF labels) clean. **Recommendation: same store, `kind` column**, because
the graph and cluster layers then see notes for free.

### 4.3 Code trees in kosha

The tree layer is generic over `(page, text)` pairs, but code's tree is not built from headings —
it is `package › module › class › function`, and kosha already has every piece of it (`mod_name`,
`lineno`, `type`, the call graph). Populating `nodes` from the AST rather than from headings makes
`toc('litesearch')` print a package outline and `read('…#12')` return a whole class, and makes
`sections()` roll code hits up to *modules* — which is the right unit for "which part of this
codebase handles X".

**Plan.** `kosha.tree`: build nodes from `mod_name` path segments at sync time, write them into
the same `nodes` schema, and let `db.toc`/`db.read`/`db.sections` work unchanged.

### 4.4 Ranking: unify kosha's `rank_results` with litesearch

`kosha.graph.rank_results` is strong and partly generic. Split it:

- **generic → litesearch**: multi-chunk file coherence boost, per-source saturation decay, the
  top-k selection loop. Rename around "source" instead of "file".
- **stays in kosha**: symbol-query detection, identifier stem boosts, test/compat/example path
  penalties, package soft-boosts.

### 4.5 Leela and chitragupta become clients

- `leela/search.py`: delete `_anchor`, `_store`, `_rows`, `_hit`, `_cluster`, `similar`,
  `clusters`, `peers` (~150 lines) and call `k.anchor/similar/peers/code_clusters`. Keep `_scan`,
  `symbols`, and the backend-selection policy. `MIN_CLUSTERABLE` disappears — litesearch's kNN
  fallback removes the reason for the threshold.
- `chitragupta`: `Library` keeps its CLI, its SKILL.md loop and its fossick acquisition, and
  delegates storage, trees and retrieval to `litesearch.tree`. Its `HashEncoder` is
  `litesearch.graph.hash_embed`; its `LateChunkEncoder` is `litesearch.utils.LateChunkFastEncode`.

### 4.6 Rishi and LLM-assisted retrieval

Every seam is already a callable, so nothing needs re-architecting — this is a list of the four
places where tokens actually buy something, in descending order of value:

1. **Node summaries** (`add_doc(summarize=…)`). `toc()` is what an agent reads when deciding
   where to look; a real summary per node is the difference between a usable table of contents
   and a list of first-300-characters. Cost is one call per node, once, at ingest.
2. **Query rewriting / decomposition** before `doc_search`. A multi-part question retrieves badly
   as one string. This wants to be an explicit `db.doc_search(..., rewrite=fn)` seam rather than
   something buried in the search path.
3. **Reranking.** `rerank_hits` already exists (flashrank cross-encoder). An LLM reranker is the
   same shape and slots in as `rerank_model=`; it is slower and usually not worth it above a good
   cross-encoder, so it should stay opt-in.
4. **LLM tree building** (`build_tree` replacement) for documents whose structure is genuinely not
   in their text — bad OCR, slide decks. The structural path handles everything else, so this is
   a fallback, not a default.

The rule worth keeping: **no litesearch or kosha code path may require an LLM.** Every one of the
above is a keyword argument with a working non-LLM default, so `pytest` stays offline and an
air-gapped install stays useful.

## 5. The atlas UI

Today the same clustering is rendered twice: leela shows a list of clusters that drills into
members, and lego/atlas shows the scatter view. Both should read the same payload, which now
exists:

```python
res = store.clusters(columns=['content','metadata'])
# res.method, res.note
# res.clusters[i] -> centroid, size, label, member_keys, members
```

**Plan.**

1. **Projection belongs in litesearch, not the UI.** Add `store.project(keys=None, dims=2)` —
   PCA over the stored vectors (numpy only; UMAP optional behind an extra). A scatter plot needs
   x/y per point and that is a retrieval concern, not a rendering one.
2. **`lego/atlas` becomes corpus-agnostic.** It takes a litesearch store name and renders
   `clusters()` + `project()`. Nothing in it should know what the corpus is.
3. **Point it at kosha** for the leela use case: `k.code_st` is a litesearch store, so the same
   block renders a code atlas with no new backend. leela then either embeds the lego view or
   keeps its list — the data contract is identical either way.
4. **Show `note`.** The UI is the reason `note` exists: "23 clusters over 8,412 chunks (usearch)"
   and "index has 340 chunks; showing nearest neighbours" are both things a user must see.

One known weakness to fix while doing this: c-TF-IDF labels on *code* come out as
`integrate_4, integrate_6, integrate_0` — unique identifiers are exactly what IDF rewards. A code
label should strip numeric suffixes and prefer the shared stem. That is a `stop=`/tokeniser
argument to `ctfidf_labels`, not a new algorithm.

## 6. Suggested order

| # | work | repo | size |
|---|---|---|---|
| 1 | ~~cluster/neighbour API~~ | litesearch | done |
| 2 | ~~document tree layer~~ | litesearch | done |
| 3 | ~~code-side similarity~~ | kosha | done |
| 4 | leela's `Search` collapses onto kosha | leela | small |
| 5 | cost-model chunking | litesearch | medium |
| 6 | `project()` + generic atlas block | litesearch + lego | medium |
| 7 | code trees | kosha | medium |
| 8 | chitragupta delegates to `litesearch.tree` | chitragupta | medium |
| 9 | `rank_results` split | litesearch + kosha | medium |
| 10 | notes + evidence packs | litesearch | small |
| 11 | Rishi seams: summaries first, then query rewriting | all | ongoing |

4 is first because it deletes code and proves the new API against a real caller. 8 is late because
chitragupta is the most invasive migration and benefits from 5 landing first.

## 7. Open questions

1. **Notes: same store or separate?** Recommendation above is same store with a `kind` column.
2. **Does `sections()` normalise by section length?** It currently sums RRF mass, which favours
   long sections. Corpus-dependent; left explicit rather than guessed.
3. **Is `lego/atlas` meant to be published?** It is referenced from litesearch but absent from the
   repo, so §5 is unverified against your code.
4. **How far does chitragupta migrate?** It could become a thin CLI + acquisition layer over
   `litesearch.tree`, or keep its own `Library` facade. The first is less code; the second keeps
   its API stable for anything already using it.
