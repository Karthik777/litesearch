# Release notes <!-- do not remove -->

## 0.1.7

### Fixed

- **`chunk_markdown` at 512 crashed on non-ASCII text.** `chonkie.FastChunker` encodes to UTF-8, takes
  byte offsets from `chonkie_core`, and decodes each byte slice; when no delimiter falls inside
  `chunk_size` bytes the core hard-cuts at an arbitrary byte, and a cut mid-codepoint raises
  `UnicodeDecodeError: 'utf-8' codec can't decode bytes in position 510-511`. Curly quotes, `naïve`,
  an em-dash, CJK or emoji are enough. 0.1.6 exposed it by lowering the default chunk size to 512 —
  at the old 4096 the chunker almost never split, so the bug was latent rather than absent.

  Raising the size to 1024 is **not** a fix: it only lowers the odds of landing mid-codepoint, which is
  worse than a crash because it then fails on a fraction of a corpus rather than on all of it.

  `data.SafeChunker` is now the default. It uses chonkie's own offsets, snapped forward to the next
  codepoint boundary, so chunking is **identical on all 2,007 page-texts of the eval corpus** — the
  0.1.6 measurements carry over unchanged — and it happens to run **2.4x faster** (235 MB/s against 97),
  because building `Chunk` objects through `BaseChunker` cost more than the splitting did. `chunk_spans`
  moves with it, and `CHUNK_SIZE` is documented as bytes, which is what `FastChunker` always meant.


## 0.1.6

### Changed by evaluation

`docs/rag_tiers.md` measures 45 store configurations over three genres that fail differently — 489
pages of EU legislation, 12 arXiv papers, and 1,275 pages of astrology books printed between 1822 and
1920 — against 351 known-item queries in five flavours, from verbatim down to five content words with
every one swapped for a WordNet synonym. Ground truth is derived from the corpus itself; no LLM is
involved. Two defaults did not survive it.

- **`chunk_markdown` chunks at 512 characters, not 4096.** `FastChunker`'s own default is 4096 and
  `add_doc` calls the chunker once per node segment, so on ordinary pages nothing was ever split and a
  "chunk" was a whole page. Page-sized against 512-character chunks costs **0.06–0.14 section MRR**
  across all three genres — the largest single effect measured. A 2,300-character chunk also overflows
  a 512-token encoder, so its tail was embedded by nothing. `chunk_spans` moves with it, and
  `CHUNK_SIZE` is the one place to change it. `FastChunker` rather than
  `RecursiveChunker(512, tokenizer='character')`, which scores ~0.01 better and chunks at 30 MB/s
  against 100 MB/s — chunking is ~0.1% of indexing cost either way, so the tiebreak is that
  `FastChunker` is the class already in use.

- **`search(fts_pre=True)` sends the FTS leg through `pre()`.** Quoting each token made FTS an
  implicit AND over the whole query, so a reworded question matched nothing and the hybrid quietly
  became vector-only. This is a **trade, not a free win**: `pre()` is worth +0.09 to +0.33 on
  paraphrased and synonym-substituted queries and costs 0.02 to 0.24 on verbatim ones. Pass
  `fts_pre=False` if your users paste exact phrases. The reranker and the vector leg still see the
  original query.

The two are coupled. Together they move `db.search` on its own defaults by **+0.10 to +0.13** weighted
section MRR (0.665→0.790 arXiv, 0.635→0.750 legislation, 0.597→0.699 books). Apart, `pre()` at page
granularity is *worse* than what it replaces on two genres of three — the verbatim penalty only
shrinks once chunks are small.

The same evaluation found that several expensive things do not pay: the PPR graph leg loses 0.07–0.11
at every `graph_w` on all three genres, `topic_nodes` contributes nothing to ranking, late chunking
over whole long documents halves the quality of the vector leg, the document tree is a wash, and
upgrading a 32M static embedder to a 300M transformer is worth 0.007 for 1,700x the indexing compute.
`SKILL.md` has been corrected where it said otherwise — in particular `peers()` is 2–3x *worse* than
`ann_neighbors()` at matched group size, the reverse of what it recommended.

### Fixed

- **`pre()` emitted invalid FTS5 for any query containing an apostrophe.** `kw()` segments with apsw's
  UAX#29 tokenizer, which keeps the apostrophe inside a word, and `add_wc` then appended a wildcard:
  `pathfinder's*` is a syntax error, so `db.search` raised `SQLITE_ERROR` rather than searching. Tokens
  that are not FTS5 barewords are now quoted.
- **`vec_search` warns when `dtype` cannot match how the vectors were stored.** A float32 store
  searched with the default `dtype=np.float16` reinterprets each float32 as two float16s, and for
  normalised embeddings every distance comes back *exactly 0.0* — no exception, no empty result, the
  vector leg silently returning rows in rowid order. `model2vec` returns float32 and `search` defaults
  to float16, so this was one line of ordinary code away.
- **`embedding_gemma` used a prompt template EmbeddingGemma was never trained on.** The model card
  specifies `title: none | text: {content}` for documents and `task: search result | query: {content}`
  for queries; litesearch shipped an Instructor-style string. This is `FastEncode`'s default model.


## 0.1.5
use ann only if there isn't a where. exactness over speed

## 0.1.4

### New

- `litesearch.tree` — a PageIndex-style document-structure layer. `db.get_tree()` adds `docs` and
  `nodes` tables beside any chunk store; `add_doc`/`add_file`/`add_dir` build a node tree per
  document (markdown headings → chapter lines → page windows) and link every chunk to a node.
  `db.doc_search` adds adaptive RRF weighting, span merging and breadcrumbs; `db.sections` rolls
  evidence up to whole sections; `db.toc`/`db.read` navigate structure with no embeddings at all.
  No LLM is required anywhere — `summarize=`, `chunker=` and `build_tree` are all replaceable.
- `store.ann_neighbors(rowid, ...)` — nearest rows to an already-indexed row, reusing the stored
  vector so nothing is re-embedded. `store.ann_vec(rowid)` exposes the vector itself.
- `store.clusters(...)` and `store.peers(rowid, ...)` — the clustering that `topic_nodes` used
  internally, now public and labelled with c-TF-IDF. Both return `method` and `note`, so a
  degraded or empty answer says why instead of looking like a broken index.

### Changed by evaluation

`nbs/07_doc_eval.ipynb` runs the stack against pre-`tree` litesearch over 486 pages of EU
legislation. Three features did not survive it unchanged:

- **`detect_mode` guards `#` in both directions.** pdf-oxide marks 16–31 lines per page as h1 on
  converted PDFs, so the first build produced a 5,242-node tree of one-line sections for the VAT
  directive. High heading density with no variation in depth is now rejected as noise, and
  `struct_levels` derives a per-document level ladder from CHAPTER/TITLE/Article words instead:
  1,983 nodes across 4 real levels.
- **`sections()` defaults to `score='max'`.** Summing RRF mass across a node is a length prior in
  disguise — long chapters outranked the precise article. Worth 0.11 MRR on verbatim queries and
  0.16 on keyword-degraded ones. `mean` and `sum` remain available.
- **`doc_search(adaptive=...)` defaults off.** Its triggers never fired across 300 natural-language
  queries, and a replacement rule keyed on FTS coverage measured worse. Kept opt-in.

The headline result is that chunk granularity, not structure, drives retrieval quality here: the
same corpus in page-sized chunks scores 0.330 MRR against 0.522 in node-scoped ones. Structure
earns its place elsewhere — a page-sized chunk names the right Article 27% of the time, a
node-linked one 93%.

### Fixed

- `ann_search` returned up to `2 × limit` rows. It still over-fetches from the index so `where`
  has candidates to filter, but now trims to `limit` as documented.

## 0.1.3
apsw fts5 instead of yake for kw

## 0.1.2
spacy models are lazy loaded

## 0.1.1
bump

## 0.1.0
graph added and fts5 stemmers added

## 0.0.37
ann latency bug fixbug fix


## 0.0.36
ann where addition


## 0.0.35
reranking and late chunking


## 0.0.34
ocr with liteparse


## 0.0.33
parallel search


## 0.0.32
ann index clear


## 0.0.31
hnsw ann with usearch


## 0.0.30
database create parent dir



## 0.0.29
offset



## 0.0.28
pre query cleaning

## 0.0.27
fix automatic uv dev



## 0.0.26
uv xtras + skill + cli



## 0.0.25
pkg origin or sub module paths for pkg2files



## 0.0.24
added progress bar to fastencode



## 0.0.23
spec fix



## 0.0.22
fix: fts query clean fix to remove fts5 specific kw from q



## 0.0.21

types removed, added exts


## 0.0.20
path stored as str



## 0.0.19
regex fix and index rewrite



## 0.0.17
ipynb + chunking



## 0.0.16
file parsing and chunking with chonkie



## 0.0.15
added codesigs support



## 0.0.14
added image and multimodal encoders


## 0.0.13
integrated with the wonderful pdf-oxide library



## 0.0.12
use uv for package management
split vec and fts search
add rrf merge with rowid
use model2vec as default static embeddings



## 0.0.11
add tokenizers



## 0.0.10
onnx gpu fix



## 0.0.9
cuda oredering



## 0.0.8
fixing fastlite imports, cuda and coreml providers.



## 0.0.7
Postfix script fixed



## 0.0.6
Improved layout, removed unwanted dependencies, added documentation



## 0.0.5
Improved db search, added fast encoding capabilities, coding assistant that you can use in examples
Pypi release for litesearch



## 0.0.4




## 0.0.3




## 0.0.2




## 0.0.1
