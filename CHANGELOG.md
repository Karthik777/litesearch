# Release notes <!-- do not remove -->

## 0.1.10
tree heading bug fix _ parallel writes into sqlite by increasing busy_window

## 0.1.9
warnings added (bug fix)

## 0.1.7
fts pre check, dtype mismatch warning, embeddinggemma template change



## 0.1.6
graph and tree need their own hash columns not generic content

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
