# Release notes <!-- do not remove -->

## Unreleased

### New

- `litesearch.sanskrit` — verse-aware chunking, sections and a graph for Sanskrit texts, built on
  `add_doc`'s new `tree_fn`/`chunk_fn` seams rather than beside them, so `toc`/`read`/`doc_search`/
  `sections`/`graph_search` work on a Sanskrit corpus unchanged.
  - **Transliteration.** `fold` reduces any of Devanagari, IAST, SLP1 or Harvard-Kyoto to one ASCII
    key; `loose` also absorbs popular spellings (`Krishna`, `Geeta`). Each chunk's fold is stored in
    the existing `metadata` column, which `get_store` already indexes for FTS — so scheme-agnostic
    search needs no new column, table or index. `to_iast`/`to_slp1`/`detect_scheme` are public.
  - **Verse segmentation.** `split_verses` handles the four layouts the corpora actually use
    (reference after the verse, reference on its own line, Devanagari with a bare number, unmetred
    prose between daṇḍas), and returns colophons, `[h: :h]` headings and speaker changes as their
    own records. The verse is never split.
  - **Metre.** `syllables`/`detect_meter` implement classical scansion; the 20 metres in `METERS` are
    derived from their gaṇas rather than transcribed. Metre doubles as the verse/prose classifier and
    as a search facet.
  - **Chunking.** `chunk_verses` builds overlapping windows of *whole* verses (`per`, `stride`);
    `max_chars` shrinks a window rather than cutting a verse. `sanskrit_chunks` chunks verse and
    prose runs separately, and a bhāṣya block inherits the citation of the verse it glosses.
  - **Sections.** `verse_tree`/`verse_mode` take structure from the citation stamped on every verse,
    then colophons, then unit headings, then fixed verse windows. Node titles are citations, or the
    names the colophons give the ladder (`Amsa 1 › Adhyaya 1`); the colophon becomes the node summary.
  - **Graph.** `verse_graph` makes every verse an entity so `follows`, `parallel`, `quotes` and
    `chandas` fit `get_graph`'s existing entity-to-entity `edges` table. `db.parallels(ref)` reports
    the same śloka elsewhere; parallels are found lexically (word-shingle Jaccard over the fold), no
    model needed. Metres above `max_meter_df` are dropped rather than turned into a corpus-wide clique.
  - **Lemmas.** `load_conllu` and `db.add_dcs` ingest Digital Corpus of Sanskrit CoNLL-U files with
    their validated lemmas indexed beside the surface text; `lemma_fn=` is the seam for any other
    segmenter. Sandhi means the surface form is often not what anyone types, so this is the one place
    real linguistics changes retrieval outright.
  - `db.verse_search` (transliteration-agnostic, and FTS-only when `emb=None`), `db.by_ref`,
    `read_text` for GRETIL `.htm`, `split_commentary` for mūla/bhāṣya files.
- `litesearch.tree` gained the seams and the book/guide structure work the above needed:
  - `add_doc(tree_fn=, chunk_fn=)` — replace the tree builder and the chunker without touching
    ingestion. `node_chunks` is now public (`_node_chunks` still aliases it).
  - **`numbered` mode** in `detect_mode`/`build_tree` — the hierarchy of a manual, guide or textbook
    is in its numbering (`3.1 Sandhi`, `3.1.2 External sandhi`), which no previous signal caught. At
    least one dot is required, so a bare `1.` stays a list item. Numbered subheadings also nest under
    `chapter` mode, and front/back matter (`Introduction`, `Bibliography`, `Index`) becomes top-level
    nodes instead of being absorbed into the last chapter of the argument.
  - `struct_re(words)` + `rank=` on `build_tree`/`detect_mode`/`struct_levels` — a corpus with a
    different hierarchy vocabulary passes a word list instead of patching a module global.
  - `strip_running_heads(pages)` — removes running heads and folios, which every matcher here would
    otherwise read as headings. Off by default in `add_doc`; the pass is lossy.
  - `TreeNode.meta` — where a custom `tree_fn` leaves what its `chunk_fn` needs.

### Fixed

- **`add_doc` no longer loses text when two documents share a passage verbatim.** Chunk ids were
  hashed over `content` alone, so the second document's copy silently replaced the first's. They are
  now hashed over `(content, doc_id)`. `process_content(..., hash_cols=)` and
  `build_graph(..., hash_cols=)` expose the choice; flat stores are unchanged. Sanskrit made this
  visible — a śloka quoted across texts is the ordinary case — but it affects any corpus with two
  editions of the same annex.
- `strip_header` no longer eats the opening pāda of a four-line metre. Only the hemistich carries a
  daṇḍa, so cutting at the first daṇḍa-terminated line dropped the first quarter of the first verse,
  which then failed to scan and was filed as prose.
- A short chunk merging into its predecessor now merges its citations and lemmas too, instead of
  leaving text in the store that `by_ref` can no longer find.

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
