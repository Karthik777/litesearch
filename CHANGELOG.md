# Release notes <!-- do not remove -->

## 0.1.23

`Index` — one route in, so that nobody has to weigh six tradeoffs to search a folder.

The library had grown four ways to answer the same question, and picking between them meant
knowing what `evals/` had already decided. `litesearch.api.Index` decides it: static
`potion-retrieval-32M` (the spread across four encoders is 0.018–0.046, and the static one *wins*
one genre while indexing ~1,700x cheaper), float16 to match the store, 512-character chunks
(+0.06 to +0.12 over page-sized), `pre()` on the FTS leg (+0.016 to +0.093), an HNSW vector leg
(−0.005 for a large speedup), and a document tree built unconditionally because its effect on
*ranking* is a wash (−0.052 to +0.011) and `toc`/`read`/`sections`/`context` are worth having for
free.

```python
ix = Index('kb.db'); ix.add('docs/'); ix.search('how does batching work')
```

One decision is left to the caller, because it is the only one the evaluation says is worth
making: `rerank=True`, worth +0.026 to +0.077 weighted MRR and positive in all twelve paired
cells, at roughly 10x the query latency. It fetches 30 candidates before the cross-encoder sees
them — reranking ten only reorders ten, and the measured gain comes from thirty.

Six methods: `add`, `add_code`, `search`, `sections`, `read`, `toc`, plus `context`. `Index.db` is
the `database()` you would have written by hand, so nothing is walled off and there is no
migration.

Nothing was removed from `core`, `data`, `utils`, `tree`, `graph` or `sanskrit`; every existing
import still resolves. What shrank is the documentation surface. `nbs/index.ipynb` was a 61-cell
tour that re-documented each module beside its own page; it is now a 22-cell front door — two
routes, the measured ladder that says which knobs are worth turning, and links out. The README
went from 776 lines to ~250 with nothing lost, since every dropped section already had a page.

The ladder itself is now in one table instead of scattered across six notebooks. Above the line
and on by default: `pre()`, 512-char chunks, ANN. Left to you: rerank. Below the line and staying
off: the tree *for ranking*, heading prefixes, deeper fanout without a reranker, late chunking
(−0.033 to −0.053), and the entity graph leg (−0.070 to −0.160 on ordinary queries, opt-in by name
only). `python -m evals.decide` reproduces all of it.

**Late chunking is now `exporti`.** The three classes stay compiled into `litesearch.utils` and
importable by name, but leave `__all__`, so `from litesearch import *` no longer offers them and
they carry no docs page. Not deleted: `evals/encoders.py` imports `AutoLateChunkFastEncode` and
`evals/run.py build_late` still builds the arm, which is what keeps it cheap to re-ask the question
on a corpus with longer-range coreference than legislation, papers or a treatise on astrology.

**vishalakshi is the nominated first caller.** It builds a litesearch-backed vault, and porting it
onto `Index` is a better test of "did this remove decisions" than any example in the notebook. The
api page lists the three things that port should answer — whether `add` covers its hand-rolled
ingest, whether `Index`'s English-tuned default encoder should follow the profile instead of the
constructor (a Sanskrit vault wants `encoder=static_embedder()`), and whether its read path belongs
on `context`/`read` rather than `search`.

Two findings worth stating outside the table. Keyword-only retrieval with `pre()` beats hybrid in
all 24 paired cells on the main query set — an artifact of that set, where every query is a lexical
transformation of the sentence it targets; `evals/multihop.py` builds the corrective and there the
vector leg reaches targets FTS cannot score at all. And the largest measured retrieval win in the
repository is not an embedding at all: the `sanskrit` FTS5 tokenizer's ASCII fold gives 1.000
Devanagari→verse recall for *every* encoder tested, which is why it is on for every store.

## 0.1.22
vishalakshi support

## 0.1.21
spec supports extra deps like rishi[all]

## 0.1.20
release

## 0.1.19

Performance across `graph`, `tree`, `core` and `sanskrit`. The question was whether [usearch],
[StringZilla] or [NumKong] could speed up the set comparisons in `graph.py`; the measured answer is
that usearch already does the vector work and the other two are slower than the stdlib on inputs
this small, so the wins came from elsewhere — SQLite write amplification, guards evaluated far more
often than they could decide anything, and pools started in the wrong places. The same three shapes
turned up in the other modules. `docs/graph_performance.md` has the numbers and
`evals/ingest_bench.py` reproduces them.

Headline: `resolve_entities` **6.0x**, `graph_search` **2.0x**, `cooccur_edges` **3.65x**,
`add_dir` over PDFs **1.75x**, `fold_token` **5.5x**.

Results are unchanged and checked rather than asserted: `build_graph` emits identical entity,
mention and edge hashes across all seven `n_workers`/`batch`/generator combinations *and* against
the pre-change code, `graph_search` returns the same rows in the same order with the same RRF
scores to twelve decimal places, and `_lex_ok` was compared against its previous form over 979,300
real entity pairs and 3,844 adversarial ones with zero disagreements.

**One exception, and it is a bug fix rather than a speedup:** chunks whose text appears under more
than one heading were embedded against the wrong one. Re-ingesting changes 4.35% of embeddings on
the eval PDF corpus. See the first entry under Fixed.

### Fixed

- **The eval harness's keyphrase graph was never connected to the store.** `build_graph` keys a
  mention on `chunk['id']` and falls back to `_slug(content)` when there is none; a **tree** store
  hashes its ids over `node_id` *and* `content`, and `run.build_graphs` selected only `content`. On
  regulation, 0 of 34,891 mentions referenced a chunk that exists. `topic_nodes` writes its mentions
  against real store ids, so the topics stayed connected while the keyphrase half did not — which is
  the entire basis of `eval_graph`'s claim that "the topics are the whole graph leg: they carry 100%
  of the PPR mass, which is why swapping the extractor moved nothing." That was this bug. With `id`
  selected the graph leg actually fires: query latency goes from ~18ms (silently falling through to
  hybrid) to ~65ms, and removing the topic nodes costs 0.003–0.015 p_mrr rather than everything.
  `evals/results/graph.json` was produced by the broken path and is regenerated.
- **`resolve_entities` is reproducible.** Two resolves of the *same database* in two processes came
  back with different partitions and merge counts drifting over a range of three. It is not HNSW,
  which is what it looks like and what the 0.1.17 notes assumed — `verify_ann_probe` shows the ANN
  candidates were stable throughout and the variance is entirely in `by_lexical`. `_toks` returns a
  `frozenset`, so its iteration order follows `PYTHONHASHSEED`; that set the insertion order of the
  inverted index in `_lexical_pairs`, hence the order pairs were proposed in, and `_uf_union` only
  accepts a merge that keeps the group a clique — an order-dependent test. Tokens are now walked in
  sorted order. Four resolves under random hash seeds return the same partition and edge table bit
  for bit; blocking recall is unchanged at 99.5% / 98.3% / 97.2%.
- **Chunks with duplicate content were embedded against the wrong heading.** Chunks are embedded as
  `heading ⏎⏎ content` and stored as bare content, but the heading was attached through a
  `{content: heading}` dict rebuilt per document — which collapses on duplicate content, so two
  chunks with the same text under different headings were both embedded with whichever heading came
  last. Headings are paired with their chunk by position now. Re-ingesting the eight-PDF eval corpus
  changes 148 of 3,402 embeddings (**4.35%**) — same ids, same content, same nodes, every changed
  row one whose text appears under more than one heading. A store built before this has those rows
  embedded against the wrong heading; re-ingesting fixes them.
- **`_collapse_edges` rewrote the edge table outside a transaction**, so it paid an fsync per insert
  batch and left a window in which a concurrent reader saw no edges at all. Both halves are one
  write now.

### Changed

- **`Database.context(graph=...)` now defaults to `False`.** The graph leg is a real capability with
  a narrow domain, so it is opt-in rather than on. Measured both ways over three genres:
  on **known-item** queries — the target contains the words you searched for — it is a straight loss
  that deepens with `graph_w` (regulatory p_mrr 0.8170 hybrid against 0.7395 / 0.6859 / 0.6463 at
  0.25 / 0.5 / 1.0), at 2–4x the latency; on **bridge** queries — the target never uses those words
  and is relevant on structural grounds — it is a significant win that also deepens with `graph_w`,
  in seven of nine paired-bootstrap comparisons (arXiv +0.0387 target MRR at `graph_w=1.0`, 95% CI
  [+0.0110, +0.0694]). Most queries are known-item, so the default was making most users pay for a
  leg they do not use. `graph_search` is unchanged: it is a method you call by name, and calling it
  is the opt-in.

- **`resolve_entities` is 6.0x faster** (11.97s → 2.00s at 8,487 entities), and three quarters of
  that was never entity matching. The `canon` write-back ran one autocommit per entity and each one
  fired apswutils' `AFTER UPDATE` trigger, re-tokenising `content` into FTS5 — 5,650 fsync'd
  reindexes to record 2,056 merges. Only rows whose canon actually moves are written now, in one
  transaction, compared against the *stored* canon so a re-resolve that undoes a merge still writes
  the row back. `_lex_ok` also evaluated its two most expensive tests first: the digit and acronym
  checks allocate four throwaway objects per call and can only veto or rescue a pair the cached
  token sets have already ruled on, which is 300,000 `_acr` calls and 300,000 `findall`s per resolve
  to change 900 answers.
- **`graph_search` is 2.0x faster** (140ms → 70ms per query at 2,000 chunks). The canonical-id map
  was a full scan of the entity table on every query to serve the mentions of twelve chunks — the
  one cost that grew with the corpus rather than the query — and is now a join. `_adjacency` reads
  raw tuples instead of building a dict per edge row, and `_ppr` flattens the adjacency into index
  arrays once and runs each round as a gather plus a `bincount` instead of twelve nested Python
  loops (34ms → 4ms). Dangling nodes still drop their mass, exactly as before.
- **`build_graph` opens one process pool per call, not one per batch.** `parallel` opens and closes
  a `ProcessPoolExecutor` around every invocation and `drain_prose` runs once per `batch`, so the
  two arguments that exist to make a large corpus finish were each making the other worse: 2,000
  chunks at `batch=200` spawned forty workers to do the work of four. Extraction 7.7s → 5.1s, and
  the cost of batching is now flat in batch size rather than rising as batches shrink. This was
  listed as open in `docs/ingestion_performance.md`.
- **`hash_embed` is 1.85x faster** and bit-identical. Same hash; the n-gram hashes are binned by one
  `bincount` rather than fancy-indexing a float row per n-gram, and an all-ASCII string is encoded
  once and sliced as bytes instead of encoded once per n-gram. n-grams are counted in *characters*,
  so the byte path is only taken when the encoding is length-preserving.
- **`_knn_clusters` fetches its vectors in one batched usearch lookup** rather than one call per key.
  `ctfidf_labels` counts cluster sizes in one pass instead of re-walking the label array per cluster.
- **`fold_token` is cached — 5.5x** (114.7ms → 20.8ms over 70,413 tokens). It runs once per token on
  every FTS write *and* every query and is three unicode normalisations plus a per-character
  category scan deep. In a `cProfile` of `add_doc` x40 the Sanskrit tokenizer chain was a third of
  the run and `fold_token` half of that, on a corpus with no Sanskrit in it at all. The eval corpus
  is 70,413 tokens and 4,153 distinct strings: 94% of the calls re-answer a question already
  answered, and it is a pure function, so the cache is a memo and nothing else.
- **`add_doc` writes the doc row and its node rows in one transaction** instead of two commits per
  document, and `delete_doc` covers all three tables in one — a reader landing between the second
  and third commit previously saw a document whose nodes had vanished. `add_doc` x150: 2.82s →
  2.25s. **`toc()` fetches every node in one query** rather than one per document, which is 26.6ms →
  16.9ms over 300 documents and the right shape for the cheap vectorless listing it is meant to be.
- **Mentions and edges upsert with one statement per batch, not two per row.**
  `insert_all(upsert=True)` emits an `INSERT OR IGNORE ... RETURNING *` *and* an
  `UPDATE ... RETURNING *` for every record, each separately prepared and each materialising a
  result row nobody reads — 23,530 statements to write 11,765 mentions. That pair is exactly what
  `ON CONFLICT DO UPDATE` does, which SQLite has had since 3.24, so `core.upsert_all` does it in
  one. `cooccur_edges` over 15,233 edges: 782ms -> 214ms (**3.65x**); `build_graph` over 2,000
  chunks 7.02s -> 6.00s.
- **`add_dir` embeds and writes across documents, not per document** (`embed_batch`, default 2,000
  chunks, flushed on a count so a large directory does not hold its own text twice). 150 markdown
  files with a static encoder: 3.08s → 2.46s. How much the encoder itself gains depends on the
  encoder and is now measurable (`ingest_bench embatch`): a static model gains ~19% and flattens by
  64 chunks a call — it is a lookup table with nothing to amortise — while an ONNX model has real
  per-call setup. Most of the gain here is the *writes* being batched alongside it.
- **`process_content` upserts through `upsert_all`.** It was still emitting apswutils' two
  statements per row, 5,912 of them for 2,956 chunks. The row id is hashed in `process_content` now
  rather than left to `hash_id=`, which is what allows the single-statement form; it is
  `hash_record` over the same columns, so ids are unchanged and a re-ingest lands on the same row.
  `upsert_all` routes values through apswutils' own `jsonify_if_needed`, so a dict `metadata` is
  stored exactly as before.
- **`add_dir` parses on a process pool and writes in the parent.** Over eight PDFs the parse is
  3.55s of a 5.31s walk, so splitting it is 6.18s -> 3.54s (**1.75x**). Over 150 markdown files the
  same "parse" is `read_text` and takes 2.7ms in total, so a pool is pure loss — forcing one costs
  2.37s against 2.24s — and `n_workers=None` declines to start one. The choice is made on how many
  files have a parse worth splitting (`PARSE_HEAVY`), not on how many files there are. Writing
  stays serial: one connection, one FTS index, one HNSW graph, none improved by contention.
  `add_file` splits into `_parse_doc` (plain data, picklable) and `_add_parsed`; a worker resolves a
  profile by name rather than re-detecting one and returns `None` if its registry lacks it, so the
  parent re-parses rather than letting a different reader through unnoticed. Ingest output is
  byte-identical serial, auto and `n_workers=4`, over both PDFs and markdown.

### New

- **`evals/multihop.py`** — the bridge query set the existing harness was missing, and the reason
  the `graph` default could be decided on evidence. Ground truth is the corpus's own structure
  rather than a similarity score: X and Y are two chunks of the same section, and the query is
  three tokens present in X and absent from Y, so nothing lexical connects it to the target. No
  extractor, embedding or entity graph is consulted in building the pairs — using the graph to pick
  them would have guaranteed the graph could answer them. Two controls run on the identical hit
  lists: `source` (FTS should find X — are these queries answerable at all) and `control` (a random
  chunk from a third document — the noise floor, which stays at ~0.000 throughout).
- **`evals/extractor_eval.py` and `evals/extractor_sig.py`** — does the keyphrase extractor change
  *retrieval*, not just the stopwatch. Each genre's tree store is cloned so chunks, embeddings and
  the ANN index are held fixed and the graph is rebuilt over the identical store once per extractor;
  `extractor_sig` then keeps the per-query reciprocal ranks and bootstraps the paired difference.
  Across 3 genres and 1,755 query-flavour pairs, **all nine pairwise comparisons straddle zero** —
  yake, yake-rust and rake-nltk are statistically indistinguishable on retrieval. So `yake-rust`'s
  6.6x is free, and rake-nltk matching at 21.6% term overlap says the walk is not discriminating on
  keyphrases. The larger finding is in the baseline row: **plain hybrid beats every graph
  configuration on every genre**, monotonically in `graph_w` (regulatory 0.8170 against 0.7395 /
  0.6859 / 0.6463 at 0.25 / 0.5 / 1.0), at 2–4x the latency. Read with the caveat that a known-item
  query set cannot test what the graph leg is for — bridging to a document the query never mentions
  — so it licenses "do not default `graph_w` on for known-item search", not "the graph leg does not
  work". Nothing is swapped; `terms_fn` remains the seam and the default is unchanged.
- **`evals/ingest_bench.py embatch`** — what the encoder gains from being handed the whole
  directory instead of one document, swept over batch size, because the answer is a property of the
  encoder rather than of litesearch.
- **`evals/ingest_bench.py kw`** — keyphrase extractors on speed *and* on agreement with the
  shipped yake, since extraction is ~99% of `prose_windows` and the reason a build needs a pool at
  all. `yake-rust` is the same algorithm at **6.6x** per call (7.5ms -> 1.1ms) and it **releases the
  GIL** — 0.46s -> 0.16s on four threads, where both pure-python candidates go *slower*. It would
  retire the process pool for extraction entirely. The catch is that it agrees with the shipped
  yake on 66.5% of terms, which is a graph change and wants a retrieval eval, not a stopwatch.
  rake-nltk is the fastest thing measured (0.5ms) and the least comparable at 21.6%: RAKE returns
  long noun phrases where YAKE returns tight keyphrases, so it is a different extractor rather than
  a cheaper one. Nothing is swapped — `build_graph(terms_fn=...)` is the seam for it and the
  default is unchanged.
- **`evals/ingest_bench.py parts`** — disables one part of `resolve_entities` at a time and
  re-resolves *the same graph*, so each row is that part's contribution. The clique guard is
  load-bearing and then some: without it 6,422 merges collapse into transitive blobs covering 14.8
  million pairs. The lexical pass carries 69% of merged pairs. The ANN pass is not purely additive —
  removing it *gains* 1,879 pairs and loses 1,284, because merging on embeddings first blocks
  lexical merges the order-dependent clique guard would otherwise accept — and nets out at four
  merges on this corpus. The acronym rule is idle on prose (identical partition without it); the
  digit guard is nearly free and catches 13 `python 3.11`/`python 3.12`-shaped merges.
- **`evals/ingest_bench.py legs`** — serial against threaded for `search`'s FTS and vector legs.
  They look independent and both drop into C, but they share one apsw connection, so SQLite
  serialises them behind its own mutex and threading is *slower* at every size and in both vector
  modes (0.60x to 0.82x). The `parallel=` flag on `search` stays deprecated, now on evidence.
- **`evals/ingest_bench.py libs`** — StringZilla and NumKong against the stdlib on the operations
  this module actually performs, with an agreement check, because a faster function that answers
  differently is not a drop-in. NumKong's `intersect` is 0.48x a Python `frozenset &` on two-to-five
  element sets; `sz.contains` is 0.81x Python's already-SIMD `in`; the one real win,
  `utf8_wordbreaks` for tokenisation at 2.14x, disagrees with apsw's UAX#29 iterator on 1.6% of
  tokens, which is 1.6% of merge decisions. Neither library is a dependency and on this evidence
  neither should be.

[usearch]: https://github.com/unum-cloud/USearch
[StringZilla]: https://github.com/ashvardanian/StringZilla
[NumKong]: https://github.com/ashvardanian/NumKong

## 0.1.18
removed apcy and added a hand rolled noun list at 89% efficiency

## 0.1.17

The graph half of the ingest work. `build_graph` was the remaining thing that could not be handed a
large corpus, and `resolve_entities` turned out to be losing merges as the corpus grew.

### Fixed

- **`_lexical_pairs` no longer skips oversized token blocks.** A block bigger than `max_group`
  yielded nothing at all, and blocks grow with the corpus — so the merges that stopped being
  proposed were exactly the ones a larger corpus added, silently, with nothing in any timing to show
  it. Against an exhaustive `_lex_ok` ground truth, blocking found 97% of valid merges at 1,377
  entities and **76% at 8,487**. Oversized blocks are now windowed on sorted names, which puts
  containment variants adjacent: 99.8% and 97.2% at the same two sizes, for 2.2x the pairs examined
  and no measurable time.
- **`max_degree` pruning breaks weight ties on `(src, dst)`.** Without it the surviving edges
  depended on the order pairs happened to be counted in, so two builds of the same corpus could
  differ. Found while checking the batched path against the in-memory one: same 2,707 edges, zero
  differences in weight or count, 30 different edges, all of them ties.
- **`build_graph` writes inside a transaction.** Entities, mentions and edges went through
  `insert_all` directly, so `insert_chunk` was 9.3s of a 36.1s build. 1,000 chunks: 21.6s -> 11.5s.

### New

- **`build_graph(n_workers=...)`** — extraction on a process pool. It is 88% of a build and a pure
  function of the chunk text, so a build is a map with a cheap serial reduce; threads are useless
  because yake and spaCy are both python and both hold the GIL. spaCy's own `n_process` is used
  rather than reinvented. 1,000 chunks on 4 cores: 11.6s -> 4.9s (yake, **2.4x**) and 19.0s -> 10.2s
  (spaCy, **1.9x**), 86 chunks/s to 207. `None` picks a pool by queue size and stays serial below
  `MIN_PARALLEL_CHUNKS`; `0` forces serial. Order is preserved and the reduce depends on it —
  `ents.setdefault` keeps the `kind` of an entity's first mention.
- **`build_graph(batch=...)`** — flushes mentions per batch and keeps co-occurrence windows in a
  scratch SQLite table rather than in memory. Windows are the term that never saturates: entity
  vocabulary flattens, mentions can be flushed, a window is one per sentence forever. PMI pair
  counting moved into SQL for the same reason, and `build_graph` now iterates `chunks` instead of
  wrapping it in `L()`, so passing a generator actually streams. Peak RSS over 8,000 chunks: 281 MB
  -> 150 MB, and the marginal cost halves as the corpus doubles rather than staying flat.

The graph is byte-identical batched or not — entities, mentions and edges are all pinned in
`nbs/05_graph.ipynb`.

## 0.1.16

Ingestion performance. `docs/ingestion_performance.md` has the measurements and
`evals/ingest_bench.py` reproduces them; the short version is that a 400-document directory went
from 112.2s to 9.2s and, more to the point, from exponent 1.64 to 0.91 — ingest is linear now.

### Changed

- **`add_doc` no longer rebuilds the ANN index per document.** It mirrors that document's keys into
  the index, the way `sync` always has. A full `rebuild_index` reads every embedding blob in the
  store, so doing it per document made a directory walk O(N²). `add_dir` now treats the walk as a
  bulk load: FTS triggers suspended for the duration, one index rebuild at the end. `delete_doc`
  drops just the deleted rows' keys. **12.2x** at 400 documents, and no longer degrading.
- **`add_doc` stops re-registering an ANN index the caller opted out of.** It calls `get_tree`
  internally, which defaults to `ann=True`, so `get_tree(store, ann=False)` was silently overridden
  on first ingest. A store that already exists now keeps whatever it was created with.
- **`process_content` always batches writes in a transaction.** apswutils commits per insert batch
  and a WAL commit is an fsync: 8,000 rows cost 7.4s that way against 1.5s in one transaction. The
  batching was already implemented but gated behind `parallel=True`, which is a concurrency flag,
  not a throughput one. `parallel` now controls only the busy window. **6.7x**.
- **`dir2chunks`/`pkg2chunks` parse on a process pool.** `file_parse` holds the GIL end to end, so
  the thread pool they used measured 0.76x — *slower than serial*. Directories below
  `MIN_PARALLEL_FILES` stay serial rather than pay for a pool. **3.3x**.

### Faster, with identical results

- **`pyparse`** hoists the line split out of the per-node loop and iterates `tree.body` instead of
  walking the whole tree to tag parents. `ast.get_source_segment` re-splits the entire file on
  every call — 68% of the time spent parsing `transformers`. **7.1–11.4x**, byte-identical across
  32,720 chunks and every flag combination.
- **`resolve_entities`** replaces 8,487 individual HNSW probes (each with its own `rowid IN (...)`
  query) with one batched probe, and memoises `_toks`, which the lexical guard was calling 252,813
  times on a few thousand strings. **2.3x**, exponent 1.35 → 1.08, same merges.

### New

- `Table.bulk_load()` — suspend a table's FTS5 triggers for a bulk insert and rebuild the index
  once on the way out (**2x**). For loads only: `rebuild` is O(rows), so it is slower than the
  triggers for a small update.
- `evals/ingest_bench.py` — ingestion benchmarks over the `evals.corpus` genres, sweeping corpus
  size so the growth exponent is visible rather than a single throughput number.

## 0.1.15
spacy batching 25% speedup

## 0.1.14
remove parallel search in sqlite. increases performance

## 0.1.13
pdf assets stored relative to the database and pdf

## 0.1.12
SafeFastChunker and fts_pre

## 0.1.11
parallel sync

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
