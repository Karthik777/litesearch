# Graph, tree and core performance

Measured on the `evals` regulatory corpus (9 EU directives, 489 pages) with `evals/ingest_bench.py`.
4 CPUs, SQLite in WAL mode, `hash_embed` standing in for an encoder so the numbers are litesearch's
own overhead rather than model inference.

The question this started from: the graph layer does a great deal of set comparison, substring
scanning and small-vector arithmetic, so would [usearch], [StringZilla] or [NumKong] make it
faster? The measured answer is that usearch already covers the part that is vector work, the other
two are slower than the stdlib on these inputs, and the actual costs were somewhere else entirely —
in SQLite write amplification, in a guard clause evaluated a hundred times more often than it could
ever decide anything, and in a process pool being torn down and rebuilt once per batch.

Every figure below is reproducible:

```
python -m evals.ingest_bench graph  --sizes 500,1000,2000
python -m evals.ingest_bench parts  --sizes 2000     # what each part of resolve_entities buys
python -m evals.ingest_bench libs                    # stringzilla / numkong against the stdlib
                                                     #   (numkong ships with usearch; for the
                                                     #    stringzilla rows: pip install stringzilla)
python -m evals.ingest_bench lexrecall
python -m evals.ingest_bench legs                    # threading search's two legs (it loses)
python -m evals.ingest_bench kw                      # keyphrase extractors, speed and agreement
                                                     #   (pip install yake-rust rake-nltk rakun2)
python -m evals.ingest_bench embatch                 # encoder gain from batching across documents
python -m evals.ingest_bench verify
```

## Headline

| | before | after | |
|---|---|---|---|
| `resolve_entities` (8,487 entities) | 11.97s | 2.00s | **6.0x** |
| `graph_search` (per query, 2,000 chunks) | 140ms | 70ms | **2.0x** |
| `build_graph` (`n_workers=4, batch=250`, 2,000 chunks) | 11.1s | 10.0s | 1.11x |
| `build_graph` extraction, `batch=200`, 2,000 chunks | 7.7s | 5.1s | **1.5x** |
| `hash_embed` (3,000 names) | 118ms | 64ms | 1.85x |
| `fold_token` (70,413 tokens) | 114.7ms | 20.8ms | **5.5x** |
| `toc()` over 300 documents | 26.6ms | 16.9ms | 1.57x |
| `add_doc` x150 | 2.82s | 2.25s | 1.25x |
| `cooccur_edges` (15,233 edges) | 782ms | 214ms | **3.65x** |
| `add_dir` over 8 PDFs | 6.18s | 3.54s | **1.75x** |
| `add_dir` over 150 md, static encoder | 3.08s | 2.46s | 1.25x |
| `build_graph` serial | 9.5s | 9.2s | — (yake-bound) |
| `search`, `fts_search` | | | unchanged |
| `clusters`, `peers`, `topic_nodes` | | | unchanged |

Results are unchanged throughout — with one deliberate exception, the duplicate-content heading
bug below, which changes 4.35% of embeddings on re-ingest. Everything else is checked rather than
asserted. `build_graph` produces
identical entity, mention and edge hashes across all seven combinations of `n_workers`, `batch` and
generator-vs-list input, and identical hashes to the pre-change code. `graph_search` returns the
same rows in the same order with the same RRF scores to twelve decimal places. `_lex_ok` was
compared against its previous form over 979,300 real entity pairs and 3,844 adversarial ones with
zero disagreements.

## Where the time actually was

### `resolve_entities` was mostly SQLite, not comparison

Three quarters of it never reached the entity-matching code at all. Component figures below are
`cProfile` cumulative time at 1,000 chunks / 5,650 entities, where the whole call went 8.58s → 2.14s.

- **The `canon` write-back was one autocommit per entity, each firing an FTS reindex.** `canon` is
  not an FTS column, but apswutils' `AFTER UPDATE` trigger fires on any update to the row, so
  every statement re-tokenised `content` into the FTS5 index — and outside a transaction apsw
  commits each statement, which in WAL mode is an fsync. 5,650 entities of which 2,056 actually
  merged: 5,650 fsync'd reindexes to record 2,056 facts. Now only rows whose canon moves are
  written, inside one transaction. 3.34s → 0.29s.
- **`_collapse_edges` deleted and reinserted the whole edge table outside a transaction** — an
  fsync per insert batch, and a window in which a concurrent reader saw no edges. Both halves are
  now one write. 2.53s → 0.23s.
- **`_lex_ok` evaluated its two most expensive tests first.** The digit check and the acronym check
  allocate four throwaway objects per call and can only ever *veto* or *rescue* a pair that the
  token overlap has already ruled on; the token sets are `lru_cache`d and settle ~99% of pairs
  outright. Running the cheap-to-decide test first is worth 300,000 `_acr` calls and 300,000
  regex `findall`s per resolve. Same answer on every pair, 1.69s → 0.58s.

The comparison against the *stored* canon rather than against the id matters: skipping rows where
`canon == id` alone would leave a stale canon behind when a re-resolve undoes a merge.

### `graph_search` was rebuilding a corpus-sized dict per query

- **The canonical-id map was a full scan of the entity table on every query**, to answer lookups
  for the mentions of twelve chunks. It is now a join on the mentions actually being read. This was
  the one cost in `graph_search` that grew with the corpus instead of with the query.
- **`_adjacency` built a dict per edge row.** A two-hop walk off twelve seeds pulls ~16k edges and
  fastlite turns each into a dict with a description lookup; the function wants three columns and
  none of the keys. Raw tuples now.
- **`_ppr` was twelve nested Python loops over an adjacency that never changes.** The iteration is
  `r ← d·Wᵀr + (1−d)·p₀`, so the edges are flattened into index arrays once and each round is a
  gather plus a `bincount`. Dangling nodes still drop their mass rather than redistributing it,
  exactly as before. 34ms → 4ms per query.

Float reassociation makes the PPR masses differ in the last bit (max relative difference 1.4e-15).
The set of entities above zero is identical, and so is the final result ordering.

### `build_graph`: `batch` and `n_workers` were charging each other

`parallel` opens and closes a `ProcessPoolExecutor` around every invocation and `drain_prose` runs
once per `batch`, so the two arguments that exist to make a large corpus finish were each making
the other worse — 2,000 chunks at `batch=200` spawned forty workers to do the work of four. The
pool is now opened once per call. Extraction at `batch=200`: 7.7s → 5.1s, and the cost of batching
is now flat in batch size instead of rising as batches get smaller.

This was listed as open in `ingestion_performance.md`; it is closed.

The serial path is unchanged, and will stay that way: yake is ~99% of `prose_windows`, so hoisting
the per-sentence `t.lower()` out of the inner loop is real and invisible.

## Would usearch, StringZilla or NumKong help?

**usearch already does the vector work**, batched, in `_ann_pairs` and `_cluster_groups`. There was
nothing left for it to take over. `_knn_clusters` now fetches its vectors in one batched key lookup
rather than one call per key, which is the last place a Python loop was wrapping it.

**NumKong is already installed** — it is a hard dependency of usearch — so using it would have been
free. It still lost. `nk.intersect` on sorted `uint32` token ids is **0.48x** a Python
`frozenset &`: the sets here hold two to five elements, and per-call FFI overhead dominates a SIMD
kernel completely at that size. NumKong is built for large arrays and these are not large arrays.

**StringZilla lost on two of three operations and was not a drop-in on the third**
(`python -m evals.ingest_bench libs`):

| operation | stdlib | stringzilla | | same answer |
|---|---|---|---|---|
| sentence split (`_sentences`) | 20.2ms | 20.0ms | 1.01x | no |
| tokenise for the lexical guard | 65.0ms | 30.3ms | **2.14x** | **98.4%** |
| term-in-sentence scan (`prose_windows`) | 20.4ms | 25.3ms | 0.81x | yes |
| token-set intersection (`_lex_ok`) | 38.8ms | 80.7ms (numkong) | 0.48x | yes |

The one real speedup, `utf8_wordbreaks` + `utf8_uncased_fold` for tokenisation, disagrees with
apsw's UAX#29 word iterator on 1.6% of tokens. `_toks` feeds the lexical guard, so a 1.6%
disagreement is 1.6% of merge decisions changed — that is a different entity resolver, not a faster
one. It is also `lru_cache`d, so the tokenisation that would be sped up is the small share that
misses the cache.

Python's `in` on `str` is already SIMD-accelerated (a two-way / `memchr` hybrid), which is why
`sz.contains` cannot beat it on sentence-length haystacks. The general shape: these libraries win on
long strings and large arrays, and the graph layer is a very large number of very small comparisons.
Neither is a dependency of litesearch, and on this evidence neither should be.

## Is every part of `resolve_entities` necessary?

`python -m evals.ingest_bench parts --sizes 2000` disables one rule at a time and re-resolves *the
same graph* — the database is built once and copied per configuration — so each row is that part's
contribution rather than a different corpus.

| configuration | wall | merges | merged pairs | vs full |
|---|---|---|---|---|
| full | 2.16s | 3,231 | 7,111 | |
| no lexical pass | 1.22s | 1,400 | 2,215 | −4,896 |
| no ANN pass | 1.94s | 3,235 | 7,706 | +1,879 / −1,284 |
| no acronym rule | 1.99s | 3,231 | 7,111 | none |
| no digit guard | 1.91s | 3,233 | 7,124 | +13 |
| no clique guard | 2.08s | 6,422 | **14,842,045** | +14,834,934 |

- **The clique guard is load-bearing and then some.** Without it 6,422 merges collapse into
  transitive blobs covering 14.8 million pairs — every entity in a chain pulled into one node. It
  is the difference between entity resolution and entity destruction.
- **The lexical pass does most of the work**: 69% of merged pairs, and it is the only pass that
  works without embeddings at all.
- **The ANN pass is not purely additive.** Removing it *gains* 1,879 pairs and loses 1,284: because
  the clique guard is order-dependent, merging on embeddings first blocks some lexical merges that
  would otherwise be accepted. Net merge count barely moves (3,231 vs 3,235). It costs ~0.2s here,
  but it also costs embedding every entity and rebuilding the index during the build, and on this
  corpus it is close to a wash. It is still the only mechanism that can reach names sharing no
  tokens, which blocking cannot do by construction — so the case for it is coverage on corpora
  with more such pairs, not merge count on this one.
- **The acronym rule is idle on prose.** Identical partition with it disabled. `_EXACT_KINDS`
  already excludes symbols and modules from resolution, so the corpora where acronyms appear are
  the corpora where the rule is skipped. It is now free (it runs only after the token test fails),
  so there is no cost argument for removing it — but it is not earning anything either.
- **The digit guard is nearly free and catches 13 bad merges** of the `python 3.11` / `python 3.12`
  kind. Keep.

## A reproducibility bug this turned up

`resolve_entities` was not reproducible. Two resolves of the *same database* in two processes came
back with different partitions and merge counts drifting over a range of three.

It was not HNSW, which is what it looks like and what the 0.1.17 notes assumed. `verify_ann_probe`
shows the ANN candidate set is stable across runs; the variance is entirely in `by_lexical`. The
cause is that `_toks` returns a `frozenset`, so its iteration order follows string hashes and
therefore `PYTHONHASHSEED`. That decided the insertion order of the inverted index in
`_lexical_pairs`, which decided the order pairs were proposed in, and `_uf_union` only accepts a
merge that keeps the group a clique — an order-dependent test.

Walking the tokens in sorted order fixes it. Four resolves of the same database under random hash
seeds now return the same partition and the same edge table, bit for bit, and blocking recall is
unchanged (99.5% / 98.3% / 97.2% at 3,254 / 5,650 / 8,487 entities).

## The same three questions, asked of `core`, `tree` and `sanskrit`

The graph findings fell into three shapes — writes outside a transaction, a pure function
recomputed, and a per-item loop that should be one query — so the other modules were checked for
the same shapes rather than re-profiled from scratch.

| | before | after | |
|---|---|---|---|
| `add_doc` x150 | 2.82s | **2.25s** | 1.25x |
| `toc()` over 300 documents | 26.6ms | **16.9ms** | 1.57x |
| `fold_token` over 70,413 tokens | 114.7ms | **20.8ms** | 5.5x |
| `search`, `fts_search` | | | unchanged |

- **`fold_token` was uncached.** It is called once per token on every FTS write *and* every query,
  and it is three unicode normalisations plus a per-character category scan deep. The eval corpus
  is 70,413 tokens and 4,153 distinct strings, so 94% of calls re-answer a question already
  answered. In a `cProfile` of `add_doc` x40 (1.84s) the Sanskrit tokenizer chain was 0.61s
  cumulative and `fold_token` half of that — on a corpus with no Sanskrit in it at all. It is a
  pure function of its argument, so an `lru_cache` is a memo and nothing else. Incidental effect on
  `build_graph`, whose entity store is also FTS-indexed: ~6%, measured with the cache stashed.
- **`add_doc` committed the doc row and the node rows separately**, so ingesting a directory paid
  two fsyncs per document where one would do. `delete_doc` had the same shape across three tables,
  with the added problem that a reader landing between the second and third commit saw a document
  whose nodes had vanished.
- **`toc()` ran one query per document** to build the listing whose entire selling point is being
  the cheap, vectorless way to look at a corpus. One query, grouped in Python.

Output is identical: doc, node and chunk rows, `toc()` at three depths, FTS results and Sanskrit
folding over mixed Devanagari/IAST all hash the same before and after (once the wall-clock
`added_at` / `uploaded_at` columns, which are not reproducible in either version, are excluded).

### Threading `search`'s two legs: measured, and it loses

`search` runs its FTS and vector legs one after the other and carries a `parallel=` flag documented
as "not used". The legs look independent and both drop into C — apsw for FTS, usearch for the ANN
probe — so a thread each is the obvious idea. `python -m evals.ingest_bench legs`:

| chunks | vector leg | serial | threaded | |
|---|---|---|---|---|
| 2,458 | exact | 5.90ms | 9.81ms | 0.60x |
| 2,458 | ann | 1.36ms | 2.03ms | 0.67x |
| 12,391 | exact | 26.00ms | 37.64ms | 0.69x |
| 12,391 | ann | 3.34ms | 4.09ms | 0.82x |

Slower at both sizes and in both modes. The legs share one apsw connection, so SQLite serialises
them behind its own mutex and the pool buys contention plus two futures instead of overlap. The
gap narrows as the vector leg grows, which is the shape you would expect if the win were coming —
but the ceiling is `min(fts, vec)` and the legs are never close to balanced. The flag stays
deprecated, now on evidence.

This is the same answer threads got in the graph layer, for a different reason: there it was the
GIL (yake is Python end to end), here it is the connection mutex. Processes remain the only thing
that has actually paid in this codebase.

## Two writes and a walk

| | before | after | |
|---|---|---|---|
| `cooccur_edges` (15,233 edges) | 782ms | **214ms** | 3.65x |
| `build_graph` (2,000 chunks) | 7.02s | **6.00s** | 1.17x |
| `add_dir` over 8 PDFs | 6.18s | **3.54s** | 1.75x |
| `add_dir` over 150 markdown files | 2.24s | 2.17s | declines the pool |

**`insert_all(upsert=True)` emits two statements per row.** An `INSERT OR IGNORE ... RETURNING *`
to create the row from its key columns, then an `UPDATE ... RETURNING *` to fill the rest — each
separately prepared, each materialising a result row nobody reads. That is 23,530 statements to
write 11,765 mentions. The net effect of the pair is exactly `ON CONFLICT DO UPDATE`, which SQLite
has had since 3.24, so `core.upsert_all` does it in one prepared statement per batch. The pure
edge-write path is 3.65x; a whole `build_graph` is 1.17x because extraction still dominates it.

**`add_dir` parsed every file in the writing process**, and the fix is only a fix for some file
types — which is why it needed measuring rather than applying. Over eight PDFs the parse is 3.55s
of a 5.31s walk, two thirds of it and embarrassingly parallel. Over 150 markdown files the same
"parse" is `read_text` and takes **2.7ms in total**, so a pool there is pure loss — forcing one
costs 2.37s against 2.24s serial. `add_dir(n_workers=None)` therefore decides on how many files
have a parse worth splitting (`PARSE_HEAVY`), not on how many files there are.

Writing stays in the parent: there is one connection, one FTS index and one HNSW graph, and none of
them is improved by being contended for. `add_file` is now a `_parse_doc` (plain data in, plain
data out, picklable) plus an `_add_parsed`, so the two halves can be scheduled separately. A worker
resolves a profile *by name* rather than re-detecting one, and returns `None` if its registry lacks
that profile, so the parent re-parses the file itself instead of letting a different reader through
unnoticed.

Ingest output is byte-identical: doc, node and chunk rows hash the same serial, auto and
`n_workers=4`, over both PDFs and markdown.

## Batching the encoder across documents

`add_dir` called the encoder once per document — about 20 chunks — and `embed_batch` (default
2,000) now widens that across documents, flushing on a chunk count so a large directory does not
hold its own text twice. How much that is worth depends entirely on the encoder, which is why it is
a parameter and not a constant (`python -m evals.ingest_bench embatch`, 2,025 chunks):

| chunks per call | static encoder |
|---|---|
| 20 (one document) | 336ms |
| 64 | 290ms |
| 256 | 272ms |
| everything, one call | 272ms |

**A static model gains ~19% and flattens by 64 chunks a call.** It is a lookup table with a mean
pool on top: there is no kernel launch and no graph execution to amortise, and model2vec already
batches internally. An ONNX model has real per-call setup and should gain more — run the bench with
the encoder you actually use before choosing a number.

End to end the encoder is not where the time is either: with the static encoder, embedding is 0.46s
of a 4.31s `add_dir` over 150 markdown files. Most of the 3.08s → 2.46s came from the *writes*
being batched alongside it, and from `process_content` now going through `upsert_all` — it was
still emitting apswutils' two statements per row, 5,912 of them for 2,956 chunks. The row id is now
hashed in `process_content` rather than left to `hash_id=`, which is what lets the single-statement
form be used; the hash is `hash_record` over the same columns, so ids are unchanged and a re-ingest
still lands on the same row.

### A bug this turned up: duplicate content took the wrong heading

Chunks are embedded as `heading ⏎⏎ content` and stored as bare content. The heading was attached
through a `{content: heading}` dict rebuilt per document, which collapses on duplicate content —
so two chunks with the same text under different headings were **both** embedded with whichever
heading came last. Pairing by position instead is what makes a cross-document batch safe, and it is
also just correct.

It is not a rare case: re-ingesting the eight-PDF corpus, 148 of 3,402 rows (**4.35%**) come back
with a different embedding. Same ids, same content, same nodes — every changed row is one whose
content appears more than once under more than one heading. If you have a store built before this,
those rows are embedded against the wrong heading and re-ingesting will fix them.

## Is there a yake that does not hold the GIL?

Yes — `yake-rust`, and it is the same algorithm. `python -m evals.ingest_bench kw`:

| extractor | serial | per chunk | 4 threads | GIL-free? | agrees with yake |
|---|---|---|---|---|---|
| yake (shipped) | 3.02s | 7.5ms | 7.16s | no | — |
| **yake-rust** | **0.46s** | **1.1ms** | **0.16s** | **yes** | 66.5% |
| rake-nltk | 0.20s | 0.5ms | 0.61s | no | 21.6% |
| rakun2 | 1.70s | 4.2ms | 2.42s | no | 23.9% |

`yake-rust` is **6.6x faster per call** and scales on *threads* (0.46s → 0.16s on four), so it
would not need the process pool `build_graph` currently carries for extraction — no fork, no
pickling, no `MIN_PARALLEL_CHUNKS` threshold. The two pure-python candidates both go *slower* on
threads, which is the GIL showing up exactly where it should.

The catch is the last column. 66.5% agreement is the same algorithm disagreeing about ranking
inside the top 12, so a swap is a graph change and wants a retrieval eval, not just a stopwatch.
rake-nltk is the fastest thing here and the least comparable: RAKE returns long noun phrases
(`'council directive 93'`) where YAKE returns tight keyphrases, so its 21.6% is a different
extractor doing a different job, not a cheaper one. Whether that is better is a quality question
`evals/run.py` can answer and this benchmark cannot.

Nothing is swapped: `build_graph(terms_fn=...)` is the seam for it, and the default is unchanged.

## What is still open

- **`build_graph` is yake-bound on the serial path and pool-bound on the parallel one.** Extraction
  is ~99% of `prose_windows` and scales 3.5x on 4 cores. Nothing in litesearch's own code is
  material next to it; the next real gain is a faster keyphrase extractor, not a faster reduce.
- ~~**Mentions and edges still go through `insert_all`.**~~ Fixed — `core.upsert_all`.
- **Extraction is still the ceiling on `build_graph`.** `yake-rust` is 6.6x per call and releases
  the GIL, so the process pool could become a thread pool; it needs a retrieval eval first, because
  it agrees with the shipped yake on 66.5% of terms.
- **The ANN rebuild and the FTS rebuild are now the two biggest slices of `add_dir`** — 1.47s and
  1.03s of 4.31s with a static encoder. Both are once-per-walk and both are doing real work
  (`usearch.compiled.add_many` builds the HNSW graph); neither is obviously wrong, and neither has
  been attacked.
- **`_adjacency` set iteration leaves PPR summation order machine-dependent** at the 1e-15 level.
  Harmless for ranking today; it would need pinning if PPR masses were ever compared across runs.
- **The ANN pass's contribution is corpus-dependent and only measured on one corpus.** Re-run
  `parts` on a code or mixed-genre store before concluding anything general from the table above.

[usearch]: https://github.com/unum-cloud/USearch
[StringZilla]: https://github.com/ashvardanian/StringZilla
[NumKong]: https://github.com/ashvardanian/NumKong
