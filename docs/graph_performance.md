# Graph layer performance

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
| `build_graph` serial | 9.5s | 9.2s | — (yake-bound) |
| `clusters`, `peers`, `topic_nodes` | | | unchanged |

Results are unchanged throughout, and that is checked rather than asserted. `build_graph` produces
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

## What is still open

- **`build_graph` is yake-bound on the serial path and pool-bound on the parallel one.** Extraction
  is ~99% of `prose_windows` and scales 3.5x on 4 cores. Nothing in litesearch's own code is
  material next to it; the next real gain is a faster keyphrase extractor, not a faster reduce.
- **Mentions and edges still go through `insert_all`.** ~1.1s of a 2,000-chunk build is apswutils
  building parameterised SQL for 22k rows. A raw `executemany` with `ON CONFLICT` would take most
  of that.
- **`_adjacency` set iteration leaves PPR summation order machine-dependent** at the 1e-15 level.
  Harmless for ranking today; it would need pinning if PPR masses were ever compared across runs.
- **The ANN pass's contribution is corpus-dependent and only measured on one corpus.** Re-run
  `parts` on a code or mixed-genre store before concluding anything general from the table above.

[usearch]: https://github.com/unum-cloud/USearch
[StringZilla]: https://github.com/ashvardanian/StringZilla
[NumKong]: https://github.com/ashvardanian/NumKong
