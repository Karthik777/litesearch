# Ingestion performance

Measured on the `evals` corpus (9 EU directives, 489 pages, 1.12M characters) with
`evals/ingest_bench.py`. 4 CPUs, SQLite in WAL mode, `hash_embed` standing in for an encoder so the
numbers are litesearch's own overhead rather than ONNX inference.

**All seven findings below are fixed as of 0.1.16.** Each section states the problem as it was
measured and then what it costs now; the "after" figures come from re-running the same benchmarks
against the fixed code. Headline: a 400-document directory went from 112.2s to 9.2s (**12.2x**),
and its growth exponent from 1.64 to 0.91 — linear, which is the property that matters at 10^6.

Every figure below is reproducible:

```
python -m evals.ingest_bench docs   --sizes 25,50,100,200
python -m evals.ingest_bench defer  --sizes 50,100,200,400
python -m evals.ingest_bench shards --sizes 400
python -m evals.ingest_bench graph  --sizes 500,1000,2000
python -m evals.ingest_bench all
```

Sizes are swept rather than run once, because per-document cost is only a *cost* if it is constant.
The exponent of the growth curve is the number to read: 1.0 is linear, and anything approaching 2.0
does not reach a million documents at any hardware budget.

## Summary

| # | Finding | Gain | Fix |
|---|---------|------|-----|
| 1 | `add_doc` rebuilt the whole ANN index per document (exponent 1.72) | **12.2x** at 400 docs, exponent → 0.91 | `add_doc` syncs its own keys; `add_dir` rebuilds once |
| 2 | `process_content` committed per insert chunk | **6.7x** (1,076 → 7,207 rows/s) | always batch in `write_txn` |
| 3 | FTS5 triggers ran per row during bulk load | **2.0x** | `Table.bulk_load()` |
| 4 | `pyparse` re-split the file once per node | **7.1–11.4x**, 0 mismatches / 32,720 chunks | hoist the split, iterate `tree.body` |
| 5 | `dir2chunks` used a thread pool for GIL-bound parsing | **3.3x** (threads were 0.76x — slower than serial) | processes above `MIN_PARALLEL_FILES` |
| 6 | PDF parsing serial and GIL-bound | **1.9x** on 4 cores | same process pool |
| 7 | `resolve_entities` superlinear (exponent 1.35) | **2.3x**, exponent → 1.08, identical merges | one batched HNSW probe, memoised `_toks` |

End to end: the document path is **12.2x** at 400 documents *and no longer degrades with corpus
size*; code ingestion is **26x** (190.4s → 7.3s over 2,169 files).

Every fix is behaviour-preserving, and the ones that could plausibly have changed results were
checked against the implementation they replaced rather than argued: `pyparse` is byte-identical
across 32,720 chunks and all four flag combinations, `resolve_entities` produces the identical merge
outcome (`merged=5896, by_ann=1813, by_lexical=4083, canonical=2591`), and `bulk_load` leaves the
same FTS row count and the same hits.

## 1. `add_doc` rebuilds the entire ANN index once per document

The dominant cost, and the only one that makes a million documents impossible rather than slow.

`add_doc` ends with `if self._ann_meta(store): g.store.rebuild_index()`. `rebuild_index` reads
*every* embedding blob in the store, reconstructs the whole HNSW graph from scratch and writes the
sidecar. Per document that is O(corpus), so a directory of N documents does O(N²) index work — to
produce an index that is only read once ingestion finishes.

```
add_dir(25 md docs)    1.64s    65.5 ms/doc
add_dir(50 md docs)    3.31s    66.3 ms/doc
add_dir(100 md docs)   8.35s    83.5 ms/doc
add_dir(200 md docs)  29.05s   145.2 ms/doc      exponent 1.38
add_dir(400 md docs) 107.14s   267.8 ms/doc      exponent 1.72
```

In a profile of the 200-document run, `usearch.compiled.add_many` is 17.4s of 34.7s — half the
wall clock — with 409,730 row fetches and 407,928 `np.frombuffer` calls, both of which are the
quadratic re-reads.

Deferring the rebuild to once per batch, same corpus, same final index (verified: identical vector
count):

```
                     per-doc rebuild   deferred rebuild
add_dir(50)                   3.11s       2.08s   1.50x
add_dir(100)                  7.77s       3.84s   2.02x
add_dir(200)                 29.49s       8.35s   3.53x
add_dir(400)                107.14s      16.04s   6.68x
exponent                       1.72        1.00
```

The final rebuild is 0.49s of the 16.04s at 400 documents. Throughput goes from 3.7 docs/s and
falling to a flat 24–26 docs/s. A repeat run on an idle machine gave 7.01x at 400 documents and
exponents of 1.64 / 0.88, so the shape is stable and the exact multiplier moves by a few percent.

Benchmark on an otherwise idle machine: a concurrent `nbdev-test` run (2 workers on 4 CPUs) was
enough to turn the 50-document case into 55.9s against 3.7s, while leaving the vector counts
identical.

**A related trap:** `add_doc` calls `self.get_tree(store, prefix)` internally, which takes
`get_tree`'s default `ann=True`. A caller who opted out with `db.get_tree('store', ann=False)` gets
the ANN index — and the per-document rebuild — registered underneath them on the first `add_doc`:

```
ann registered after get_tree(ann=False): False
ann registered after add_dir:             True
```

`sync()` already did this correctly: `_sync_index` adds and removes keys incrementally rather than
rebuilding. `add_doc` was the path that did not use it.

**Fixed.** `add_doc` now mirrors only its own document's keys into the index, `add_dir` treats the
walk as a bulk load (one `bulk_load` for FTS, one `rebuild_index` at the end), and `_tree_ann`
stops `add_doc` re-registering an index the caller opted out of. `delete_doc` drops just the
deleted rows' keys instead of rebuilding. Re-running the same benchmark:

```
add_dir(25 md docs)    0.80s    32.1 ms/doc
add_dir(50 md docs)    1.21s    24.1 ms/doc
add_dir(100 md docs)   2.16s    21.6 ms/doc
add_dir(200 md docs)   5.06s    25.3 ms/doc
add_dir(400 md docs)   9.21s    23.0 ms/doc      exponent 0.91
```

112.2s → 9.2s at 400 documents (**12.2x**), and per-document cost is now flat rather than climbing.
The index invariant — `index.size` equals the number of embedded chunks after a bulk load, after a
single `add_file`, and after a `delete_doc` — is asserted in `nbs/06_tree.ipynb`.

## 2. Writes commit per insert chunk

`process_content(parallel=False)` — the default, and what `add_doc` uses — calls `insert_all`
outside any transaction, so apswutils commits per chunk and WAL fsyncs each time. 8,000 rows:

```
no fts, autocommit      4.56s    1,754 rows/s
no fts, one txn         0.50s   16,023 rows/s     9.1x
fts, autocommit         7.44s    1,076 rows/s
fts, one txn            1.46s    5,462 rows/s     5.1x
```

The win was already implemented — `process_content(parallel=True)` wraps chunked
`BEGIN IMMEDIATE` transactions and reaches 7,207 rows/s, **6.7x** the default — but it sat behind a
parameter named for concurrency, and neither `add_doc` nor `sync` passed it. The batching is a pure
throughput property; only the widened busy timeout has anything to do with parallelism.

**Fixed.** `process_content` always batches through `write_txn`; `parallel` now controls only the
busy window, which is what it always meant. End to end through the benchmark's embed-and-store
path, where `hash_embed` is now the limiting factor rather than the writes:

```
                        before     after
process_content(2000)    527/s     913/s
process_content(8000)    513/s     912/s
process_content(32000)   469/s     912/s
exponent                  1.04      1.00
```

## 3. FTS5 triggers cost 2x during bulk load

With the transaction fix in place, the remaining gap between `no fts` (16,023 rows/s) and
`fts` (5,462 rows/s) is the per-row trigger maintaining the FTS index. Creating the FTS index
*after* the bulk insert instead:

```
triggers during insert   1.09s   8,000 rows   fts_rows=8000   hits('vessel')=17
enable_fts afterwards     0.53s   8,000 rows   fts_rows=8000   hits('vessel')=17
```

2.05x, with an identical FTS row count and identical search results — the index is fully populated,
not skipped. This only applies to bulk load; `rebuild` is O(rows in the table), so wrapping a
two-row update in it is strictly slower than letting the triggers run.

**Fixed.** `Table.bulk_load()` is a context manager that suspends the triggers and rebuilds once on
exit, replaying the captured trigger DDL verbatim so the tokenizer chain survives. `add_dir` uses
it for the whole walk.

## 4. `pyparse` re-splits every file once per node

`ast.get_source_segment(code, n)` splits the entire source into lines on every call, and `pyparse`
calls it once per chunk it emits — O(top-level defs x file size) per file. In a profile of 400
files from `transformers`, `ast._splitlines_no_ff` is **47.3s of 70.0s (68%)**.

Two further passes are redundant. `pyparse` walks the whole tree tagging every node with a `parent`
attribute, then filters on `is_p_mod` — "parent is the Module" — which is what `tree.body` already
means. The walk and the tagging together are another ~12s of that 70s.

Hoisting the line split to once per file (using the same `ast._splitlines_no_ff`, so the byte-offset
semantics are preserved exactly) and iterating `tree.body`:

**Fixed**, and verified against the implementation it replaced over all 2,169 python files:

```
                                    old        new           chunks        mismatches
default                          159.80s     22.36s   7.1x   14,444 both        0
imports=True, assigns=True       281.89s     24.66s  11.4x   32,720 both        0
```

`nbs/02_data.ipynb` pins `_seg` against `ast.get_source_segment` on the cases that actually differ
between the two line-splitters: a decorated def, a one-line class, a form feed, and a non-ascii
line before the node.

## 5. `dir2chunks` and `pkg2chunks` use a thread pool for CPU-bound work

Both call `parallel(file_parse, ..., threadpool=True)`. `file_parse` is `ast.parse` plus pure-Python
tree walking, all of which holds the GIL. Over 2,170 files:

```
serial       145.30s    14.9 files/s
thread(4)    190.44s    11.4 files/s    0.76x   (slower than serial)
process(4)    43.10s    50.4 files/s    3.37x
```

The thread pool is not merely ineffective, it is a 24% penalty from contention.

**Fixed.** `_parse_files` puts both on a process pool once a directory is worth one — below
`MIN_PARALLEL_FILES` (64) the pool costs more to start than the parse it saves, so small
directories stay serial. `threadpool=` and `n_workers=` remain available. With the `pyparse` fix
on top, the same 2,169 files:

```
serial (n_workers=0)      17.67s
threads (old default)     24.00s
processes (new default)    7.34s
```

190.4s on the old default against 7.34s — **26x** for code ingestion end to end.

## 6. PDF parsing does not thread either

`pdf_parse` is Rust (pdf-oxide) but holds the GIL, and `add_dir` walks files serially. Over the
8 corpus PDFs (489 pages):

```
serial         4.07s   120.2 pages/s
thread(4)      4.41s   111.0 pages/s   0.92x
process(4)     2.14s   228.3 pages/s   1.90x
thread(8)      3.92s   124.8 pages/s   1.04x
process(8)     2.68s   182.7 pages/s   1.52x
```

Single-document parse rates for reference: 120 pages/s on the text-heavy directives, 34 pages/s on
an image-heavy arXiv paper.

## 7. Graph build is linear; entity resolution is not

`build_graph` holds at ~40 chunks/s across sizes (exponent 0.99), but `resolve_entities` grows at
exponent **1.35**:

```
build_graph(500)    12.71s   3,254 ents    resolve  3.61s
build_graph(1000)   24.97s   5,650 ents    resolve  7.43s
build_graph(2000)   50.22s   8,487 ents    resolve 13.26s
```

`_lexical_pairs` was the obvious suspect and the profile said otherwise. Of 23.0s: 10.3s was
`ann_search` — one HNSW probe *plus one `rowid IN (...)` query* per entity, 8,487 of each — and
4.2s was `_toks`, called 252,813 times on the same few thousand strings because the lexical guard
re-tokenises both sides of every candidate pair.

**Fixed** without touching the blocking strategy, so the candidate set is unchanged: `_ann_pairs`
does one batched probe over the whole matrix (usearch parallelises internally) with a single row
fetch, and `_toks` is `lru_cache`d.

```
              before    after
n=500          3.61s    2.01s
n=1000         7.43s    3.52s
n=2000        13.26s    5.65s     2.3x
exponent        1.35     1.08
```

The merge outcome is identical at every size — `merged=5896, by_ann=1813, by_lexical=4083,
canonical=2591` at n=2000, matching the pre-change run exactly.

Still open: `build_graph` holds at ~40 chunks/s (~7 hours per million chunks on one core) and
accumulates `ents`, `mens`, `edges` and `wins` in memory for the whole call, so it cannot be handed
a million chunks in one go regardless of speed. Feed it in batches.

## On sharding: one database per profile

Sharding helped, but mostly because it divided the quadratic. 400 documents, measured *before* the
fixes:

```
                    per-doc rebuild    deferred rebuild
1 shard                  111.82s            19.76s
4 shards                  33.19s            17.43s
8 shards                  23.76s            17.53s
```

Unfixed, 8 shards buys 4.7x. Fixed, sharding buys ~12% and then flattens — and **the fix alone on a
single database (19.76s) beats 8-way sharding without it (23.76s)**.

Federated search pays for the shards on the read side, linearly:

```
1 shard     3.8 ms/query
2 shards    5.2 ms/query
4 shards    8.4 ms/query
8 shards   15.7 ms/query
```

So the ordering mattered, and it has now been done in that order. With the rebuild fixed, shard for
the reasons sharding is actually good — bounding the resident HNSW index (usearch holds vectors in
memory, so ~1M x 512 dims of float16 is ~1 GB plus graph overhead per index), isolating tenants,
allowing independent re-ingest, and enabling cross-process parallelism. Sharding for ingest
*throughput* is paying query latency for a problem that had a cheaper fix.

## What is still open at 10^6

- `build_graph` at ~40 chunks/s, single-core, accumulating in memory for the whole call. Batch it.
- Embedding, which these numbers deliberately exclude. `hash_embed` stands in for the encoder; a
  real model is the dominant cost once litesearch's own overhead is out of the way, and it is the
  part that wants a GPU or a process pool.
- `add_dir` walks files serially. `_parse_files` shows the shape of the fix; the document path has
  not had it applied, because PDF parsing and SQLite writing want different pool sizes.
