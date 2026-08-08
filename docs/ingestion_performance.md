# Ingestion performance

Measured on the `evals` corpus (9 EU directives, 489 pages, 1.12M characters) with
`evals/ingest_bench.py`. 4 CPUs, SQLite in WAL mode, `hash_embed` standing in for an encoder so the
numbers are litesearch's own overhead rather than ONNX inference.

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

| # | Finding | Measured | Scope |
|---|---------|----------|-------|
| 1 | `add_doc` rebuilds the whole ANN index per document | exponent **1.72**; 6.7x slower at 400 docs, unbounded | docs, PDFs, folders |
| 2 | `process_content` commits per insert chunk | **6.7x** with one transaction | every write path |
| 3 | FTS5 triggers during bulk load | **2.0x** building FTS after the insert | every write path |
| 4 | `pyparse` re-splits the file per node | **7.2x**, byte-identical output | code files |
| 5 | `dir2chunks` uses a thread pool for CPU-bound parsing | thread **0.76x** vs serial, process **3.37x** | code files, folders |
| 6 | PDF parsing is serial and GIL-bound | process pool **1.90x** on 4 cores | PDFs, folders |
| 7 | `resolve_entities` is superlinear | exponent **1.35** | graphs |

Compounded, the document path is ~6.7x at 400 documents and grows with corpus size; the code path is
~7.2x serial and ~24x with a process pool.

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

`sync()` already does this correctly: `_sync_index` adds and removes keys incrementally rather than
rebuilding. `add_doc` is the path that does not use it.

## 2. Writes commit per insert chunk

`process_content(parallel=False)` — the default, and what `add_doc` uses — calls `insert_all`
outside any transaction, so apswutils commits per chunk and WAL fsyncs each time. 8,000 rows:

```
no fts, autocommit      4.56s    1,754 rows/s
no fts, one txn         0.50s   16,023 rows/s     9.1x
fts, autocommit         7.44s    1,076 rows/s
fts, one txn            1.46s    5,462 rows/s     5.1x
```

The win is already implemented — `process_content(parallel=True)` wraps chunked
`BEGIN IMMEDIATE` transactions and reaches 7,207 rows/s, **6.7x** the default — but it is behind a
parameter named for concurrency, and neither `add_doc` nor `sync` passes it. The batching is a pure
throughput property; only the widened busy timeout has anything to do with parallelism.

## 3. FTS5 triggers cost 2x during bulk load

With the transaction fix in place, the remaining gap between `no fts` (16,023 rows/s) and
`fts` (5,462 rows/s) is the per-row trigger maintaining the FTS index. Creating the FTS index
*after* the bulk insert instead:

```
triggers during insert   1.09s   8,000 rows   fts_rows=8000   hits('vessel')=17
enable_fts afterwards     0.53s   8,000 rows   fts_rows=8000   hits('vessel')=17
```

2.05x, with an identical FTS row count and identical search results — the index is fully populated,
not skipped. This only applies to bulk load into a new or empty store; incremental updates still
want the triggers.

## 4. `pyparse` re-splits every file once per node

`ast.get_source_segment(code, n)` splits the entire source into lines on every call, and `pyparse`
calls it once per chunk it emits — O(top-level defs x file size) per file. In a profile of 400
files from `transformers`, `ast._splitlines_no_ff` is **47.3s of 70.0s (68%)**.

Two further passes are redundant. `pyparse` walks the whole tree tagging every node with a `parent`
attribute, then filters on `is_p_mod` — "parent is the Module" — which is what `tree.body` already
means. The walk and the tagging together are another ~12s of that 70s.

Hoisting the line split to once per file (using the same `ast._splitlines_no_ff`, so the byte-offset
semantics are preserved exactly) and iterating `tree.body`:

```
pyparse        23.37s   2,525 chunks
pyparse_fast    3.25s   2,525 chunks    7.2x
identical chunks (content + metadata): 2525/2525
```

Verified byte-identical across `imports=True`, `assigns=True` and both together.

## 5. `dir2chunks` and `pkg2chunks` use a thread pool for CPU-bound work

Both call `parallel(file_parse, ..., threadpool=True)`. `file_parse` is `ast.parse` plus pure-Python
tree walking, all of which holds the GIL. Over 2,170 files:

```
serial       145.30s    14.9 files/s
thread(4)    190.44s    11.4 files/s    0.76x   (slower than serial)
process(4)    43.10s    50.4 files/s    3.37x
```

The thread pool is not merely ineffective, it is a 24% penalty from contention.

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

The superlinear term is `_lexical_pairs`, which enumerates all pairs within each shared-token block.
`max_group=60` caps a single block but not the number of blocks, and blocks grow with the corpus.

40 chunks/s is also simply slow in absolute terms: a million chunks is ~7 hours of graph
extraction on one core. `build_graph` accumulates `ents`, `mens`, `edges` and `wins` in memory for
the entire call, so it also cannot be handed a million chunks in one go.

## On sharding: one database per profile

Sharding helps today, but mostly because it divides the quadratic. 400 documents:

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

So the ordering matters: fix the per-document rebuild first, then shard for the reasons sharding is
actually good — bounding the resident HNSW index (usearch holds vectors in memory, so ~1M x 512 dims
of float16 is ~1 GB plus graph overhead per index), isolating tenants, allowing independent
re-ingest, and enabling cross-process parallelism. Sharding for ingest *throughput* is paying
query latency for a problem that has a cheaper fix.
