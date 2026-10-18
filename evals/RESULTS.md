# Eval results

Numbers behind the defaults, with the method. `python -m evals.decide` reproduces the retrieval
sweeps from `evals/results/*.json`; `python -m evals.multihop --evaluate` reproduces the bridge
set below. Paired bootstrap over queries, a CI spanning zero reported as no difference.

## The entity graph leg does not earn its place (removed from `context`)

The `context(graph=True)` seam that dispatched to vruksha's `graph_search` was removed. The graph
leg measures negative on the standard set (−0.070 to −0.160 weighted MRR) and buys almost nothing
on the one set built to favour it, at 3-5x latency and a ~21% cost to ordinary retrieval.

### Method

`build_tree` + `build_graph` (vruksha `build_graph` + `resolve_entities` + `topic_nodes`) over the
`c512`/`bge-small`/`tree` store for two genres: regulatory (EU legislation, real hierarchy and
cross-reference) and arxiv (entity-rich prose, the graph's best case). Retrieval scored with
`evals/multihop.py`: 120 bridge queries per genre where the answer shares no token with the query,
so FTS cannot reach it and the graph leg has room to bridge. `target` is the bridge, `source` the
easy chunk FTS should nail, reciprocal rank over the top 50.

### Bridge retrieval: graph vs hybrid+tree

| genre | strategy | target MRR | target hit | source MRR | ms |
|---|---|---|---|---|---|
| regulatory | hybrid | 0.0106 | 0.150 | 0.209 | 11 |
| regulatory | graph w=0.25 | 0.0125 | 0.158 | 0.198 | 58 |
| regulatory | graph w=0.5 | 0.0135 | 0.150 | 0.171 | 57 |
| regulatory | graph w=1.0 | 0.0147 | 0.192 | 0.162 | 59 |
| arxiv | hybrid | 0.0546 | 0.550 | 0.376 | 10 |
| arxiv | graph w=0.25 | 0.0591 | 0.533 | 0.326 | 33 |
| arxiv | graph w=0.5 | 0.0658 | 0.525 | 0.306 | 32 |
| arxiv | graph w=1.0 | 0.0689 | 0.608 | 0.296 | 31 |

Bridge gain at the best weight is +0.004 (regulatory) and +0.014 (arxiv) absolute target MRR. Over
the same weights `source` MRR drops ~21% (0.209 to 0.162; 0.376 to 0.296): turning the graph up to
win a bridge makes ordinary retrieval worse. Latency is 3-5x. On the standard query set (not
bridges) the leg is a flat loss, `evals/results/graph.json`.

### Why: the graph is co-occurrence keyphrase soup, not knowledge

`build_graph` on these stores: 14,070 keyphrase entities and 5,821 edges (arxiv). Sampled:

- entities are yake fragments and generic words: `auth`, `text`, `supplies`, `needed to ensure`
  (regulatory); `susan`, `wei`, `john`, `lan-guage processing` (a hyphenation artifact) (arxiv).
- top-degree hubs are domain stopwords and PDF furniture: `graph`, `distance`, `table`, `figure`.
- highest-weight (w=1.0) edges are (a) fragments of one phrase yake split
  (`including large-scale computer` — `including large-scale` — `computer resources`) and
  (b) co-authors from reference lists (`landauer` — `deerwester` — `dumais`).

No typed relations, no hierarchy. The edges are statistical association, which is why the leg
cannot bridge on grounds the vector leg does not already cover.

### Decision

Cut from litesearch: `context(graph=True)` is gone. `get_graph`, `topic_nodes` and `clusters`
stay: they are ANN clustering for topic maps, not the retrieval graph leg, and are measured
separately (`evals/results/cluster.json`). `vruksha` remains an external package for anyone who
wants to build the graph directly. A typed graph (relation edges, not co-occurrence) built with an
LLM is the only version that could clear this bar, and it must clear it on this same harness
without the source-MRR regression before it ships.

## A typed (LLM-built) graph does clear the bar, on cross-reference bridges

Prototype, not shipped. The PMI graph loses because its edges are co-occurrence. A graph whose
edges are typed relations extracted by an LLM is a different artifact, and on the one query class
the whole graph idea exists for (word-disjoint cross-reference bridges) it wins by a wide margin.

### Method

18 cross-reference-dense chunks of one EU directive (dir_2006_112) handed to Claude, which
extracted typed entities and relations (`refers_to`, `defines`, `part_of`, `subject_to`, ...) plus
a node summary per chunk, canonicalizing article and concept names so a reference collapses to one
node. 117 entities, 106 relations over 18 chunks. Retrieval over the full regulatory store (3585
chunks). Bridges: query = distinctive tokens of a referencing chunk absent from the referenced
article; target = that article's chunks. `typed` = hybrid RRF-fused with a leg that follows the
LLM `refers_to` edges from the top hybrid seeds. Reciprocal rank over top 50.

Ground truth built two ways: from the LLM edges (25 bridges) and, to remove circularity, from a
regex over the raw text independent of the LLM (18 bridges).

### Cross-reference bridge retrieval, full regulatory store

| ground truth | method | target MRR | target hit |
|---|---|---|---|
| LLM edges | hybrid | 0.0066 | 0.12 |
| LLM edges | pmi-graph w=1 | 0.0091 | 0.08 |
| LLM edges | typed-graph | 0.2069 | 0.72 |
| regex (independent) | hybrid | 0.0028 | 0.11 |
| regex (independent) | pmi-graph w=1 | 0.0053 | 0.17 |
| regex (independent) | typed-graph | 0.1942 | 0.61 |

The typed leg reaches the referenced article 61-72% of the time against 8-17% for hybrid and the
PMI graph, ~30-70x the MRR. On the regex ground truth the LLM's `refers_to` recall was 1.00: it
extracted every cross-reference a regex finds, plus looser ones a regex misses. The PMI graph adds
nothing here: co-occurrence cannot represent "Article X refers to Article Y".

### Caveats and the bar for shipping

One directive, 18 source chunks, ~20 bridges, regulatory (cross-reference dense). The typed leg is
a hand-built RRF expansion, not a productionised `graph_search`, and the graph was extracted over
the 18 chunks, not the whole corpus. It costs an LLM pass at ingest, so it lives in vruksha (the typed-graph package,
model injected by the caller), not in litesearch core. Before shipping it must hold on a full
build across genres, on prose (arxiv) as well as legal, and against the same source-MRR
regression check the PMI leg failed. But the direction is now measured: typed edges bridge, co-occurrence does not.

### Reproducible: a local model builds the typed graph (rishi + llama.cpp)

`evals/typed_graph.py` extracts the graph with a local model through rishi/urai (llama.cpp GGUF),
prompt adapted from artifact-pyramids. Fixtures in `evals/cache/`: the 18 chunks (`typed_subset`)
and two extractions, Claude and Qwen2.5-1.5B-Instruct. `evaluate(sub, tg)` scores the bridges.

| extractor | refers_to recall vs regex | typed target hit | hybrid | pmi |
|---|---|---|---|---|
| Claude | 1.00 | 0.61-0.81 | 0.08-0.12 | 0.08-0.17 |
| Qwen2.5-1.5B (local) | 0.11 | 0.17 (1.00 on its own edges) | 0.06 | 0.06 |

The value tracks extraction quality. The 1.5B model, on CPU through rishi, recovers ~11% of
cross-references but where it emits an edge the typed leg reaches the target every time; overall it
still triples hybrid's hit and lifts MRR ~38x. A larger local model closes the gap toward Claude's
0.61-0.81. So the local-model path is real, and extraction recall (model size, prompt, or a
regex-assisted seed for article citations) is the lever, not the retrieval mechanism.

### Model size is the lever; a 4B local model matches Claude

Same 18 chunks, same bridges. `evaluate` on each extraction; `seed_regex` adds regex-detected
`Article N` citations (perfect precision) before scoring.

| extractor | refers_to recall | typed hit (raw) | typed hit (+regex seed) |
|---|---|---|---|
| Qwen2.5-1.5B | 0.11 | 0.17 | 0.74 |
| Qwen3-4B-Instruct-2507 | 0.67 | 0.87 | 0.86 |
| Claude | 1.00 | 0.65-0.81 | - |
| hybrid / pmi baseline | - | 0.08-0.20 | - |

Qwen3-4B (local, CPU, via rishi) matches or beats Claude on this set and needs no seed. The regex
seed is what rescues a weak model: it lifts Qwen2.5-1.5B from 0.17 to 0.74, so citation extraction
can be handed to a regex and the LLM left to type concepts. Fixtures: `typed_graph_qwen3-4b.json`.

gemma-3n-E4B-it (Q4 GGUF) loads but emits gibberish under llama-cpp-python 0.3.30: its MatFormer /
per-layer-embedding architecture is not correctly supported there. Run gemma on its native LiteRT
runtime (rishi has one), not a llama.cpp GGUF.

### Do all three: comprehensive citation seed + better model + incremental

"Article N" alone is flaky. The seed is now a pattern set over Article, Annex, Chapter, Title,
Section and Regulation/Directive act numbers (`CITE` in `evals/typed_graph.py`), canonicalized so a
reference collapses to one node. It runs beside the model, not instead of it, and the graph is
built incrementally (new chunks upsert onto existing canonical nodes).

Bridges now target every cited heading kind, so the set is larger and harder.

| extractor | bridges | typed hit | hybrid | pmi |
|---|---|---|---|---|
| Qwen2.5-1.5B raw | 3 | 1.00 | 0.00 | 0.00 |
| Qwen2.5-1.5B + comprehensive seed | 23 | 0.61 | 0.13 | 0.13 |
| Qwen3-4B raw | 14 | 0.71 | 0.14 | 0.21 |
| Qwen3-4B + comprehensive seed | 22 | 0.64 | 0.09 | 0.14 |
| Claude + comprehensive seed | 27 | 0.63 | 0.07 | 0.11 |

The seed is what lets a weak model cover the whole citation surface: Qwen2.5-1.5B goes from 3
traversable bridges to 23 at 0.61 hit against 0.13 for hybrid. A strong model reaches most of them
on its own; the seed guarantees the structured ones regardless of model. Loose references ("that
Directive", "the preceding paragraph") still need the model, which is where its size earns out.

## One shared connection against a connection per thread

`database(shared=True)` stays off by default. With SQLite's default settings, a connection per thread is slower than one shared connection for FTS reads, and faster for brute-force vector scans. With SQLite's allocation statistics off, a connection per thread is 6.1 times faster for FTS at 8 threads. A shared connection holds one copy of the HNSW index; a connection per thread holds one copy each.

### Method

`python -m evals.shared_conn --rows 20000 --threads 1,2,4,8 --repeats 3 --n fts=400,vec=20,ann=400`, then the same command with `--memstatus 0`. The store holds 20,000 rows of 60 Zipf-distributed pseudo-words, one random 256-dim float16 unit vector per row, the default tokenizer chain, and an HNSW index (12.6 MB sidecar). FTS queries OR two words from frequency ranks 100 to 1,000 and return the top 10 by bm25. Vector queries return the top 10 by cosine, by brute-force scan (`vec_search`) or by HNSW (`ann_search`). Each thread runs its queries after a warm-up query and a common start. Each cell is the median of 3 runs. The latency runs use 4 FTS readers for 5 seconds on a fresh copy of the store, with and without one writer committing 200-row `process_content` transactions back to back. Memory and open cost come from a fresh process that opens 8 connections and loads the HNSW index on each. Machine: macOS 26.5 on a 14-core arm64 Mac, Python 3.13.1 with the GIL, SQLite 3.53.4.

### Read throughput, queries per second

| reads | threads | shared | own | own / shared | shared, memstatus 0 | own, memstatus 0 | own / shared, memstatus 0 |
|---|---|---|---|---|---|---|---|
| FTS | 1 | 1,116 | 1,131 | 1.01x | 1,303 | 1,308 | 1.00x |
| FTS | 2 | 1,118 | 851 | 0.76x | 1,188 | 2,455 | 2.07x |
| FTS | 4 | 1,134 | 496 | 0.44x | 1,140 | 4,197 | 3.68x |
| FTS | 8 | 608 | 441 | 0.73x | 1,114 | 6,820 | 6.12x |
| vector scan | 1 | 64 | 69 | 1.08x | 71 | 71 | 1.00x |
| vector scan | 2 | 67 | 95 | 1.41x | 72 | 85 | 1.17x |
| vector scan | 4 | 70 | 150 | 2.14x | 53 | 131 | 2.46x |
| vector scan | 8 | 70 | 176 | 2.51x | 73 | 168 | 2.31x |
| HNSW | 1 | 2,960 | 3,020 | 1.02x | 3,161 | 3,318 | 1.05x |
| HNSW | 2 | 4,337 | 4,492 | 1.04x | 4,789 | 4,727 | 0.99x |
| HNSW | 4 | 4,283 | 4,552 | 1.06x | 4,883 | 5,284 | 1.08x |
| HNSW | 8 | 4,705 | 2,748 | 0.58x | 4,076 | 2,587 | 0.63x |

### FTS latency, 4 readers

| connection | memstatus | writer | reads/s | p50 ms | p99 ms | max ms | rows written/s |
|---|---|---|---|---|---|---|---|
| shared | 1 | no | 1,009 | 3.90 | 7.36 | 17.8 | |
| shared | 1 | yes | 39 | 105.98 | 193.13 | 198.4 | 5,520 |
| own | 1 | no | 486 | 7.66 | 17.73 | 25.6 | |
| own | 1 | yes | 274 | 13.44 | 33.01 | 45.4 | 1,640 |
| shared | 0 | no | 1,168 | 3.34 | 6.53 | 16.2 | |
| shared | 0 | yes | 38 | 105.95 | 194.15 | 196.8 | 5,560 |
| own | 0 | no | 4,530 | 0.82 | 1.76 | 3.3 | |
| own | 0 | yes | 2,387 | 1.58 | 3.41 | 17.3 | 4,720 |

One connection costs 1.4 to 1.5 ms to open, 4.8 ms to load the HNSW index, and 17.1 MB resident with the index loaded. The setting does not change these.

### Why a connection per thread is slow by default

SQLite keeps allocation statistics behind one process-wide mutex that every allocation takes. A pilot on 2,000 rows ran the same FTS query through raw apsw with SQLite's C tokenizer (`porter unicode61`), so neither litesearch nor a Python tokenizer was involved. One thread ran 5,104 queries per second. Four threads with a connection each ran 1,721. Four processes with a connection each ran 19,526. With `apsw.config(apsw.SQLITE_CONFIG_MEMSTATUS, 0)` before the first connection, four threads ran 12,706. Through litesearch, a connection per thread fell from 1 to 4 threads with `porter unicode61` (3,159 to 1,209 queries per second) as with the default chain (2,409 to 838), so the Python tokenizer is not the cause.

The setting is process-wide. It has to run before any connection opens in the process, as `apsw.shutdown(); apsw.config(apsw.SQLITE_CONFIG_MEMSTATUS, 0); apsw.initialize()`. With it off, `apsw.memory_used()` returns 0. litesearch does not set it.

### Reading the tables

A writer on a shared connection holds it for each whole transaction. Python's lock does not guarantee waiters an order, and the writer in this benchmark starts its next transaction at once. Readers then wait a median 106 ms. On separate connections, WAL lets readers run beside the writer.

`ann_search` runs the usearch search outside the connection lock, so a shared connection keeps pace with separate connections up to 4 threads. At 8 threads separate connections fall to 0.6 times shared. The cause of that drop is not investigated.

The shared FTS drop from 1,134 to 608 queries per second at 8 threads with statistics on did not repeat with statistics off (1,114). It is not investigated.
