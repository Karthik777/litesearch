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
the 18 chunks, not the whole corpus. It costs an LLM pass at ingest, so it belongs in a layer that
already has a model (vishalakshi), not in litesearch core. Before shipping it must hold on a full
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
