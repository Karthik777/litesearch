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
