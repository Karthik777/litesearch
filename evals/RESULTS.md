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

## A static-embedding query router does not beat a coin flip (prototype, not shipped)

Prototype in `evals/router_spike.py`, not wired into `Index` or `database()`. A static embedder
classifies each query and escalates the hard ones to a dearer encoder. Against random escalation at
the same rate the router measures +0.0028 at best and -0.0147 at worst, every interval that matters
spanning zero. `python -m evals.router_spike` reproduces it.

### The premise it started from is wrong about Laya

convaiinnovations/laya is not an embedding model and not a Jina competitor. It is a
ModernBERT-large backbone (421M) trained with RLCD to answer typed yes/no and scoring questions in
one forward pass, and its Router is a sub-millisecond language and script detector that sends Latin
text to the ModernBERT checkpoint and non-Latin scripts to an mmBERT-base one. It routes between
two classifiers by language, not between embedding models by content. minishlab/potion-multilingual-128M
is a real static embedder, Model2Vec distilled from BAAI/bge-m3, 256 dimensions, 101 languages, and
it needs no new dependency because `model2vec` is already a core one. The regulatory corpus is 9
English EU documents, so there is no language to route on. What does carry over from Laya is the
shape: a near-zero-cost classifier over the input gates an expensive resource. That is the version
measured here.

### Method

genre `regulatory`, `c512`, flat stores, `hybrid-pre`, k=10, section-level reciprocal rank per
query from `score_one`. 120 source sentences x 5 flavours = 600 query instances, weighted by
`report.WEIGHTS`. Oracle label per instance: 1 where the dear encoder's RR is strictly higher.
Ties stay cheap (81% of instances on the first pair, 79% on the second), which is the tie-break
that favours the cheap arm and therefore the conservative one for a cost router.

Router features are the L2-normalised static query vector of the raw query text. Two classifiers,
neither needing a new dependency: nearest centroid over the two classes, and closed-form ridge
least squares on +-1 labels. 5-fold cross-validation grouped by source sentence, so the five
flavours of one sentence never straddle the split. The escalation threshold is a quantile of the
training scores, so the operating point is out of sample too. Paired bootstrap over source
sentences, 10,000 resamples, a CI spanning zero reported as no difference.

Two encoder pairs. The first is the one the idea is about. The second exists because the encoder
sweep above already puts `egemma-300m` below `bge-small` on all five regulatory flavours, so the
first pair has no quality gradient to climb and a null result there would say nothing about
routing.

### Quality against cost, held out

`index s` is the embedding cost of the store or stores the arm needs; a routed arm needs both.

| pair | arm | escalation | weighted u_mrr | query ms | index s |
|---|---|---|---|---|---|
| bge-small -> egemma-300m | fixed cheap | 0.00 | 0.7814 | 16.0 | 159 |
| bge-small -> egemma-300m | fixed dear | 1.00 | 0.7819 | 68.9 | 868 |
| bge-small -> egemma-300m | routed, potion-32M + ridge | 0.10 | 0.7761 | 21.3 | 1027 |
| bge-small -> egemma-300m | random escalation at 0.10 | 0.10 | 0.7814 | 21.3 | 1027 |
| bge-small -> egemma-300m | oracle | 0.10 | 0.8288 | 21.3 | 1027 |
| potion-32M -> bge-small | fixed cheap | 0.00 | 0.7513 | 0.11 | 0.3 |
| potion-32M -> bge-small | fixed dear | 1.00 | 0.7814 | 16.0 | 159 |
| potion-32M -> bge-small | routed, potion-32M + ridge | 0.12 | 0.7575 | 2.1 | 159 |
| potion-32M -> bge-small | random escalation at 0.12 | 0.12 | 0.7547 | 2.1 | 159 |
| potion-32M -> bge-small | oracle | 0.13 | 0.8266 | 2.1 | 159 |

On the first pair `egemma-300m` is worth +0.0005 weighted section MRR over `bge-small`, 95% CI
[-0.0248, +0.0253], at 4.3x the query cost and 5.5x the index cost. There is nothing to escalate
to. On the second pair the gradient is real, +0.0301 [+0.0018, +0.0580], and a perfect router would
reach 0.8266 for 2.1 ms a query, which is the headroom the whole idea is after.

### The random-escalation check, which is the finding

The bar is `(1-r)*cheap + r*dear`, what a coin flip at the same escalation rate `r` earns in
expectation. Natural rate is the training base rate.

| pair | router | clf | rate | precision | routed | random | diff | 95% CI |
|---|---|---|---|---|---|---|---|---|
| bge-small -> egemma-300m | potion-32M | centroid | 0.057 | 0.059 | 0.7790 | 0.7814 | -0.0024 | [-0.0078, +0.0020] |
| bge-small -> egemma-300m | potion-32M | ridge | 0.097 | 0.034 | 0.7761 | 0.7814 | -0.0054 | [-0.0120, +0.0004] |
| bge-small -> egemma-300m | potion-multilingual-128M | centroid | 0.068 | 0.073 | 0.7759 | 0.7814 | -0.0055 | [-0.0128, +0.0005] |
| bge-small -> egemma-300m | potion-multilingual-128M | ridge | 0.083 | 0.120 | 0.7812 | 0.7814 | -0.0002 | [-0.0056, +0.0052] |
| potion-32M -> bge-small | potion-32M | centroid | 0.085 | 0.235 | 0.7548 | 0.7538 | +0.0010 | [-0.0086, +0.0106] |
| potion-32M -> bge-small | potion-32M | ridge | 0.115 | 0.275 | 0.7575 | 0.7547 | +0.0028 | [-0.0087, +0.0143] |
| potion-32M -> bge-small | potion-multilingual-128M | centroid | 0.097 | 0.190 | 0.7514 | 0.7542 | -0.0028 | [-0.0121, +0.0068] |
| potion-32M -> bge-small | potion-multilingual-128M | ridge | 0.120 | 0.222 | 0.7543 | 0.7549 | -0.0006 | [-0.0128, +0.0109] |

Base rates are 0.098 and 0.130. Escalation precision on the first pair is 0.03 to 0.12, at or below
chance. On the second pair it is 0.19 to 0.28, above chance, and still not enough to move the
weighted score past the coin flip. Forcing the rate to 0.05, 0.15, 0.30 and 0.55 does not rescue
it: the best cell over all 32 (pair, router, classifier, rate) combinations is +0.0085
[-0.0028, +0.0197] and the worst is -0.0147 [-0.0253, -0.0045], the losing cells all on the pair
with no gradient, where escalating at all is a tax.

The multilingual router buys nothing on English text, as expected. Its eight cells lie inside the
same intervals as `potion-32M`'s.

### What the router actually learns is the flavour

Escalation rate by flavour, `potion-32M` + ridge, second pair, beside the oracle rate and the
lexical overlap `queries.overlap` computes:

| flavour | escalated | oracle | lexical overlap |
|---|---|---|---|
| verbatim | 0.058 | 0.033 | 1.000 |
| degraded | 0.008 | 0.050 | 0.760 |
| keyword | 0.167 | 0.183 | 0.260 |
| paraphrase | 0.083 | 0.075 | 0.302 |
| kw_para | 0.258 | 0.308 | 0.041 |

corr(escalate, overlap) is -0.221 against corr(oracle, overlap) -0.258. The router has learned that
queries with little surface overlap are the ones a better encoder might save, which is the flavour
axis restated. That signal is real and it is also the only one: within a flavour the per-query
decision is noise, which is why an above-chance precision produces a weighted score on the coin
flip line. 81% of instances are ties, so the classifier is fitting a 10-13% minority class from 480
training instances of a 512-dimensional lookup vector.

### Decision

Off. The spike stays in `evals/`; nothing in `litesearch/` changed and no default moved. The
direction measures negative on the pair it was proposed for and no better than random on the pair
that has a gradient, so it does not clear the bar the graph leg failed either.

What would have to change before it is worth another run. A query signal that is not the query's
own surface form, since a static lookup vector of the query text carries the flavour and little
else. A quality gap that is worth routing across, which regulatory does not have between
`bge-small` and `egemma-300m`, so the gap has to be found on another genre first. And a label with
more than 10% positives, since 81% ties means most queries do not care which encoder answers them.
Routing also doubles the index, 1027 s of embedding against 159 s, which a +0.047 oracle ceiling
has to pay for before any classifier is discussed.
