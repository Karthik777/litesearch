# RAG configuration tiers for litesearch

Measured on three document genres — EU legislation, arXiv papers, and astrology books printed
between 1822 and 1920 — on 4 CPU cores with no GPU. 351 known-item queries in five flavours, 45
stores, no LLM anywhere in the ground truth. Every number here comes from `python -m evals.run` in
this repository.

## The short version

Two of the cheapest knobs in the library are worth more than everything expensive in it.

| change | worth | costs |
|---|---|---|
| chunk at 256–512 chars instead of the default | **+0.06 to +0.14** section MRR | nothing — a chunker argument |
| route the FTS leg through `pre()` | **+0.09 to +0.33** on reworded queries | nothing, but see the trade below |
| add a flashrank reranker | +0.03 to +0.08 | ~40ms per query |
| upgrade potion-32M → egemma-300m | **+0.007** | 1,700× the indexing compute |
| add the PPR graph leg | **−0.07 to −0.11** | 5–8× the query latency |
| late-chunk instead of embedding chunks independently | **−0.03 to −0.05**, and −0.18 on the vector leg alone | 3× the indexing time |

The single best configuration measured on every genre uses **no embeddings at all**: FTS5 with
`pre()`, at 256–512 character chunks, scoring 0.880 / 0.807 / 0.777 at 2–9ms. The best configuration
that uses embeddings beats it on exactly one genre, by 0.007, at 6× the latency and after an
indexing pass.

That result is real but it is not the whole truth, and [the caveat](#four-caveats) explains why.

---

## The tiers

### Tier 0 — lexical (no model, no index)

```python
db = database('corpus.db')
store = db.get_store('store')                       # no ann=, no embeddings
store.insert_all([dict(content=c) for c in chunks]) # chunks at 512 chars
hits = store.fts_search(pre(q), ['content'], 'rank', 10, quote=False)
```

**0.880 / 0.807 / 0.750** (arxiv / regulatory / astrology), **2–9ms**, zero indexing cost, and the
database is a fraction of the size. This is the baseline every other tier has to beat, and on this
corpus most of them do not.

Use it when your corpus is in your users' own vocabulary — legislation, standards, code, API docs,
anything where the terms are terms of art and people type them.

### Tier 1 — cheap hybrid (static embeddings)

```python
enc   = static_retrieval_embedder()                 # potion-retrieval-32M
store = db.get_store('store', ann=True, ndim=512, dtype=np.float32)
# dtype=np.float32 — model2vec returns float32; see the dtype trap below
hits  = db.search(q, enc.encode([q])[0].tobytes(), ann=True, dtype=np.float32)
```

**6,100 chunks/s** to index — the whole 4.1M-character corpus in under a second — and `ann=True`
costs nothing measurable in quality (0.783 vs 0.792 on arXiv) at half the latency of the exact scan.

Add this over Tier 0 when queries are reworded rather than quoted. It buys +0.10 to +0.24 on
paraphrased queries and gives back 0.02 to 0.15 on verbatim ones.

### Tier 2 — the one to default to (cheap embeddings + cross-encoder)

```python
enc = static_retrieval_embedder()
hits = db.search(q, qv, columns=['content'], limit=10, dtype=np.float32,
                 reranking=True)                    # flashrank ms-marco-TinyBERT-L-2-v2
```

**0.851 / 0.809 / 0.734** at **~45ms**. The reranker is worth +0.03 to +0.06 over hybrid alone and,
crucially, **it is worth more than any encoder upgrade**: potion-32M + reranker matches
egemma-300m + reranker to within 0.007 on every genre, and beats it on astrology.

If you are going to spend latency, spend it here and not on the encoder.

### Tier 3 — quality ceiling

```python
enc = FastEncode()                                  # EmbeddingGemma, 2048 ctx
```

**0.858 / 0.812 / 0.730** at ~60ms — and 3.6 chunks/s to index, against potion's 6,100. On this
corpus that is **1,700× the indexing compute for +0.007**, and on astrology it is worse.

Reach for Tier 3 only when you have measured Tier 2 failing on your own queries. "A bigger encoder
will help" was not true here even once.

### Per-genre overrides

| genre | chunk size | notes |
|---|---|---|
| regulatory | **c512** | the one genre where embeddings + reranker beat pure lexical (0.814 vs 0.807), and the only one where the tree layer is a small positive (+0.010) |
| arXiv papers | **c256** | finer is better all the way down; the tree layer is a net negative here |
| old books | **c256** | biggest granularity effect of the three: page → c256 is +0.08 |

Mixing all three in one store costs about **0.01** and doubles latency, with 5–17% of top-10 hits
coming from the wrong genre (worst on `kw_para`, where the astro-ph papers and the astrology books
genuinely collide). Separate stores per genre if you can filter cheaply; one store is defensible.

---

## What each feature actually did

### Chunk granularity — the largest effect measured

litesearch's default is **not** a considered choice: `chunk_markdown` uses
`FastChunker(chunk_size=4096)` and `add_doc` feeds it one node segment at a time, so on pages of
1.7–3.5k characters it almost never splits. `page` in this table *is* the default.

Section MRR, `fts-pre`:

| genre | page | c1024 | c512 | c256 |
|---|---|---|---|---|
| arxiv | 0.819 | 0.864 | 0.879 | **0.880** |
| regulatory | 0.735 | 0.779 | **0.807** | 0.780 |
| astrology | 0.697 | 0.731 | 0.750 | **0.777** |

A second reason to care: at `page` size a chunk is ~2,300 characters ≈ 575 tokens, which **overflows
bge-small's 512-token window**. Above 512 characters you are also paying to embed text the model
never sees.

### `pre()` on the FTS leg — a trade, not a free win

`db.search` quotes every token, making the FTS leg an implicit AND over the whole query. `pre()`
turns it into `word* OR word*`. The library ships `pre()` and `search` does not call it.

Averaged over encoders and genres, per flavour:

| flavour | Δ from using `pre()` |
|---|---|
| verbatim | **−0.02 to −0.24** |
| degraded | −0.01 to −0.26 |
| keyword | −0.05 to −0.24 |
| paraphrase | **+0.05 to +0.36** |
| kw_para | **+0.09 to +0.33** |

So this is a genuine decision, not an oversight to fix: **conjunctive quoting wins when users paste
exact phrases; `pre()` wins when they type questions.** The penalty shrinks sharply at fine
granularity (at c256 on arXiv, verbatim costs only 0.016) and is worst at page size. If you chunk
finely, `pre()` is nearly free upside.

This is why it should be a parameter on `search` rather than a changed default — and why the
`hybrid-pre` column in this report is not automatically the recommendation for your traffic.

Control: `hybrid-pre-deep`, identical but fusing 30 candidates per leg instead of 10, scores *worse*
everywhere (0.776 vs 0.808 on arXiv). The `pre()` gain is not a candidate-depth artifact — extra
depth is itself a penalty when the vector leg is weak.

### The encoder — almost irrelevant

With a reranker on top, across a ~1,700× range in indexing cost:

| encoder | arxiv | regulatory | astrology | chunks/s |
|---|---|---|---|---|
| egemma-300m | **0.858** | 0.812 | 0.730 | 3.6 |
| jina-v2-sm | 0.856 | **0.814** | 0.728 | 24 |
| bge-small | 0.852 | 0.805 | 0.720 | 16 |
| potion-32M (static) | 0.851 | 0.809 | **0.734** | 6,100 |

A 32M static lookup table with no attention and no context wins one genre and loses the other two by
under 0.01. The vector leg *alone* is the weakest component measured anywhere — astrology `vec`
scores 0.26–0.38 against 0.75 for keyword search.

Note this table only became trustworthy after fixing a bug: `embedding_gemma` shipped with a prompt
template EmbeddingGemma was never trained on. egemma coming last on the vector leg was the clue.

### The tree layer — a wash

`tree` against `flat` at identical encoder and granularity, weighted section MRR:

| genre | Δ |
|---|---|
| regulatory | **+0.003 to +0.011** |
| astrology | −0.010 to +0.005 |
| arxiv | −0.052 to −0.008 |

Two sub-results worth having:

- **`tree-nohead` beats `tree` on arXiv** (0.817 vs 0.798). Embedding the heading path with the
  chunk is a small net negative there — the heading is repeated across every chunk of a section, so
  it dilutes rather than contextualises.
- **Recovering the headings pdf-oxide throws away does not help retrieval.** pdf-oxide renders paper
  section headings as bold runs (`**3.1** **Encoder** **and** **Decoder** **Stacks**`), invisible to
  `build_tree`, so *Attention Is All You Need* gets a 5-node tree. Promoting those lines takes arXiv
  from 65 to 356 nodes and touches neither other genre — and moves retrieval by +0.004, i.e. noise.
  If you want that fix, justify it by `toc()` and `read()` navigation, which this does not measure.

Also worth knowing: the VAT directive still builds **1,128 empty nodes out of 1,983**. Lines like
`Article 5` inside running text open a node, so a chunk following a cross-reference is filed under a
stub rather than under the Article it belongs to.

### The graph leg — negative at every weight

| genre | hybrid-pre | graph w=0.25 | w=0.5 | w=1.0 |
|---|---|---|---|---|
| arxiv | **0.798** | 0.707 | 0.688 | 0.672 |
| regulatory | **0.789** | 0.676 | 0.663 | 0.644 |
| astrology | **0.673** | 0.599 | 0.589 | 0.566 |

Monotonic in `graph_w`, so the optimum is `graph_w=0`. At 55–107ms against 8–13ms. The loss holds on
`kw_para` — the no-shared-vocabulary case the graph leg was written for.

The graph itself builds cheaply (175s / 151s / 359s plus resolution and topics) and
`resolve_entities` merges 61–63% of surface variants. The finding is not that the graph is badly
built; it is that entity co-occurrence is not evidence of relevance to a query.

**`topic_nodes` contributes nothing to ranking.** Deleting every topic entity and re-running moves
the score by **+0.003 to +0.006** — marginally better without them.

### Clustering — good, but not at what it is advertised for

`clusters()` is sound as a map of a corpus:

| genre | clusters | coverage | purity (doc) | NMI (doc) | seconds |
|---|---|---|---|---|---|
| regulatory | 178 | 0.99 | 0.70 | 0.31 | 0.3 |
| arxiv | 123 | 0.99 | 0.59 | 0.39 | 0.1 |
| astrology | 242 | 0.99 | 0.54 | 0.23 | 0.4 |
| mixed | 544 | 0.99 | 0.57 | 0.43 | 0.4 |

On the mixed store it separates the three genres at **0.97 purity**, the `usearch` HNSW cut works at
every size tested (no kNN fallback needed), and the c-TF-IDF labels are 120–400× more frequent inside
their cluster than in the corpus. Cheap, honest, useful.

**But `peers()` is 2–3× worse than `ann_neighbors()`**, at matched group size, on every genre:

| genre | `peers` | `ann_neighbors` |
|---|---|---|
| regulatory | 0.046 | **0.124** |
| arxiv | 0.273 | **0.453** |
| astrology | 0.109 | **0.287** |
| mixed | 0.370 | **0.627** |

`SKILL.md` currently says to reach for `peers` over `ann_neighbors` when the question is "where else
did we already do this?", on the reasoning that k-NN returns `limit` rows whether or not they are
related while a cluster returns a family. Measured, the family is *less* related than the same number
of nearest neighbours. Group size was matched precisely so this could not be a `limit` artifact.

### Late chunking — negative on all three genres, and it damages the vectors

jina-v2-sm (8192-token context) at c512, late chunking against embedding the same chunks
independently:

| genre | hybrid-pre: independent → late | vec only: independent → late |
|---|---|---|
| arxiv | 0.826 → **0.781** (−0.045) | 0.605 → **0.430** (−0.175) |
| regulatory | 0.781 → **0.728** (−0.053) | 0.622 → **0.437** (−0.185) |
| astrology | 0.672 → **0.638** (−0.034) | 0.384 → **0.198** (−0.186) |

The vector-only column is the diagnostic, and it is not a subtle ranking effect: late chunking
roughly **halves** the quality of the vector leg. The chunk vectors come out less discriminative, and
the hybrid only looks less damaged because the FTS leg carries it.

The mechanism is visible in the tier the encoder chose. Every document here lands in `encode_auto`'s
`longer` tier — 28,600-character windows over documents of 200+ pages. Mean-pooling a span's token
embeddings when those embeddings were computed over 8,192 tokens of surrounding text makes every span
in a window look like its window. Late chunking's premise is that a chunk should see its context;
when the "context" is a whole VAT directive or a whole book, that context is not coherent and the
pooling homogenises rather than informs.

**What this does not say.** `nbs/04_latechunk_eval.ipynb` measures late chunking on LongEmbed
(narrativeqa, 2wikimqa), where the task is picking the right *document* and documents are far shorter
than these. That result can hold while this one does. The untested middle case is the interesting one:
late-chunk **within a section** rather than within a whole document, so the pooled context is one
coherent passage. That is a small change to `doc_spans` and it is the variant worth trying before
concluding anything about the technique itself.

It also costs ~3× the indexing time of independent embedding, because every window is a full
8192-token forward pass — 838s for the astrology corpus against 201s.

Two mechanical notes from doing it:

- The pooling loop looks like the bottleneck and is not. Replacing the per-span scan of token offsets
  with binary search gives **bit-identical vectors** (`max_abs_diff` 0.0) and **1.0–1.3×**. The cost
  is the ONNX passes.
- One 8192-token pass over a whole document needs ~2 GB per attention matrix. Batch size has to fall
  as the context window rises — the `fulldoc` control could not be built on a 15 GB machine at all.

---

## Three bugs found on the way

All three are fixed on this branch, each with a regression cell in its notebook.

1. **`pre()` emitted invalid FTS5 for any query containing an apostrophe.** `kw()` segments with
   apsw's UAX#29 tokenizer, which keeps the apostrophe inside a word, and `add_wc` appended a
   wildcard: `pathfinder's*` is a syntax error, so `db.search` **raised** rather than searching. 7 of
   1,755 query/flavour pairs, all possessives.
2. **A float32 store searched with the default `dtype=np.float16` returns every distance as exactly
   0.0.** No exception, no empty result — the vector leg silently returns rows in rowid order and a
   hybrid search degrades to FTS only. `model2vec` returns float32 and `search` defaults to float16,
   so this is one line of ordinary code away. It is live in `nbs/04_latechunk_eval.ipynb`, whose
   `potion` rows therefore measure FTS with a dead vector leg.
3. **`embedding_gemma` used a prompt template EmbeddingGemma was never trained on.** The model card
   specifies `title: none | text: {content}` and `task: search result | query: {content}`; litesearch
   shipped an Instructor-style string. This is `FastEncode`'s default model and the one `SKILL.md`
   recommends for best quality.

---

## Four caveats

In order of how much they should change your reading.

1. **These are known-item queries, not questions.** Each has exactly one right answer that exists
   verbatim somewhere in the corpus. Real traffic includes multi-hop questions, questions with
   several valid answers, and questions with none. Nothing here measures those, and they are where
   embeddings are most likely to earn their cost.
2. **Four of the five flavours are lexical transformations of the target text**, which biases the
   whole evaluation towards the FTS leg. `kw_para` is the corrective — five content words, every one
   swapped for a WordNet synonym, mean lexical overlap 0.04–0.10 — and it is the column to weight if
   your users type questions. Even there, keyword search wins on two genres of three.
3. **WordNet substitutes senses, not meanings.** *Table proper to the climate* became *mesa right to
   the clime*. Some `kw_para` queries are unanswerable by anything, which depresses that column
   equally for every configuration — it is a floor, not a ceiling.
4. **Section-level numbers for arXiv are weak.** pdf-oxide gives a paper about four headings, so a
   "section" there averages five pages and section accuracy nearly equals document accuracy.

The honest summary: **on known-item lookup over these three corpora, embeddings did not earn their
cost, and the two free knobs did.** A question-shaped query set would narrow that gap. It would have
to narrow it by a great deal to change the ordering of the tiers.

---

## Method

**Corpus.** 2,007 pages / 4.1M characters over 27 documents. Regulatory: 8 EU directives and
regulations, 489 pages, real TITLE › CHAPTER › Article hierarchy and heavy term repetition between
documents. arXiv: 12 papers, cs.\* and astro-ph. Astrology: 7 Project Gutenberg books, 1,275 pages,
archaic prose and `CHAPTER XI` lines instead of markup, one book with no headings at all. The
astro-ph slice is deliberate: it shares vocabulary with the astrology books without sharing meaning.

**Ground truth.** Each query derives from one sentence occurring exactly once in the corpus. Scoring
is keyed on **word 5-grams unique to one section**, at three levels — passage (the hit overlaps the
target sentence by five consecutive words), section (the hit belongs to the target Article / CHAPTER
/ paper section), document. Nothing in the metric moves when chunk size does, which matters because
chunk size turned out to be the largest effect. Coarse chunks are *favoured* by the passage metric,
which runs against the conclusion reached.

**The score column** is a weighted mean over flavours: `verbatim` 0.10, `degraded` 0.15, and 0.25
each for `keyword`, `paraphrase` and `kw_para`. Equal weighting would let `verbatim` — a regression
check that FTS answers near perfectly — carry a fifth of every ranking. Per-flavour numbers are shown
throughout so the weighting can be argued with.

## Reproducing

```bash
uv sync --extra eval --group dev
python -m evals.fetch_corpus            # arXiv + Gutenberg; the regulatory PDFs ship with the repo
python -m evals.run build_core build_tree build_late build_graph
python -m evals.run eval_grain eval_encoder eval_strategy eval_structure \
                    eval_late eval_graph eval_mixed eval_cluster
python -m evals.tables > tables.md      # every table
python -m evals.decide                  # paired feature contrasts and the cost frontier
```

Stores land in `evals/dbs/` (one SQLite file per configuration, ~1.5 GB total), results in
`evals/results/*.json`. Every phase is resumable: a store that exists is not rebuilt. Budget about
four hours on 4 cores, nearly all of it ONNX encoding.
