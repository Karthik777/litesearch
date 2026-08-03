# RAG configuration tiers for litesearch

Measured on three document genres — EU legislation, arXiv papers, and astrology books printed
between 1822 and 1920 — on 4 CPU cores with no GPU. Every number in this document was produced by
`python -m evals.run` in this repository and can be reproduced from it.

<!-- FINDINGS -->

## What was measured, and what that means

**The corpus.** 2,007 pages and 4.1M characters over 27 documents, in three genres that fail
differently:

<!-- CORPUS TABLE -->

The three were not chosen for variety alone. Regulatory text has real hierarchy and enormous
term repetition between documents; arXiv PDFs lose their structure in conversion and keep their
notation; the astrology books are long, archaic, weakly structured, and — this part is deliberate —
share their vocabulary with the astro-ph papers without sharing their meaning. A retriever that
cannot tell *the transit of Venus through the seventh house* from *the transit of Venus observed at
Greenwich* will say so on this corpus.

**Ground truth, without an LLM.** Every query is derived from one sentence that occurs exactly once
in the corpus, so its passage and its section are known exactly. Five flavours degrade the wording
progressively; all five are asked of the same 351 source sentences, so every comparison is paired.

| flavour | how it is made | mean lexical overlap with the answer |
|---|---|---|
| `verbatim` | the sentence itself | 1.00 |
| `degraded` | its rarest third of content words deleted | 0.75 |
| `keyword` | five mid-frequency content words, in order | 0.26–0.60 |
| `paraphrase` | content words swapped for WordNet synonyms | 0.30–0.42 |
| `kw_para` | five content words, every one of them swapped | 0.04–0.10 |

Scoring happens at three levels, all keyed on word 5-grams that are unique to one section, so no
metric moves when the chunk size does:

- **passage** — the returned text overlaps the target sentence by five consecutive words;
- **section** — the returned text belongs to the target Article / CHAPTER / paper section;
- **document** — the returned text comes from the right document.

The **score** column throughout is a weighted mean over flavours: `verbatim` 0.10, `degraded` 0.15,
and 0.25 each for the three where the query is not a copy of the answer. Per-flavour numbers are
always shown beside it.

### Four caveats, in order of how much they should change your reading

1. **These are known-item queries, not questions.** Each has exactly one right answer that exists
   verbatim somewhere. Real RAG traffic includes multi-hop questions, questions with several valid
   answers, and questions with none. Nothing here measures those.
2. **Four of five flavours are lexical transformations of the target**, which biases the whole
   evaluation towards the FTS leg. `kw_para` is the corrective, and it is the flavour to weight if
   your users type questions. Read the ordering of configurations rather than the absolute numbers.
3. **WordNet substitutes senses, not meanings.** *Table proper to the climate* became *mesa right to
   the clime*. Some `kw_para` queries are therefore unanswerable by anything, which depresses that
   column for every configuration equally — it is a floor, not a ceiling.
4. **Section-level numbers for arXiv are weak.** pdf-oxide gives a paper about four headings, so a
   "section" there averages five pages and section accuracy nearly equals document accuracy. See
   the structure findings.

## Reproducing

```bash
uv sync --extra eval --group dev
python -m evals.fetch_corpus            # arXiv + Project Gutenberg; regulatory PDFs ship with the repo
python -m evals.run build_core build_tree build_late build_graph
python -m evals.run eval_grain eval_encoder eval_strategy eval_structure eval_late eval_graph eval_mixed eval_cluster
python -m evals.tables > tables.md
```

Stores land in `evals/dbs/` (one SQLite file per configuration, ~1.5 GB in total) and results in
`evals/results/*.json`. Every phase is resumable: a store that exists is not rebuilt.
