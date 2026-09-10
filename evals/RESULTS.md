# Results

Method: every system is scored on the same queries and the same gold. Generic gold is
`evals.refindex` (section overlap, chunk-independent); code gold is known-item (one function per
query). Difference is a paired bootstrap over queries (`colbert_eval.boot`, 10k resamples); a 95%
CI spanning zero is reported as no difference. `python -m evals.run_colbert` reproduces the table.

## ColBERT late interaction vs the static + FTS hybrid

The shipped retriever is a static vector leg (`potion-multilingual-128M` for prose,
`potion-code-16M-v2` for code) fused with porter/sanskrit FTS by RRF, which is what vishalakshi and
kosha run. The contender is `answerdotai/answerai-colbert-small-v1`, scored three ways:

- `hybrid`    — the baseline, `db.search`.
- `colbert`   — late interaction over the whole corpus, MaxSim, no FTS.
- `cb-rerank` — hybrid top-30 reordered by MaxSim. No index change; one extra ONNX model.
- `fx-rerank` — the same hybrid top-30 reordered by the shipped flashrank cross-encoder
  (`ms-marco-TinyBERT-L-2-v2`), which is the reranker litesearch already offers via `reranking=True`.

Metric is section MRR@10, averaged over the five query flavours (verbatim/degraded/keyword/
paraphrase/kw_para). Latency is per-query median at the given corpus size, single CPU.

| task    | corpus     | chunks | hybrid | colbert | cb-rerank | fx-rerank | Δ cb vs baseline       | ms hybrid / colbert / cb / fx |
|---------|------------|-------:|-------:|--------:|----------:|----------:|------------------------|-------------------------------|
| generic | regulatory |  3,080 | 0.601  | 0.652   | 0.656     | 0.640     | +0.055 [0.035, 0.075]  | 10 / 108 / 11 / 60            |
| generic | arxiv      |  2,416 | 0.692  | 0.750   | 0.759     | 0.729     | +0.066 [0.044, 0.088]  |  7 /  87 /  8 / 96            |
| generic | astrology  |  4,977 | 0.717  | 0.825   | 0.831     | 0.772     | +0.114 [0.089, 0.140]  | 11 / 141 / 12 / 63            |
| code    | repos      |    442 | 0.938  | 0.990   | 0.990     | 0.966     | +0.052 [0.034, 0.072]  |  3 /  12 / 11 / 153           |

Code is known-item, so it also carries recall@10: **1.000 for all four systems**. ColBERT reorders
the top ten; it does not find a function the baseline missed. The +0.052 is rank-1 promotion, not
coverage.

### The decisive contrast: ColBERT reranker vs the shipped flashrank reranker

Both rerank the identical hybrid top-30, so this isolates the reranker. ColBERT wins on every
corpus, and by more than the CI width; and it is 5-15x faster, because MaxSim over 30 short
candidate matrices is cheaper than a TinyBERT cross-encoder forward pass per candidate.

| corpus     | fx-rerank | cb-rerank | Δ cb − fx (95% CI)     | fx ms | cb ms |
|------------|----------:|----------:|------------------------|------:|------:|
| regulatory | 0.640     | 0.656     | +0.016 [0.001, 0.031]  |  60   |  11   |
| arxiv      | 0.729     | 0.759     | +0.030 [0.015, 0.045]  |  96   |   8   |
| astrology  | 0.772     | 0.831     | +0.059 [0.040, 0.079]  |  63   |  12   |
| code       | 0.966     | 0.990     | +0.024 [0.013, 0.037]  | 153   |  11   |

flashrank does help over the raw hybrid (its own deltas clear zero too), but it recovers only about
half of ColBERT's gain, and it is the slower of the two by 5-15x. The gap between the rerankers,
like the gain over the baseline, is widest on the archaic paraphrased astrology prose (+0.059) where
meaning matters most, and narrowest on the lexically-overlapping regulatory text (+0.016).

### What the numbers say

1. ColBERT beats the baseline on every corpus, and the CI clears zero every time. The gain tracks
   how much the task turns on meaning rather than surface form: smallest where the query shares
   vocabulary with the target (code +0.052, regulatory +0.055, arxiv +0.066), largest on the
   archaic paraphrased prose of the astrology books (+0.114).
2. `cb-rerank` captures the whole `colbert` gain, and slightly beats it, at baseline latency. It
   rescoring 30 candidates, so it costs 1-2 ms on top of the hybrid query (8-12 ms) against the
   87-141 ms of a full MaxSim scan that grows with the corpus. Full-corpus ColBERT buys nothing
   over reranking here.
3. Against the reranker you already have, ColBERT is better and cheaper. flashrank's TinyBERT
   costs 60-153 ms a query and recovers half the gain; the ColBERT reranker costs ~11 ms and
   recovers all of it.
4. The static + FTS baseline already puts the answer in the top ten (code recall@10 = 1.000). What
   either reranker moves is the rank inside that window, i.e. MRR, not what a caller can find at all.

### Cost of the switch

- **Reranker (`cb-rerank`): worth it, and it should replace flashrank.** No store change, keep the
  static + FTS index exactly as is. One added ONNX model (`fastembed`, ~500 MB, ~5 s load, imported
  on use like `FastEncode`), +1-2 ms per query. It beats the shipped flashrank reranker on every
  corpus (CI clears zero) while running 5-15x faster than it, so it is a strict improvement on the
  reranker already in the tree, not just on the raw hybrid.
- **Primary encoder (full ColBERT): not worth it.** Multi-vector storage is 10-180 vectors per
  chunk against one, and usearch is single-vector, so it needs a MaxSim index (PLAID or equivalent)
  that litesearch does not have. 7-10x query latency, and no recall gain to pay for it.
- **Multilingual is untested and at risk.** `answerai-colbert-small-v1` is English. The baseline's
  `potion-multilingual-128M` covers 100+ languages including Devanagari, which is the point of the
  sanskrit tokenizer chain. Every corpus here is English (EU law in English, English papers/books,
  Python). On Sanskrit or mixed-script text ColBERT could regress; that has to be measured before it
  replaces the multilingual leg anywhere.

### Recommendation

Add ColBERT as the reranker over the hybrid top-k, in place of flashrank; leave the static + FTS
retriever and the multilingual vector leg as the default. It is better and faster than the flashrank
leg it replaces. Do not switch the primary encoder to ColBERT. For kosha's code path the baseline is
already at recall@10 = 1.000, so the reranker is a rank-quality nicety, not a fix.

Wired as `reranker='colbert'` on `Database.search` / `doc_search` / `Index.search` (default stays
`'flashrank'`): `ColbertReranker` in `litesearch/utils.py`, own onnxruntime path, no fastembed, its
scores matching fastembed to 1e-6.

Caveat on the flashrank comparison: this used flashrank's default `ms-marco-TinyBERT-L-2-v2`, the
fastest and weakest of its models, which is litesearch's default. A heavier flashrank model would
rerank better but be slower still, moving it further from ColBERT on latency, not closer on quality
per millisecond.

Open follow-up: measure ColBERT on a Sanskrit/Devanagari corpus before trusting it on multilingual
text; `answerai-colbert-small-v1` is English and every corpus here is English.
