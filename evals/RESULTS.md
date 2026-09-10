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

Metric is section MRR@10, averaged over the five query flavours (verbatim/degraded/keyword/
paraphrase/kw_para). Latency is per-query median at the given corpus size, single CPU.

| task    | corpus     | chunks | hybrid | colbert | cb-rerank | Δ rerank (95% CI)      | ms hybrid / colbert / rerank |
|---------|------------|-------:|-------:|--------:|----------:|------------------------|------------------------------|
| generic | regulatory |  3,080 | 0.601  | 0.652   | 0.656     | +0.055 [0.035, 0.075]  | 12 / 106 / 14                |
| generic | arxiv      |  2,416 | 0.692  | 0.750   | 0.759     | +0.066 [0.044, 0.088]  |  7 /  79 /  9                |
| generic | astrology  |  4,977 | 0.717  | 0.825   | 0.831     | +0.114 [0.089, 0.140]  | 12 / 142 / 15                |
| code    | repos      |    442 | 0.938  | 0.990   | 0.990     | +0.052 [0.034, 0.072]  |  3 /  10 /  6                |

Code is known-item, so it also carries recall@10: **1.000 for all three systems**. ColBERT reorders
the top ten; it does not find a function the baseline missed. The +0.052 is rank-1 promotion, not
coverage.

### What the numbers say

1. ColBERT beats the baseline on every corpus, and the CI clears zero every time. The gain tracks
   how much the task turns on meaning rather than surface form: smallest where the query shares
   vocabulary with the target (code +0.052, regulatory +0.055, arxiv +0.066), largest on the
   archaic paraphrased prose of the astrology books (+0.114).
2. `cb-rerank` captures the whole `colbert` gain, and slightly beats it, at baseline latency. It
   rescoring 30 candidates, so it costs 2-3 ms on top of the hybrid query (9-15 ms) against the
   79-142 ms of a full MaxSim scan that grows with the corpus. Full-corpus ColBERT buys nothing
   over reranking here.
3. The static + FTS baseline already puts the answer in the top ten (code recall@10 = 1.000). What
   ColBERT moves is the rank inside that window, i.e. MRR, not what a caller can find at all.

### Cost of the switch

- **Reranker (`cb-rerank`): worth it.** No store change, keep the static + FTS index exactly as is.
  One added ONNX model (`fastembed`, ~500 MB, ~5 s load, imported on use like `FastEncode`), +2-3 ms
  per query. A significant, reproducible MRR gain across four corpora.
- **Primary encoder (full ColBERT): not worth it.** Multi-vector storage is 10-180 vectors per
  chunk against one, and usearch is single-vector, so it needs a MaxSim index (PLAID or equivalent)
  that litesearch does not have. 7-10x query latency, and no recall gain to pay for it.
- **Multilingual is untested and at risk.** `answerai-colbert-small-v1` is English. The baseline's
  `potion-multilingual-128M` covers 100+ languages including Devanagari, which is the point of the
  sanskrit tokenizer chain. Every corpus here is English (EU law in English, English papers/books,
  Python). On Sanskrit or mixed-script text ColBERT could regress; that has to be measured before it
  replaces the multilingual leg anywhere.

### Recommendation

Add ColBERT as an optional reranker over the hybrid top-k, beside the existing flashrank leg; leave
the static + FTS retriever and the multilingual vector leg as the default. Do not switch the primary
encoder to ColBERT. For kosha's code path the baseline is already at recall@10 = 1.000, so the
reranker is a rank-quality nicety, not a fix.

Open follow-ups: compare `cb-rerank` against the existing flashrank `rerank` strategy on the same
corpora (this study compared against the un-reranked hybrid, the shipped default); and measure
ColBERT on a Sanskrit/Devanagari corpus before trusting it on multilingual text.
