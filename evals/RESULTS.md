# Results

## Is a multimodal page retriever a viable replacement for parse-then-embed?

`Hcompany/NeoMME-*-Retriever`, four variants, against the route litesearch ships: `pdf_parse`
(pdflite, with liteparse behind the OCR branch) to markdown pages, chunk, embed, FTS plus vectors.
Reproduce with `python -m evals.visual_eval prepare parse visual text ocr report`.

### Method

112 pages, five of the nine regulatory documents in `examples/pdfs`, whole documents so the
distractors are the ones a query would really face. 97 known-item queries from `evals.queries` in
five flavours over the same source sentences, paired. `kw_para` is the one flavour that is not a
lexical transform of the target text and the one to weight.

Both routes return pages, so the metric is the page: a query is answered when the page holding its
source sentence is in the top 10. Chunk hits fold to their page before ranking, which favours the
parse route. The visual route gets an exhaustive MaxSim scan and no lexical leg, which favours it
on quality and is charged for on latency.

### Quality, MRR@10 on the page

| configuration | verbatim | degraded | keyword | paraphrase | kw_para | mean | ms |
|---|---|---|---|---|---|---|---|
| bge-small / f512 / fts-pre | 0.756 | 0.757 | 0.731 | 0.731 | 0.374 | 0.670 | 4.4 |
| neomme-800m-late / image | 0.744 | 0.732 | 0.661 | 0.725 | 0.418 | 0.656 | 48.1 |
| neomme-260m-late / image | 0.747 | 0.727 | 0.671 | 0.719 | 0.379 | 0.649 | 32.9 |
| bge-small / f512 / hybrid-pre-deep | 0.746 | 0.706 | 0.636 | 0.708 | 0.379 | 0.635 | 5.9 |
| neomme-260m-late / text | 0.741 | 0.730 | 0.649 | 0.715 | 0.313 | 0.629 | 19.3 |
| neomme-800m-late / text | 0.742 | 0.714 | 0.625 | 0.710 | 0.346 | 0.627 | 36.5 |
| potion-32M / page / fts-pre | 0.770 | 0.697 | 0.653 | 0.681 | 0.284 | 0.617 | 1.8 |
| bge-small / f512 / vec | 0.699 | 0.627 | 0.527 | 0.635 | 0.259 | 0.549 | 1.1 |
| neomme-800m-dense / text | 0.578 | 0.453 | 0.385 | 0.433 | 0.243 | 0.418 | 31.9 |
| neomme-260m-dense / image | 0.535 | 0.444 | 0.363 | 0.444 | 0.206 | 0.398 | 10.6 |
| neomme-800m-dense / image | 0.535 | 0.452 | 0.343 | 0.429 | 0.200 | 0.392 | 30.2 |
| bge-small / page / vec | 0.542 | 0.398 | 0.325 | 0.405 | 0.196 | 0.373 | 0.3 |

Paired bootstrap over queries, 10,000 resamples, 95% CI, against `bge-small/f512/fts-pre`:

| variant | verbatim | degraded | keyword | paraphrase | kw_para |
|---|---|---|---|---|---|
| neomme-800m-late | -0.012 ns | -0.025 ns | -0.070 loses | -0.006 ns | +0.044 ns |
| neomme-260m-late | -0.009 ns | -0.030 ns | -0.060 loses | -0.012 ns | +0.005 ns |
| neomme-260m-dense | -0.221 loses | -0.313 loses | -0.369 loses | -0.288 loses | -0.168 loses |
| neomme-800m-dense | -0.221 loses | -0.305 loses | -0.389 loses | -0.302 loses | -0.174 loses |

Late interaction over page images ties the text route on four of five flavours and loses on
`keyword` by 0.060 to 0.070. It reaches that without a tokenizer, a chunker or a text layer. The
dense variants lose every flavour by 0.17 to 0.39 and sit below bge-small's vector leg on its own,
so the single-vector pooling is where this family stops working, not the modality.

Image against text for the same model, same bootstrap: no difference on four flavours, image ahead
on `kw_para` by +0.066 (260M) and +0.072 (800M). Page layout carries signal, and only where the
query has no lexical anchor left.

### Cost, 112 pages

| route | prep s | embed s | s/page | vectors MB |
|---|---|---|---|---|
| parse potion-32M / page | 0.6 | 0.1 | 0.009 | 4.8 |
| parse bge-small / page | 0.6 | 20.7 | 0.193 | 4.8 |
| parse bge-small / f512 | 0.6 | 24.8 | 0.233 | 6.2 |
| visual neomme-260m-late / image | 5.7 | 152.1 | 1.409 | 63.1 |
| visual neomme-800m-late / image | 0.9 | 312.2 | 2.796 | 63.1 |
| visual neomme-260m-dense / image | 3.7 | 151.9 | 1.389 | 0.2 |
| visual neomme-800m-dense / image | 0.9 | 301.1 | 2.697 | 0.4 |

The parse-route MB column is the whole store, text and FTS index included. The visual column is
vectors alone: 563 kB per page at 2202 patches by 128 dimensions in float16, 13x the entire parse
store, holding no text, so no snippets, no FTS leg and no cross-encoder rerank.

Query latency is the harder number. 33 ms at 112 pages is an exhaustive MaxSim scan, and litesearch
has no ANN index for multi-vector, so it grows linearly: about 3 seconds per query at 10,000 pages.

### The OCR branch, 8 scanned pages

`pdf_parse` is 0.006 s/page on born-digital PDFs, so on the corpus above the parse step is 3% of
the parse route and 0.4% of the default route. Skipping it saves nothing. The case for skipping it
is a scan, which this measures by rasterising eight pages into a text-free PDF.

| step | s/page |
|---|---|
| pdf_parse, text layer | 0.029 |
| pdf_parse ocr=on (liteparse plus tesseract) | 1.743 |
| neomme-260m-late on the same page images | 2.276 |
| neomme-800m-late on the same page images | 5.633 |

Two runs, for the run-to-run spread: 1.743 and 1.739 for OCR, 2.276 and 2.359 for the 260M.

One footgun found while measuring, not fixed here. `ocr_selection='auto'` never fires on this
file. `needs_ocr` looks for a `> [OCR REQUIRED` marker from pdf-oxide, and on a PDF built by
`images_to_pdf` pdf-oxide emits `![Image 1 from page 1](images/page1_1.png)` instead, so `auto`
returns 408 characters of image links for 8 pages and reports success. Whether a real scanner PDF
carries the marker is not tested here; the OCR numbers above use `ocr_selection='on'`.

OCR is the cheaper of the two. Even against the whole OCR route, 1.743 plus 0.193 for bge-small at
1.936 s/page, the 260M page encoder at 2.276 s/page does not pay. The synthetic scan renders larger
than the born-digital page, so the visual figures here run about 1.6x their table-above values;
the ordering does not change.

### Verdict

Not viable as a replacement, on CPU, on this corpus. The late-interaction variants are a real
result in that they match a tuned lexical-plus-vector route with no text at all, but they cost
150x the ingest, 8x the query latency at 112 pages with linear growth after, and 13x the storage,
and they give up FTS, snippets and reranking to do it. The dense variants are not competitive on
quality either.

Where it could still pay, untested here: a corpus that is genuinely scanned and whose OCR is worse
than tesseract on clean rasters, figures and tables that the text layer loses, or a GPU, where the
ingest ratio is the number that changes.
