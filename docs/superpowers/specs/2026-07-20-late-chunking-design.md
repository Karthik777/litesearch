# Late Chunking for litesearch — Design

Date: 2026-07-20
Status: Approved (design phase)

## Goal

Implement **late chunking** as encoder extensions to `FastEncode`, and build an
accuracy benchmark comparing it against three other embedding techniques on a
long-context retrieval dataset, scored through litesearch's real hybrid search.

Late chunking: embed the whole document once to get per-token embeddings, *then*
split and mean-pool per chunk. Each chunk vector retains full-document context,
which naive chunk-then-embed loses at chunk boundaries.

## Non-goals

- No new/bigger embedding model. nomic-embed-text-v1.5 (8192 ctx) is sufficient.
- No changes to `FastEncodeImage` / `FastEncodeMultimodal` (chunk-context loss is
  a text-only concern).
- No hyperparameter search / MTEB-scale sweep. One dataset subset, one model.

## Component 1 — Late-chunking encoders

Source of truth: `nbs/03_utils.ipynb` (nbdev). Regenerate `litesearch/utils.py`
with `nbdev_prepare`. All classes subclass `FastEncode` and reuse its
`_input_names`, `tti`, `prompt`, `normalize`, `dtype`, and ONNX session.

### `LateChunkFastEncode(FastEncode)`

Single-pass late chunking for docs within the context window.

- `_token_embeddings(text) -> (token_embs, offsets, mask)`
  - Tokenize with `self.tok.encode(text, add_special_tokens=True)` (single
    sequence — captures `enc.offsets`, which the batched `_enc` path discards).
  - Build ONNX inputs conditionally on `_input_names` and `tti`
    (`input_ids`, optional `attention_mask`, optional `token_type_ids`).
  - Return raw per-token embeddings `(seq_len, dim)`, char offsets, attention mask.
- `encode_late_chunks(text, spans, prompt=None) -> np.ndarray (n_chunks, dim)`
  - `spans`: list of `(start_char, end_char)` in the ORIGINAL `text`.
  - Apply the document prompt template (`self.prompt['document']`) and account for
    the char shift the prefix introduces (`prefix_len`).
  - For each span, select token indices whose offset overlaps the span
    (`e > cs and s < ce`, masked), mean-pool them.
  - Normalize if `self.normalize`; cast to `self.dtype`.

### `LongLateChunkFastEncode(LateChunkFastEncode)`

Docs longer than the context window: slide overlapping windows, pool each chunk
per window, and combine by **token-weighted average** so boundary chunks (seen in
two windows) blend both sides' context proportional to token contribution.

- `_make_windows(text, window_chars, overlap_chars)` — stepped windows covering
  the full text incl. tail.
- `encode_long_document(text, spans, window_chars=None, overlap_chars=None, prompt=None)`
  - Defaults: `window_chars ≈ (max_seq_len-8) * 3.5`, `overlap_chars = window_chars//5`.
  - Accumulate `chunk_sums` and `chunk_weights`; emit weighted mean, normalize, cast.

### `AutoLateChunkFastEncode(LongLateChunkFastEncode)`

Routes by a cheap tokenizer-only count (no ONNX run).

- `_count_tokens(text)` — count with truncation temporarily disabled, restore after.
- `encode_auto(text, spans, prompt=None, long_ratio=4.0, **kw) -> (embs, tier)`
  - `normal` (≤ ctx): `encode_late_chunks`
  - `long` (≤ ctx*long_ratio): `encode_long_document` (default ~20% overlap)
  - `longer` (> ctx*long_ratio): `encode_long_document` with tighter overlap

### `chunk_spans` helper

`nbs/02_data.ipynb` — `chunk_spans(text, chunker=None) -> L[(start,end,text)]`.
Complements `chunk_markdown` (which maps to `.text` only, dropping offsets). Uses a
chonkie chunker; reads `c.start_index`, `c.end_index`, `c.text`. This is the single
source of chunk spans shared by every eval method, so all methods chunk identically.

### Tests (in the notebook, per nbdev + CLAUDE.md)

- Offset→token mapping: a known short text, spans over two sentences, assert each
  chunk vector differs from naive per-chunk embedding but stays unit-norm.
- Sum-of-spans covering the whole doc ≈ behaves sanely; single full-span late chunk
  ≈ close to the normal `_mp` document embedding (sanity, not exact equality).
- `encode_auto` returns the expected tier for short / medium / long inputs.
- Window combination: a chunk straddling a window boundary gets a nonzero weighted
  vector; a chunk in one window equals its single-window pooling.

## Component 2 — Accuracy eval

New notebook: `nbs/04_latechunk_eval.ipynb` (exploratory eval; may stay
`#| eval: false` for CI but runnable locally).

### Data

- LongEmbed via HF `datasets` (`dwzhu/LongEmbed`). Tasks: NarrativeQA and
  2WikiMultihopQA. Cap corpus/query counts (config constants) for runtime.
- Format per task: `corpus` (doc_id → text), `queries` (qid → text),
  `qrels` (qid → set of relevant doc_ids). **Relevance is doc-level.**

### Methods (all share `chunk_spans` output per doc)

1. **naive** — `encode_document(chunk_texts)`; store one row per chunk.
2. **full-doc** — one vector per whole doc via `encode_document([doc])`; one row per doc.
3. **late-chunk** — `AutoLateChunkFastEncode.encode_auto(doc, spans)`; one row per chunk,
   stored chunk text = the raw chunk (for FTS), embedding = late-chunk vector.
4. **contextual** — `rishi.Chat` (local Gemma via litert_lm) generates a short
   situating blurb per chunk given the doc; prepend blurb to chunk text, then
   `encode_document`. One row per chunk. Blurbs cached to disk keyed by
   `(doc_id, chunk_idx)` so reruns are cheap and offline.

### Retrieval — litesearch hybrid RRF

- One `database()` store per method. Rows: `content`, `embedding`, `metadata`
  (JSON with `doc_id`, `chunk_idx`).
- Query with `db.search(query, encode_query(query).tobytes())` (default RRF hybrid).
- **Chunk→doc aggregation:** collapse ranked chunk hits to parent `doc_id`, keeping
  each doc's best (lowest) rank, to produce a doc-level ranking comparable across all
  four methods. Full-doc method is already doc-level.

### Metrics

- nDCG@10 and Recall@{1,5,10} against qrels, averaged over queries.
- Output: one table, rows = methods, cols = metrics, plus per-task breakdown.

### Success criterion

The harness runs end to end and produces the comparison table. The *hypothesis*
(late chunking ≥ naive on long-doc nDCG@10) is what the table tests — a result
showing otherwise is still a valid outcome, not a bug.

## Dependencies

- `uv add rishi` (contextual baseline, local litert Gemma).
- `datasets` for LongEmbed (add if not present).
- chonkie, onnxruntime, tokenizers, numpy already present.

## Risks / open notes

- LongEmbed doc lengths exercise the `long`/`longer` tiers — good coverage of the
  windowing path, but slower; the doc cap controls this.
- Offsets require the fast tokenizer's default behavior; nomic's tokenizer preserves
  char offsets, so no extra config expected. Verify in the first test cell.
- rishi first run downloads Gemma weights (~minutes); blurb cache mitigates reruns.
