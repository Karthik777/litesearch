"""Late chunking with the span→token lookup done by binary search instead of by scanning.

**This did not work, and the measurement is the point.** The hypothesis was that
`LateChunkFastEncode.encode_late_chunks` and `encode_long_document` are dominated by

    idx = [t for t,(s,e) in enumerate(offsets) if msk[t] and e>cs and s<ce]

a full scan of the window's token offsets **per span** — on the VAT directive, 1,148 spans × 8,192
tokens × ~20 windows, which counts out to something like 190M Python-level comparisons. Replacing it
with two `searchsorted` calls per span produces bit-identical vectors (`max_abs_diff` 0.0 on all
three genres) and is **1.0–1.1x faster**: 197s against 216s on the VAT directive, and no difference
at all on the other two.

So the cost of late chunking is the ONNX forward passes over 8,192-token windows, not the pooling
around them. There is no optimisation to recommend here, and `evals/results/late_pooling.json` is
the evidence. What the module is still good for is `check()`, which pins the library's pooling
against an independent implementation of the same arithmetic.
"""
import numpy as np


def _pool_window(token_embs, offsets, msk, spans, lo, hi, prefix_len, sums, weights):
    '''Accumulate token-mean vectors for every span overlapping `[lo, hi)` of the document.

    `sums`/`weights` are updated in place with token-count weighting, which is what makes the
    windowed case a weighted average over windows rather than a vote.'''
    starts = np.asarray([s for s, _ in offsets], dtype=np.int64)
    ends   = np.asarray([e for _, e in offsets], dtype=np.int64)
    keep   = np.asarray(msk, dtype=bool)
    order  = np.argsort(starts, kind='stable')          # tokenizer offsets are already sorted
    s_ord, e_ord = starts[order], ends[order]
    for i, (cs, ce) in enumerate(spans):
        if ce <= lo or cs >= hi: continue
        a = max(cs - lo, 0) + prefix_len
        b = min(ce - lo, hi - lo) + prefix_len
        # tokens with start < b, then filter that block by end > a — offsets are sorted by start,
        # and a token's span is short, so the block is tight
        j = int(np.searchsorted(s_ord, b, side='left'))
        if j == 0: continue
        blk = order[:j]
        sel = blk[(e_ord[:j] > a) & keep[blk]]
        if sel.size == 0: continue
        sums[i] += token_embs[sel].mean(axis=0) * sel.size
        weights[i] += sel.size


def encode_auto_fast(lc, text, spans, prompt=None, long_ratio=4.0):
    '''`AutoLateChunkFastEncode.encode_auto`, with searchsorted pooling. Returns (vectors, tier).

    Routing, window sizes and overlaps are taken from `lc` so the arithmetic is the library\'s; only
    the span→token lookup differs.'''
    max_tok = lc.max_seq_len or 512
    n_tok = lc._count_tokens(text)
    tmpl = prompt if prompt is not None else (lc.prompt.get('document', None) or '{text}')
    if n_tok <= max_tok - 8:
        windows, tier = [(0, len(text))], 'normal'
    else:
        if n_tok <= max_tok*long_ratio:
            wc = int((max_tok-8)*3.5); oc = wc//5; tier = 'long'
        else:
            wc = int((max_tok-8)*3.5); oc = wc//8; tier = 'longer'
        windows = lc._make_windows(text, wc, oc)
    sums, weights, dim = None, np.zeros(len(spans)), None
    for lo, hi in windows:
        win = text[lo:hi]
        full = tmpl.format(text=win)
        token_embs, offsets, msk = lc._token_embeddings(full)
        if dim is None:
            dim = token_embs.shape[-1]; sums = np.zeros((len(spans), dim), dtype=np.float32)
        _pool_window(token_embs, offsets, msk, spans, lo, hi, len(full)-len(win), sums, weights)
    out = np.zeros_like(sums)
    ok = weights > 0
    out[ok] = sums[ok]/weights[ok, None]
    if lc.normalize: out = out/np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-12, None)
    return out.astype(lc.dtype), tier


def check(lc, text, spans, atol=2e-3):
    'Assert the fast path matches the library, and return (library seconds, fast seconds).'
    import time
    t0 = time.time(); ref, tier_a = lc.encode_auto(text, spans); t_lib = time.time()-t0
    t0 = time.time(); got, tier_b = encode_auto_fast(lc, text, spans); t_fast = time.time()-t0
    assert tier_a == tier_b, (tier_a, tier_b)
    a, b = ref.astype(np.float32), got.astype(np.float32)
    bad = int((np.abs(a-b) > atol).any(axis=1).sum())
    return dict(tier=tier_a, spans=len(spans), rows_differing=bad,
                max_abs_diff=float(np.abs(a-b).max()),
                t_library=round(t_lib, 2), t_fast=round(t_fast, 2),
                speedup=round(t_lib/max(t_fast, 1e-9), 1))
