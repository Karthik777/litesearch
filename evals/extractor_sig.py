"""Are the gaps between keyphrase extractors real, or within noise?

`extractor_eval` reports aggregates, and the spread between extractors there is 2–3 points of
p_mrr while the gap between *any* graph configuration and plain hybrid is 15–26. Two or three
points on 120 queries is exactly the size of difference that deserves a paired test rather than an
eyebrow, so this scores the same queries per extractor, keeps the per-query reciprocal ranks, and
bootstraps the paired difference.

Paired because every extractor answers the identical query set over the identical chunks: the
variance between queries is enormous next to the variance between extractors, and pairing removes
it. A 95% interval that straddles zero means the ranking in the aggregate table is noise.

    python -m evals.extractor_sig
    python -m evals.extractor_sig --genres regulatory --topics both
"""
from __future__ import annotations
import argparse, json
import numpy as np

from litesearch import database

from . import corpus as C
from .build import db_path
from .queries import build as build_queries, FLAVOURS
from .refindex import ref
from .run import RESULTS, BASE_GRAIN, qvecs
from .score import COLS, K, key_grams, score_one
from .extractor_eval import ENCODER, _mode, available

SEED = 20260803


def per_query_rr(genre, mode, graph_w=0.5, flavours=FLAVOURS, limit=K):
    '''Reciprocal rank of the target passage for every (flavour, query), as one flat array.

    The array is the point: `aggregate` averages these away, and the average is what cannot be
    tested. Order is fixed by `flavours` then query index, so two runs line up element-wise.'''
    p = db_path(genre, BASE_GRAIN, ENCODER, mode)
    if not p.exists(): return None
    db, qs, qv = database(str(p)), build_queries(genre), qvecs(genre, ENCODER)
    r = ref(genre)
    kgs = [key_grams(q, r) for q in qs]
    out = []
    for fl in flavours:
        for q, v, kg in zip(qs, qv[fl], kgs):
            try: hits = db.graph_search(q[fl], v.tobytes(), columns=COLS, limit=limit, graph_w=graph_w) or []
            except Exception: hits = []
            rank = score_one(hits, q, r, limit, kg)[0]
            out.append(0.0 if rank is None else 1.0/(rank+1))
    db.conn.close()
    return np.array(out)


def boot(a, b, n=10_000, seed=SEED):
    '''Bootstrap the paired mean difference `a - b`, resampling *queries* rather than scores.

    Returns `(mean, lo, hi, p)` where `p` is the two-sided share of resamples on the wrong side of
    zero — a permutation-flavoured read of "could this ordering have come out the other way".'''
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(1.0, p))


def run(genres=None, names=None, graph_w=0.5, topics='no'):
    genres = list(genres or C.GENRES)
    exts = list(available(names))
    modes = {n: _mode(n) for n in exts}
    print(f'\n== paired bootstrap of per-query reciprocal rank (graph_w={graph_w}, topics={topics}) ==')
    print('   the eval leaves the clones with topics already removed, so run this after it\n')
    out = []
    for g in genres:
        rr = {n: per_query_rr(g, m, graph_w) for n, m in modes.items()}
        rr = {n: v for n, v in rr.items() if v is not None}
        if len(rr) < 2: print(f'  {g}: need two extractors built; skipping'); continue
        n_q = len(next(iter(rr.values())))
        print(f'  {g} ({n_q} query-flavour pairs)')
        for n, v in rr.items(): print(f'    {n:<12} mean RR {v.mean():.4f}')
        for i, a in enumerate(exts):
            for b in exts[i+1:]:
                if a not in rr or b not in rr: continue
                m, lo, hi, p = boot(rr[a], rr[b])
                sig = 'significant' if (lo > 0 or hi < 0) else 'not distinguishable'
                print(f'    {a:>10} - {b:<12} {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  p={p:.3f}  {sig}')
                out.append(dict(genre=g, a=a, b=b, diff=m, lo=lo, hi=hi, p=p,
                                significant=bool(lo > 0 or hi < 0), n=n_q, graph_w=graph_w, topics=topics))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS/'extractor_sig.json').write_text(json.dumps(out, indent=1))
    print(f'\n  -> {RESULTS/"extractor_sig.json"} ({len(out)} comparisons)')
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--genres', default=None)
    p.add_argument('--extractors', default=None)
    p.add_argument('--graph-w', type=float, default=0.5)
    p.add_argument('--topics', default='no', help='label only; reflects the state the clones are in')
    a = p.parse_args(argv)
    return run(genres=a.genres.split(',') if a.genres else None,
               names=a.extractors.split(',') if a.extractors else None,
               graph_w=a.graph_w, topics=a.topics)


if __name__ == '__main__': main()
