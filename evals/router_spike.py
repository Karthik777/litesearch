"""Can a static-embedding classifier decide, per query, when to escalate to a dearer encoder?

The premise correction first, because the idea arrived attached to the wrong model.
convaiinnovations/laya is not an embedding model: it is a ModernBERT-large classifier that answers
typed yes/no questions in one pass, and its "router" is a <0.5ms script detector that sends Latin
text to the ModernBERT checkpoint and everything else to an mmBERT one. It routes between
*classifiers* by *language*. This corpus is English-only (`corpus.GENRE_UNIT` is EU legislation),
so there is no language to route on. What transfers is the shape: a near-zero-cost classifier over
the input gates access to an expensive resource.

So the testable version is a cost router. Per query, a static embedder (`potion-32M` or
`potion-multilingual-128M`, lookup tables, no forward pass) predicts whether the dearer encoder
would beat the cheaper one on that query, and the query goes to whichever is predicted. Three arms,
held out: always cheap, always dear, routed. The bar is the random-escalation line,
`(1-r)*cheap + r*dear`, which is what any coin flip at the same escalation rate `r` earns in
expectation. A router that does not clear that line is a coin flip with a centroid attached.

    python -m evals.router_spike
    python -m evals.router_spike --pair potion-32M,bge-small --router potion-32M --clf ridge
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np

from litesearch import database

from .build import db_path
from .queries import build as build_queries, FLAVOURS, overlap
from .refindex import ref
from .report import WEIGHTS
from .run import RESULTS, BASE_GRAIN, qvecs
from .score import K, STRATEGIES, key_grams, score_one

GENRE, STRATEGY = 'regulatory', 'hybrid-pre'
# Two pairs. The first is the one the idea is about. The second exists because on regulatory the
# encoder sweep already puts egemma-300m *below* bge-small on all five flavours, so the first pair
# has no quality gradient to climb; potion-32M to bge-small does.
PAIRS = (('bge-small', 'egemma-300m'), ('potion-32M', 'bge-small'))
ROUTERS = ('potion-32M', 'potion-multilingual-128M')
SEED, FOLDS = 20260803, 5

CACHE = Path(__file__).parent/'cache'

# Flavour weights renormalised over the five, so a per-sentence weighted score is directly the
# weighted section MRR every other table in RESULTS.md reports.
WFL = np.array([WEIGHTS[f] for f in FLAVOURS]); WFL = WFL/WFL.sum()


# ------------------------------------------------------------------ per-query retrieval
def per_query_u(genre, encoder, strategy=STRATEGY, limit=K, refresh=False):
    'Section-level reciprocal rank for every (flavour, query) over one encoder store, cached.'
    f = CACHE/f'router_u_{genre}__{encoder}__{strategy}.json'
    if f.exists() and not refresh: return np.array(json.loads(f.read_text()))
    p = db_path(genre, BASE_GRAIN, encoder, 'flat')
    if not p.exists(): raise FileNotFoundError(f'{p} — run build_flat({genre!r}, {BASE_GRAIN!r}, {encoder!r})')
    db, qs, qv = database(str(p)), build_queries(genre), qvecs(genre, encoder)
    r, fn = ref(genre), STRATEGIES[strategy]
    kgs = [key_grams(q, r) for q in qs]
    out = []
    for fl in FLAVOURS:
        for q, v, kg in zip(qs, qv[fl], kgs):
            hits = fn(db, q[fl], v.tobytes(), limit) or []
            u = score_one(hits, q, r, limit, kg)[1]
            out.append(0.0 if u is None else 1.0/(u+1))
    db.conn.close()
    CACHE.mkdir(parents=True, exist_ok=True); f.write_text(json.dumps(out))
    return np.array(out)


def features(genre, router):
    'L2-normalised static query vectors, flavour-major, in the same order as `per_query_u`.'
    qv = qvecs(genre, router)
    x = np.concatenate([qv[fl] for fl in FLAVOURS]).astype(np.float32)
    return x/np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)


def labels(rr_cheap, rr_dear):
    'Escalate only where the dear encoder strictly wins. Ties stay cheap, which is the cheap arm\'s side.'
    return (rr_dear > rr_cheap).astype(np.int8)


# ------------------------------------------------------------------ classifiers
def fit_centroid(X, y):
    'Nearest-centroid: score is cosine to the escalate centroid minus cosine to the keep centroid.'
    u = lambda v: v/max(float(np.linalg.norm(v)), 1e-6)
    c1 = u(X[y == 1].mean(0)) if (y == 1).any() else np.zeros(X.shape[1], np.float32)
    c0 = u(X[y == 0].mean(0)) if (y == 0).any() else np.zeros(X.shape[1], np.float32)
    w = c1 - c0
    return lambda Z: Z@w


def fit_ridge(X, y, lam=1.0):
    'Closed-form least squares on +-1 labels with an intercept. No scikit-learn, no iteration.'
    A = np.hstack([X, np.ones((len(X), 1), np.float32)])
    t = np.where(y == 1, 1.0, -1.0).astype(np.float32)
    w = np.linalg.solve(A.T@A + lam*np.eye(A.shape[1], dtype=np.float32), A.T@t)
    return lambda Z: np.hstack([Z, np.ones((len(Z), 1), np.float32)])@w


CLASSIFIERS = {'centroid': fit_centroid, 'ridge': fit_ridge}


# ------------------------------------------------------------------ trained static router (model2vec)
# Part 2: instead of generic potion-32M vectors + a classifier bolted on after, fine-tune the vectors
# themselves on the escalation label with `model2vec.train.StaticModelForClassification`, then it is
# still a static (lookup-table) model at inference time. Same starting point as the `potion-32M`
# router arm above, so the only variable is whether the weights are task-tuned.
M2V_BASE = 'minishlab/potion-retrieval-32M'
M2V_SEEDS = (42, 20260803, 7)


def texts_all(genre):
    'Query text, flavour-major, same order as `per_query_u` and `features`.'
    qs = build_queries(genre)
    return np.array([q[fl] for fl in FLAVOURS for q in qs])


def fit_m2v(X_txt, y, freeze, seed):
    '''Fit `StaticModelForClassification` from potion-32M's vectors, return a `predict_proba` scorer.

    `freeze` gates the embedding table (`nn.Embedding.from_pretrained(..., freeze=freeze)`); the
    constructor's `freeze_weights` arg is a false friend, it only freezes a per-token weighting
    scalar, not the vectors, so it is left at its default and `freeze` is the real toggle here.
    '''
    from model2vec import StaticModel
    from model2vec.train import StaticModelForClassification
    sm = StaticModel.from_pretrained(M2V_BASE)
    clf = StaticModelForClassification.from_static_model(model=sm, freeze=freeze)
    clf.fit(list(X_txt), [int(v) for v in y], random_seed=seed)
    idx1 = list(clf.classes_).index(1) if 1 in clf.classes_ else 1
    return lambda Z: clf.predict_proba(list(Z))[:, idx1]


def oof_escalate_m2v(texts, y, n_q, freeze, seed, rate=None, k=FOLDS, fold_seed=SEED):
    'Held-out escalate/keep decision from a fold-local `StaticModelForClassification` fit, no leakage.'
    fold_q = folds(n_q, k, fold_seed)
    fold = np.tile(fold_q, len(FLAVOURS))
    esc = np.zeros(len(y), bool)
    for i in range(k):
        tr, te = fold != i, fold == i
        s = fit_m2v(texts[tr], y[tr], freeze, seed)
        r = float(y[tr].mean()) if rate is None else rate
        st = s(texts[tr])
        thr = np.quantile(st, 1-r) if 0 < r < 1 else (np.inf if r <= 0 else -np.inf)
        esc[te] = s(texts[te]) > thr
    return esc


def run_m2v(genre=GENRE, pair=('potion-32M', 'bge-small'), freeze=False, seeds=M2V_SEEDS, rate=None):
    'One arm (frozen or trained), a handful of seeds, same scoring as `run`.'
    cheap, dear = pair
    qs = build_queries(genre); n_q = len(qs)
    rr_c, rr_d = per_query_u(genre, cheap), per_query_u(genre, dear)
    y = labels(rr_c, rr_d)
    texts = texts_all(genre)
    ov = np.array([overlap(q[fl], q['key']) for fl in FLAVOURS for q in qs])
    tag = 'frozen-vectors (head-only)' if freeze else 'trained-vectors'

    print(f'\n== {genre} · {cheap} -> {dear} · model2vec router, {tag} ==')
    rows = []
    for sd in seeds:
        esc = oof_escalate_m2v(texts, y, n_q, freeze, sd, rate)
        a = arms(rr_c, rr_d, esc, n_q)
        m, lo, hi, p = boot(a['routed'], a['random'])
        prec = float(y[esc].mean()) if esc.any() else float('nan')
        rec = float(esc[y.astype(bool)].mean()) if y.any() else float('nan')
        r_esc = float(np.corrcoef(esc.astype(float), ov)[0, 1])
        row = dict(genre=genre, cheap=cheap, dear=dear, router='model2vec', freeze=freeze, seed=sd,
                   strategy=STRATEGY, n=len(y), rate=a['rate'], base_rate=float(y.mean()),
                   precision=prec, recall=rec,
                   **{f'{k}_mrr': float(a[k].mean()) for k in ('cheap', 'dear', 'routed', 'random', 'oracle')},
                   routed_vs_random=dict(diff=m, lo=lo, hi=hi, p=p, significant=bool(lo > 0 or hi < 0)),
                   corr_escalate_overlap=r_esc)
        print(f'   seed {sd:<10} rate {a["rate"]:.3f} precision {prec:.3f} recall {rec:.3f}  '
              f'routed {a["routed"].mean():.4f}  random {a["random"].mean():.4f}  '
              f'gain {m:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.3f}  corr(esc,overlap) {r_esc:+.3f}')
        rows.append(row)
    return rows


def run_m2v_suite(genre=GENRE, pair=('potion-32M', 'bge-small'), seeds=M2V_SEEDS):
    'Both arms (frozen control, trained), merged additively into router_spike.json under "m2v".'
    rows = run_m2v(genre, pair, freeze=True, seeds=seeds) + run_m2v(genre, pair, freeze=False, seeds=seeds)
    RESULTS.mkdir(parents=True, exist_ok=True)
    f = RESULTS/'router_spike.json'
    out = json.loads(f.read_text()) if f.exists() else dict(arms=[], sweep=[])
    out['m2v'] = rows
    f.write_text(json.dumps(out, indent=1))
    print(f'\n  -> {f}')
    return rows


# ------------------------------------------------------------------ held-out routing
def folds(n_q, k=FOLDS, seed=SEED):
    'Fold id per source sentence. Splitting on the sentence keeps its five flavours on one side.'
    rng = np.random.default_rng(seed)
    f = np.zeros(n_q, np.int8)
    f[rng.permutation(n_q)] = np.arange(n_q) % k
    return f


def oof_escalate(X, y, n_q, fit, rate=None, k=FOLDS, seed=SEED):
    '''Held-out escalate/keep decision per query instance.

    The threshold is a quantile of the *training* scores, so both the fit and the operating point
    are out of sample. `rate=None` uses the training base rate.'''
    fold_q = folds(n_q, k, seed)
    fold = np.tile(fold_q, len(FLAVOURS))
    esc = np.zeros(len(y), bool)
    for i in range(k):
        tr, te = fold != i, fold == i
        s = fit(X[tr], y[tr])
        r = float(y[tr].mean()) if rate is None else rate
        st = s(X[tr])
        thr = np.quantile(st, 1-r) if 0 < r < 1 else (np.inf if r <= 0 else -np.inf)
        esc[te] = s(X[te]) > thr
    return esc


# ------------------------------------------------------------------ scoring
def per_sentence(x, n_q):
    'Flavour-weighted score per source sentence: the unit the bootstrap resamples.'
    return (WFL[:, None]*np.asarray(x, float).reshape(len(FLAVOURS), n_q)).sum(0)


def arms(rr_cheap, rr_dear, esc, n_q):
    'Per-sentence weighted section MRR for cheap, dear, routed, expected-random and oracle.'
    r = float(esc.mean())
    routed = np.where(esc, rr_dear, rr_cheap)
    rand = (1-r)*rr_cheap + r*rr_dear                 # a coin flip at rate r, in expectation
    return dict(rate=r,
                cheap=per_sentence(rr_cheap, n_q), dear=per_sentence(rr_dear, n_q),
                routed=per_sentence(routed, n_q), random=per_sentence(rand, n_q),
                oracle=per_sentence(np.maximum(rr_cheap, rr_dear), n_q))


def boot(a, b, n=10_000, seed=SEED):
    '''Bootstrap the paired mean difference `a - b`, resampling source sentences.

    Same method as `extractor_sig.boot`, copied rather than imported because that module currently
    fails at import (`litesearch.core._in` no longer exists).'''
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2*min((means <= 0).mean(), (means >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(1.0, p))


def qry_ms(encoder, texts, n=40):
    'Median single-query encode milliseconds, one query at a time as a live caller would.'
    from .encoders import enc
    e = enc(encoder)
    e.qry([texts[0]])
    ts = []
    for t in texts[:n]:
        t0 = time.time(); e.qry([t]); ts.append((time.time()-t0)*1000)
    return float(np.median(ts))


# ------------------------------------------------------------------ run
def run(genre=GENRE, pair=PAIRS[0], router='potion-32M', clf='centroid', rate=None, cost=True):
    cheap, dear = pair
    qs = build_queries(genre)
    n_q = len(qs)
    rr_c, rr_d = per_query_u(genre, cheap), per_query_u(genre, dear)
    y = labels(rr_c, rr_d)
    X = features(genre, router)
    esc = oof_escalate(X, y, n_q, CLASSIFIERS[clf], rate)
    a = arms(rr_c, rr_d, esc, n_q)

    print(f'\n== {genre} · {BASE_GRAIN} · {STRATEGY} · {cheap} -> {dear} · router {router}/{clf} ==')
    print(f'   {n_q} sentences x {len(FLAVOURS)} flavours = {len(y)} query instances, '
          f'{FOLDS}-fold grouped by sentence')
    print(f'   oracle label: dear strictly better on {y.mean():.3f} of instances '
          f'(ties stay cheap: {(rr_c == rr_d).mean():.3f} of instances are ties)')
    print(f'   escalation rate {a["rate"]:.3f}, precision '
          f'{y[esc].mean() if esc.any() else float("nan"):.3f} against base rate {y.mean():.3f}')

    print('\n   weighted section MRR (held out)')
    for k in ('cheap', 'dear', 'routed', 'random', 'oracle'):
        print(f'     {k:<8} {a[k].mean():.4f}')

    out = dict(genre=genre, cheap=cheap, dear=dear, router=router, clf=clf, strategy=STRATEGY, n=len(y),
               rate=a['rate'], base_rate=float(y.mean()),
               precision=float(y[esc].mean()) if esc.any() else None,
               **{f'{k}_mrr': float(a[k].mean()) for k in ('cheap', 'dear', 'routed', 'random', 'oracle')})

    print('\n   paired bootstrap over source sentences, 95% CI')
    for lab, b in (('routed - cheap', 'cheap'), ('routed - dear', 'dear'),
                   ('routed - random', 'random'), ('dear - cheap', None)):
        aa, bb = (a['dear'], a['cheap']) if b is None else (a['routed'], a[b])
        m, lo, hi, p = boot(aa, bb)
        sig = 'significant' if (lo > 0 or hi < 0) else 'no difference'
        print(f'     {lab:<16} {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]  p={p:.3f}  {sig}')
        out[lab.replace(' - ', '_vs_').replace(' ', '')] = dict(diff=m, lo=lo, hi=hi, p=p,
                                                                significant=bool(lo > 0 or hi < 0))

    print('\n   escalation rate by flavour, and what the router is keying on')
    ov = np.array([overlap(q[fl], q['key']) for fl in FLAVOURS for q in qs])
    by_fl = {}
    for i, fl in enumerate(FLAVOURS):
        sl = slice(i*n_q, (i+1)*n_q)
        by_fl[fl] = dict(escalated=float(esc[sl].mean()), oracle=float(y[sl].mean()),
                         overlap=float(ov[sl].mean()))
        print(f'     {fl:<11} escalated {by_fl[fl]["escalated"]:.3f}  '
              f'oracle {by_fl[fl]["oracle"]:.3f}  lexical overlap {by_fl[fl]["overlap"]:.3f}')
    r_esc = float(np.corrcoef(esc.astype(float), ov)[0, 1])
    r_orc = float(np.corrcoef(y.astype(float), ov)[0, 1])
    print(f'     corr(escalate, overlap) {r_esc:+.3f}   corr(oracle, overlap) {r_orc:+.3f}')
    out['by_flavour'] = by_fl; out['corr_escalate_overlap'] = r_esc; out['corr_oracle_overlap'] = r_orc

    if cost:
        texts = [q['keyword'] for q in qs]
        ms = {e: qry_ms(e, texts) for e in dict.fromkeys((router, cheap, dear))}
        routed_ms = ms[router] + (1-a['rate'])*ms[cheap] + a['rate']*ms[dear]
        print(f'\n   query embed ms: {router} {ms[router]:.2f}  {cheap} {ms[cheap]:.2f}  '
              f'{dear} {ms[dear]:.2f}  routed {routed_ms:.2f}')
        out['ms'] = ms; out['routed_ms'] = routed_ms
    return out


def sweep(genre=GENRE, pair=PAIRS[0], rates=(0.1, 0.2, 0.3, 0.5), routers=ROUTERS,
          clfs=tuple(CLASSIFIERS)):
    'Routed against the random-escalation line at fixed escalation rates.'
    cheap, dear = pair
    qs = build_queries(genre); n_q = len(qs)
    rr_c, rr_d = per_query_u(genre, cheap), per_query_u(genre, dear)
    y = labels(rr_c, rr_d)
    print(f'\n== escalation-rate sweep, {genre} · {cheap} -> {dear} ==')
    print(f'   {"router":<26} {"clf":<9} {"rate":>5} {"routed":>8} {"random":>8} {"gain":>8}  95% CI')
    out = []
    for rt in routers:
        X = features(genre, rt)
        for cl in clfs:
            for r in rates:
                esc = oof_escalate(X, y, n_q, CLASSIFIERS[cl], r)
                a = arms(rr_c, rr_d, esc, n_q)
                m, lo, hi, p = boot(a['routed'], a['random'])
                print(f'   {rt:<26} {cl:<9} {a["rate"]:>5.2f} {a["routed"].mean():>8.4f} '
                      f'{a["random"].mean():>8.4f} {m:>+8.4f}  [{lo:+.4f}, {hi:+.4f}] p={p:.3f}')
                out.append(dict(genre=genre, cheap=cheap, dear=dear, router=rt, clf=cl,
                                rate=a['rate'], routed=float(a['routed'].mean()),
                                random=float(a['random'].mean()), gain=m, lo=lo, hi=hi, p=p,
                                significant=bool(lo > 0 or hi < 0)))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--genre', default=GENRE)
    p.add_argument('--pair', default=None, help='cheap,dear; default: every pair in PAIRS')
    p.add_argument('--router', default=None, help='default: every router in ROUTERS')
    p.add_argument('--clf', default=None, help='centroid or ridge; default: both')
    p.add_argument('--rate', type=float, default=None, help='escalation rate; default: the base rate')
    p.add_argument('--no-cost', action='store_true')
    p.add_argument('--no-sweep', action='store_true')
    p.add_argument('--m2v', action='store_true',
                    help='run the trained-static-vectors arm (model2vec) instead of the generic-router sweep')
    a = p.parse_args(argv)
    if a.m2v:
        pr = tuple(a.pair.split(',')) if a.pair else ('potion-32M', 'bge-small')
        return run_m2v_suite(a.genre, pr)
    pairs = [tuple(a.pair.split(','))] if a.pair else list(PAIRS)
    routers = [a.router] if a.router else list(ROUTERS)
    clfs = [a.clf] if a.clf else list(CLASSIFIERS)
    out = [run(a.genre, pr, rt, cl, a.rate, cost=not a.no_cost)
           for pr in pairs for rt in routers for cl in clfs]
    sw = [] if a.no_sweep else [r for pr in pairs for r in sweep(a.genre, pr, routers=routers, clfs=clfs)]
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS/'router_spike.json').write_text(json.dumps(dict(arms=out, sweep=sw), indent=1))
    print(f'\n  -> {RESULTS/"router_spike.json"}')
    return out, sw


if __name__ == '__main__': main()
