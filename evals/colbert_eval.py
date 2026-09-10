"""ColBERT (late interaction) against the shipped static + FTS hybrid, on generic and code retrieval.

The question is narrow: is `answerdotai/answerai-colbert-small-v1` worth switching to over the
`potion-multilingual-128M` / `potion-code-16M-v2` static vector leg fused with FTS by RRF, which is
what vishalakshi and kosha run today. So every system here is scored on the same queries and the
same gold, and the paired bootstrap over queries decides difference (`evals.extractor_sig.boot`).

Three systems per task:

- `hybrid`    — the baseline: static vectors + porter/sanskrit FTS, merged with RRF. `db.search`.
- `colbert`   — late interaction over the whole corpus, MaxSim scored, no FTS.
- `cb-rerank` — the deployable middle: take the hybrid top `FANOUT` and reorder them by MaxSim.

Generic gold is `evals.refindex` (passage/section overlap, chunk-independent). Code gold is
known-item: one function per query, a hit is correct when its `doc_id` is that function.
"""
import json, time
from pathlib import Path
import numpy as np

from litesearch import database
from litesearch.utils import static_embedder
from litesearch.core import rrf_merge, rerank_hits

from . import corpus as C
from .build import flat_chunks
from .queries import build as build_queries, FLAVOURS
from .refindex import ref
from .score import score_one, key_grams, aggregate, COLS


def boot(a, b, n=10_000, seed=20260803):
    'Paired bootstrap of the mean of `a - b`, resampling queries. Returns (mean, lo, hi, two-sided p).'
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(1.0, p))

RESULTS = Path(__file__).parent/'results'
CACHE   = Path(__file__).parent/'cache/colbert'
DT      = np.float16
K       = 10
FANOUT  = 30                       # candidates the reranker sees, matching db.search's RERANK_FANOUT
CB_MODEL = 'answerdotai/answerai-colbert-small-v1'
STATIC   = {'generic': 'minishlab/potion-multilingual-128M', 'code': 'minishlab/potion-code-16M-v2'}


# ------------------------------------------------------------------ ColBERT
_CB = {}
def _cb():
    'Memoised fastembed late-interaction model; the ONNX session is expensive to build.'
    if 'm' not in _CB:
        from fastembed import LateInteractionTextEmbedding
        _CB['m'] = LateInteractionTextEmbedding(CB_MODEL)
    return _CB['m']


def cb_doc(texts, cap=180):
    'Document multi-vectors, one (n_tok, dim) float32 array per text, token axis capped at `cap`.'
    return [np.asarray(v, dtype=np.float32)[:cap] for v in _cb().embed(list(texts), batch_size=32)]

def cb_qry(texts):
    'Query multi-vectors (ColBERT pads queries to 32 tokens with [MASK]).'
    return [np.asarray(v, dtype=np.float32) for v in _cb().query_embed(list(texts))]


def _pack(mvs):
    'List of (n_tok, dim) into one padded (N, Lmax, dim) tensor plus a (N, Lmax) validity mask.'
    dim = mvs[0].shape[1]; L = max(m.shape[0] for m in mvs)
    D = np.zeros((len(mvs), L, dim), np.float32); M = np.zeros((len(mvs), L), bool)
    for i, m in enumerate(mvs): D[i, :m.shape[0]] = m; M[i, :m.shape[0]] = True
    return D, M


def maxsim_scores(Q, D, M):
    'MaxSim of one query (Lq, dim) against every packed doc: sum over query tokens of max over doc tokens.'
    s = np.einsum('nld,qd->nlq', D, Q, optimize=True)     # (N, Lmax, Lq)
    s = np.where(M[:, :, None], s, -1e9)
    return s.max(axis=1).sum(axis=1)                       # (N,)


# ------------------------------------------------------------------ baseline store
def build_store(path, texts, doc_ids, pages, model):
    'A litesearch flat store (static vectors + FTS), i.e. exactly the vishalakshi/kosha index.'
    sm = static_embedder(model)
    vecs = np.asarray(sm.encode(list(texts)), dtype=DT)
    db = database(str(path))
    st = db.get_store('store', hash=True, ann=False, ndim=vecs.shape[1], dtype=DT, doc_id=str, page=int)
    st.insert_all([dict(content=t, embedding=v.tobytes(), doc_id=d, page=p)
                   for t, v, d, p in zip(texts, vecs, doc_ids, pages)],
                  upsert=True, hash_id='id', hash_id_columns=['content'])
    return db, sm            # db.search defaults to the exact vec scan; no ANN index needed at 3k chunks


# ------------------------------------------------------------------ rankers
def rank_hybrid(db, sm, q, limit=K):
    'db.search: static vector leg + FTS leg, RRF-merged. The baseline.'
    qv = np.asarray(sm.encode([q]), dtype=DT)[0]
    return db.search(q, qv.tobytes(), columns=COLS, limit=limit) or []

def rank_colbert(qmv, D, M, meta, limit=K):
    'Exhaustive MaxSim over the corpus. `meta` is the aligned list of hit dicts.'
    sc = maxsim_scores(qmv, D, M)
    top = np.argsort(-sc)[:limit]
    return [meta[i] for i in top]

def rank_rerank(db, sm, qmv, cb_docs, cmap, q, limit=K, fanout=FANOUT):
    'Hybrid top-`fanout`, reordered by MaxSim against the candidate chunks only.'
    cand = [h for h in rank_hybrid(db, sm, q, fanout) if h.get('content') in cmap]
    if not cand: return []
    D, M = _pack([cb_docs[cmap[h['content']]] for h in cand])
    sc = maxsim_scores(qmv, D, M)
    return [cand[i] for i in np.argsort(-sc)[:limit]]

def rank_flashrank(db, sm, q, limit=K, fanout=FANOUT):
    'The same hybrid top-`fanout`, reordered by the shipped flashrank cross-encoder. The decisive arm.'
    cand = rank_hybrid(db, sm, q, fanout)
    return rerank_hits(q, cand, None, limit) if cand else []


# ------------------------------------------------------------------ generic eval (refindex gold)
def _generic_corpus(genre, chunking='c512'):
    'Chunks with an index `_i`, and per-chunk hit-dict metadata carrying content/doc_id.'
    ch = flat_chunks(genre, chunking)
    meta = [dict(content=c['content'], doc_id=c['doc_id'], page=c['page'], _i=i) for i, c in enumerate(ch)]
    return ch, meta

def eval_generic(genre, chunking='c512', flavours=FLAVOURS, limit=K):
    'Baseline / colbert / cb-rerank on one genre. Returns (rows, rr) where rr holds paired per-query RR.'
    ch, meta = _generic_corpus(genre, chunking)
    texts = [c['content'] for c in ch]
    p = Path(CACHE)/f'store_{genre}_{chunking}.db'; p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists(): [Path(f).unlink() for f in __import__('glob').glob(f'{p}*')]
    db, sm = build_store(p, texts, [c['doc_id'] for c in ch], [c['page'] for c in ch], STATIC['generic'])
    cb_docs = cb_corpus(f'{genre}_{chunking}', texts)
    D, M = _pack(cb_docs); cmap = {t: i for i, t in enumerate(texts)}
    r, qs = ref(genre), build_queries(genre)
    kgs = [key_grams(q, r) for q in qs]
    systems = ('hybrid', 'colbert', 'cb-rerank', 'fx-rerank')
    rr = {s: [] for s in systems}; lat = {s: [] for s in systems}
    for fl in flavours:
        qtexts = [q[fl] for q in qs]
        qmvs = cb_qry(qtexts)
        for q, qmv, kg in zip(qs, qmvs, kgs):
            for s in systems:
                t0 = time.time()
                if s == 'hybrid':     hits = rank_hybrid(db, sm, q[fl], limit)
                elif s == 'colbert':  hits = rank_colbert(qmv, D, M, meta, limit)
                elif s == 'cb-rerank':hits = rank_rerank(db, sm, qmv, cb_docs, cmap, q[fl], limit)
                else:                 hits = rank_flashrank(db, sm, q[fl], limit)
                lat[s].append((time.time()-t0)*1000)
                u = score_one(hits, q, r, limit, kg)[1]          # section-level rank (the headline axis)
                rr[s].append(1.0/(u+1) if u is not None else 0.0)
    db.conn.close()
    rows = _rows(genre, 'generic', systems, rr, lat, len(qs)*len(flavours), len(ch))
    return rows, rr


# ------------------------------------------------------------------ code eval (known-item gold)
def eval_code(corpus_path, flavours=('summary',), limit=K):
    'Baseline / colbert / cb-rerank on the code corpus. Gold is the function a query was written from.'
    data = json.loads(Path(corpus_path).read_text())
    ch, qs = data['chunks'], data['queries']
    texts = [c['content'] for c in ch]; ids = [c['doc_id'] for c in ch]
    meta = [dict(content=c['content'], doc_id=c['doc_id'], page=0, _i=i) for i, c in enumerate(ch)]
    p = Path(CACHE)/'store_code.db'; p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists(): [Path(f).unlink() for f in __import__('glob').glob(f'{p}*')]
    db, sm = build_store(p, texts, ids, [0]*len(ch), STATIC['code'])
    cb_docs = cb_corpus('code', texts)
    D, M = _pack(cb_docs); cmap = {t: i for i, t in enumerate(texts)}
    systems = ('hybrid', 'colbert', 'cb-rerank', 'fx-rerank')
    rr = {s: [] for s in systems}; hit = {s: [] for s in systems}; lat = {s: [] for s in systems}
    qtexts = [q['query'] for q in qs]; qmvs = cb_qry(qtexts)
    for q, qmv in zip(qs, qmvs):
        gold = q['doc_id']
        for s in systems:
            t0 = time.time()
            if s == 'hybrid':     hits = rank_hybrid(db, sm, q['query'], limit)
            elif s == 'colbert':  hits = rank_colbert(qmv, D, M, meta, limit)
            elif s == 'cb-rerank':hits = rank_rerank(db, sm, qmv, cb_docs, cmap, q['query'], limit)
            else:                 hits = rank_flashrank(db, sm, q['query'], limit)
            lat[s].append((time.time()-t0)*1000)
            ranks = [i for i, h in enumerate(hits) if h.get('doc_id') == gold]
            rr[s].append(1.0/(ranks[0]+1) if ranks else 0.0)
            hit[s].append(1.0 if ranks else 0.0)
    db.conn.close()
    rows = _rows('code', 'code', systems, rr, lat, len(qs), len(ch), hit=hit)
    return rows, rr


# ------------------------------------------------------------------ ColBERT corpus cache
def cb_corpus(tag, texts):
    'Cache ColBERT document multi-vectors as a ragged npz (concatenated tokens + lengths).'
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE/f'cb_{tag}.npz'
    if f.exists():
        z = np.load(f); toks, lens = z['toks'], z['lens']
        if lens.sum() == len(toks) and len(lens) == len(texts):
            out, o = [], 0
            for l in lens: out.append(toks[o:o+l]); o += l
            return out
    t0 = time.time(); mvs = cb_doc(texts); dt = time.time()-t0
    toks = np.concatenate(mvs); lens = np.array([m.shape[0] for m in mvs])
    np.savez(f, toks=toks, lens=lens)
    print(f'  colbert encoded {len(texts)} chunks in {dt:.0f}s ({len(texts)/dt:.0f}/s)', flush=True)
    return mvs


# ------------------------------------------------------------------ rows + bootstrap
def _rows(name, task, systems, rr, lat, nq, nchunk, hit=None):
    out = []
    for s in systems:
        r = dict(task=task, corpus=name, system=s, n=nq, chunks=nchunk,
                 mrr=round(float(np.mean(rr[s])), 4),
                 ms_p50=round(float(np.median(lat[s])), 2), ms_p90=round(float(np.percentile(lat[s], 90)), 2))
        if hit is not None: r['hit10'] = round(float(np.mean(hit[s])), 4)
        out.append(r)
    return out


def _pb(a, b):
    d, lo, hi, pv = boot(np.array(a), np.array(b))
    return dict(delta=round(d, 4), lo=round(lo, 4), hi=round(hi, 4), p=round(pv, 4), diff=not (lo <= 0 <= hi))

def paired(rr, base='hybrid'):
    'Each system minus the baseline, plus the decisive cb-rerank minus fx-rerank. CI over zero = no difference.'
    out = {s: _pb(v, rr[base]) for s, v in rr.items() if s != base}
    if 'cb-rerank' in rr and 'fx-rerank' in rr:
        out['cb-rerank_vs_fx-rerank'] = _pb(rr['cb-rerank'], rr['fx-rerank'])
    return out


def save(rows, contrasts, fname='colbert.json'):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS/fname).write_text(json.dumps(dict(rows=rows, contrasts=contrasts), indent=1))
    print(f'  -> {RESULTS/fname}')
