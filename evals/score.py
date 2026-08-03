"""Retrieval strategies and the metrics they are scored on.

Three levels of "right answer", all derived from `refindex` and none of them dependent on how the
store under test chunked the corpus:

- **passage** — a hit is correct if its text overlaps the source sentence by five consecutive words,
  counting only 5-grams unique to the target section. Coarse chunks hold more text and so are
  *favoured* by this metric; it is a floor for fine chunking, not a thumb on its side.
- **section** — a hit is correct if any 5-gram it contains belongs to the target Article / CHAPTER /
  paper section. This is the question a reader actually asks of a document.
- **document** — a hit is correct if it comes from the right document. Easy here (8–12 documents per
  genre) and reported only to show where a configuration has collapsed.
"""
import time
import numpy as np

from litesearch.core import rrf_merge
from litesearch.data import pre

from .refindex import ref, nrm, grams

K = 10

# A passage counts as found when it overlaps the target sentence by five consecutive words. Strict
# containment of the whole sentence would be unmeasurable for 256-character chunks — the sentence
# straddles two of them — and would score fine chunking at zero for a reason that has nothing to do
# with retrieval. Overlap still favours big chunks, which have more surface to overlap with; that
# bias is left in place because it runs against the conclusion this evaluation reaches.
def overlaps(text, key_grams, max_words=900):
    if not key_grams: return False
    return any(g in key_grams for g in grams(text, max_words))


# ------------------------------------------------------------------ metrics
def _rank(hits, ok, k=K):
    'Index of the first hit satisfying `ok`, or None.'
    for i, h in enumerate(hits[:k]):
        if ok(h): return i
    return None


def score_one(hits, q, r, k=K, key_grams=None):
    'Passage / section / document ranks for one query.'
    unit = q['unit']
    kg = key_grams if key_grams is not None else set(grams(q['key']))
    # flat stores carry the document title in `doc_id`, tree stores the library's hash of it, and
    # the mixed store prefixes the title with its genre
    title = q.get('doc_title') or ''
    docs = {q['doc'], title}
    ok_doc = lambda h: (d_ := (h.get('doc_id') or '\0')) in docs or (title and d_.endswith(f':{title}'))
    p = _rank(hits, lambda h: overlaps(h.get('content'), kg), k)
    u = _rank(hits, lambda h: unit in r.units_of_text(h.get('content')), k)
    d = _rank(hits, ok_doc, k)
    return p, u, d


def contamination(hits, genre, k=K):
    'Share of the top-k hits from a different genre. Only meaningful on the mixed store.'
    got = [(h.get('doc_id') or '').split(':', 1)[0] for h in hits[:k]]
    got = [g for g in got if g]
    return sum(1 for g in got if g != genre)/len(got) if got else 0.0


def aggregate(ranks, k=K):
    'MRR@k and hit rates from a list of (passage, section, doc) ranks.'
    n = max(1, len(ranks))
    def m(idx, pre_):
        rs = [t[idx] for t in ranks]
        return {f'{pre_}_mrr':  sum(1/(x+1) for x in rs if x is not None)/n,
                f'{pre_}_hit1': sum(1 for x in rs if x == 0)/n,
                f'{pre_}_hit5': sum(1 for x in rs if x is not None and x < 5)/n,
                f'{pre_}_hit{k}': sum(1 for x in rs if x is not None)/n}
    return {**m(0, 'p'), **m(1, 'u'), **m(2, 'd')}


METRICS = ('p_mrr', 'p_hit1', 'p_hit5', 'u_mrr', 'u_hit1', 'u_hit5', 'd_hit1')


# --------------------------------------------------------------- strategies
# Every strategy takes (db, q, qv, limit) and returns ranked rows carrying `content` and `doc_id`.
COLS = ['content', 'doc_id', 'page']


def s_fts(db, q, qv, limit=K, **kw):
    'FTS5 only, with the default token quoting — an implicit AND over every token in the query.'
    return db.t.store.fts_search(db.quote_fts(q), ['rowid'] + COLS, 'rank', limit)


def s_fts_pre(db, q, qv, limit=K, **kw):
    'FTS5 only, through `pre()`: keywords, wildcards, OR.'
    p = pre(q)
    return db.t.store.fts_search(p, ['rowid'] + COLS, 'rank', limit, quote=False) if p else []


def s_vec(db, q, qv, limit=K, **kw):
    'Exact vector scan.'
    return db.t.store.vec_search(qv, ['rowid'] + COLS, limit=limit)


def s_ann(db, q, qv, limit=K, **kw):
    'HNSW approximate vector search.'
    return db.t.store.ann_search(qv, ['rowid'] + COLS, limit)


def s_hybrid(db, q, qv, limit=K, **kw):
    'What `db.search` does out of the box: quoted FTS + exact vector, merged with RRF.'
    return db.search(q, qv, columns=COLS, limit=limit) or []


def s_hybrid_pre(db, q, qv, limit=K, rrf_k=60, depth=None, **kw):
    '''Hybrid with `pre()` on the FTS leg only, at the same candidate depth as `db.search`.

    Kept separate from `s_hybrid` because `db.search` sends one string to both legs: pass a
    preprocessed query and the reranker (and any future lexical scoring) sees `word* OR word*`.

    `depth` defaults to `limit`, matching what `db.search` fetches per leg. That equality is the
    whole point — fusing 30 candidates per leg and returning 10 beats fusing 10 and returning 10
    regardless of how the query was written, so a deeper `pre()` arm would credit preprocessing for
    a gain that belongs to candidate depth.'''
    d = depth or limit
    fts = s_fts_pre(db, q, qv, d)
    vec = db.t.store.vec_search(qv, ['rowid'] + COLS, limit=d)
    return rrf_merge(fts, vec, rrf_k, limit)


def s_hybrid_pre_deep(db, q, qv, limit=K, fanout=30, **kw):
    'The same, fusing `fanout` candidates per leg. Isolates candidate depth from everything else.'
    return s_hybrid_pre(db, q, qv, limit, depth=max(fanout, limit))


def s_hybrid_ann(db, q, qv, limit=K, **kw):
    'Hybrid with the ANN index doing the vector leg.'
    return db.search(q, qv, columns=COLS, limit=limit, ann=True) or []


def s_rerank(db, q, qv, limit=K, fanout=30, **kw):
    '''Hybrid, then a flashrank cross-encoder over the merged list.

    Compare against `hybrid-pre-deep`, not `hybrid-pre`: both fuse `fanout` candidates, so the
    difference is the cross-encoder and nothing else.'''
    hits = s_hybrid_pre(db, q, qv, fanout, depth=fanout)
    from litesearch.core import rerank_hits
    return rerank_hits(q, hits, None, limit)


def s_doc_search(db, q, qv, limit=K, **kw):
    'The tree layer: weighted RRF, span merging, breadcrumbs. Needs a tree store.'
    return db.doc_search(q, qv, columns=COLS, limit=limit)


def s_sections(db, q, qv, limit=K, **kw):
    '''`sections()` rolled back out to passage shape so it can be scored on the same axis.

    Each section contributes its snippets, concatenated, in section rank order — which is exactly
    what a caller would hand to a model.'''
    secs = db.sections(q, qv, limit=limit, per=3)
    if not secs: return []
    nids = [s['node_id'] for s in secs if s.get('node_id')]
    docs = {}
    if nids:
        from litesearch.core import _in
        docs = {r['id']: r['doc_id'] for r in db.t.nodes(select='id, doc_id', where=_in('id', nids))}
    return [dict(content='\n\n'.join(s.get('snippets') or []), page=None,
                 node_id=s.get('node_id'), doc_id=docs.get(s.get('node_id'), ''))
            for s in secs]


def s_graph(db, q, qv, limit=K, graph_w=0.5, **kw):
    'Hybrid plus a personalized-PageRank leg over the entity graph.'
    return db.graph_search(q, qv, columns=COLS, limit=limit, graph_w=graph_w)


STRATEGIES = {
    'fts': s_fts, 'fts-pre': s_fts_pre, 'vec': s_vec, 'ann': s_ann,
    'hybrid': s_hybrid, 'hybrid-pre': s_hybrid_pre, 'hybrid-pre-deep': s_hybrid_pre_deep,
    'hybrid-ann': s_hybrid_ann,
    'rerank': s_rerank, 'doc_search': s_doc_search, 'sections': s_sections,
    'graph': s_graph,
}
TREE_ONLY = ('doc_search', 'sections')
GRAPH_ONLY = ('graph',)


# ------------------------------------------------------------------ runner
def key_grams(q, r):
    '''The 5-grams of the source sentence that are unique to its section.

    Filtering to unique grams is what stops the passage metric from firing on `of the european union
    and`: a chunk retrieved from the wrong Article shares plenty of legislative boilerplate with the
    right one, and counting that as a hit would flatter every configuration equally and the coarse
    ones most.'''
    gs = grams(q['key'])
    keep = {g for g in gs if r.gram2u.get(g) == q['unit']}
    return keep or set(gs)


def eval_store(db, genre, queries, flavour, qvecs, strategy='hybrid', limit=K, **kw):
    'Run one strategy over one store for one flavour. Returns metrics + median query latency.'
    r, fn, ranks, lat, cont = ref(genre), STRATEGIES[strategy], [], [], []
    kgs = [key_grams(q, r) for q in queries]
    for q, qv in zip(queries, qvecs):
        t0 = time.time()
        try: hits = fn(db, q[flavour], qv.tobytes(), limit, **kw) or []
        except Exception as ex:
            hits = []
            if len(ranks) == 0: print(f'    ! {strategy} raised {type(ex).__name__}: {ex}', flush=True)
        lat.append((time.time()-t0)*1000)
        ranks.append(score_one(hits, q, r, limit, kgs[len(ranks)]))
        cont.append(contamination(hits, genre, limit))
    out = aggregate(ranks, limit)
    out['ms_p50'] = float(np.median(lat)); out['ms_p90'] = float(np.percentile(lat, 90))
    out['cross_genre'] = float(np.mean(cont)) if cont else 0.0
    out['n'] = len(ranks)
    return out


def fmt(name, m, width=34):
    return (f"{name:<{width}} p_mrr {m['p_mrr']:.3f}  p@1 {m['p_hit1']:.3f}  "
            f"u_mrr {m['u_mrr']:.3f}  u@1 {m['u_hit1']:.3f}  d@1 {m['d_hit1']:.3f}  "
            f"{m['ms_p50']:>6.1f}ms")
