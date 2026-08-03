"""Run the evaluation in phases. Each phase is resumable and writes one JSON file.

    python -m evals.run build_core        # stores for the granularity + encoder sweeps
    python -m evals.run build_tree        # tree / tree-nohead stores
    python -m evals.run build_late        # late chunking + fulldoc controls
    python -m evals.run build_graph       # entity graph + topic nodes over the tree stores
    python -m evals.run eval_grain        # granularity x flavour
    python -m evals.run eval_encoder      # encoder x flavour
    python -m evals.run eval_strategy     # retrieval strategy x flavour
    python -m evals.run eval_structure    # flat vs tree vs tree-nohead
    python -m evals.run eval_late         # naive vs late vs fulldoc
    python -m evals.run eval_cluster      # clustering, intrinsically
"""
import json, sys, time
from pathlib import Path
import numpy as np

from litesearch import database

from . import corpus as C
from .build import (build_flat, build_tree, build_late, build_fulldoc, db_path, show, slug)
from .encoders import enc
from .queries import build as build_queries, FLAVOURS
from .score import eval_store, STRATEGIES, fmt

RESULTS = Path(__file__).parent/'results'
QVEC    = Path(__file__).parent/'cache/qvec'

GRAINS   = ('page', 'c1024', 'c512', 'c256')
ENCODERS = ('potion-32M', 'bge-small', 'jina-v2-sm', 'egemma-300m', 'nomic-v1.5')
BASE_GRAIN = 'c512'          # the granularity every non-granularity sweep holds fixed


# ------------------------------------------------------------------ helpers
def qvecs(genre, encoder, refresh=False):
    'Cached query embeddings, one array per flavour. Invalidated when the query set changes.'
    QVEC.mkdir(parents=True, exist_ok=True)
    f = QVEC/f'{genre}__{encoder}.npz'
    qs = build_queries(genre)
    if f.exists() and not refresh:
        z = np.load(f)
        if set(z.files) == set(FLAVOURS) and len(z[FLAVOURS[0]]) == len(qs):
            return {k: z[k] for k in z.files}
        print(f'  qvec cache for {genre}/{encoder} is stale; re-encoding', flush=True)
    e = enc(encoder)
    out = {fl: e.qry([q[fl] for q in qs]) for fl in FLAVOURS}
    np.savez(f, **out)
    return out


def save(phase, rows):
    RESULTS.mkdir(parents=True, exist_ok=True)
    f = RESULTS/f'{phase}.json'
    old = json.loads(f.read_text()) if f.exists() else []
    seen = {tuple(sorted((k, str(v)) for k, v in r.items() if k in _KEY)) for r in rows}
    keep = [r for r in old
            if tuple(sorted((k, str(v)) for k, v in r.items() if k in _KEY)) not in seen]
    f.write_text(json.dumps(keep + rows, indent=1))
    print(f'  -> {f} ({len(keep)+len(rows)} rows)')

_KEY = ('genre', 'mode', 'chunking', 'encoder', 'strategy', 'flavour', 'graph_w')


def run_store(genre, chunking, encoder, mode, strategies, flavours=FLAVOURS, **kw):
    'Every (strategy, flavour) over one store. Skips silently if the store was never built.'
    p = db_path(genre, chunking, encoder, mode)
    if not p.exists(): print(f'  missing {p.stem}'); return []
    db, qs, qv = database(str(p)), build_queries(genre), qvecs(genre, encoder)
    rows = []
    for st in strategies:
        for fl in flavours:
            m = eval_store(db, genre, qs, fl, qv[fl], st, **kw)
            rows.append(dict(genre=genre, mode=mode, chunking=chunking, encoder=encoder,
                             strategy=st, flavour=fl, **{k: round(v, 4) for k, v in m.items()},
                             **{k: v for k, v in kw.items()}))
    return rows


# ------------------------------------------------------------------ builds
def build_core():
    'Granularity sweep (cheap encoders) plus the c512 store for every encoder.'
    for g in C.GENRES:
        for ch in GRAINS:
            for e in ('potion-32M', 'bge-small'):
                print(show(build_flat(g, ch, e)), flush=True)
    for g in C.GENRES:
        for e in ENCODERS:
            print(show(build_flat(g, BASE_GRAIN, e)), flush=True)


def build_trees():
    for g in C.GENRES:
        for e in ('potion-32M', 'bge-small', 'jina-v2-sm'):
            print(show(build_tree(g, BASE_GRAIN, e, with_heading=True)), flush=True)
        print(show(build_tree(g, BASE_GRAIN, 'bge-small', with_heading=False)), flush=True)
        print(show(build_tree(g, BASE_GRAIN, 'bge-small', bold=True)), flush=True)


def build_lates():
    'Late chunking needs a context window; `fulldoc` is the control at the other extreme.'
    for g in C.GENRES:
        print(f'  {g} late:', flush=True)
        print(show(build_late(g, BASE_GRAIN, 'jina-v2-sm')), flush=True)
        print(show(build_fulldoc(g, 'jina-v2-sm')), flush=True)


def build_graphs():
    '''Entity graph + topic nodes beside the tree store, in the same database file.

    The graph is keyed on the hash of chunk content, which is exactly the `id` a hash store
    already uses, so it attaches to the chunks that are there rather than re-chunking.'''
    from litesearch import build_graph, resolve_entities, topic_nodes, spacy_pipe, graph_stats
    nlp = None
    try: nlp = spacy_pipe()
    except Exception as ex: print(f'  spaCy unavailable ({ex}); falling back to yake', flush=True)
    e = enc('potion-32M')            # entity-name vectors only; the cheap one is the right one
    out = {}
    for g in C.GENRES:
        for encn in ('bge-small',):
            p = db_path(g, BASE_GRAIN, encn, 'tree')
            if not p.exists(): print(f'  missing {p.stem}'); continue
            db = database(str(p))
            rows = list(db.t.store(select='content'))
            t0 = time.time()
            st = build_graph(db, rows, store='store', nlp=nlp, prose=True, code=False,
                             emb_fn=lambda ts, **kw: e.doc(ts))
            t_g = time.time()-t0
            t0 = time.time(); res = resolve_entities(db, store='store'); t_r = time.time()-t0
            t0 = time.time(); tn = topic_nodes(db, store='store'); t_t = time.time()-t0
            out[f'{g}__{encn}'] = dict(genre=g, encoder=encn, **st, resolved=res, topics=tn,
                                       t_graph=round(t_g, 1), t_resolve=round(t_r, 1),
                                       t_topics=round(t_t, 1), stats=graph_stats(db, 'store'))
            print(f"  {g}/{encn}: {st} resolve={res} topics={tn} "
                  f"({t_g:.0f}s + {t_r:.0f}s + {t_t:.0f}s)", flush=True)
    (RESULTS/'graph_build.json').parent.mkdir(parents=True, exist_ok=True)
    (RESULTS/'graph_build.json').write_text(json.dumps(out, indent=1, default=str))


# ------------------------------------------------------------------ evals
def eval_grain():
    rows = []
    for g in C.GENRES:
        for ch in GRAINS:
            for e in ('potion-32M', 'bge-small'):
                r = run_store(g, ch, e, 'flat', ('hybrid', 'hybrid-pre', 'vec', 'fts-pre'))
                for x in r: print('  ' + fmt(f"{g}/{ch}/{e}/{x['strategy']}/{x['flavour']}", x, 52))
                rows += r
    save('grain', rows)


def eval_encoder():
    rows = []
    for g in C.GENRES:
        for e in ENCODERS:
            r = run_store(g, BASE_GRAIN, e, 'flat', ('vec', 'hybrid', 'hybrid-pre'))
            for x in r: print('  ' + fmt(f"{g}/{e}/{x['strategy']}/{x['flavour']}", x, 52))
            rows += r
    save('encoder', rows)


def eval_strategy():
    rows = []
    for g in C.GENRES:
        for e in ('potion-32M', 'bge-small'):
            r = run_store(g, BASE_GRAIN, e, 'flat',
                          ('fts', 'fts-pre', 'vec', 'ann', 'hybrid', 'hybrid-ann', 'hybrid-pre', 'rerank'))
            for x in r: print('  ' + fmt(f"{g}/{e}/{x['strategy']}/{x['flavour']}", x, 52))
            rows += r
    save('strategy', rows)


def eval_structure():
    rows = []
    for g in C.GENRES:
        for e in ('potion-32M', 'bge-small', 'jina-v2-sm'):
            rows += run_store(g, BASE_GRAIN, e, 'flat', ('hybrid', 'hybrid-pre'))
            rows += run_store(g, BASE_GRAIN, e, 'tree',
                              ('hybrid', 'hybrid-pre', 'doc_search', 'sections'))
        for m in ('tree-nohead', 'tree-bold'):
            rows += run_store(g, BASE_GRAIN, 'bge-small', m,
                              ('hybrid', 'hybrid-pre', 'doc_search', 'sections'))
    for x in rows: print('  ' + fmt(f"{x['genre']}/{x['mode']}/{x['encoder']}/{x['strategy']}/{x['flavour']}", x, 58))
    save('structure', rows)


def eval_late():
    rows = []
    for g in C.GENRES:
        rows += run_store(g, BASE_GRAIN, 'jina-v2-sm', 'flat', ('hybrid-pre', 'vec'))
        rows += run_store(g, BASE_GRAIN, 'jina-v2-sm', 'late', ('hybrid-pre', 'vec'))
        rows += run_store(g, 'doc', 'jina-v2-sm', 'fulldoc', ('hybrid-pre', 'vec'))
    for x in rows: print('  ' + fmt(f"{x['genre']}/{x['mode']}/{x['strategy']}/{x['flavour']}", x, 52))
    save('late', rows)


def eval_graph():
    '''The graph leg at three weights, then the same graph with its topic nodes removed.

    Removing the topics is how the clustering layer's contribution *to ranking* gets isolated:
    `topic_nodes` writes one entity per cluster plus a mention per member, so deleting exactly those
    rows leaves the spaCy/PMI graph intact and answers "did clustering the index help retrieval, or
    only help a human read the corpus".'''
    rows = []
    for g in C.GENRES:
        for w in (0.25, 0.5, 1.0):
            rows += run_store(g, BASE_GRAIN, 'bge-small', 'tree', ('graph',), graph_w=w)
        rows += run_store(g, BASE_GRAIN, 'bge-small', 'tree', ('hybrid', 'hybrid-pre'))
    for x in rows: print('  ' + fmt(f"{x['genre']}/{x['strategy']}{x.get('graph_w','')}/{x['flavour']}", x, 52))
    save('graph', rows)
    # --- now without topic nodes
    notopic = []
    for g in C.GENRES:
        p = db_path(g, BASE_GRAIN, 'bge-small', 'tree')
        if not p.exists(): continue
        db = database(str(p))
        tids = [r['id'] for r in db.t.entities(select='id', where="kind='topic'")]
        if not tids: print(f'  {g}: no topic nodes to remove'); continue
        from litesearch.core import _in
        db.t.mentions.delete_where(where=_in('entity_id', tids))
        db.t.entities.delete_where(where=_in('id', tids))
        print(f'  {g}: removed {len(tids)} topic nodes', flush=True)
        r = run_store(g, BASE_GRAIN, 'bge-small', 'tree', ('graph',), graph_w=0.5)
        for x in r: x['mode'] = 'tree-notopic'
        notopic += r
        for x in r: print('  ' + fmt(f"{g}/graph-notopic/{x['flavour']}", x, 52))
    save('graph_notopic', notopic)


def eval_mixed():
    '''One store per genre against one store holding all three.

    The realistic library is mixed — regulations, papers and books in the same database — and the
    astro-ph papers were chosen so that the mixed store has a genuine confusion to make: they talk
    about Mars, conjunctions and ascending nodes without meaning any of it the way the astrology
    books do.'''
    from .build import build_flat
    rows = []
    for e in ('potion-32M', 'bge-small'):
        print(show(build_flat('mixed', BASE_GRAIN, e)), flush=True)
    for g in C.GENRES:
        for e in ('potion-32M', 'bge-small'):
            for mode_genre, tag in ((g, 'single'), ('mixed', 'mixed')):
                p = db_path(mode_genre, BASE_GRAIN, e, 'flat')
                if not p.exists(): print(f'  missing {p.stem}'); continue
                db, qs, qv = database(str(p)), build_queries(g), qvecs(g, e)
                for st in ('hybrid-pre', 'rerank'):
                    for fl in FLAVOURS:
                        m = eval_store(db, g, qs, fl, qv[fl], st)
                        rows.append(dict(genre=g, mode=tag, chunking=BASE_GRAIN, encoder=e,
                                         strategy=st, flavour=fl,
                                         **{k: round(v, 4) for k, v in m.items()}))
                        print('  ' + fmt(f'{g}/{tag}/{e}/{st}/{fl}', rows[-1], 50)
                              + f"  xgenre {rows[-1]['cross_genre']:.3f}")
    save('mixed', rows)


def eval_cluster():
    from .cluster_eval import run as cluster_run
    cluster_run()


PHASES = dict(build_core=build_core, build_tree=build_trees, build_late=build_lates,
              build_graph=build_graphs, eval_grain=eval_grain, eval_encoder=eval_encoder,
              eval_strategy=eval_strategy, eval_structure=eval_structure, eval_late=eval_late,
              eval_graph=eval_graph, eval_cluster=eval_cluster, eval_mixed=eval_mixed)

if __name__ == '__main__':
    for name in (sys.argv[1:] or ['--help']):
        if name not in PHASES: print(__doc__); raise SystemExit(1)
        print(f'=== {name}', flush=True)
        t0 = time.time(); PHASES[name](); print(f'=== {name} done in {time.time()-t0:.0f}s', flush=True)
