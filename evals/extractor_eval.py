"""Does the keyphrase extractor change *retrieval*, or only the stopwatch?

`ingest_bench kw` says `yake-rust` is 6.6x the shipped yake per chunk, releases the GIL, and returns
66.5% of the same terms; `rake-nltk` is faster still and returns 21.6% of them. None of that says
whether the graph leg answers questions any better or worse, which is the only reason the extractor
exists. This module builds the *same* tree store's graph once per extractor and scores the graph
leg on each.

Two conditions per extractor, and the second is the one that matters:

- **with topics** — the graph as `build_graph` + `topic_nodes` leaves it. `run.eval_graph` already
  observed that on regulation the topic nodes carry essentially all of the PPR mass, which would
  make the extractor unable to move the metric whatever it extracted. That claim is re-checked here
  rather than inherited.
- **without topics** — the topic entities and their mentions deleted, leaving only the keyphrase
  and PMI graph. This is the condition in which the extractor's terms are the *only* thing feeding
  the walk, so it is where a difference has to show up if there is one.

Reported beside a `hybrid` baseline with no graph leg at all, because "the graph leg is worth
nothing here" and "the extractors are indistinguishable" look the same on the graph rows alone.

    python -m evals.extractor_eval                    # every extractor installed, every genre
    python -m evals.extractor_eval --genres regulatory --extractors yake,yake-rust
"""
from __future__ import annotations
import argparse, json, shutil, time
from pathlib import Path

from fastcore.all import first

from litesearch import database
from litesearch.core import _in

from . import corpus as C
from .build import db_path
from .encoders import enc
from .run import RESULTS, BASE_GRAIN, run_store, save

ENCODER = 'bge-small'


# ------------------------------------------------------------------ extractors
def _yake(topk_default=12):
    from litesearch.graph import _yake_terms
    return lambda text, topk=topk_default: _yake_terms(text, topk)

def _yake_rust(topk_default=12):
    import yake_rust
    y = yake_rust.Yake(language='en', ngrams=3)
    def f(text, topk=topk_default):
        try: return [w for w, _ in y.get_n_best(text, n=topk)]
        except Exception: return []
    return f

def _rake(topk_default=12):
    from rake_nltk import Rake
    import nltk
    for pkg in ('stopwords', 'punkt', 'punkt_tab'):
        try: nltk.download(pkg, quiet=True)
        except Exception: pass
    def f(text, topk=topk_default):
        try:
            r = Rake(max_length=3); r.extract_keywords_from_text(text)
            return r.get_ranked_phrases()[:topk]
        except Exception: return []
    return f

EXTRACTORS = {'yake': _yake, 'yake-rust': _yake_rust, 'rake-nltk': _rake}


def available(names=None):
    'The extractors that actually import, as `{name: fn}`. Missing ones are skipped, not fatal.'
    out = {}
    for n in (names or EXTRACTORS):
        try: out[n] = EXTRACTORS[n]()
        except Exception as e: print(f'  {n}: unavailable ({type(e).__name__}), skipping')
    return out


# ------------------------------------------------------------------ builds
def clone_store(src, dst):
    '''Copy a built store so a graph can be rebuilt over identical chunks.

    Two things make this more than a `shutil.copy`, and both were silently wrong the first time:

    - **The WAL.** `build_tree` leaves its connection open, so the `.db` file on its own is missing
      whatever has not been checkpointed — copying it alone gave 2,283 of 3,579 chunks, and every
      graph was then built over a two-thirds corpus.
    - **The ANN sidecar path is absolute and stored *inside* the database**, in `usearch_indices`.
      A plain copy therefore points at the *original's* `.usearch` file, so every clone shared one
      index, with 3,579 keys over a 2,283-row table. The sidecar is copied and the row repointed.
    '''
    s = database(str(src)); s.conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    m = s._ann_meta('store'); s.conn.close()
    shutil.copy(src, dst)
    if m and m['path'] and Path(m['path']).exists():
        side = f'{dst}.store.usearch'
        shutil.copy(m['path'], side)
        d = database(str(dst)); d.t.usearch_indices.update(dict(name='store', path=side)); d.conn.close()


def build_for(genre, name, fn, force=False):
    '''Clone the genre's tree store and build its graph with `fn` as the term extractor.

    A clone rather than a rebuild: the chunks, their embeddings and the ANN index are held fixed so
    the only thing that differs between two runs is the graph built over them. `mode` carries the
    extractor name, which is what puts each variant at its own `db_path`.'''
    from litesearch.graph import build_graph, resolve_entities, topic_nodes, graph_stats
    src, dst = db_path(genre, BASE_GRAIN, ENCODER, 'tree'), db_path(genre, BASE_GRAIN, ENCODER, _mode(name))
    if not src.exists(): print(f'  missing {src.stem} — run `python -m evals.run build_tree`'); return None
    if dst.exists() and not force: return dict(genre=genre, extractor=name, skipped=True)
    for f in Path(dst).parent.glob(f'{dst.stem}.db*'): f.unlink()
    clone_store(src, dst)
    e = enc('potion-32M')
    db = database(str(dst))
    n_st, n_ix = first(db.q('select count(*) c from store'))['c'], db.get_index('store').size
    assert n_st == n_ix, f'{dst.stem}: clone has {n_st} chunks but an index of {n_ix} — copy is wrong'
    # `id`, not just `content`: `build_graph` keys mentions on `chunk['id']` and falls back to
    # `_slug(content)`, but a tree store hashes ids over `node_id` *and* `content`, so the fallback
    # matched nothing and the keyphrase graph was fully disconnected from the store
    rows = list(db.t.store(select='id, content'))
    t0 = time.time()
    st = build_graph(db, rows, store='store', prose=True, code=False,
                     emb_fn=lambda ts, **kw: e.doc(ts), terms_fn=fn)
    t_g = time.time() - t0
    res, tn = resolve_entities(db, store='store'), topic_nodes(db, store='store')
    stats = graph_stats(db, 'store')
    db.conn.close()
    print(f'  {genre}/{name}: {st} resolve={res} topics={tn} ({t_g:.0f}s)', flush=True)
    return dict(genre=genre, extractor=name, build_s=round(t_g, 1), **st,
                resolved=res, topics=tn, stats=stats)


def _mode(name): return f'tree-x-{name.replace("-", "")}'


def drop_topics(genre, name):
    'Delete the topic entities and their mentions, leaving the keyphrase/PMI graph alone.'
    p = db_path(genre, BASE_GRAIN, ENCODER, _mode(name))
    db = database(str(p))
    tids = [r['id'] for r in db.t.entities(select='id', where="kind='topic'")]
    if tids:
        db.t.mentions.delete_where(where=_in('entity_id', tids))
        db.t.entities.delete_where(where=_in('id', tids))
    db.conn.close()
    return len(tids)


# ------------------------------------------------------------------ eval
def run(genres=None, names=None, graph_w=0.5, force=False, out='extractor'):
    genres = list(genres or C.GENRES)
    exts = available(names)
    if not exts: print('no extractors available'); return []
    print(f'\n== building one graph per extractor ({", ".join(exts)}) ==', flush=True)
    builds = [b for g in genres for n, fn in exts.items() if (b := build_for(g, n, fn, force))]

    rows = []
    print(f'\n== graph leg, topics intact (graph_w={graph_w}) ==', flush=True)
    for g in genres:
        # the no-graph baseline: without it, "every extractor scores the same" is unreadable
        for r in run_store(g, BASE_GRAIN, ENCODER, 'tree', ('hybrid',)):
            r['extractor'] = '(hybrid, no graph leg)'; r['topics'] = None; rows.append(r)
        # the same baseline over a *clone*, which must match the line above exactly. It is the only
        # thing that shows the clone carried the whole store: the first version of this eval copied
        # two thirds of the corpus and every extractor still scored identically, which read as a
        # null result rather than as the broken harness it was.
        if exts:
            for r in run_store(g, BASE_GRAIN, ENCODER, _mode(next(iter(exts))), ('hybrid',)):
                r['extractor'] = '(hybrid on a clone — control)'; r['topics'] = None; rows.append(r)
        for n in exts:
            for r in run_store(g, BASE_GRAIN, ENCODER, _mode(n), ('graph',), graph_w=graph_w):
                r['extractor'] = n; r['topics'] = True; rows.append(r)

    print(f'\n== graph leg, topic nodes removed ==', flush=True)
    for g in genres:
        for n in exts:
            k = drop_topics(g, n)
            for r in run_store(g, BASE_GRAIN, ENCODER, _mode(n), ('graph',), graph_w=graph_w):
                r['extractor'] = n; r['topics'] = False; r['n_topics_removed'] = k; rows.append(r)

    _report(rows, genres, exts)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS/f'{out}_build.json').write_text(json.dumps(builds, indent=1, default=str))
    save(out, rows)
    return rows


def _report(rows, genres, exts):
    'Averaged over flavours, then `paraphrase` alone — it is the only flavour that breaks surface overlap.'
    def table(title, pick):
        print(f'\n== {title} ==')
        hdr = f"  {'genre':<12}{'topics':<8}{'extractor':<32}{'p_mrr':>8}{'p@1':>8}{'u_mrr':>8}{'ms_p50':>9}"
        print(hdr); print('  ' + '-' * (len(hdr) - 2))
        for g in genres:
            for topics in (None, True, False):
                for n in (['(hybrid, no graph leg)', '(hybrid on a clone — control)'] if topics is None else list(exts)):
                    sel = [r for r in rows if r['genre'] == g and r['extractor'] == n
                           and r.get('topics') is topics and pick(r)]
                    if not sel: continue
                    m = lambda k: sum(r[k] for r in sel)/len(sel)
                    lbl = {None: 'n/a', True: 'yes', False: 'no'}[topics]
                    print(f"  {g:<12}{lbl:<8}{n:<32}{m('p_mrr'):>8.4f}{m('p_hit1'):>8.4f}"
                          f"{m('u_mrr'):>8.4f}{m('ms_p50'):>8.1f}ms")
    table('all query flavours, averaged', lambda r: True)
    table('paraphrase only (the flavour that breaks surface overlap)',
          lambda r: r['flavour'] == 'paraphrase')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--genres', default=None, help='comma-separated; default all')
    p.add_argument('--extractors', default=None, help='comma-separated; default all installed')
    p.add_argument('--graph-w', type=float, default=0.5)
    p.add_argument('--force', action='store_true', help='rebuild graphs that already exist')
    a = p.parse_args(argv)
    return run(genres=a.genres.split(',') if a.genres else None,
               names=a.extractors.split(',') if a.extractors else None,
               graph_w=a.graph_w, force=a.force)


if __name__ == '__main__': main()
