"""Turn the measured results into the comparisons a tier decision actually rests on.

Three things a table of scores does not tell you:

- **the marginal value of each feature**, paired across genres and flavours, rather than the score of
  whichever configuration happened to win;
- **the Pareto frontier** of quality against cost, since a configuration that is 0.01 better and 40x
  dearer is not a tier, it is a footnote;
- **where the win comes from**, because a feature that helps `verbatim` and hurts `kw_para` is worse
  than useless — production traffic is the second kind.
"""
import json
from collections import defaultdict

from . import report as R

FL = R.FLAVOURS


def _key(r): return (r['genre'], r.get('mode'), r.get('chunking'), r['encoder'], r['strategy'],
                     r.get('graph_w'))


def index(*phases):
    'All result rows from several phases, keyed by configuration, flavour -> metrics.'
    out = defaultdict(dict)
    for p in phases:
        for r in R.load(p): out[_key(r)][r['flavour']] = r
    return out


def score(rows, metric='u_mrr'):
    got = {f: v[metric] for f, v in rows.items() if f in R.WEIGHTS}
    if not got: return None
    w = sum(R.WEIGHTS[f] for f in got)
    return sum(R.WEIGHTS[f]*v for f, v in got.items())/w


def contrast(a, b, phases, metric='u_mrr', match=('genre',)):
    '''Paired difference between two configuration predicates, per genre and per flavour.

    `a` and `b` are callables over a result row. Only configurations that differ in exactly the
    dimension under test are compared, which is why the harness always builds the control.
    '''
    idx = index(*phases)
    rows_a = {k: v for k, v in idx.items() if a(dict(zip(('genre','mode','chunking','encoder','strategy','graph_w'), k)))}
    rows_b = {k: v for k, v in idx.items() if b(dict(zip(('genre','mode','chunking','encoder','strategy','graph_w'), k)))}
    out = []
    for ka, va in rows_a.items():
        da = dict(zip(('genre','mode','chunking','encoder','strategy','graph_w'), ka))
        for kb, vb in rows_b.items():
            db = dict(zip(('genre','mode','chunking','encoder','strategy','graph_w'), kb))
            if any(da[m] != db[m] for m in match): continue
            per_fl = {f: round(va[f][metric] - vb[f][metric], 4)
                      for f in FL if f in va and f in vb}
            if not per_fl: continue
            sa, sb = score(va, metric), score(vb, metric)
            out.append(dict(genre=da['genre'], a=_label(da), b=_label(db),
                            delta=round(sa-sb, 4), a_score=round(sa, 4), b_score=round(sb, 4),
                            per_flavour=per_fl))
    return sorted(out, key=lambda r: (r['genre'], -r['delta']))


def _label(d):
    return '·'.join(str(d[k]) for k in ('mode', 'chunking', 'encoder', 'strategy')
                    if d.get(k) is not None) + (f"·w{d['graph_w']}" if d.get('graph_w') else '')


def best_per_genre(phases, metric='u_mrr', top=5):
    idx = index(*phases)
    by = defaultdict(list)
    for k, v in idx.items():
        s = score(v, metric)
        if s is not None: by[k[0]].append((s, _label(dict(zip(('genre','mode','chunking','encoder','strategy','graph_w'), k))), v))
    return {g: sorted(v, key=lambda t: -t[0])[:top] for g, v in by.items()}


def cost_of(genre, chunking, encoder, mode):
    from .build import slug
    return R.build_stats().get(slug(genre, chunking, encoder, mode))


def frontier(phases, metric='u_mrr'):
    '''Quality against measured cost per configuration: indexing seconds and median query ms.

    Indexing cost is per 1,000 chunks so it can be compared across granularities, and it is the
    number that decides whether a corpus can be re-indexed at all.'''
    idx = index(*phases)
    out = []
    for k, v in idx.items():
        g, mode, ch, e, st, gw = k
        s = score(v, metric)
        if s is None: continue
        c = cost_of(g, ch, e, mode) if ch else None
        ms = sorted(r['ms_p50'] for r in v.values())
        out.append(dict(genre=g, config=_label(dict(zip(('genre','mode','chunking','encoder','strategy','graph_w'), k))),
                        score=round(s, 4), ms_p50=round(ms[len(ms)//2], 1),
                        embed_s_per_1k=round(c['t_embed']/max(c['chunks'], 1)*1000, 2) if c else None,
                        chunks=c['chunks'] if c else None, mb=round(c['bytes']/1e6, 1) if c else None))
    return sorted(out, key=lambda r: (r['genre'], -r['score']))


def show(title, rows, keys):
    print(f'\n### {title}')
    if not rows: print('  (no data)'); return
    print('  ' + '  '.join(f'{k}' for k in keys))
    for r in rows:
        print('  ' + '  '.join(str(r.get(k)) for k in keys))


ALL = ('grain', 'encoder', 'strategy', 'structure', 'late', 'graph', 'graph_notopic', 'mixed')

if __name__ == '__main__':
    print('# Feature contrasts (weighted section MRR)\n')

    show('pre() on the FTS leg vs the default token quoting',
         contrast(lambda d: d['strategy'] == 'hybrid-pre', lambda d: d['strategy'] == 'hybrid',
                  ('strategy', 'grain', 'encoder'), match=('genre', 'mode', 'chunking', 'encoder')),
         ('genre', 'a', 'b', 'a_score', 'b_score', 'delta', 'per_flavour'))

    show('cross-encoder rerank vs hybrid-pre alone',
         contrast(lambda d: d['strategy'] == 'rerank', lambda d: d['strategy'] == 'hybrid-pre',
                  ('strategy', 'encoder'), match=('genre', 'mode', 'chunking', 'encoder')),
         ('genre', 'a', 'b', 'a_score', 'b_score', 'delta', 'per_flavour'))

    show('tree vs flat, same encoder and granularity',
         contrast(lambda d: d['mode'] == 'tree', lambda d: d['mode'] == 'flat',
                  ('structure',), match=('genre', 'chunking', 'encoder', 'strategy')),
         ('genre', 'a', 'b', 'a_score', 'b_score', 'delta'))

    show('late chunking vs independent chunk embedding',
         contrast(lambda d: d['mode'] == 'late', lambda d: d['mode'] == 'flat',
                  ('late',), match=('genre', 'chunking', 'encoder', 'strategy')),
         ('genre', 'a', 'b', 'a_score', 'b_score', 'delta', 'per_flavour'))

    show('ANN vector leg vs the exact scan',
         contrast(lambda d: d['strategy'] == 'hybrid-ann', lambda d: d['strategy'] == 'hybrid',
                  ('strategy',), match=('genre', 'mode', 'chunking', 'encoder')),
         ('genre', 'a', 'b', 'a_score', 'b_score', 'delta'))

    print('\n\n# Best configurations per genre\n')
    for g, rows in best_per_genre(ALL).items():
        print(f'  {g}')
        for s, lab, v in rows: print(f'    {s:.4f}  {lab}')

    print('\n\n# Quality against cost\n')
    show('frontier', frontier(ALL),
         ('genre', 'config', 'score', 'ms_p50', 'embed_s_per_1k', 'chunks', 'mb'))
