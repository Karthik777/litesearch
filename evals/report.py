"""Aggregate the phase results into markdown tables.

One opinion is baked in here: the **weighted score**. Averaging the five flavours equally would let
`verbatim` — the flavour that exists only as a regression check, and which FTS answers near
perfectly — carry 20% of every ranking. The weights below put three quarters of the mass on the
three flavours where the query is not a copy of the answer, because that is what production traffic
looks like. Every per-flavour number is printed too, so the weighting can be argued with rather than
having to be trusted.
"""
import json
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).parent/'results'
STATS   = Path(__file__).parent/'cache/build_stats.json'

WEIGHTS = {'verbatim': 0.10, 'degraded': 0.15, 'keyword': 0.25, 'paraphrase': 0.25, 'kw_para': 0.25}
FLAVOURS = tuple(WEIGHTS)


def load(phase):
    f = RESULTS/f'{phase}.json'
    return json.loads(f.read_text()) if f.exists() else []


def build_stats():
    return json.loads(STATS.read_text()) if STATS.exists() else {}


def weighted(rows, metric='u_mrr'):
    'Weighted mean of `metric` over flavours. Missing flavours are dropped and the weights renormalised.'
    got = {r['flavour']: r[metric] for r in rows if r.get('flavour') in WEIGHTS}
    if not got: return None
    w = sum(WEIGHTS[f] for f in got)
    return sum(WEIGHTS[f]*v for f, v in got.items())/w


def group(rows, by):
    'Group rows by a tuple of keys.'
    out = defaultdict(list)
    for r in rows: out[tuple(r.get(k) for k in by)].append(r)
    return out


def table(rows, by, metric='u_mrr', label=None, extra=('ms_p50',), sort_by_score=True):
    '''Markdown table: one row per `by` group, one column per flavour, plus the weighted score.

    `extra` columns are taken from the group's median row so latency is reported once per
    configuration rather than once per flavour.'''
    g = group(rows, by)
    hdr = ['·'.join(str(x) for x in by)] + list(FLAVOURS) + ['**score**'] + list(extra)
    lines = ['| ' + ' | '.join(hdr) + ' |', '|' + '---|'*len(hdr)]
    items = []
    for k, rs in g.items():
        sc = weighted(rs, metric)
        if sc is None: continue
        by_fl = {r['flavour']: r[metric] for r in rs}
        cells = [f"{by_fl.get(f, float('nan')):.3f}" if f in by_fl else '–' for f in FLAVOURS]
        ex = []
        for c in extra:
            vals = sorted(r[c] for r in rs if r.get(c) is not None)
            ex.append(f'{vals[len(vals)//2]:.1f}' if vals else '–')
        items.append((sc, '· '.join(str(x) for x in k), cells, ex))
    items.sort(key=lambda t: -t[0] if sort_by_score else 0)
    for sc, name, cells, ex in items:
        lines.append('| ' + ' | '.join([name] + cells + [f'**{sc:.3f}**'] + ex) + ' |')
    return (f'\n**{label or metric}**\n\n' if label else '\n') + '\n'.join(lines) + '\n'


def cost_table(keys=None):
    'Build cost per store: chunks, embed seconds, chunks/s, bytes.'
    st = build_stats()
    hdr = ['store', 'chunks', 'mean chars', 'embed s', 'chunks/s', 'MB']
    lines = ['| ' + ' | '.join(hdr) + ' |', '|' + '---|'*len(hdr)]
    for k, v in sorted(st.items()):
        if keys and not any(x in k for x in keys): continue
        cps = v['chunks']/v['t_embed'] if v['t_embed'] else float('inf')
        lines.append(f"| `{k}` | {v['chunks']} | {v['mean_chars']} | {v['t_embed']:.1f} | "
                     f"{cps:,.0f} | {v['bytes']/1e6:.1f} |")
    return '\n'.join(lines) + '\n'


def summary(phase, by, metric='u_mrr', **kw):
    rows = load(phase)
    if not rows: return f'\n_(no results for `{phase}`)_\n'
    return table(rows, by, metric, **kw)


if __name__ == '__main__':
    import sys
    phase = sys.argv[1] if len(sys.argv) > 1 else 'grain'
    by = sys.argv[2].split(',') if len(sys.argv) > 2 else ['genre', 'chunking', 'encoder', 'strategy']
    metric = sys.argv[3] if len(sys.argv) > 3 else 'u_mrr'
    print(summary(phase, by, metric, label=f'{phase} — {metric}'))
