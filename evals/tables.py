"""Emit every measured table as markdown, for pasting into the report.

    python -m evals.tables > /tmp/tables.md
"""
from . import report as R


def section(title, body): return f'\n## {title}\n{body}'


def main():
    out = []
    out.append(section('Corpus', _corpus()))
    out.append(section('Chunk granularity (section MRR)',
                       R.summary('grain', ['genre', 'chunking', 'encoder', 'strategy'])))
    out.append(section('Chunk granularity (passage MRR)',
                       R.summary('grain', ['genre', 'chunking', 'encoder', 'strategy'], 'p_mrr')))
    out.append(section('Encoder', R.summary('encoder', ['genre', 'encoder', 'strategy'])))
    out.append(section('Retrieval strategy',
                       R.summary('strategy', ['genre', 'encoder', 'strategy'])))
    out.append(section('Structure: flat vs tree',
                       R.summary('structure', ['genre', 'mode', 'encoder', 'strategy'])))
    out.append(section('Late chunking', R.summary('late', ['genre', 'mode', 'strategy'])))
    out.append(section('Graph leg', R.summary('graph', ['genre', 'strategy', 'graph_w'])))
    out.append(section('One store or three', R.summary('mixed', ['genre', 'mode', 'encoder', 'strategy'])))
    out.append(section('Cross-genre contamination in the mixed store',
                       R.summary('mixed', ['genre', 'mode', 'encoder', 'strategy'], 'cross_genre')))
    out.append(section('Clustering', _cluster()))
    out.append(section('Build cost', R.cost_table()))
    return '\n'.join(out)


def _corpus():
    from . import corpus as C
    from .refindex import ref
    from .queries import build, FLAVOURS, overlap
    lines = ['| genre | docs | pages | chars | sections | ≥300ch | queries | overlap by flavour |',
             '|---|---|---|---|---|---|---|---|']
    for g in C.GENRES:
        s, r, qs = C.stats(g), ref(g), build(g)
        ov = ' / '.join(f'{sum(overlap(q[f], q["key"]) for q in qs)/len(qs):.2f}' for f in FLAVOURS)
        lines.append(f"| {g} | {s['docs']} | {s['pages']} | {s['chars']/1e6:.2f}M | "
                     f"{len(r.units)} | {len(r.big_units)} | {len(qs)} | {ov} |")
    return '\n'.join(lines) + '\n\n(overlap order: ' + ', '.join(FLAVOURS) + ')\n'


def _cluster():
    import json
    f = R.RESULTS/'cluster.json'
    if not f.exists(): return '_(not run)_\n'
    d = json.loads(f.read_text())
    a = ['**clusters()**\n',
         '| genre | encoder | method | clusters | coverage | median size | max size | purity (doc) '
         '| purity (section) | NMI (doc) | label distinctiveness | s |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for c in d['clusters']:
        a.append(f"| {c['genre']} | {c['encoder']} | {c['method']} | {c['n_clusters']} | "
                 f"{c['coverage']} | {c['size_med']} | {c['size_max']} | {c['purity_doc']} | "
                 f"{c.get('purity_sec', c.get('purity_genre','–'))} | {c['nmi_doc']} | "
                 f"{c['label_dist']} | {c['t_cluster']} |")
    b = ['\n**peers() against ann_neighbors() at matched size**\n',
         '| genre | encoder | label | peers precision | kNN precision | mean group | method | ms (peers) | ms (kNN) |',
         '|---|---|---|---|---|---|---|---|---|']
    for p in d['peers']:
        b.append(f"| {p['genre']} | {p['encoder']} | {p['label']} | {p['peers_prec']} | "
                 f"{p['knn_prec']} | {p['mean_group']} | {list(p['methods'])} | "
                 f"{p['ms_peers']} | {p['ms_knn']} |")
    c = ['\n**Sample labels**\n']
    for x in d['clusters']:
        c.append(f"- `{x['genre']}/{x['encoder']}`: " + '; '.join(x['labels'][:4]))
    return '\n'.join(a + b + c) + '\n'


if __name__ == '__main__':
    print(main())
