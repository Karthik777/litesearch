"""Reproduce the ColBERT vs static+FTS comparison. `python -m evals.run_colbert [genre ...]`.

Generic genres need their corpus fetched (`python -m evals.fetch_corpus`); the code corpus is built
from the three repos' Python on the fly (`evals.code_corpus`). Writes one JSON per corpus under
`results/` and prints the table.
"""
import json, sys
import numpy as np

from . import colbert_eval as CE
from . import code_corpus


def _line(r):
    hit = f"  recall@10 {r['hit10']:.3f}" if 'hit10' in r else ''
    print(f"  {r['corpus']:<11} {r['system']:<10} mrr {r['mrr']:.4f}  "
          f"p50 {r['ms_p50']:>6.1f}ms  p90 {r['ms_p90']:>6.1f}ms{hit}")

def _contrast(con):
    for s, c in con.items():
        verdict = 'different' if c['diff'] else 'no difference'
        print(f"    {s:<10} Δ{c['delta']:+.4f}  CI [{c['lo']:+.4f}, {c['hi']:+.4f}]  {verdict}")


def run_generic(genres):
    for g in genres:
        rows, rr = CE.eval_generic(g)
        con = CE.paired(rr)
        for r in rows: _line(r)
        _contrast(con)
        CE.save(rows, {g: con}, f'colbert_generic_{g}.json')

def run_code():
    corpus = code_corpus.build()
    rows, rr = CE.eval_code(corpus)
    con = CE.paired(rr)
    for r in rows: _line(r)
    _contrast(con)
    CE.save(rows, {'code': con}, 'colbert_code.json')


if __name__ == '__main__':
    args = sys.argv[1:] or ['regulatory', 'arxiv', 'astrology', 'code']
    gen = [a for a in args if a != 'code']
    if gen: run_generic(gen)
    if 'code' in args: run_code()
