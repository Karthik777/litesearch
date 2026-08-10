"""A bridge query set: the answer is a passage that never uses the query's words.

Every query in `evals/queries.py` is a lexical transformation of the sentence it is looking for, so
the target is always findable by surface overlap and the graph leg can only add noise. That is why
the graph leg loses there, and it is also why losing there proves nothing: `graph_search` exists to
reach a passage the query does *not* name, by walking `query term → shared entity → passage`. This
builds the set that asks that question.

    anchor tokens A ─── appear in chunk X and *nowhere* in chunk Y
    query           ─── the anchor tokens alone
    target          ─── Y, relevant to X on grounds independent of any word it shares

Everything turns on that last clause, so the ground truth is the corpus's own structure rather than
a similarity score: **X and Y are two chunks of the same section**. A section is an author's own
statement that its contents belong together, `build_tree` recovers it from headings and `node_id`
records it — none of which involves an extractor, an embedding, or the entity graph. Had the graph
been consulted to pick the pairs, it would be guaranteed to answer them.

Y contains no token of A, so nothing lexical connects the query to the target and FTS can only
reach X. Whether the *vector* leg reaches Y anyway is not a flaw in the set, it is the comparison
worth running: if embeddings already bridge, the graph leg has nothing left to add.

**Two controls, because a bridge metric is easy to fool.**

- `control` — the same query scored against a random chunk from a third document. Anything above
  ~0 here means the metric is matching noise and the table should be thrown away.
- `source` — the same query scored against X, which FTS should nail. It separates "the graph
  cannot bridge" from "these queries are gibberish".

An earlier version paired chunks by rare-token overlap across documents. It survives as
`--crossdoc` but is not the default: on this corpus most such pairs are two reference lists sharing
author surnames, two boilerplate headers, or one hyphenation artifact (`estab`/`lished`) occurring
twice — bridges that are real lexically and meaningless otherwise.

    python -m evals.multihop                      # build and show the sets
    python -m evals.multihop --evaluate           # hybrid vs the graph leg over them
"""
from __future__ import annotations
import argparse, json, random, re, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from litesearch import database

from . import corpus as C
from .build import db_path
from .encoders import enc
from .queries import _STOP
from .refindex import _WORD
from .run import RESULTS, BASE_GRAIN
from .score import COLS, K

CACHE = Path(__file__).parent/'cache'
ENCODER = 'bge-small'
SEED = 20260803

MIN_LEN   = 4
N_ANCHOR  = 3         # anchor tokens per query
MIN_CHARS = 200       # skip stub chunks
DF_MAX    = 40        # an anchor has to be specific enough to be worth searching for
MIN_DOCS  = 2         # a token in one document only is usually a hyphenation artifact

_YEAR = re.compile(r'\b(?:19|20)\d{2}\b')


def _is_refs(text):
    '''A bibliography chunk. Two reference lists share author surnames without sharing a subject.
    Years are the cheap tell: prose cites a date or two, a reference list is made of them.'''
    return len(_YEAR.findall(text or '')) >= 5


def _toks(text):
    'Lowercased content tokens. Hyphens are dropped: PDF line-breaking invents `trans-former`.'
    return {w.lower() for w in _WORD.findall(text or '')
            if len(w) >= MIN_LEN and '-' not in w and w.lower() not in _STOP}


def build(genre, chunking=BASE_GRAIN, encoder=ENCODER, mode='tree', n=120, seed=SEED,
          refresh=False, crossdoc=False):
    'Bridge queries for one store, cached. Targets are chunk ids, so a set belongs to its store.'
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = 'crossdoc' if crossdoc else 'section'
    f = CACHE/f'multihop_{tag}_{genre}__{mode}__{chunking}__{encoder}.json'
    if f.exists() and not refresh: return json.loads(f.read_text())[:n]
    p = db_path(genre, chunking, encoder, mode)
    if not p.exists(): print(f'  missing {p.stem}'); return []
    db = database(str(p))
    rows = [r for r in db.t.store(select='id, doc_id, node_id, content')
            if len(r['content'] or '') >= MIN_CHARS and not _is_refs(r['content'])]
    db.conn.close()
    tok = {r['id']: _toks(r['content']) for r in rows}
    doc = {r['id']: r['doc_id'] for r in rows}
    df = Counter(t for ts in tok.values() for t in ts)
    seen_doc = set()
    for i, ts in tok.items():
        for t in ts: seen_doc.add((t, doc[i]))
    docs_of = Counter(t for t, _ in seen_doc)
    ok = {t for t, c in df.items() if 2 <= c <= DF_MAX and docs_of[t] >= MIN_DOCS}

    pairs, bysec = [], defaultdict(list)
    if crossdoc:
        inv = defaultdict(list)
        for i, ts in tok.items():
            for t in (ts & ok): inv[t].append(i)
        for x in tok:
            cand = Counter(y for t in (tok[x] & ok) for y in inv[t] if y != x and doc[y] != doc[x])
            pairs += [(x, y, sh) for y, sh in cand.most_common(4) if sh >= 2]
    else:
        bysec = defaultdict(list)
        for r in rows: bysec[r['node_id']].append(r['id'])
        for ids in bysec.values():
            if len(ids) < 2: continue
            pairs += [(a, b, 0) for a in ids for b in ids if a != b]
    sec_of = defaultdict(list)
    for r in rows: sec_of[r['id']] = [i for i in bysec[r['node_id']]] if not crossdoc else []

    rng = random.Random(seed)
    rng.shuffle(pairs)
    all_ids = sorted(tok)
    out, used = [], set()
    for x, y, sh in pairs:
        if len(out) >= n: break
        if (x, y) in used: continue
        anchor = sorted((tok[x] & ok) - tok[y], key=lambda t: df[t])[:N_ANCHOR]
        if len(anchor) < N_ANCHOR: continue
        assert not (set(anchor) & tok[y]), 'anchor leaked into the target'
        # every sibling of X in the section, minus any that leaks an anchor token. Pinning one
        # chunk made the metric all-or-nothing against thousands of candidates; the section is
        # what the ground truth actually asserts is relevant, so any sibling counts as a bridge
        sibs = [i for i in sec_of[x] if i != x and not (set(anchor) & tok[i])] if not crossdoc else [y]
        # a random chunk from a third document, also free of the anchor — the noise floor
        pool = [i for i in rng.sample(all_ids, min(60, len(all_ids)))
                if doc[i] not in (doc[x], doc[y]) and not (set(anchor) & tok[i])]
        if not pool: continue
        used.add((x, y))
        if not sibs: continue
        out.append(dict(query=' '.join(anchor), target=y, siblings=sibs, source=x, control=pool[0],
                        doc_target=doc[y], doc_source=doc[x], shared=sh))
    f.write_text(json.dumps(out, indent=1))
    return out[:n]


# ------------------------------------------------------------------ scoring
DEEP = 50   # a bridge that lands at rank 30 is still a bridge; k=10 could not see one at all

def _rr(hits, want, k=DEEP):
    'Reciprocal rank of the first hit in `want` (a set, or a single id).'
    w = want if isinstance(want, (set, frozenset, list, tuple)) else (want,)
    w = set(w)
    for i, h in enumerate(hits[:k]):
        if h.get('id') in w: return 1.0/(i+1)
    return 0.0


def evaluate(genres=None, weights=(0.25, 0.5, 1.0), n=120, crossdoc=False, out='multihop'):
    '''Hybrid against the graph leg on bridge queries, plus the two controls.

    `target` is the number that decides whether the graph leg earns its place; `control` is the
    noise floor and `source` the sanity check, both scored off the identical hit lists.'''
    genres = list(genres or C.GENRES)
    rows = []
    print(f"\n== bridge queries ({'cross-document' if crossdoc else 'same-section'}) ==")
    print(f"  {'genre':<12}{'strategy':<14}{'target MRR':>11}{'target hit':>11}"
          f"{'source MRR':>11}{'control MRR':>12}{'ms':>9}")
    print('  ' + '-' * 80)
    for g in genres:
        qs = build(g, n=n, crossdoc=crossdoc)
        p = db_path(g, BASE_GRAIN, ENCODER, 'tree')
        if not qs or not p.exists(): print(f'  {g}: no queries or no store'); continue
        db, e = database(str(p)), enc(ENCODER)
        qv = e.qry([q['query'] for q in qs])
        cols = list(dict.fromkeys(COLS + ['id']))
        for label, w in [('hybrid', None)] + [(f'graph w={w}', w) for w in weights]:
            tgt, src, ctl, lat = [], [], [], []
            for q, v in zip(qs, qv):
                t0 = time.time()
                hits = (db.search(q['query'], v.tobytes(), columns=cols, limit=DEEP) if w is None
                        else db.graph_search(q['query'], v.tobytes(), columns=cols, limit=DEEP, graph_w=w)) or []
                lat.append((time.time()-t0)*1000)
                tgt.append(_rr(hits, q.get('siblings') or q['target']))
                src.append(_rr(hits, q['source'])); ctl.append(_rr(hits, q['control']))
            m, h1 = float(np.mean(tgt)), float(np.mean([x > 0 for x in tgt]))
            print(f"  {g:<12}{label:<14}{m:>11.4f}{h1:>11.4f}"
                  f"{np.mean(src):>11.4f}{np.mean(ctl):>12.4f}{np.median(lat):>7.1f}ms")
            rows.append(dict(genre=g, strategy=label, graph_w=w, n=len(qs),
                             kind=('crossdoc' if crossdoc else 'section'),
                             target_mrr=m, target_hit1=h1, source_mrr=float(np.mean(src)),
                             control_mrr=float(np.mean(ctl)), ms_p50=float(np.median(lat)),
                             target_rr=[float(x) for x in tgt]))
        db.conn.close()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS/f'{out}.json').write_text(json.dumps(rows, indent=1))
    print(f'\n  -> {RESULTS/f"{out}.json"} ({len(rows)} rows)')
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--genres', default=None)
    p.add_argument('-n', type=int, default=120)
    p.add_argument('--refresh', action='store_true')
    p.add_argument('--crossdoc', action='store_true', help='the weaker rare-token pairing')
    p.add_argument('--evaluate', action='store_true')
    a = p.parse_args(argv)
    gs = a.genres.split(',') if a.genres else C.GENRES
    if a.refresh or not a.evaluate:
        for g in gs:
            qs = build(g, n=a.n, refresh=a.refresh, crossdoc=a.crossdoc)
            print(f'\n== {g}: {len(qs)} bridge queries ==')
            for q in qs[:3]: print(f"  query {q['query']!r}  target {q['target'][:10]}  source {q['source'][:10]}")
    if a.evaluate: evaluate(genres=gs, n=a.n, crossdoc=a.crossdoc)
    return 0


if __name__ == '__main__': main()
