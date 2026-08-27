"""Does an embedding store want markdown or plain text?

`fossick.to_md` runs readability then html2text and hands litesearch markdown. `fossick.to_text`
runs resiliparse and hands it plain text at a seventh of the cost. Both feed the same chunker and
the same encoder, so the question is whether the markdown syntax and the headings it preserves are
worth anything to retrieval, or whether they are tokens the encoder pays for and does not use.

Paired, on purpose. Ground truth is a set of source sentences that survive *both* conversions, so
neither pipeline is scored on text the other never saw, and every query is asked of both stores.
Coverage — which sentences a pipeline drops — is reported separately, because it is a different
failure from ranking the right chunk too low.

    python -m evals.htmlmd fetch     # download the page corpus (once)
    python -m evals.htmlmd build     # convert it both ways, cache the result
    python -m evals.htmlmd run [encoder]
"""
import json, random, re, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
from litesearch import database
from litesearch.data import chunk_markdown

from .encoders import enc, DT
from .queries import degrade, keywordise, paraphrase, synonyms, _content_words
from .refindex import nrm, grams, _SENT, _WORD
from .score import overlaps

HERE    = Path(__file__).parent
CORPUS  = HERE/'corpus/html'          # fetched, not committed: see `fetch`
CACHE   = HERE/'cache/htmlmd.json'    # derived from CORPUS, also not committed
RESULTS = HERE/'results/htmlmd.json'
STORES  = HERE/'cache'

#: 35 pages over documentation, reference, encyclopedia and spec prose. Chosen because readability
#: and resiliparse disagree about them in different ways, not because they are representative of
#: the web: a corpus of news and aggregators would flatter resiliparse and one of clean article
#: pages would flatter readability.
URLS = '''
https://en.wikipedia.org/wiki/Markdown
https://en.wikipedia.org/wiki/Retrieval-augmented_generation
https://en.wikipedia.org/wiki/Hierarchical_navigable_small_world
https://en.wikipedia.org/wiki/SQLite
https://en.wikipedia.org/wiki/Tree-sitter_(parser_generator)
https://en.wikipedia.org/wiki/PageRank
https://en.wikipedia.org/wiki/Word_embedding
https://en.wikipedia.org/wiki/Locality-sensitive_hashing
https://docs.python.org/3/library/functions.html
https://docs.python.org/3/library/sqlite3.html
https://docs.python.org/3/library/asyncio-task.html
https://docs.python.org/3/library/itertools.html
https://docs.python.org/3/tutorial/classes.html
https://docs.python.org/3/library/json.html
https://peps.python.org/pep-0008/
https://peps.python.org/pep-0484/
https://peps.python.org/pep-0703/
https://www.sqlite.org/fts5.html
https://www.sqlite.org/whentouse.html
https://www.sqlite.org/wal.html
https://docs.astral.sh/uv/concepts/projects/
https://docs.astral.sh/ruff/rules/
https://fastapi.tiangolo.com/tutorial/first-steps/
https://fastapi.tiangolo.com/async/
https://requests.readthedocs.io/en/latest/user/quickstart/
https://numpy.org/doc/stable/user/basics.broadcasting.html
https://numpy.org/doc/stable/user/absolute_beginners.html
https://scikit-learn.org/stable/modules/clustering.html
https://pandas.pydata.org/docs/user_guide/indexing.html
https://developer.mozilla.org/en-US/docs/Web/HTTP/Caching
https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Regular_expressions
https://git-scm.com/docs/git-rebase
https://www.postgresql.org/docs/current/indexes-types.html
https://redis.io/docs/latest/develop/data-types/
'''.split()


def fetch():
    'Download the corpus. Pages are not committed; this is the reproduction step.'
    import hashlib, httpx
    CORPUS.mkdir(parents=True, exist_ok=True)
    ok = 0
    with httpx.Client(follow_redirects=True, timeout=30,
                      headers={'User-Agent': 'Mozilla/5.0 (litesearch-eval)'}) as c:
        for u in URLS:
            f = CORPUS/f'{hashlib.blake2b(u.encode(), digest_size=6).hexdigest()}.html'
            if f.exists(): ok += 1; continue
            try: r = c.get(u)
            except Exception as e: print('  fail', type(e).__name__, u); continue
            if r.status_code == 200 and len(r.text) > 3000:
                f.write_text(r.text, errors='replace')
                f.with_suffix('.url').write_text(u); ok += 1
            else: print('  skip', r.status_code, u)
    print(f'{ok}/{len(URLS)} pages in {CORPUS}')
SEED    = 20260827
N_QUERIES = 400
FLAVOURS = ('verbatim', 'degraded', 'keyword', 'paraphrase')
ARMS = ('md', 'txt')


def convert():
    'Every page in the corpus, converted both ways.'
    from fossick.core import to_md, to_text
    if not any(CORPUS.glob('*.html')): raise SystemExit('no corpus: run `python -m evals.htmlmd fetch` first')
    out = {}
    for f in sorted(CORPUS.glob('*.html')):
        html = f.read_text(errors='replace')
        url = (f.with_suffix('.url')).read_text() if f.with_suffix('.url').exists() else f.stem
        try: md, txt = to_md(html), to_text(html)
        except Exception as e: print('  convert failed', url, e); continue
        if len(md.strip()) < 500 or len(txt.strip()) < 500:
            print(f'  thin, dropped: {url} md={len(md)} txt={len(txt)}'); continue
        out[url] = dict(md=md, txt=txt)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out))
    print(f'{len(out)} pages converted -> {CACHE}')
    return out


def load():
    if not CACHE.exists(): return convert()
    return json.loads(CACHE.read_text())


def sents(text):
    'Normalised sentences of a text, long enough to be unambiguous ground truth.'
    return [nrm(s) for s in _SENT.findall(text or '')]


def gramset(text): return set(grams(text))


def pick_queries(docs, n=N_QUERIES):
    '''Source sentences that survive both conversions, unique across the corpus.

    A sentence counts as surviving a conversion when every one of its 5-grams is present in that
    conversion's output. Matching on grams rather than on the string absorbs the whitespace and
    entity differences between the two converters without letting a near-miss through.
    '''
    rng = random.Random(SEED)
    seen = Counter()
    per_doc = {}
    for url, d in docs.items():
        gm, gt = gramset(d['md']), gramset(d['txt'])
        keep = []
        for s in sents(d['md']):
            g = grams(s)
            if len(g) < 8: continue
            if all(x in gm for x in g) and all(x in gt for x in g): keep.append(s)
        per_doc[url] = keep
        for s in keep: seen[s] += 1
    pool = [(u, s) for u, ss in per_doc.items() for s in ss if seen[s] == 1]
    rng.shuffle(pool)
    # spread across documents rather than taking whatever the shuffle piles up
    by_doc, out = {}, []
    for u, s in pool: by_doc.setdefault(u, []).append((u, s))
    while len(out) < n and any(by_doc.values()):
        for u in list(by_doc):
            if by_doc[u]: out.append(by_doc[u].pop())
            if len(out) >= n: break
    return out


def coverage(docs):
    'What share of each arm\'s sentences the other arm also carries.'
    tot = {a: 0 for a in ARMS}; kept = {a: 0 for a in ARMS}
    chars = {a: 0 for a in ARMS}
    for d in docs.values():
        g = {a: gramset(d[a]) for a in ARMS}
        for a in ARMS:
            chars[a] += len(d[a])
            other = 'txt' if a == 'md' else 'md'
            for s in sents(d[a]):
                gs = grams(s)
                if len(gs) < 8: continue
                tot[a] += 1
                if all(x in g[other] for x in gs): kept[a] += 1
    return {a: dict(sentences=tot[a], also_in_other=kept[a],
                    retained=round(kept[a]/max(tot[a],1), 4), chars=chars[a]) for a in ARMS}


def build_store(docs, arm, encoder, path):
    'One store per arm, same chunker and encoder, content stored as that arm converted it.'
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    for sfx in ('-wal','-shm'): Path(str(path)+sfx).unlink(missing_ok=True)
    db = database(str(path), sem_search=True)
    tbl = db.get_store('store')
    tbl.add_column('doc_id', str)
    e = enc(encoder)
    rows, t0 = [], time.time()
    for url, d in docs.items():
        for ch in chunk_markdown(d[arm]):
            if ch.strip(): rows.append(dict(content=ch, doc_id=url))
    vecs = e.doc([r['content'] for r in rows])
    for r, v in zip(rows, vecs): r['embedding'] = np.asarray(v, dtype=DT).tobytes()
    tbl.insert_all(rows)
    return db, tbl, dict(chunks=len(rows), build_s=round(time.time()-t0, 1))


def run(encoder='potion-32M', strategies=('hybrid','vector'), n=N_QUERIES):
    docs = load()
    print(f'{len(docs)} pages, encoder={encoder}')
    cov = coverage(docs)
    for a in ARMS:
        c = cov[a]
        print(f'  {a:<4} {c["chars"]:>9,} chars  {c["sentences"]:>5,} sentences  '
              f'{c["retained"]:.1%} of them also present in the other arm')

    qs = pick_queries(docs, n)
    print(f'{len(qs)} paired queries (sentences present in both arms, unique in the corpus)')

    # flavour inputs: df over the md arm's sentences, synonyms over the query words
    df = Counter()
    for d in docs.values():
        for s in set(sents(d['md'])): df.update(set(w.lower() for w in _WORD.findall(s)))
    syn = synonyms()
    rng = random.Random(SEED)
    built = []
    for u, s in qs:
        para, nswap = paraphrase(s, syn, rng)
        q = dict(doc=u, key=s, verbatim=s, degraded=degrade(s, df), keyword=keywordise(s, df),
                 paraphrase=para, n_swap=nswap)
        # a paraphrase that swapped nothing is a verbatim query wearing a different label
        if nswap >= 3 and all(q[f].strip() for f in FLAVOURS): built.append(q)
    print(f'{len(built)} usable after flavour generation')

    e = enc(encoder)
    qvec = {f: e.qry([q[f] for q in built]) for f in FLAVOURS}
    out, rr = [], {}
    for arm in ARMS:
        db, tbl, meta = build_store(docs, arm, encoder, STORES/f'htmlmd_{arm}.db')
        print(f'  {arm}: {meta["chunks"]:,} chunks, built in {meta["build_s"]}s')
        for strat in strategies:
            for f in FLAVOURS:
                ranks, lat = [], []
                for i, q in enumerate(built):
                    kg = set(grams(q['key']))
                    t0 = time.perf_counter()
                    if strat == 'hybrid':
                        hits = db.search(q[f], qvec[f][i].tobytes(), columns=['content','doc_id'],
                                         limit=10, dtype=DT) or []
                    else:
                        hits = tbl.vec_search(qvec[f][i].tobytes(), ['rowid','content','doc_id'],
                                              dtype=DT, limit=10)
                    lat.append((time.perf_counter()-t0)*1000)
                    r = next((j for j, h in enumerate(hits[:10])
                              if overlaps(h.get('content'), kg)), None)
                    ranks.append(r)
                n_ = max(1, len(ranks))
                rr[(arm, strat, f)] = [0.0 if x is None else 1/(x+1) for x in ranks]
                out.append(dict(arm=arm, strategy=strat, flavour=f, encoder=encoder,
                    mrr=round(sum(1/(x+1) for x in ranks if x is not None)/n_, 4),
                    hit1=round(sum(1 for x in ranks if x == 0)/n_, 4),
                    hit5=round(sum(1 for x in ranks if x is not None and x < 5)/n_, 4),
                    hit10=round(sum(1 for x in ranks if x is not None)/n_, 4),
                    ms_p50=round(float(np.percentile(lat, 50)), 2), n=len(ranks), **meta))
        db.close()
    sig = paired(rr)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    RESULTS.write_text(json.dumps(prev + [dict(coverage=cov, rows=out, sig=sig, encoder=encoder,
                                               n=len(built))], indent=1))
    show(out); show_sig(sig)
    return out


def paired(rr, iters=5000):
    '''Bootstrap the per-query difference in reciprocal rank, txt minus md.

    Paired and resampled by query, because the two arms answer the same queries: an unpaired test
    would be dominated by how hard each query is rather than by which arm answered it better.
    '''
    rng = np.random.default_rng(SEED)
    out = {}
    for strat in dict.fromkeys(k[1] for k in rr):
        for f in FLAVOURS:
            a, b = rr.get(('md', strat, f)), rr.get(('txt', strat, f))
            if not a or not b: continue
            d = np.array(b) - np.array(a)
            idx = rng.integers(0, len(d), size=(iters, len(d)))
            boot = d[idx].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            out[f'{strat}/{f}'] = dict(delta=round(float(d.mean()), 4),
                                       lo=round(float(lo), 4), hi=round(float(hi), 4),
                                       crosses_zero=bool(lo <= 0 <= hi))
    return out


def show_sig(sig):
    print(f'\n{"strategy/flavour":<24}{"delta mrr":>10}{"95% CI":>20}   verdict')
    for k, v in sig.items():
        verdict = 'indistinguishable' if v['crosses_zero'] else ('txt better' if v['delta'] > 0 else 'md better')
        print(f'{k:<24}{v["delta"]:>+10.4f}   [{v["lo"]:+.4f}, {v["hi"]:+.4f}]   {verdict}')


def show(rows):
    print(f'\n{"strategy":<9}{"flavour":<12}{"md mrr":>8}{"txt mrr":>9}{"delta":>8}   '
          f'{"md hit1":>8}{"txt hit1":>9}')
    for strat in dict.fromkeys(r['strategy'] for r in rows):
        for f in FLAVOURS:
            g = {r['arm']: r for r in rows if r['strategy'] == strat and r['flavour'] == f}
            if len(g) < 2: continue
            d = g['txt']['mrr'] - g['md']['mrr']
            print(f'{strat:<9}{f:<12}{g["md"]["mrr"]:>8.3f}{g["txt"]["mrr"]:>9.3f}{d:>+8.3f}   '
                  f'{g["md"]["hit1"]:>8.3f}{g["txt"]["hit1"]:>9.3f}')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'run'
    if cmd == 'fetch': fetch()
    elif cmd == 'build': convert()
    else: run(*(sys.argv[2:] or []))
