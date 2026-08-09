"""Ingestion throughput benchmarks for litesearch.

Measures the *write* side of the stack — parsing, chunking, embedding, SQLite upserts, ANN index
maintenance, graph extraction — rather than retrieval quality, which `evals/run.py` already covers.
Text comes from the same `evals.corpus` genres the retrieval evals use, repartitioned into as many
documents as a run needs, so a throughput number is measured on prose that chunks and trees the way
the real corpus does.

`hash_embed` is the default stand-in for an encoder: it keeps ONNX inference out of the numbers so
what is left is litesearch's own overhead. `--encoder potion` puts a real model back in to show the
split between the two.

Sizes are swept rather than run once, because per-document cost is only a *cost* if it is constant.
The exponent of the growth curve is the number to read: 1.0 is linear, and anything approaching 2.0
cannot reach a million documents at any hardware budget.

    python -m evals.ingest_bench docs --sizes 25,50,100,200
    python -m evals.ingest_bench code --dir litesearch
    python -m evals.ingest_bench graph --sizes 500,1000,2000
    python -m evals.ingest_bench all
"""
from __future__ import annotations
import argparse, cProfile, io, json, os, pstats, shutil, tempfile, time
from pathlib import Path

import numpy as np

from fastcore.all import L, first, defaults

from litesearch.core import database, process_content
from litesearch.graph import build_graph, hash_embed, resolve_entities, rrf_all
from litesearch.data import dir2chunks

# ---------------------------------------------------------------- timing helpers

class Timer:
    'Wall + CPU time for a block, in seconds.'
    def __enter__(self):
        self.t0, self.c0 = time.perf_counter(), time.process_time()
        return self
    def __exit__(self, *a):
        self.wall = time.perf_counter() - self.t0
        self.cpu = time.process_time() - self.c0

def _fit_exponent(xs, ys):
    '''Slope of log(y) on log(x) — the empirical exponent of the growth curve.

    1.0 means linear (what ingestion should be), 2.0 means each item pays for every item before it.
    Reported instead of raw times because "is it linear" is the only question that matters when the
    target corpus is 10^6 documents and the benchmark can only run 10^2.'''
    xs = [x for x, y in zip(xs, ys) if x > 0 and y > 0]
    ys = [y for y in ys if y > 0]
    if len(xs) < 2: return float('nan')
    lx, ly = np.log(np.array(xs, float)), np.log(np.array(ys, float))
    return float(np.polyfit(lx, ly, 1)[0])

def _row(name, n, wall, extra=''):
    rate = n / wall if wall else float('inf')
    print(f'  {name:<34} n={n:<8,d} {wall:8.3f}s  {rate:10,.1f}/s  {extra}')
    return dict(step=name, n=n, wall=wall, rate=rate)

# ---------------------------------------------------------------- corpora

GENRE = os.environ.get('LITESEARCH_BENCH_GENRE', 'regulatory')

def _corpus_pages(genre=None):
    'Every page of a genre as a flat list of texts, cached across calls.'
    from . import corpus as C
    g = genre or GENRE
    if g not in _corpus_pages.cache:
        docs = C.load(g)
        _corpus_pages.cache[g] = [t for v in docs.values() for _, t in v if (t or '').strip()]
    return _corpus_pages.cache[g]
_corpus_pages.cache = {}

def corpus_docs(n, pages_per_doc=3, genre=None):
    '''`n` documents of `pages_per_doc` real corpus pages each, cycling the corpus when it runs out.

    Repartitioning rather than duplicating whole documents matters for the store benchmarks: a
    hash-id store upserts by content hash, so a corpus of byte-identical documents measures
    deduplication rather than insertion.'''
    pages = _corpus_pages(genre)
    out = []
    for i in range(n):
        sel = [pages[(i*pages_per_doc + j) % len(pages)] for j in range(pages_per_doc)]
        # a unique heading per document keeps chunk hashes distinct once the corpus wraps
        out.append([(0, f'# Document {i}\n\n' + sel[0])] + list(enumerate(sel[1:], start=1)))
    return out

def corpus_chunks(n, chars=800, genre=None):
    '''`n` chunk dicts of real prose, each holding *different* text.

    Sliding the window within a page as well as across pages matters for the graph benchmarks: with
    a fixed `page[:chars]` slice the corpus wraps after ~500 chunks and every later chunk repeats an
    earlier one, so entity and edge counts flatten and the growth curve measures deduplication
    instead of extraction.'''
    pages = _corpus_pages(genre)
    P, out = len(pages), []
    for i in range(n):
        pg = pages[i % P]
        off = ((i // P) * chars) % max(1, len(pg))
        txt = (pg[off:off+chars] or pg[:chars])
        if len(txt) < chars // 2: txt = pg[:chars]          # tail slices too short to be a chunk
        out.append(dict(content=f'[{i}] ' + txt))
    return out

def mk_doc_corpus(dst: Path, n, pages_per_doc=3, genre=None):
    'Write `n` markdown files of real corpus text, for the `add_dir` path.'
    dst.mkdir(parents=True, exist_ok=True)
    for i, pages in enumerate(corpus_docs(n, pages_per_doc, genre)):
        (dst/f'doc_{i:05d}.md').write_text('\n\n'.join(t for _, t in pages))
    return dst

def hash_emb_fn(ndim=256):
    'Deterministic embedder with no model download — isolates litesearch overhead from inference.'
    return lambda texts, **kw: hash_embed(texts, ndim=ndim)

def real_emb_fn():
    'The static potion encoder from evals.encoders — cheapest real model, float16 out.'
    from .encoders import enc
    e = enc('potion-32M')
    return lambda texts, **kw: e.doc(list(texts))

# ---------------------------------------------------------------- benchmarks

def bench_docs(sizes=(25, 50, 100), emb=None, tmp: Path = None, ann=True, workdir=None):
    '''`add_dir` over markdown docs at increasing corpus sizes.

    The point of sweeping sizes rather than timing one run: per-document cost is only meaningful if
    it is constant. Anything that touches the whole store per document shows up here as an exponent
    above 1.'''
    print(f'\n== add_dir / tree ingestion (ann={ann}) ==')
    rows, per_doc = [], []
    for n in sizes:
        root = Path(tempfile.mkdtemp(dir=workdir))
        corpus = mk_doc_corpus(root/'corpus', n, pages_per_doc=3)
        db = database(str(root/'bench.db'))
        db.get_tree('store', ann=ann)
        with Timer() as t: res = db.add_dir(str(corpus), emb_fn=emb)
        chunks = sum(r.get('chunks', 0) for r in res)
        rows.append(_row(f'add_dir({n} md docs)', n, t.wall, f'{chunks:,} chunks  {t.wall/n*1000:.1f} ms/doc'))
        per_doc.append(t.wall/n)
        db.conn.close(); shutil.rmtree(root, ignore_errors=True)
    print(f'  -> total-time exponent {_fit_exponent(sizes, [r["wall"] for r in rows]):.2f} '
          f'(1.0 = linear), per-doc cost grew {per_doc[-1]/per_doc[0]:.1f}x over {sizes[0]}->{sizes[-1]} docs')
    return rows

def bench_deferred_index(sizes=(50, 100, 200), emb=None, workdir=None):
    '''What `add_dir` costs when the ANN index is built once instead of once per document.

    `add_doc` ends with `if self._ann_meta(store): g.store.rebuild_index()`, and `rebuild_index`
    reads every embedding blob in the store and reconstructs the whole HNSW graph from scratch. Per
    document that is O(corpus), so a directory of N documents does O(N^2) index work for an index
    that is only read once ingestion finishes. Deferring it is measured here rather than argued:
    the same corpus, the same final index, one rebuild at the end.'''
    print('\n== add_dir: per-doc rebuild_index vs one deferred rebuild ==')
    rows = []
    for n in sizes:
        base = _add_dir_run(n, emb, workdir, defer=False)
        defer = _add_dir_run(n, emb, workdir, defer=True)
        rows.append(_row(f'add_dir({n}) per-doc rebuild', n, base['wall'], f'{base["idx"]:,} vectors'))
        rows.append(_row(f'add_dir({n}) deferred rebuild', n, defer['wall'],
                         f'{defer["idx"]:,} vectors  incl. {defer["t_rebuild"]:.2f}s final rebuild'))
        print(f'  -> {base["wall"]/defer["wall"]:.2f}x faster at n={n}')
    b = [r['wall'] for r in rows[0::2]]
    d = [r['wall'] for r in rows[1::2]]
    print(f'  -> exponent: per-doc {_fit_exponent(sizes, b):.2f}  deferred {_fit_exponent(sizes, d):.2f}')
    return rows

def _add_dir_run(n, emb, workdir, defer):
    'One add_dir run, optionally suppressing rebuild_index until the end.'
    from apswutils.db import Table
    root = Path(tempfile.mkdtemp(dir=workdir))
    corpus = mk_doc_corpus(root/'corpus', n, pages_per_doc=3)
    db = database(str(root/'bench.db'))
    db.get_tree('store', ann=True)
    real, t_rebuild = Table.rebuild_index, 0.0
    if defer: Table.rebuild_index = lambda self, dtype=None: 0
    try:
        with Timer() as t: db.add_dir(str(corpus), emb_fn=emb)
        wall = t.wall
    finally: Table.rebuild_index = real
    if defer:
        with Timer() as t2: db.t.store.rebuild_index()
        t_rebuild, wall = t2.wall, wall + t2.wall
    idx = db.get_index('store').size
    db.conn.close(); shutil.rmtree(root, ignore_errors=True)
    return dict(wall=wall, idx=idx, t_rebuild=t_rebuild)

def bench_shards(n=400, shards=(1, 2, 4, 8), emb=None, workdir=None, queries=25):
    '''One store per shard versus one monolithic store, for the same corpus.

    The case for sharding is not that SQLite gets slow — it is that every whole-index operation in
    the ingest path costs O(rows in that index). `add_doc` rebuilds the HNSW graph per document and
    `resolve_entities` is superlinear in entity count, so both are paid against the size of whichever
    index the document lands in. K shards divide that size by K, which turns a quadratic into K
    independent quadratics of 1/K^2 the work each. The read side has to pay for it: a federated query
    runs K searches and fuses them, so the cost is measured here too.'''
    print(f'\n== sharded vs monolithic ingest ({n} docs) ==')
    rows, docs = [], corpus_docs(n, pages_per_doc=3)
    for k in shards:
        root = Path(tempfile.mkdtemp(dir=workdir))
        dbs, per = [], (n + k - 1) // k
        with Timer() as t:
            for s in range(k):
                db = database(str(root/f'shard{s}.db'))
                db.get_tree('store', ann=True)
                for i, pages in enumerate(docs[s*per:(s+1)*per]):
                    db.add_doc(pages, title=f'doc {s*per+i}', source=f'doc{s*per+i}', emb_fn=emb)
                dbs.append(db)
        nrow = sum(first(d.q('select count(*) c from store'))['c'] for d in dbs)
        ing = _row(f'ingest into {k} shard(s)', n, t.wall, f'{nrow:,} chunks, {per} docs/shard')
        qv = np.asarray(hash_embed(['transport of dangerous goods by vessel'], ndim=256)).ravel()
        with Timer() as t:
            for _ in range(queries):
                hits = [h for d in dbs for h in (d.search('transport of dangerous goods by vessel',
                        qv.tobytes(), columns=['id','content'], limit=20, ann=True) or [])]
                rrf_all([hits], limit=20)
        _row(f'federated search over {k} shard(s)', queries, t.wall, f'{t.wall/queries*1000:.1f} ms/query')
        rows.append(ing)
        for d in dbs: d.conn.close()
        shutil.rmtree(root, ignore_errors=True)
    base = rows[0]['wall']
    for r, k in zip(rows, shards):
        print(f'  -> {k} shard(s): {base/r["wall"]:.2f}x vs monolithic ingest')
    return rows

def bench_store(sizes=(2000, 8000, 32000), emb=None, workdir=None, ann=False):
    'Raw `process_content` upsert throughput into a hash-id store with FTS triggers live.'
    print(f'\n== process_content upsert (ann={ann}) ==')
    rows = []
    for n in sizes:
        root = Path(tempfile.mkdtemp(dir=workdir))
        db = database(str(root/'bench.db'))
        st = db.get_store('store', hash=True, ann=ann)
        content = corpus_chunks(n, chars=600)
        with Timer() as t: process_content(st, content, embed=bool(emb), emb_fn=emb)
        rows.append(_row(f'process_content({n} rows)', n, t.wall))
        db.conn.close(); shutil.rmtree(root, ignore_errors=True)
    print(f'  -> exponent {_fit_exponent(sizes, [r["wall"] for r in rows]):.2f}')
    return rows

def bench_code(dirs=('litesearch',), workdir=None, emb=None):
    'Code parsing (`dir2chunks`) and then storing the chunks.'
    print('\n== code ingestion ==')
    rows = []
    for d in dirs:
        with Timer() as t: chunks = dir2chunks(d)
        rows.append(_row(f'dir2chunks({d})', len(chunks), t.wall, f'{t.cpu:.2f}s cpu'))
        root = Path(tempfile.mkdtemp(dir=workdir))
        db = database(str(root/'bench.db'))
        st = db.get_store('store', hash=True)
        cl = [dict(content=c['content'], metadata=json.dumps(c['metadata'], default=str)) for c in chunks]
        with Timer() as t: process_content(st, cl, embed=bool(emb), emb_fn=emb)
        rows.append(_row(f'store code chunks({d})', len(cl), t.wall))
        db.conn.close(); shutil.rmtree(root, ignore_errors=True)
    return rows

def bench_pdf(pdf='nbs/pdfs/attention_is_all_you_need.pdf', workdir=None, emb=None, reps=1):
    'PDF parse + tree ingestion for one file, so parse cost is separable from store cost.'
    from litesearch.data import pdf_parse
    print('\n== pdf ingestion ==')
    if not Path(pdf).exists():
        print(f'  (skipped: {pdf} not found)'); return []
    rows, root = [], Path(tempfile.mkdtemp(dir=workdir))
    with Timer() as t: pages = pdf_parse(pdf, out_path=root/'assets')
    rows.append(_row('pdf_parse (pages)', len(pages), t.wall))
    db = database(str(root/'bench.db')); db.get_tree('store', ann=True)
    with Timer() as t: res = db.add_doc(list(enumerate(pages)), 'attention', emb_fn=emb)
    rows.append(_row('add_doc (pdf)', res.get('chunks', 0), t.wall, f'{res.get("nodes",0)} nodes'))
    db.conn.close(); shutil.rmtree(root, ignore_errors=True)
    return rows

def bench_graph(sizes=(500, 1000, 2000), workdir=None, emb=None, n_workers=0, batch=None):
    'build_graph + resolve_entities over real prose chunks. `n_workers=None` picks a pool by size.'
    print(f'\n== graph build (n_workers={n_workers}, batch={batch}) ==')
    rows = []
    for n in sizes:
        root = Path(tempfile.mkdtemp(dir=workdir))
        db = database(str(root/'bench.db')); db.get_store('store', hash=True, ann=True)
        chunks = corpus_chunks(n, chars=800)
        with Timer() as t: st = build_graph(db, chunks, emb_fn=emb or hash_emb_fn(),
                                            n_workers=n_workers, batch=batch)
        rows.append(_row(f'build_graph({n} chunks)', n, t.wall, f'{st["entities"]:,} ents {st["edges"]:,} edges'))
        with Timer() as t: rs = resolve_entities(db)
        rows.append(_row(f'resolve_entities({st["entities"]} ents)', st['entities'], t.wall, f'{rs["merged"]} merged'))
        db.conn.close(); shutil.rmtree(root, ignore_errors=True)
    walls = [r['wall'] for r in rows if r['step'].startswith('build_graph')]
    print(f'  -> build_graph exponent {_fit_exponent(sizes, walls):.2f}')
    return rows

def bench_chunk(n=2000):
    'Chunker throughput alone, and the cost of constructing a chunker per call.'
    from litesearch.data import chunk_markdown, SafeFastChunker
    print('\n== chunking ==')
    texts = [c['content'] for c in corpus_chunks(n, chars=4000)]
    nchars = sum(map(len, texts))
    with Timer() as t: [chunk_markdown(x) for x in texts]
    a = _row('chunk_markdown (chunker=None)', n, t.wall, f'{nchars/t.wall/1e6:.1f} MB/s')
    ck = SafeFastChunker(chunk_size=512)
    with Timer() as t: [chunk_markdown(x, ck) for x in texts]
    b = _row('chunk_markdown (shared chunker)', n, t.wall, f'{nchars/t.wall/1e6:.1f} MB/s')
    print(f'  -> reusing one chunker: {a["wall"]/b["wall"]:.2f}x')
    return [a, b]

def bench_shard_reads(n=1200, shards=(1,2,4,8,16), queries=60, reps=3, workdir=None, emb=None):
    '''Read cost of sharding, fanned out to every shard against routed to one.

    The two are opposite answers, which is why "does sharding cost reads" has no single one. A
    fan-out pays K searches and a fusion, and K searches over N/K rows cost more than one over N
    because HNSW is O(log N) — there is no size at which splitting the index makes the *total* work
    smaller. Routing to the shard that holds the profile pays one search over N/K rows, which is
    strictly less than one over N. So the question is never "how many shards" but "does the query
    know which shard".'''
    from concurrent.futures import ThreadPoolExecutor
    emb = emb or hash_emb_fn()
    docs = corpus_docs(n, pages_per_doc=3)
    qs = ['transport of dangerous goods by vessel', 'value added tax on imported services',
          'protection of workers from carcinogens', 'recovery and resilience facility funding',
          'inland waterway safety certificate rules']
    qvs = [np.asarray(hash_embed([q], ndim=256)).ravel().tobytes() for q in qs]
    hit = lambda db, i: db.search(qs[i%len(qs)], qvs[i%len(qs)], columns=['id','content'], limit=20, ann=True) or []
    def best(fn):
        fn(0)                                                    # warm the page cache and the index
        return min(min_run(fn, queries) for _ in range(reps))
    def min_run(fn, q):
        t = time.perf_counter()
        for i in range(q): fn(i)
        return (time.perf_counter()-t)/q*1000
    print(f'\n== read cost of sharding ({n} docs, {queries} queries x{reps}, best-of-{reps}, ms/query) ==')
    print(f"{'shards':>7} {'chunks/shard':>13} {'fanout':>9} {'fanout+threads':>15} {'routed':>8}")
    rows = []
    for k in shards:
        root = Path(tempfile.mkdtemp(dir=workdir))
        dbs, per = [], (n + k - 1) // k
        for s in range(k):
            db = database(str(root/f'shard{s}.db')); db.get_tree('store', ann=True)
            for i, pages in enumerate(docs[s*per:(s+1)*per]):
                db.add_doc(pages, title=f'doc {s*per+i}', source=f'doc{s*per+i}', emb_fn=emb, index=False)
            db.t.store.rebuild_index(); dbs.append(db)
        nper = first(dbs[0].q('select count(*) c from store'))['c']
        ex = ThreadPoolExecutor(max_workers=min(k, 8))           # persistent: a per-query pool is its own cost
        fan  = lambda i: rrf_all([hit(d, i) for d in dbs], limit=20)
        fanp = lambda i: rrf_all(list(ex.map(lambda d: hit(d, i), dbs)), limit=20)
        routed = lambda i: hit(dbs[i % k], i)                    # profile known: one shard, no fusion
        a, b, c = best(fan), best(fanp), best(routed)
        print(f'{k:>7} {nper:>13,} {a:>8.2f}ms {b:>14.2f}ms {c:>7.2f}ms')
        rows.append(dict(shards=k, chunks_per_shard=nper, fanout=a, fanout_threads=b, routed=c))
        ex.shutdown()
        for d in dbs: d.conn.close()
        shutil.rmtree(root, ignore_errors=True)
    return rows

# ---------------------------------------------------------------- equivalence checks
# Three of the ingest fixes could in principle have changed results rather than just timings. Each is
# pinned here against the implementation it replaced, so "identical" is a thing you can re-run.

def bench_lexical_recall(sizes=(250, 500, 1000, 2000), lex=0.34, max_group=60, workdir=None):
    '''What fraction of the merges `_lex_ok` accepts does the blocking actually propose?

    Ground truth is exhaustive — every unordered pair of entity names the guard accepts, O(E²).
    Blocking exists only to avoid enumerating that set, so the number that matters is how much of
    it survives, and it has to be measured against corpus size: a recall that is fine at a thousand
    entities and poor at ten thousand looks like nothing at all from inside a single run.'''
    from litesearch.graph import _lexical_pairs, _lex_ok
    print(f'\n== lexical blocking recall (max_group={max_group}) ==')
    print(f"{'chunks':>7} {'entities':>9} {'truth pairs':>12} {'recall':>8} {'pairs examined':>15}")
    out = []
    for n in sizes:
        root = Path(tempfile.mkdtemp(dir=workdir))
        db = database(str(root/'g.db')); db.get_store('store', hash=True, ann=True)
        build_graph(db, corpus_chunks(n, chars=800), emb_fn=hash_emb_fn())
        rows = [r for r in db.t.entities(select='id, content, kind') if r['kind'] not in ('symbol','module','topic')]
        name = {r['id']: r['content'] for r in rows}
        ids = sorted(name)
        truth = {(a, b) for i, a in enumerate(ids) for b in ids[i+1:] if _lex_ok(name[a], name[b], lex)}
        got = {(a, b) if a < b else (b, a) for a, b in _lexical_pairs(name, max_group)}
        rec = len(got & truth)/len(truth) if truth else float('nan')
        print(f'{n:>7} {len(name):>9,} {len(truth):>12,} {rec:>7.1%} {len(got):>15,}')
        out.append(dict(n=n, entities=len(name), truth=len(truth), recall=rec, examined=len(got)))
        db.conn.close(); shutil.rmtree(root, ignore_errors=True)
    return out

def bench_resolve_parts(n=2000, workdir=None, examples=3):
    '''Which parts of `resolve_entities` are carrying their weight.

    Every row disables exactly one thing and re-resolves *the same graph* — the database is built
    once and copied per configuration — so the delta is that part's contribution and not a
    different corpus. `merges` is the count, `only here` is how many merged pairs no other
    configuration in the table produced, and `partition` is the fingerprint of which entities ended
    up together; two rows with the same fingerprint did the same job at different prices.

    Reported because "it is in the pipeline" is not evidence that it is doing anything. Two of the
    rules turn out to be nearly free and two of them turn out to be load-bearing, and neither fact
    was visible from a timing of the whole call.'''
    import hashlib, copy
    from litesearch import graph as G
    print(f'\n== resolve_entities: what each part contributes ({n} chunks) ==')
    root = Path(tempfile.mkdtemp(dir=workdir)); src = root/'built'
    src.mkdir(parents=True)
    db = database(str(src/'g.db')); db.get_store('store', hash=True, ann=True)
    st = build_graph(db, corpus_chunks(n, chars=800), emb_fn=hash_emb_fn()); db.conn.close()

    def pairs_of(d):
        'The merge partition as a set of (member, representative) facts, plus its fingerprint.'
        grp = {}
        for x in d.t.entities(select='id, canon'): grp.setdefault(x['canon'] or x['id'], []).append(x['id'])
        part = sorted(tuple(sorted(v)) for v in grp.values() if len(v) > 1)
        merged = {(g_[i], g_[j]) for g_ in part for i in range(len(g_)) for j in range(i+1, len(g_))}
        return merged, hashlib.sha256(repr(part).encode()).hexdigest()[:10]

    # each patch turns off one rule; `full` is the shipped behaviour. The replacement is built from
    # the original rather than closing over the module attribute, which by then is the replacement.
    no_acr    = lambda orig: (lambda s: '\x00')        # an acronym that can never equal a name
    no_nums   = lambda orig: (lambda s: frozenset())   # every pair agrees on digits
    no_clique = lambda orig: (lambda par, rank, a, b, name=None, ok=None, members=None, max_check=32:
                              orig(par, rank, a, b))   # union without the clique test
    cfgs = [('full',                   dict(), None),
            ('no lexical pass',        dict(lexical=False), None),
            ('no ANN pass',            dict(thresh=-1.0), None),   # every distance fails `dist > thresh`
            ('no acronym rule',        dict(), ('_acr', no_acr)),
            ('no digit guard',         dict(), ('_nums', no_nums)),
            ('no clique guard',        dict(), ('_uf_union', no_clique))]
    rows, got = [], {}
    for label, kw, patch_ in cfgs:
        d = root/label.replace(' ', '_'); shutil.rmtree(d, ignore_errors=True); shutil.copytree(src, d)
        dbc = database(str(d/'g.db'))
        old = getattr(G, patch_[0]) if patch_ else None
        if patch_: setattr(G, patch_[0], patch_[1](old))
        try:
            with Timer() as t: r = resolve_entities(dbc, **kw)
        finally:
            if patch_: setattr(G, patch_[0], old)
        merged, h = pairs_of(dbc)
        got[label] = merged
        rows.append(dict(cfg=label, wall=t.wall, partition=h, pairs=len(merged), **r))
        dbc.conn.close(); shutil.rmtree(d, ignore_errors=True)
    base = got['full']
    print(f"  {'configuration':<18} {'wall':>7} {'merges':>8} {'pairs':>8} {'vs full':>16} {'partition':>11}")
    for row in rows:
        m = got[row['cfg']]
        delta = '' if row['cfg'] == 'full' else f'+{len(m-base):,}/-{len(base-m):,}'
        print(f"  {row['cfg']:<18} {row['wall']:6.2f}s {row['merged']:>8,} {len(m):>8,} {delta:>16} {row['partition']:>11}")
        row['gained'], row['lost'] = len(m - base), len(base - m)
    # a name for the pairs each rule is solely responsible for, so the counts can be sanity-checked
    dbc = database(str(src/'g.db'))
    nm = {x['id']: x['content'] for x in dbc.t.entities(select='id, content')}
    for row in rows:
        lost = sorted(base - got[row['cfg']])[:examples]
        if lost: print(f"    merges only '{'full' if True else ''}' keeps vs {row['cfg']}: "
                       + '; '.join(f'{nm.get(a)!r}~{nm.get(b)!r}' for a, b in lost))
    dbc.conn.close(); shutil.rmtree(root, ignore_errors=True)
    return rows


def bench_string_libs(n=1500, workdir=None):
    '''Do StringZilla and NumKong beat the stdlib on the comparisons this module actually does?

    Asked because the graph layer looks like it should be their home ground — token-set overlap,
    substring scans, sentence splitting, thousands of tiny comparisons — and because "a SIMD
    library exists for this" is not the same claim as "this workload is SIMD-bound". Both libraries
    are measured on data taken from a real build rather than on synthetic strings, and any
    disagreement with the stdlib result is reported, because a faster function that answers
    differently is not a drop-in.

    usearch is not in the table: it is already doing the vector work here, batched, in
    `_ann_pairs` and `_cluster_groups`.'''
    from litesearch.graph import _sentences, _toks, prose_windows, _yake_terms
    def have(m):
        try: return __import__(m)
        except ImportError: return None
    sz, nk = have('stringzilla'), have('numkong')
    print(f'\n== stringzilla / numkong on the graph workload ==')
    print(f'  stringzilla {getattr(sz,"__version__","(not installed)")}   numkong {getattr(nk,"__version__","(not installed)")}')
    texts = [c['content'] for c in corpus_chunks(n, chars=800)]
    out = []
    def cmp(name, base, cand, agree):
        with Timer() as t: rb = base()
        with Timer() as u: rc = cand()
        ok = agree(rb, rc)
        print(f'  {name:<38} stdlib {t.wall*1000:8.1f} ms   candidate {u.wall*1000:8.1f} ms   '
              f'{t.wall/u.wall:5.2f}x   same answer: {ok}')
        out.append(dict(op=name, stdlib=t.wall, candidate=u.wall, speedup=t.wall/u.wall, agrees=bool(ok)))

    if sz:
        cmp('sentence split (_sentences)',
            lambda: [len(_sentences(t)) for t in texts],
            lambda: [len([s for s in (str(x).strip() for x in sz.utf8_sentences(t)) if s]) for t in texts],
            lambda a, b: a == b)
        words = [w for t in texts[:300] for w in t.split() if w][:20000]
        from apsw.unicode import word_iter, casefold
        cmp('tokenise for the lexical guard',
            lambda: [frozenset(casefold(x) for x in word_iter(w) if any(c.isalnum() for c in x)) for w in words],
            lambda: [frozenset(str(x) for x in sz.utf8_wordbreaks(sz.utf8_uncased_fold(w)) if str(x).isalnum()) for w in words],
            lambda a, b: sum(x == y for x, y in zip(a, b)) / len(a))
        sents = [s for t in texts[:400] for s in _sentences(t)]
        terms = [t for t in _yake_terms(texts[0], 12)] or ['the', 'of']
        low = [t.lower() for t in terms]
        cmp('term-in-sentence scan (prose_windows)',
            lambda: [sum(1 for t in low if t in s.lower()) for s in sents],
            lambda: [sum(1 for t in low if sz.contains(s.lower(), t)) for s in sents],
            lambda a, b: a == b)
    if nk:
        names = [x['content'] for x in [dict(content=' '.join(t.split()[:3])) for t in texts]]
        sets = [np.sort(np.array(sorted(abs(hash(t)) % (1 << 31) for t in _toks(s)), dtype=np.uint32)) for s in names]
        sets = [s for s in sets if len(s)]
        fs = [frozenset(s.tolist()) for s in sets]
        import random; random.seed(0)
        prs = [(random.randrange(len(sets)), random.randrange(len(sets))) for _ in range(200_000)]
        cmp('token-set intersection (_lex_ok)',
            lambda: sum(len(fs[i] & fs[j]) for i, j in prs),
            lambda: sum(int(nk.intersect(sets[i], sets[j])) for i, j in prs),
            lambda a, b: a == b)
    return out


def bench_search_legs(sizes=(120, 600), workdir=None, reps=30):
    '''Does overlapping the FTS and vector legs of `search` on a thread pool help? No.

    `search` runs its two legs one after the other and carries a deprecated `parallel=` flag that
    does nothing. The legs look independent and both drop into C — apsw for FTS, usearch for the
    ANN probe — so a thread each is the obvious idea, and it is measured here rather than argued
    about. It loses at both corpus sizes and in both vector modes, because the legs share one apsw
    connection: SQLite serialises them behind its own mutex, so the thread pool buys contention and
    two futures instead of overlap. Recorded so the flag stays deprecated on evidence.'''
    from concurrent.futures import ThreadPoolExecutor
    from litesearch.graph import hash_embed
    print('\n== search legs: serial vs threaded ==')
    print(f"  {'chunks':>8} {'vector leg':>11} {'serial':>9} {'threaded':>9} {'change':>9}")
    out, ex = [], ThreadPoolExecutor(2)
    for n in sizes:
        root = Path(tempfile.mkdtemp(dir=workdir))
        db = database(str(root/'t.db')); db.get_tree('store', ann=True)
        emb = hash_emb_fn()
        for i, pages in enumerate(corpus_docs(n, pages_per_doc=3)):
            db.add_doc(pages, title=f'D{i}', emb_fn=emb, index=False)
        db.t.store.rebuild_index()
        nc = first(db.q('select count(*) c from store'))['c']
        qv = np.asarray(hash_embed(['transport of dangerous goods'], ndim=256)).ravel().tobytes()
        tbl = db.t.store
        for label, use_ann in (('exact', False), ('ann', True)):
            vleg = ((lambda: tbl.ann_search(qv, ['rowid','content'], 20)) if use_ann else
                    (lambda: tbl.vec_search(qv, ['rowid','content'], limit=20)))
            fleg = lambda: tbl.fts_search('dangerous OR goods OR transport', ['rowid','content'], 'rank', 20)
            ser = lambda: (fleg(), vleg())
            par = lambda: ((lambda f: (f, vleg()))(ex.submit(fleg)))[0].result()
            ts = []
            for f in (ser, par):
                f()
                with Timer() as t:
                    for _ in range(reps): f()
                ts.append(t.wall/reps)
            print(f'  {nc:>8,} {label:>11} {ts[0]*1000:8.2f}ms {ts[1]*1000:8.2f}ms {ts[0]/ts[1]:8.2f}x')
            out.append(dict(chunks=nc, leg=label, serial=ts[0], threaded=ts[1], speedup=ts[0]/ts[1]))
        db.conn.close(); shutil.rmtree(root, ignore_errors=True)
    ex.shutdown()
    return out


def bench_extractors(n=400, topk=12, workdir=None):
    '''Keyphrase extractors against the yake `build_graph` ships with, on speed *and* on agreement.

    Extraction is ~99% of `prose_windows` and the reason a build needs a process pool at all, so
    "is there a faster one, and does it release the GIL" is the question with the most headroom
    behind it. Both halves are measured: the 4-thread column is the GIL test (a pure-python
    extractor cannot beat 1.0x there, and will usually lose to contention), and the overlap column
    is the one that decides whether a swap is a speedup or a different graph.

    Install what you want compared — each is skipped if absent:

        pip install yake-rust rake-nltk rakun2
    '''
    from concurrent.futures import ThreadPoolExecutor
    from litesearch.graph import _yake_terms
    texts = [c['content'] for c in corpus_chunks(n, chars=800)]
    cands = [('yake (shipped)', _yake_terms)]
    try:
        import yake_rust
        yr = yake_rust.Yake(language='en', ngrams=3)
        cands.append(('yake-rust', lambda t, k=topk: [w for w, _ in yr.get_n_best(t, n=k)]))
    except Exception: pass
    try:
        from rake_nltk import Rake
        import nltk
        for pkg in ('stopwords', 'punkt', 'punkt_tab'):
            try: nltk.download(pkg, quiet=True)
            except Exception: pass
        def _rk(t, k=topk):
            r = Rake(max_length=3); r.extract_keywords_from_text(t)
            return r.get_ranked_phrases()[:k]
        cands.append(('rake-nltk', _rk))
    except Exception: pass
    try:
        from rakun2 import RakunKeyphraseDetector
        cfg = dict(num_keywords=topk, merge_threshold=1.1, alpha=0.3, token_prune_len=3)
        cands.append(('rakun2', lambda t, k=topk: [w for w, _ in
                      RakunKeyphraseDetector(cfg, verbose=False).find_keywords(t, input_type='string')]))
    except Exception: pass

    print(f'\n== keyphrase extractors ({n} chunks, {defaults.cpus} cores) ==')
    print(f"  {'extractor':<18}{'serial':>9}{'per chunk':>11}{'4 threads':>11}{'GIL-free?':>11}{'agrees w/ yake':>16}")
    base, out = None, []
    for label, fn in cands:
        try: fn(texts[0])
        except Exception as e: print(f'  {label:<18} unavailable: {type(e).__name__}'); continue
        with Timer() as t: res = [fn(x) for x in texts]
        with ThreadPoolExecutor(4) as ex:
            with Timer() as u: list(ex.map(fn, texts))
        sets = [{str(w).lower() for w in r} for r in res]
        if base is None: base, ov = sets, '—'
        else:
            den = sum(len(a) for a in base)
            ov = f'{sum(len(a & b) for a, b in zip(base, sets))/den:.1%}' if den else 'n/a'
        gil = 'yes' if t.wall/u.wall > 1.2 else 'no'
        print(f'  {label:<18}{t.wall:8.2f}s{t.wall/len(texts)*1000:10.1f}ms{u.wall:10.2f}s'
              f'{gil:>11}{ov:>16}')
        out.append(dict(extractor=label, serial=t.wall, threaded=u.wall, per_chunk=t.wall/len(texts),
                        gil_free=gil == 'yes', overlap=ov))
    print('  (overlap is share of the shipped extractor\'s terms the candidate also returns; a low\n'
          '   number means a different graph, not a faster one)')
    return out


def bench_embed_batch(n=200, sizes=(20, 64, 256, 2000, 0), workdir=None):
    '''How much the encoder gains from being handed the whole directory instead of one document.

    `add_dir` used to call the encoder once per document — about 20 chunks — and `embed_batch` now
    widens that across documents. How much that is worth depends entirely on the encoder, which is
    why it is a parameter and not a constant: a static model is a lookup table with nothing to
    amortise and flattens by ~64 chunks a call, while an ONNX model has real per-call setup. Run
    this with the encoder you actually use before choosing a number. `0` means one call.'''
    from litesearch.utils import static_retrieval_embedder, doc_encoder
    enc = doc_encoder(static_retrieval_embedder())
    texts = []
    for pages in corpus_docs(n, pages_per_doc=3):
        for _, t in pages: texts += [t[i:i+800] for i in range(0, len(t), 800)]
    texts = [x for x in texts if x.strip()]
    enc(texts[:8])
    print(f'\n== static encoder, {len(texts):,} chunks, avg {sum(map(len, texts))//len(texts)} chars ==')
    out = []
    for bs in sizes:
        b = [texts] if not bs else [texts[i:i+bs] for i in range(0, len(texts), bs)]
        b = [x for x in b if x]
        best = min((_t.wall for _t in (_time_calls(enc, b) for _ in range(3))), default=0.0)
        lbl = 'one call, everything' if not bs else f'{bs} per call'
        print(f'  {lbl:<26} {best*1000:8.1f} ms  ({len(b)} calls)')
        out.append(dict(batch=bs, wall=best, calls=len(b)))
    return out

def _time_calls(fn, batches):
    with Timer() as t:
        for b in batches: fn(b)
    return t


def bench_graph_memory(sizes=(2000, 4000, 8000), batch=500, workdir=None):
    '''Peak RSS of `build_graph`, listed input against a generator with `batch` set.

    Reported as marginal KB/chunk between sizes rather than as a total, because the total is
    dominated by the interpreter and the question is only whether the slope flattens.'''
    import resource
    print(f'\n== build_graph memory (batch={batch}) ==')
    print(f"{'chunks':>7} {'mode':>16} {'wall':>8} {'rss delta':>11} {'KB/chunk':>9}")
    out = []
    for n in sizes:
        for label, bt, gen in (('list, no batch', None, False), ('generator, batched', batch, True)):
            root = Path(tempfile.mkdtemp(dir=workdir))
            db = database(str(root/'g.db')); db.get_store('store', hash=True, ann=True)
            src = (c for c in corpus_chunks(n, chars=800)) if gen else corpus_chunks(n, chars=800)
            base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
            with Timer() as t: build_graph(db, src, emb_fn=hash_emb_fn(), batch=bt)
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
            print(f'{n:>7} {label:>16} {t.wall:>7.2f}s {peak-base:>10.1f}MB {(peak-base)*1024/n:>8.1f}')
            out.append(dict(n=n, mode=label, wall=t.wall, rss=peak-base))
            db.conn.close(); shutil.rmtree(root, ignore_errors=True)
    print('  note: RSS is a high-water mark within one process, so run this as its own process per '
          'size for a clean read — `python -m evals.ingest_bench gmem --sizes N`.')
    return out

def _pyparse_pre_0116(p=None, code=None, imports=False, assigns=False):
    'The `pyparse` that shipped up to 0.1.15, verbatim, as the reference for the rewrite.'
    import ast
    from ast import get_source_segment as gs
    from fastcore.all import L, type2str
    try: code = Path(p).read_text(encoding='utf-8') if not code else code
    except UnicodeDecodeError: return L()
    tree = ast.parse(code)
    [setattr(c,'parent',n) for n in ast.walk(tree) for c in ast.iter_child_nodes(n)]
    def meta(xtra=None): return dict(path=str(p), uploaded_at=Path(p).stat().st_mtime if p else None, **(xtra or {}))
    def n2c(n):
        return dict(content=gs(code, n).strip(), metadata=meta(dict(name=getattr(n,'name',None), lang='.py',
            type=type2str(n.__class__), lineno=getattr(n,'lineno',None), end_lineno=getattr(n,'end_lineno',None))))
    def is_mod(n): return isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    def is_assign(n): return assigns and isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value
    def is_p_mod(n): return getattr(getattr(n,'parent',None),'__class__',None) == ast.Module
    def ok(n): return is_p_mod(n) and (is_mod(n) or is_assign(n) or (imports and isinstance(n, ast.ImportFrom)))
    return L(ast.walk(tree)).filter(ok).map(n2c)

def verify_pyparse(dir='litesearch', flags=(dict(), dict(imports=True, assigns=True))):
    'Chunk-for-chunk comparison of `pyparse` against the pre-0.1.16 implementation, with timings.'
    from litesearch.data import dir2files, pyparse
    files = [f for f in dir2files(dir) if f.suffix == '.py']
    print(f'\n== pyparse equivalence over {len(files)} files in {dir} ==')
    out = []
    for kw in flags:
        with Timer() as t: a = [_pyparse_pre_0116(f, **kw) for f in files]
        with Timer() as u: b = [pyparse(f, **kw) for f in files]
        na, nb = sum(map(len, a)), sum(map(len, b))
        mism = sum(1 for x, y in zip(a, b) for cx, cy in zip(x, y) if cx != cy)
        print(f'  {str(kw):<34} old {t.wall:7.2f}s  new {u.wall:5.2f}s  {t.wall/u.wall:5.1f}x   '
              f'chunks {na} vs {nb}   mismatches {mism}')
        out.append(dict(flags=kw, old=t.wall, new=u.wall, chunks_old=na, chunks_new=nb, mismatches=mism))
    return out

def _partition_hash(db):
    '''Fingerprint of *which entities were merged together*, independent of which one won the name.

    Not the id->canon map: `_uf_union` breaks rank ties by the order pairs arrive, and usearch's
    HNSW probe is both approximate and threaded, so the representative of a group varies run to run
    — before this change as much as after. The partition is the thing that has to be stable, and it
    is what a caller actually depends on.'''
    import hashlib
    groups = {}
    for x in db.t.entities(select='id, canon'): groups.setdefault(x['canon'] or x['id'], []).append(x['id'])
    part = sorted(tuple(sorted(v)) for v in groups.values())
    return hashlib.sha256(repr(part).encode()).hexdigest()[:12]

def verify_ann_probe(n=2000, k=8, reps=3, workdir=None):
    '''The batched probe against the per-entity loop it replaced, over one *fixed* index.

    This is the check that actually settles it. `verify_resolve` rebuilds the index per run and so
    cannot separate "the batching changed the candidates" from "usearch built a different graph":
    HNSW construction is multithreaded and not reproducible, which moves `resolve_entities` by
    ~0.1% between full rebuilds — before this change as much as after. Hold the index still and the
    two probes have to agree exactly, and they do.'''
    from litesearch.core import _rid
    from litesearch.graph import _ann_pairs
    root = Path(tempfile.mkdtemp(dir=workdir))
    db = database(str(root/'g.db')); db.get_store('store', hash=True, ann=True)
    st = build_graph(db, corpus_chunks(n, chars=800), emb_fn=hash_emb_fn())
    ents = L(db.t.entities(select=f'{_rid()}, id, content, freq, embedding, kind'))
    looped = lambda: {(r['id'], h['id']) for r in ents if r['embedding']
                      for h in db.t.entities.ann_search(r['embedding'], columns=['id','content'], limit=k)}
    batched = lambda: {(r['id'], h['id']) for r, h, d in _ann_pairs(db.t.entities, ents, k)}
    ls, bs = [looped() for _ in range(reps)], [batched() for _ in range(reps)]
    print(f'\n== ann probe equivalence ({st["entities"]} entities, fixed index, k={k}) ==')
    print(f'  looped  x{reps}: {len(ls[0])} pairs, stable: {all(x == ls[0] for x in ls)}')
    print(f'  batched x{reps}: {len(bs[0])} pairs, stable: {all(x == bs[0] for x in bs)}')
    print(f'  identical: {ls[0] == bs[0]}   symmetric difference: {len(ls[0] ^ bs[0])}')
    out = dict(entities=st['entities'], pairs=len(ls[0]), identical=ls[0] == bs[0],
               symmetric_difference=len(ls[0] ^ bs[0]))
    db.conn.close(); shutil.rmtree(root, ignore_errors=True)
    return out

def verify_resolve(sizes=(500, 1000, 2000), workdir=None, reps=1):
    '''Merge outcome of `resolve_entities`, as counts plus a partition fingerprint.

    The batched HNSW probe changed how candidates are fetched, not which ones, so both have to come
    back what the per-entity loop produced. Pass `reps>1` to check stability run to run.'''
    print('\n== resolve_entities outcome ==')
    out = []
    for n in sizes:
        for rep in range(reps):
            root = Path(tempfile.mkdtemp(dir=workdir))
            db = database(str(root/'g.db')); db.get_store('store', hash=True, ann=True)
            st = build_graph(db, corpus_chunks(n, chars=800), emb_fn=hash_emb_fn())
            with Timer() as t: r = resolve_entities(db)
            h = _partition_hash(db)
            print(f'  n={n:5d} ents={st["entities"]:5d}  {t.wall:6.2f}s  merged={r["merged"]} '
                  f'ann={r["by_ann"]} lex={r["by_lexical"]} canonical={r["canonical"]}  partition={h}')
            out.append(dict(n=n, rep=rep, wall=t.wall, partition=h, **r))   # r already carries `entities`
            db.conn.close(); shutil.rmtree(root, ignore_errors=True)
    return out

def verify_bulk_load(n=8000, workdir=None):
    'Same rows, same FTS index, same hits — with the triggers live and with `bulk_load`.'
    print('\n== bulk_load equivalence ==')
    out = []
    for defer in (False, True):
        root = Path(tempfile.mkdtemp(dir=workdir))
        db = database(str(root/'b.db')); st = db.get_store('store', hash=True)
        rows = corpus_chunks(n, chars=600)
        with Timer() as t:
            if defer:
                with st.bulk_load(): process_content(st, rows, embed=False)
            else: process_content(st, rows, embed=False)
        r = dict(bulk_load=defer, wall=t.wall,
                 rows=first(db.q('select count(*) c from store'))['c'],
                 fts_rows=first(db.q('select count(*) c from store_fts'))['c'],
                 hits=len(st.fts_search('vessel', columns=['id'], limit=5000)))
        print(f'  bulk_load={str(defer):<5} {r["wall"]:5.2f}s  rows={r["rows"]}  fts_rows={r["fts_rows"]}  '
              f'hits(vessel)={r["hits"]}')
        out.append(r)
        db.conn.close(); shutil.rmtree(root, ignore_errors=True)
    return out

def profile(fn, top=25, sort='tottime'):
    'cProfile a callable and print the hottest frames.'
    pr = cProfile.Profile(); pr.enable(); fn(); pr.disable()
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats(sort).print_stats(top)
    print(s.getvalue())

# ---------------------------------------------------------------- cli

def _sizes(s): return tuple(int(x) for x in s.split(','))

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('bench', choices=['docs','defer','shards','reads','store','code','pdf','graph','chunk',
                                     'lexrecall','parts','libs','legs','kw','embatch','gmem','verify','all'])
    p.add_argument('--sizes', type=_sizes, default=None)
    p.add_argument('--dir', default='litesearch')
    p.add_argument('--encoder', choices=['hash','fast','none'], default='hash')
    p.add_argument('--no-ann', action='store_true')
    p.add_argument('--workers', type=int, default=0, help='graph extraction workers; 0 serial')
    p.add_argument('--batch', type=int, default=None, help='build_graph chunks per flush')
    p.add_argument('--workdir', default=os.environ.get('LITESEARCH_BENCH_DIR'))
    p.add_argument('--profile', action='store_true')
    p.add_argument('--out', default=None, help='write results as json')
    a = p.parse_args(argv)
    emb = dict(hash=hash_emb_fn, fast=real_emb_fn, none=lambda: None)[a.encoder]()
    ann = not a.no_ann
    rows = []
    def run(fn, **kw): rows.extend(fn(**kw) or [])
    if a.bench in ('docs','all'):  run(bench_docs, sizes=a.sizes or (25,50,100), emb=emb, ann=ann, workdir=a.workdir)
    if a.bench in ('defer','all'): run(bench_deferred_index, sizes=a.sizes or (50,100,200), emb=emb, workdir=a.workdir)
    if a.bench in ('shards','all'): run(bench_shards, n=(a.sizes or (400,))[0], emb=emb, workdir=a.workdir)
    if a.bench in ('reads','all'):  run(bench_shard_reads, n=(a.sizes or (1200,))[0], emb=emb, workdir=a.workdir)
    if a.bench == 'verify':
        verify_bulk_load(workdir=a.workdir); verify_resolve(workdir=a.workdir); verify_pyparse(a.dir)
    if a.bench in ('store','all'): run(bench_store, sizes=a.sizes or (2000,8000,32000), emb=emb, workdir=a.workdir, ann=False)
    if a.bench in ('code','all'):  run(bench_code, dirs=(a.dir,), workdir=a.workdir, emb=emb)
    if a.bench in ('chunk','all'): run(bench_chunk, n=(a.sizes or (2000,))[0])
    if a.bench in ('graph','all'): run(bench_graph, sizes=a.sizes or (500,1000,2000), workdir=a.workdir,
                                      n_workers=a.workers, batch=a.batch)
    if a.bench in ('lexrecall','all'): run(bench_lexical_recall, sizes=a.sizes or (250,500,1000,2000), workdir=a.workdir)
    if a.bench in ('parts','all'): run(bench_resolve_parts, n=(a.sizes or (2000,))[0], workdir=a.workdir)
    if a.bench in ('libs','all'):  run(bench_string_libs, n=(a.sizes or (1500,))[0], workdir=a.workdir)
    if a.bench in ('legs','all'):  run(bench_search_legs, sizes=a.sizes or (120,600), workdir=a.workdir)
    if a.bench in ('kw','all'):    run(bench_extractors, n=(a.sizes or (400,))[0], workdir=a.workdir)
    if a.bench in ('embatch','all'): run(bench_embed_batch, n=(a.sizes or (200,))[0], workdir=a.workdir)
    if a.bench == 'gmem': bench_graph_memory(sizes=a.sizes or (2000,4000,8000), workdir=a.workdir)
    if a.bench in ('pdf','all'):   run(bench_pdf, workdir=a.workdir, emb=emb)
    if a.out: Path(a.out).write_text(json.dumps(rows, indent=1))
    return rows

if __name__ == '__main__': main()
