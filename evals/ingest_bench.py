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

from fastcore.all import first

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

def bench_graph(sizes=(500, 1000, 2000), workdir=None, spacy=False, emb=None):
    'build_graph + resolve_entities over synthetic prose chunks.'
    print(f'\n== graph build (spacy={spacy}) ==')
    nlp = None
    if spacy:
        from litesearch.graph import spacy_pipe
        nlp = spacy_pipe()
        if nlp is None: print('  (spaCy unavailable; using the yake fallback)')
    rows = []
    for n in sizes:
        root = Path(tempfile.mkdtemp(dir=workdir))
        db = database(str(root/'bench.db')); db.get_store('store', hash=True, ann=True)
        chunks = corpus_chunks(n, chars=800)
        with Timer() as t: st = build_graph(db, chunks, emb_fn=emb or hash_emb_fn(), nlp=nlp)
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

def profile(fn, top=25):
    'cProfile a callable and print the hottest cumulative frames.'
    pr = cProfile.Profile(); pr.enable(); fn(); pr.disable()
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(top)
    print(s.getvalue())

# ---------------------------------------------------------------- cli

def _sizes(s): return tuple(int(x) for x in s.split(','))

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('bench', choices=['docs','defer','shards','store','code','pdf','graph','chunk','all'])
    p.add_argument('--sizes', type=_sizes, default=None)
    p.add_argument('--dir', default='litesearch')
    p.add_argument('--encoder', choices=['hash','fast','none'], default='hash')
    p.add_argument('--no-ann', action='store_true')
    p.add_argument('--spacy', action='store_true')
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
    if a.bench in ('store','all'): run(bench_store, sizes=a.sizes or (2000,8000,32000), emb=emb, workdir=a.workdir, ann=False)
    if a.bench in ('code','all'):  run(bench_code, dirs=(a.dir,), workdir=a.workdir, emb=emb)
    if a.bench in ('chunk','all'): run(bench_chunk, n=(a.sizes or (2000,))[0])
    if a.bench in ('graph','all'): run(bench_graph, sizes=a.sizes or (500,1000,2000), workdir=a.workdir, spacy=a.spacy)
    if a.bench in ('pdf','all'):   run(bench_pdf, workdir=a.workdir, emb=emb)
    if a.out: Path(a.out).write_text(json.dumps(rows, indent=1))
    return rows

if __name__ == '__main__': main()
