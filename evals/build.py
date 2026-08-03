"""Build one litesearch store per configuration, on disk, resumably.

A configuration is `(genre, chunking, encoder, mode)`. `mode` is `flat`, `tree`, `tree-nohead`,
`late` or `fulldoc`. Every build records what it cost — chunking, embedding, indexing, bytes on
disk — because a tier that ignores cost is not a tier, it is a leaderboard.

On chunk granularity: `page` is not an arbitrary choice, it is what litesearch does by default.
`chunk_markdown` uses `FastChunker(chunk_size=4096)` and `add_doc` feeds it one node segment at a
time, so on a corpus whose pages run 1.7–3.5k characters the default chunker almost never splits
anything. Everything finer has to be asked for.
"""
import json, time
import numpy as np
from pathlib import Path

from chonkie import RecursiveChunker, FastChunker
from litesearch import database
from litesearch.data import chunk_markdown

from . import corpus as C
from .encoders import enc, DT

DBS   = Path(__file__).parent/'dbs'
STATS = Path(__file__).parent/'cache/build_stats.json'

# Character-based recursive chunking: model-agnostic, and the number in the name is the number a
# practitioner would actually set.
CHUNKERS = {
    'page':  None,                                                   # one chunk per page
    'c1024': lambda: RecursiveChunker(chunk_size=1024, tokenizer='character'),
    'c512':  lambda: RecursiveChunker(chunk_size=512,  tokenizer='character'),
    'c256':  lambda: RecursiveChunker(chunk_size=256,  tokenizer='character'),
    'xl4096': lambda: FastChunker(chunk_size=4096),                   # the library default chunker
}


def slug(genre, chunking, encoder, mode): return f'{genre}__{mode}__{chunking}__{encoder}'


def db_path(genre, chunking, encoder, mode):
    DBS.mkdir(parents=True, exist_ok=True)
    return DBS/f'{slug(genre, chunking, encoder, mode)}.db'


def _clean(path):
    import glob
    for f in [str(path)] + glob.glob(f'{path}*'):
        if Path(f).exists() and Path(f).is_file(): Path(f).unlink()


# --------------------------------------------------------------- chunking
def flat_chunks(genre, chunking):
    'Chunk every page of every document. Chunks never straddle a page — neither do litesearch\'s.'
    ck = CHUNKERS[chunking]() if CHUNKERS[chunking] else None
    out = []
    for title, pages in C.load(genre).items():
        for pg, txt in pages:
            if not (txt or '').strip(): continue
            parts = [txt] if ck is None else [c for c in chunk_markdown(txt, ck) if c.strip()]
            for c in parts: out.append(dict(content=c, doc_id=title, page=pg))
    return out


def doc_spans(genre, chunking):
    '''Whole-document text plus `(start, end)` chunk spans, for late chunking.

    Pages are joined rather than encoded separately on purpose: the entire claim of late chunking is
    that a chunk vector should see the document around it, so the document is what gets tokenised.'''
    from litesearch.data import chunk_spans
    ck = CHUNKERS[chunking]() if CHUNKERS[chunking] else None
    out = {}
    for title, pages in C.load(genre).items():
        offs, buf, n = [], [], 0
        for pg, txt in pages:
            t = (txt or '')
            offs.append((n, n+len(t), pg)); buf.append(t); n += len(t) + 2
        full = '\n\n'.join(buf)
        spans = ([(s, e, full[s:e]) for s, e, _ in offs] if ck is None
                 else [(s, e, t) for s, e, t in chunk_spans(full, ck) if t.strip()])
        def page_of(s):
            for a, b, pg in offs:
                if a <= s < b: return pg
            return offs[-1][2] if offs else 0
        out[title] = dict(text=full, spans=[(s, e, t, page_of(s)) for s, e, t in spans])
    return out


# --------------------------------------------------------------- encoding
def encode_sorted(e, texts):
    '''Encode length-sorted, then restore the original order.

    `FastEncode` pads each batch to its longest member, so a batch holding one 2000-character chunk
    and thirty-one 200-character ones costs as much as thirty-two long ones. Sorting first is worth
    a measured 1.2x on this corpus and changes no vector — it is free throughput that every caller
    of `encode_document` is currently leaving on the table.'''
    txts = list(texts)
    if not txts: return np.zeros((0, e.dim), dtype=DT)
    order = sorted(range(len(txts)), key=lambda i: len(txts[i]))
    v = e.doc([txts[i] for i in order])
    out = np.empty_like(v)
    out[np.asarray(order)] = v
    return out


# --------------------------------------------------------------- builders
def build_flat(genre, chunking, encoder, force=False):
    'Chunk, embed independently, insert, index. The ordinary pipeline.'
    p = db_path(genre, chunking, encoder, 'flat')
    if p.exists() and not force: return dict(path=str(p), skipped=True)
    _clean(p)
    e, t0 = enc(encoder), time.time()
    chunks = flat_chunks(genre, chunking)
    t_chunk = time.time()-t0
    t0 = time.time()
    vecs = encode_sorted(e, [c['content'] for c in chunks])
    t_embed = time.time()-t0
    t0 = time.time()
    db = database(str(p))
    st = db.get_store('store', hash=True, ann=True, ndim=e.dim, dtype=DT, doc_id=str, page=int)
    st.insert_all([dict(content=c['content'], embedding=v.tobytes(), doc_id=c['doc_id'], page=c['page'])
                   for c, v in zip(chunks, vecs)], upsert=True, hash_id='id', hash_id_columns=['content'])
    t_insert = time.time()-t0
    t0 = time.time(); n_idx = st.rebuild_index(); t_index = time.time()-t0
    return _record(genre, chunking, encoder, 'flat', p, len(chunks), n_idx,
                   t_chunk, t_embed, t_insert, t_index, chunks)


def build_tree(genre, chunking, encoder, with_heading=True, bold=False, force=False):
    'The `litesearch.tree` path: a node tree per document, chunks linked to nodes.'
    mode = 'tree-bold' if bold else ('tree' if with_heading else 'tree-nohead')
    p = db_path(genre, chunking, encoder, mode)
    if p.exists() and not force: return dict(path=str(p), skipped=True)
    _clean(p)
    e = enc(encoder)
    ck = CHUNKERS[chunking]() if CHUNKERS[chunking] else None
    db = database(str(p))
    db.get_tree('store', ndim=e.dim, dtype=DT)
    emb = lambda ts, **kw: encode_sorted(e, ts)
    docs = C.load(genre)
    if bold:
        from .boldhead import promote_docs
        docs, _ = promote_docs(docs)
    t0 = time.time()
    for title, pages in docs.items():
        db.add_doc(pages, title=title, source=title, kind='doc', emb_fn=emb,
                   chunker=ck, with_heading=with_heading)
    t_all = time.time()-t0
    rows = list(db.t.store(select='content'))
    n_idx = db.get_index('store').size
    nnodes = len(list(db.t.nodes(select='id')))
    return _record(genre, chunking, encoder, mode, p, len(rows), n_idx,
                   0.0, t_all, 0.0, 0.0, rows, nodes=nnodes)


def build_late(genre, chunking, encoder, fast=True, force=False):
    '''Late chunking: embed the document, then pool token vectors per chunk span.

    `fast=True` uses `evals.fastlate`, which computes the same vectors with binary search instead of
    a per-span scan of the window's token offsets. The library's own path is O(spans × tokens) per
    window, which on these documents is minutes per book of pure Python — see `evals.fastlate.check`
    for the equivalence assertion and the measured ratio.'''
    p = db_path(genre, chunking, encoder, 'late')
    if p.exists() and not force: return dict(path=str(p), skipped=True)
    _clean(p)
    e = enc(encoder)
    assert e.late is not None, f'{encoder} has no context window to late-chunk over'
    t0 = time.time(); docs = doc_spans(genre, chunking); t_chunk = time.time()-t0
    rows, tiers, t_embed = [], {}, 0.0
    pool = (lambda txt, sp: __import__('evals.fastlate', fromlist=['x']).encode_auto_fast(e.late, txt, sp)) \
           if fast else (lambda txt, sp: e.late.encode_auto(txt, sp))
    for title, d in docs.items():
        t0 = time.time()
        vs, tier = pool(d['text'], [(s, en) for s, en, _, _ in d['spans']])
        t_embed += time.time()-t0
        tiers[tier] = tiers.get(tier, 0) + 1
        for (s, en, txt, pg), v in zip(d['spans'], vs):
            rows.append(dict(content=txt, embedding=np.asarray(v, dtype=DT).tobytes(),
                             doc_id=title, page=pg))
        print(f'    late {title[:34]:<34} {len(d["spans"]):>5} spans  tier={tier}  '
              f'{time.time()-t0:>6.1f}s', flush=True)
    t0 = time.time()
    db = database(str(p))
    st = db.get_store('store', hash=True, ann=True, ndim=e.dim, dtype=DT, doc_id=str, page=int)
    st.insert_all(rows, upsert=True, hash_id='id', hash_id_columns=['content'])
    t_insert = time.time()-t0
    t0 = time.time(); n_idx = st.rebuild_index(); t_index = time.time()-t0
    return _record(genre, chunking, encoder, 'late', p, len(rows), n_idx,
                   t_chunk, t_embed, t_insert, t_index, rows, tiers=tiers, fast_pool=fast)


def build_fulldoc(genre, encoder, force=False):
    'One vector per document. The control that shows what chunking is worth.'
    p = db_path(genre, 'doc', encoder, 'fulldoc')
    if p.exists() and not force: return dict(path=str(p), skipped=True)
    _clean(p)
    e = enc(encoder)
    docs = {t: '\n\n'.join(x or '' for _, x in pages) for t, pages in C.load(genre).items()}
    t0 = time.time(); vecs = encode_sorted(e, list(docs.values())); t_embed = time.time()-t0
    db = database(str(p))
    st = db.get_store('store', hash=True, ann=True, ndim=e.dim, dtype=DT, doc_id=str, page=int)
    rows = [dict(content=t, embedding=v.tobytes(), doc_id=k, page=0)
            for (k, t), v in zip(docs.items(), vecs)]
    st.insert_all(rows, upsert=True, hash_id='id', hash_id_columns=['content'])
    n_idx = st.rebuild_index()
    return _record(genre, 'doc', encoder, 'fulldoc', p, len(rows), n_idx, 0.0, t_embed, 0.0, 0.0, rows)


# --------------------------------------------------------------- bookkeeping
def _record(genre, chunking, encoder, mode, p, n_chunks, n_idx,
            t_chunk, t_embed, t_insert, t_index, rows, **extra):
    import glob
    lens = [len(r['content'] or '') for r in rows]
    size = sum(Path(f).stat().st_size for f in glob.glob(f'{p}*') if Path(f).is_file())
    rec = dict(genre=genre, chunking=chunking, encoder=encoder, mode=mode, path=str(p),
               chunks=n_chunks, indexed=n_idx, mean_chars=round(sum(lens)/max(len(lens), 1)),
               max_chars=max(lens or [0]), t_chunk=round(t_chunk, 2), t_embed=round(t_embed, 2),
               t_insert=round(t_insert, 2), t_index=round(t_index, 2),
               bytes=size, **extra)
    all_ = json.loads(STATS.read_text()) if STATS.exists() else {}
    all_[slug(genre, chunking, encoder, mode)] = rec
    STATS.parent.mkdir(parents=True, exist_ok=True)
    STATS.write_text(json.dumps(all_, indent=1))
    return rec


def show(rec):
    if rec.get('skipped'): return f"  skip {Path(rec['path']).stem}"
    return (f"  {slug(rec['genre'], rec['chunking'], rec['encoder'], rec['mode']):<46} "
            f"{rec['chunks']:>6} chunks (mean {rec['mean_chars']:>5}ch) "
            f"embed {rec['t_embed']:>7.1f}s  idx {rec['t_index']:>5.1f}s  "
            f"{rec['bytes']/1e6:>6.1f}MB")
