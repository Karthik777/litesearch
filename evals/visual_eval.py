"""Is a multimodal page retriever a viable replacement for parse-then-embed?

The two routes answer the same question over the same pages and are scored on the same target:

- **parse** — `pdf_parse` (pdflite/liteparse) turns the PDF into one markdown page per page, the
  page is chunked and embedded by a small text encoder, and `litesearch` answers with FTS, vectors
  or the hybrid of both.
- **visual** — the page is rendered to an image and embedded by NeoMME. No OCR, no chunker, no
  tokenizer chain. Retrieval is an exhaustive MaxSim scan over page vectors.

Both return pages, so the metric is the page: a query is answered when the page holding its source
sentence appears in the top k. Chunk-level hits are folded to their page first, which is generous
to the parse route (three chunks of one page cost it one slot, not three).

The visual route is scored with an exhaustive scan and no lexical leg. That is deliberately
generous on quality and honest on cost: the scan time is measured and reported.

    python -m evals.visual_eval prepare      # parse, queries, page images
    python -m evals.visual_eval parse        # the parse route
    python -m evals.visual_eval visual       # the NeoMME variants
    python -m evals.visual_eval report
"""
import json, sys, time
from pathlib import Path
import numpy as np

from . import corpus as C
from . import queries as Q

HERE    = Path(__file__).parent
CACHE   = HERE/'cache'
RESULTS = HERE/'results'
GENRE   = 'regsub'
K       = 10

# Five of the nine regulatory documents, 112 pages. Whole documents rather than a page sample, so
# the distractors a query has to beat are the ones it would face in the real store.
SUBSET = ('dir_1993_13_2022-05-28_eng', 'dir_1993_83_2019-06-06_eng', 'dir_1996_9_2019-06-06_eng',
          'reg_2021_694_2023-09-21_eng', 'reg_2021_523_2024-03-01_eng')

NEOMME = {
    'neomme-260m-late':  ('Hcompany/NeoMME-260M-Retriever-ST-late',  'late'),
    'neomme-260m-dense': ('Hcompany/NeoMME-260M-Retriever-ST-dense', 'dense'),
    'neomme-800m-late':  ('Hcompany/NeoMME-800M-Retriever-ST-late',  'late'),
    'neomme-800m-dense': ('Hcompany/NeoMME-800M-Retriever-ST-dense', 'dense'),
}
DPI = 150            # 1241x1754 for A4; 2202 image patches at patch_size 32


# ------------------------------------------------------------------ corpus
def pdfs(): return [HERE.parent/'examples/pdfs'/f'{n}.pdf' for n in SUBSET]

def parse_pages(refresh=False):
    'Parse the subset with `pdf_parse` and cache it as the `regsub` genre. Returns seconds parsing.'
    f = CACHE/f'pages_{GENRE}.json'
    if f.exists() and not refresh: return json.loads((CACHE/'parse_cost.json').read_text())['s']
    from litesearch.data import pdf_parse
    CACHE.mkdir(parents=True, exist_ok=True)
    docs, t0 = {}, time.time()
    for p in pdfs():
        t = time.time(); pages = list(enumerate(pdf_parse(str(p))))
        print(f'  parsed {p.stem:<32} {len(pages):>4} pages  {time.time()-t:>6.1f}s', flush=True)
        docs[p.stem] = pages
    el = time.time()-t0
    f.write_text(json.dumps(docs))
    (CACHE/'parse_cost.json').write_text(json.dumps(dict(s=el, pages=sum(len(v) for v in docs.values()))))
    return el


def register():
    'Wire the subset in as a genre. Synonyms are cached separately so the real ones stay untouched.'
    C.GENRE_UNIT[GENRE] = C.GENRE_UNIT['regulatory']
    Q.SAMPLING[GENRE] = Q.SAMPLING['regulatory']
    f = CACHE/f'synonyms_{GENRE}.json'
    def syn():
        if f.exists(): return json.loads(f.read_text())
        from .refindex import ref
        words = sorted({w for w, n in ref(GENRE).df.items()
                        if n >= 3 and w not in Q._STOP and len(w) > 3})
        return Q.build_synonyms(words, out=f)
    Q.synonyms = syn


def queries(n=150, refresh=False):
    'The query set over the subset, five flavours, paired.'
    register()
    return Q.build(GENRE, n=n, refresh=refresh)


def page_index():
    '`[(doc, page)]` in a fixed order, and the text of each page.'
    docs = C.load(GENRE)
    keys = [(t, pg) for t in SUBSET for pg, _ in docs[t]]
    text = {(t, pg): txt for t in SUBSET for pg, txt in docs[t]}
    return keys, text


# ------------------------------------------------------------------ scoring
def rank_of(pages, target, k=K):
    'Index of `target` in a deduped page list, or None.'
    seen, i = set(), 0
    for p in pages:
        if p in seen: continue
        seen.add(p)
        if p == target: return i if i < k else None
        i += 1
        if i >= k: return None
    return None


def agg(ranks, k=K):
    n = max(1, len(ranks))
    return dict(mrr=sum(1/(r+1) for r in ranks if r is not None)/n,
                hit1=sum(1 for r in ranks if r == 0)/n,
                hit5=sum(1 for r in ranks if r is not None and r < 5)/n,
                hit10=sum(1 for r in ranks if r is not None)/n)

def rr(ranks): return np.array([0.0 if r is None else 1.0/(r+1) for r in ranks])


def boot(a, b, n=10_000, seed=20260803):
    'Paired bootstrap of the mean difference `a - b`, resampling queries.'
    d, rng = a - b, np.random.default_rng(20260803 if seed is None else seed)
    m = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def save(phase, rows):
    RESULTS.mkdir(parents=True, exist_ok=True)
    f = RESULTS/f'{phase}.json'
    old = json.loads(f.read_text()) if f.exists() else []
    key = lambda r: (r.get('route'), r.get('encoder'), r.get('chunking'), r.get('strategy'),
                     r.get('modality'), r.get('dpi'), r.get('flavour'))
    seen = {key(r) for r in rows}
    f.write_text(json.dumps([r for r in old if key(r) not in seen] + rows, indent=1))
    print(f'  -> {f} ({len(rows)} new rows)')


def load(phase):
    f = RESULTS/f'{phase}.json'
    return json.loads(f.read_text()) if f.exists() else []


# --------------------------------------------------------------- parse route
def parse_route(encoder='bge-small', chunking='page', strategies=('fts-pre', 'vec', 'hybrid-pre-deep'),
                force=False):
    'Build the litesearch store over the parsed pages and score every strategy on the page metric.'
    from litesearch import database
    from .build import build_flat, db_path
    from .run import qvecs
    from .score import COLS
    from .score import STRATEGIES
    qs = queries()
    st = build_flat(GENRE, chunking, encoder, force=force)
    qv = qvecs(GENRE, encoder)
    db = database(str(db_path(GENRE, chunking, encoder, 'flat')))
    rows = []
    for strat in strategies:
        fn = STRATEGIES[strat]
        for fl in Q.FLAVOURS:
            ranks, lat = [], []
            for q, v in zip(qs, qv[fl]):
                t = time.time()
                try: hits = fn(db, q[fl], v.tobytes(), K) or []
                except Exception: hits = []
                lat.append((time.time()-t)*1000)
                pages = [(h.get('doc_id'), h.get('page')) for h in hits]
                ranks.append(rank_of(pages, (q['doc_title'], q['page'])))
            rows.append(dict(route='parse', encoder=encoder, chunking=chunking, strategy=strat,
                             modality='text', dpi=None, flavour=fl, n=len(qs),
                             ms_p50=float(np.median(lat)), ranks=ranks, **agg(ranks)))
            print(f'  {encoder:<12} {chunking:<5} {strat:<16} {fl:<10} '
                  f'mrr={rows[-1]["mrr"]:.3f} hit1={rows[-1]["hit1"]:.3f} {np.median(lat):.1f}ms', flush=True)
    db.conn.close()
    return rows, st


# -------------------------------------------------------------- visual route
def images(dpi=DPI):
    'Render every subset page. Returns `(keys, images, seconds)`.'
    import pypdfium2 as pdfium
    keys, imgs, t0 = [], [], time.time()
    for p in pdfs():
        d = pdfium.PdfDocument(str(p))
        for i in range(len(d)):
            imgs.append(d[i].render(scale=dpi/72).to_pil()); keys.append((p.stem, i))
    return keys, imgs, time.time()-t0


def load_model(name):
    'The ST wrapper the variant needs: MultiVectorEncoder for late interaction, plain for dense.'
    import torch
    torch.set_num_threads(4)
    repo, kind = NEOMME[name]
    if kind == 'late':
        from sentence_transformers import MultiVectorEncoder
        return MultiVectorEncoder(repo), kind
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(repo), kind


def encode_docs(m, kind, payload, batch=4):
    'Embed page images or page texts, timed. Returns `(vectors, seconds, bytes)`.'
    t0 = time.time()
    v = m.encode_document(payload, batch_size=batch, show_progress_bar=False)
    el = time.time()-t0
    nb = int(sum(x.numel() for x in v)*2) if kind == 'late' else int(v.numel()*2)
    return v, el, nb


def visual_route(name, modality='image', dpi=DPI, force=False):
    'Encode the pages once, then score every flavour by exhaustive MaxSim / cosine.'
    import torch
    qs = queries()
    keys, text = page_index()
    cache = CACHE/f'neomme/{name}__{modality}__{dpi}.pt'
    cache.parent.mkdir(parents=True, exist_ok=True)
    m, kind = load_model(name)
    if cache.exists() and not force:
        d = torch.load(cache, weights_only=False); dv, t_enc, nbytes, t_prep = d['v'], d['t'], d['b'], d['prep']
    else:
        if modality == 'image': ikeys, payload, t_prep = images(dpi); assert ikeys == keys
        else: payload, t_prep = [text[k] for k in keys], 0.0
        dv, t_enc, nbytes = encode_docs(m, kind, payload)
        torch.save(dict(v=dv, t=t_enc, b=nbytes, prep=t_prep), cache)
    rows = []
    for fl in Q.FLAVOURS:
        t0 = time.time(); qv = m.encode_query([q[fl] for q in qs], show_progress_bar=False)
        t_q = (time.time()-t0)/len(qs)*1000
        t0 = time.time(); sim = m.similarity(qv, dv); t_s = (time.time()-t0)/len(qs)*1000
        order = torch.argsort(torch.as_tensor(sim), dim=1, descending=True)[:, :K].tolist()
        ranks = [rank_of([keys[j] for j in row], (q['doc_title'], q['page']))
                 for q, row in zip(qs, order)]
        rows.append(dict(route='visual', encoder=name, chunking='page', strategy='maxsim',
                         modality=modality, dpi=dpi if modality == 'image' else None, flavour=fl,
                         n=len(qs), ms_p50=t_q+t_s, ranks=ranks,
                         t_encode=t_enc, t_prep=t_prep, bytes=nbytes, n_pages=len(keys), **agg(ranks)))
        print(f'  {name:<18} {modality:<5} {fl:<10} mrr={rows[-1]["mrr"]:.3f} '
              f'hit1={rows[-1]["hit1"]:.3f} {t_q+t_s:.1f}ms', flush=True)
    return rows


# ------------------------------------------------------------------ report
def label(r):
    return (f"{r['encoder']}/{r['chunking']}/{r['strategy']}" if r['route'] == 'parse'
            else f"{r['encoder']}/{r['modality']}")


def quality(rows):
    'One row per configuration: MRR@10 per flavour, the mean, and median query latency.'
    by = {}
    for r in rows: by.setdefault(label(r), {})[r['flavour']] = r
    out = [(k, {f: v[f]['mrr'] for f in Q.FLAVOURS if f in v},
            float(np.mean([v[f]['mrr'] for f in Q.FLAVOURS if f in v])),
            float(np.median([v[f]['ms_p50'] for f in Q.FLAVOURS if f in v]))) for k, v in by.items()]
    return sorted(out, key=lambda t: -t[2])


def ingest():
    'Seconds and bytes to make 112 pages searchable, both routes.'
    parse_s = json.loads((CACHE/'parse_cost.json').read_text())
    bs = json.loads((CACHE/'build_stats.json').read_text())
    out = []
    for r in load('visual_parse'):
        if r['flavour'] != Q.FLAVOURS[0] or r['strategy'] != 'fts-pre': continue
        b = bs.get(f"{GENRE}__flat__{r['chunking']}__{r['encoder']}")
        if not b: continue
        out.append((f"parse  {r['encoder']}/{r['chunking']}", parse_s['s'],
                    b['t_chunk']+b['t_embed'], b['t_insert']+b['t_index'], b['bytes']))
    for r in load('visual_neomme'):
        if r['flavour'] != Q.FLAVOURS[0]: continue
        out.append((f"visual {label(r)}", r['t_prep'], r['t_encode'], 0.0, r['bytes']))
    return out


def compare(rows, a, b):
    'Paired bootstrap of MRR difference `a - b`, per flavour, over the same queries.'
    ix = {(label(r), r['flavour']): r for r in rows}
    out = []
    for fl in Q.FLAVOURS:
        ra, rb = ix.get((a, fl)), ix.get((b, fl))
        if not (ra and rb): continue
        out.append((fl,) + boot(rr(ra['ranks']), rr(rb['ranks'])))
    return out


def report():
    rows = load('visual_parse') + load('visual_neomme')
    if not rows: print('nothing to report'); return
    n = rows[0]['n']
    hdr = ' | '.join(f'{f:>10}' for f in Q.FLAVOURS)
    print(f'\n== MRR@10 on the page, {n} queries x 5 flavours, {len(page_index()[0])} pages ==\n')
    print(f"{'configuration':<40} | {hdr} | {'mean':>6} | {'ms':>6}")
    print('-'*(40+len(hdr)+20))
    for k, per, mean, ms in quality(rows):
        cells = ' | '.join(f'{per.get(f, float("nan")):>10.3f}' for f in Q.FLAVOURS)
        print(f'{k:<40} | {cells} | {mean:>6.3f} | {ms:>6.1f}')
    print(f"\n== ingest, {len(page_index()[0])} pages ==\n")
    print(f"{'route':<40} | {'prep s':>7} | {'embed s':>8} | {'index s':>8} | {'s/page':>7} | {'MB':>7}")
    print('-'*90)
    npg = len(page_index()[0])
    for k, p, e, i, b in sorted(ingest(), key=lambda t: t[1]+t[2]):
        print(f'{k:<40} | {p:>7.1f} | {e:>8.1f} | {i:>8.2f} | {(p+e+i)/npg:>7.3f} | {b/1e6:>7.1f}')
    best_p = quality([r for r in rows if r['route'] == 'parse'])[0][0]
    for k, *_ in quality([r for r in rows if r['route'] == 'visual']):
        print(f'\n== paired bootstrap, {k} minus {best_p} (95% CI) ==')
        for fl, d, lo, hi in compare(rows, k, best_p):
            tag = 'no difference' if lo <= 0 <= hi else ('visual wins' if d > 0 else 'parse wins')
            print(f'  {fl:<12} {d:>+7.3f}  [{lo:>+.3f}, {hi:>+.3f}]  {tag}')


PHASES = dict(
    prepare = lambda: (print(f'parse: {parse_pages():.1f}s'), print(f'{len(queries())} queries')),
    parse   = lambda: save('visual_parse', sum((parse_route(e, c)[0] for e, c in
                                                (('bge-small', 'page'), ('bge-small', 'f512'),
                                                 ('potion-32M', 'page'))), [])),
    visual  = lambda: save('visual_neomme', sum((visual_route(n) for n in NEOMME), [])),
    text    = lambda: save('visual_neomme', sum((visual_route(n, 'text') for n in NEOMME), [])),
    report  = report,
)

if __name__ == '__main__':
    for p in (sys.argv[1:] or ['report']): PHASES[p]()
