"""Does the clustering layer produce a map of the corpus, or just groups?

`clusters()` and `peers()` answer questions about shape rather than about a query, so ranking
metrics do not apply. What can be measured is whether the groups line up with structure that
already exists — the document a chunk came from, the section it came from, the genre it came from —
and whether `peers` beats the k-NN it degrades to.

Four measurements:

- **purity / NMI** against document, section and (on the mixed store) genre labels. Purity alone
  rewards tiny clusters, so NMI is reported beside it; a configuration that scores well on both is
  actually recovering structure.
- **coverage** — the share of indexed vectors that land in a kept cluster. A clustering that is
  perfectly pure over 6% of the corpus is not a map of the corpus.
- **`peers` against `ann_neighbors`** at matched size. `peers` claims to return a family rather
  than a ranked list; the test is whether its members share a section more often than the same
  number of nearest neighbours do.
- **label distinctiveness** — how much more often a cluster's c-TF-IDF label terms occur inside it
  than in the corpus at large. Labels are what a human reads, and a label made of corpus-wide
  boilerplate is worse than no label.
"""
import json, math, random, time
from collections import Counter
from pathlib import Path

from litesearch import database

from . import corpus as C
from .build import db_path
from .refindex import ref, _WORD
from .run import BASE_GRAIN, RESULTS

SEED = 20260803


# ------------------------------------------------------------------ metrics
def purity(groups, lab):
    'Weighted mean of the largest label share in each group, over assigned members only.'
    tot = num = 0
    for g in groups:
        ls = [lab[m] for m in g if m in lab]
        if not ls: continue
        tot += len(ls); num += Counter(ls).most_common(1)[0][1]
    return num/tot if tot else 0.0


def nmi(groups, lab):
    'Normalised mutual information between a clustering and a labelling.'
    pairs = [(i, lab[m]) for i, g in enumerate(groups) for m in g if m in lab]
    if not pairs: return 0.0
    n = len(pairs)
    ci, li = Counter(c for c, _ in pairs), Counter(l for _, l in pairs)
    joint = Counter(pairs)
    hc = -sum(v/n*math.log(v/n) for v in ci.values())
    hl = -sum(v/n*math.log(v/n) for v in li.values())
    mi = sum(v/n*math.log((v/n)/((ci[c]/n)*(li[l]/n))) for (c, l), v in joint.items())
    return mi/math.sqrt(hc*hl) if hc > 0 and hl > 0 else 0.0


def label_distinctiveness(res, rows_text, top=4):
    '''Mean ratio of in-cluster to corpus-wide occurrence rate for a cluster\'s label terms.

    1.0 means the label describes the corpus, not the cluster.'''
    corpus_df, n_docs = Counter(), len(rows_text)
    for t in rows_text.values():
        for w in {w.lower() for w in _WORD.findall(t or '')}: corpus_df[w] += 1
    out = []
    for cl in res.clusters:
        terms = [t.strip() for t in (cl.label or '').split(',')][:top]
        terms = [t for t in terms if t]
        if not terms: continue
        mem = [rows_text.get(k, '') for k in cl.member_keys if k in rows_text]
        if not mem: continue
        for t in terms:
            inr = sum(1 for x in mem if t in (x or '').lower())/len(mem)
            outr = corpus_df.get(t, 0)/max(1, n_docs)
            if outr > 0: out.append(inr/outr)
    return sum(out)/len(out) if out else 0.0


# ------------------------------------------------------------------ labels
def _labels(db, genre):
    'rowid -> (doc label, section label, genre label) for every row in the store.'
    rows = list(db.t.store(select='rowid as rowid, content, doc_id'))
    r = None if genre == 'mixed' else ref(genre)
    doc, sec, gen, txt = {}, {}, {}, {}
    for x in rows:
        rid = x['rowid']
        txt[rid] = x['content']
        doc[rid] = x['doc_id'] or '?'
        gen[rid] = (x['doc_id'] or ':').split(':', 1)[0]
        if r is not None:
            us = r.units_of_text(x['content'], max_units=1)
            if us: sec[rid] = us[0]
    return doc, sec, gen, txt


# ------------------------------------------------------------------ runners
def eval_clusters(genre, encoder, mode='flat', chunking=BASE_GRAIN, min_size=3):
    p = db_path(genre, chunking, encoder, mode)
    if not p.exists(): return None
    db = database(str(p))
    doc, sec, gen, txt = _labels(db, genre)
    t0 = time.time()
    res = db.t.store.clusters(min_size=min_size, columns=['doc_id'])
    t_cl = time.time()-t0
    groups = [list(c.member_keys) for c in res.clusters]
    sizes = sorted(len(g) for g in groups)
    n_idx = db.get_index('store').size
    assigned = len({m for g in groups for m in g})
    out = dict(genre=genre, encoder=encoder, mode=mode, chunking=chunking,
               method=res.method, note=res.note, n_clusters=len(groups),
               indexed=n_idx, coverage=round(assigned/max(1, n_idx), 3),
               size_med=sizes[len(sizes)//2] if sizes else 0, size_max=sizes[-1] if sizes else 0,
               t_cluster=round(t_cl, 2),
               purity_doc=round(purity(groups, doc), 3), nmi_doc=round(nmi(groups, doc), 3),
               label_dist=round(label_distinctiveness(res, txt), 2),
               labels=[f'{c.size}: {c.label}' for c in res.clusters[:6]])
    if sec:
        out |= dict(purity_sec=round(purity(groups, sec), 3), nmi_sec=round(nmi(groups, sec), 3))
    if genre == 'mixed':
        out |= dict(purity_genre=round(purity(groups, gen), 3), nmi_genre=round(nmi(groups, gen), 3))
    return out


def eval_peers(genre, encoder, mode='flat', chunking=BASE_GRAIN, n=120, limit=15):
    '''`peers` against `ann_neighbors` at matched size, on the same rows.

    Matched size matters: k-NN always returns `limit` rows, a cluster returns however many it has,
    and comparing a 4-member family to 15 neighbours would measure the limit, not the method.'''
    p = db_path(genre, chunking, encoder, mode)
    if not p.exists(): return None
    db = database(str(p))
    doc, sec, gen, _ = _labels(db, genre)
    rng = random.Random(SEED)
    keys = sorted(sec or doc)
    rng.shuffle(keys); keys = keys[:n]
    ps, ks, sizes, notes, t_p, t_k = [], [], [], Counter(), 0.0, 0.0
    lab = sec or doc
    for k in keys:
        t0 = time.time(); pr = db.t.store.peers(k, limit=limit, columns=['doc_id']); t_p += time.time()-t0
        notes[pr.method or 'none'] += 1
        hits = [h['rowid'] for h in pr.hits]
        if not hits: continue
        sizes.append(len(hits))
        ps.append(sum(1 for h in hits if lab.get(h) == lab.get(k))/len(hits))
        t0 = time.time()
        nb = db.t.store.ann_neighbors(k, limit=len(hits), columns=['doc_id'])
        t_k += time.time()-t0
        if nb: ks.append(sum(1 for h in nb if lab.get(h['rowid']) == lab.get(k))/len(nb))
    m = lambda v: round(sum(v)/len(v), 3) if v else 0.0
    return dict(genre=genre, encoder=encoder, mode=mode, chunking=chunking, n=len(ps),
                label='section' if sec else 'document',
                peers_prec=m(ps), knn_prec=m(ks), mean_group=m(sizes),
                methods=dict(notes), ms_peers=round(t_p/max(1, len(keys))*1000, 2),
                ms_knn=round(t_k/max(1, len(keys))*1000, 2))


def run():
    from .build import build_flat
    rows_c, rows_p = [], []
    for g in C.ALL:
        for e in ('potion-32M', 'bge-small'):
            if g == 'mixed': print(show_or_build(g, e), flush=True)
            c = eval_clusters(g, e)
            if c:
                rows_c.append(c)
                print(f"  {g}/{e}: {c['n_clusters']:>4} clusters ({c['method']}) cov {c['coverage']:.2f} "
                      f"med {c['size_med']} max {c['size_max']}  purity doc {c['purity_doc']} "
                      f"sec {c.get('purity_sec','-')} nmi_doc {c['nmi_doc']} labeldist {c['label_dist']} "
                      f"{c['t_cluster']}s", flush=True)
                for l in c['labels']: print(f'      {l}', flush=True)
            pr = eval_peers(g, e)
            if pr:
                rows_p.append(pr)
                print(f"  {g}/{e}: peers {pr['peers_prec']} vs knn {pr['knn_prec']} "
                      f"({pr['label']}, mean group {pr['mean_group']}, {pr['methods']}) "
                      f"{pr['ms_peers']}ms vs {pr['ms_knn']}ms", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS/'cluster.json').write_text(json.dumps(dict(clusters=rows_c, peers=rows_p), indent=1))
    print(f"  -> {RESULTS/'cluster.json'}")


def show_or_build(genre, encoder):
    from .build import build_flat, show
    return show(build_flat(genre, BASE_GRAIN, encoder))


if __name__ == '__main__':
    run()
