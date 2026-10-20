"""Does rewriting the query help a static embedder? Algorithmic rewriters only, no LLM.

A potion query vector is a weighted mean of token vectors: no order, no negation, no context. So a
rewrite is a choice of token set. Corpus: vedicreader section meanings (English, real library
text). Query flavours are templates over one sentence of the target, so ground truth is exact:

| flavour   | what it is                                                | what a rewrite should fix |
|-----------|-----------------------------------------------------------|---------------------------|
| `keyword` | five mid-frequency content words, in order                | nothing, the control      |
| `sparse`  | three of them                                             | too little signal         |
| `verbose` | the keywords inside a chatty request                      | filler diluting the mean  |
| `negated` | the keywords plus "but not" and two words of another doc  | negation the mean ignores |

`verbose` and `negated` are synthetic and built to favour a rewriter. Read them as the upper bound
of the cheap rewrites, `keyword` as the cost of applying them to a query that needs none.

`python -m evals.rewrite` writes `evals/results/rewrite.json`.
"""
import json, os, random, re, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

from .queries import _STOP, _content_words
from .refindex import _WORD, _SENT

TEXTS = Path(os.environ.get('VR_TEXTS', '/home/user/vedicreader/static/vedic_texts'))
OUT = Path(__file__).parent/'results'/'rewrite.json'
SEED, K, PER_CAT, N_DOCS, N_Q = 20261020, 10, 1200, 6000, 1000
NEG = re.compile(r'\b(?:not|no|without|except|excluding|never)\b(?:\s+\w+){0,2}', re.I)
FILLER = ['i would like to find the passage that explains {} and what it says about it',
          'can you tell me something about {} please',
          'what is the text where {} comes up and how is it described',
          'show me anything related to {} that i could read']


# ---------------------------------------------------------------- corpus
def load_docs(n=N_DOCS, per_cat=PER_CAT, seed=SEED):
    'Up to `per_cat` unique section meanings per category, sampled to `n`: `{id: text}`.'
    seen, by = set(), {}
    for f in sorted(TEXTS.glob('**/*.json')):
        try: d = json.loads(f.read_text())
        except Exception: continue
        if not isinstance(d, dict): continue
        for s in d.get('sections', []):
            m = (s.get('meaning') or '').strip()
            if len(m.split()) < 40 or m in seen: continue
            seen.add(m); by.setdefault(d.get('category'), []).append((f'{d["id"]}/{s["name"]}', m))
    r = random.Random(seed); out = []
    for c, v in by.items(): r.shuffle(v); out += v[:per_cat]
    r.shuffle(out)
    return dict(out[:n])


# ---------------------------------------------------------------- queries
def mid_words(sent, df, n):
    'The middle frequency band of `sent`s content words, `n` of them, in text order.'
    ws = list(dict.fromkeys(w.lower() for w in _content_words(sent)))
    if len(ws) < n: return None
    mid = sorted(ws, key=lambda w: df[w])[len(ws)//4: len(ws)*3//4 + 1]
    keep = set(mid[:n]) if len(mid) >= n else set(sorted(ws, key=lambda w: df[w])[:n])
    return [w for w in ws if w in keep]


def make_queries(docs, n=N_Q, seed=SEED):
    'One query per flavour for `n` target docs: a list of dicts with `id`, `keyword`, `sparse`, `verbose`, `negated`.'
    r = random.Random(seed); ids = list(docs)
    df = Counter(w.lower() for t in docs.values() for w in set(_content_words(t)))
    out = []
    for i in r.sample(ids, len(ids)):
        sents = [s for s in _SENT.split(docs[i]) if len(_content_words(s)) >= 6]
        if not sents: continue
        kw = mid_words(r.choice(sents), df, 5)
        if not kw: continue
        other = _content_words(docs[r.choice(ids)])
        out.append(dict(id=i, keyword=' '.join(kw), sparse=' '.join(kw[:3]), verbose=r.choice(FILLER).format(' '.join(kw)),
                        negated=' '.join(kw) + ' but not ' + ' '.join(r.sample(other, 2)) if len(other) > 2 else ' '.join(kw)))
        if len(out) == n: break
    return out


# ---------------------------------------------------------------- rewriters, query string -> query string
def strip(q):
    'Drop negated clauses, stopwords and repeats.'
    return ' '.join(dict.fromkeys(w for w in _WORD.findall(NEG.sub(' ', q)) if w.lower() not in _STOP))

def no_neg(q): return ' '.join(NEG.sub(' ', q).split())
def no_stop(q): return ' '.join(w for w in _WORD.findall(q) if w.lower() not in _STOP)
def raw(q): return q


class Dense:
    'Exact cosine search over the static embedder, the vector leg alone.'
    def __init__(self, enc, docs):
        self.ids, self.enc = list(docs), enc
        self.D = self.norm(enc(list(docs.values())))
        self.texts = docs

    @staticmethod
    def norm(v): v = np.asarray(v, dtype=np.float32); return v/np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)

    def vec(self, qs): return self.norm(self.enc(list(qs)))
    def top(self, v, k=K): return np.argsort(-(self.D@v))[:k]
    def rank(self, v, gold):
        'Zero-based rank of doc `gold` under query vector `v`.'
        s = self.D@v; return int((s > s[self.ids.index(gold)]).sum())


def expand(dn, q, k=5, n=3, ncand=150):
    'Pseudo-relevance feedback in token space: add the top-k docs words that most raise cosine to their centroid.'
    v = dn.vec([q])[0]; cen = dn.norm(dn.D[dn.top(v, k)].mean(0))
    cands = [w for w, _ in Counter(w.lower() for i in dn.top(v, k) for w in _content_words(dn.texts[dn.ids[i]])).most_common(ncand)]
    cur = q
    for _ in range(n):
        cs = [c for c in cands if c not in cur.split()]
        if not cs: break
        cur += ' ' + cs[int(np.argmax(dn.vec([f'{cur} {c}' for c in cs])@cen))]
    return cur


def rocchio(dn, q, k=5, beta=.5):
    'The vector-space version: shift the query toward the mean of the top-k docs.'
    v = dn.vec([q])[0]; return dn.norm(v + beta*dn.D[dn.top(v, k)].mean(0))


def oracle(dn, q, gold):
    'Ceiling for any token-subset rewrite: drop each word that lowers the gold rank, one pass. Uses the answer.'
    ws = q.split(); best = dn.rank(dn.vec([q])[0], gold)
    for i in range(len(ws)):
        t = ' '.join(ws[:i] + ws[i+1:])
        if not t: continue
        r = dn.rank(dn.vec([t])[0], gold)
        if r < best: best, ws = r, ws[:i] + ws[i+1:]
    return ' '.join(ws)


# ---------------------------------------------------------------- scoring
def metrics(ranks):
    'MRR@K, recall@K and hit@1 from zero-based ranks (inf for a miss).'
    r = np.asarray(ranks, dtype=float); rr = np.where(r < K, 1/(r + 1), 0)
    return dict(mrr=float(rr.mean()), recall=float((r < K).mean()), hit1=float((r == 0).mean()))

def boot(a, b, n=2000, seed=0):
    'Paired bootstrap over queries of mean(b - a): `(diff, lo, hi)`, 95%.'
    d = np.asarray(b) - np.asarray(a); g = np.random.default_rng(seed)
    m = d[g.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def rr_of(rank): return 1/(rank + 1) if rank < K else 0.


def hybrid_ranks(docs, qs, rws, encoder=None):
    'Ranks under `Index.search` (FTS + vector, RRF) for each rewriter, one in-memory Index.'
    from litesearch.api import Index
    ix = Index(encoder=encoder, ann=False); ix.add(docs)
    def rk(q, gold):
        for i, h in enumerate(ix.search(q, limit=K)):
            if h['heading'].startswith(gold + ' › '): return i
        return 99
    return {n: [rk(rw(q[f]) if f else rw(q), q['id']) for q in qs] for n, (rw, f) in rws.items()}


def run(n_docs=N_DOCS, n_q=N_Q, hybrid=True):
    'Run every rewriter on every flavour; write `OUT`.'
    from litesearch.utils import static_embedder
    m = static_embedder(); enc = lambda t: m.encode(list(t))
    docs = load_docs(n_docs); qs = make_queries(docs, n_q); dn = Dense(enc, docs)
    print(f'{len(docs)} docs, {len(qs)} queries', flush=True)
    res = {}
    for fl in ['keyword', 'sparse', 'verbose', 'negated']:
        t0 = time.time(); rk = {}
        def ranks(rw): return [dn.rank(dn.vec([rw(q[fl])])[0], q['id']) for q in qs]
        rk['raw'] = ranks(raw); rk['no_stop'] = ranks(no_stop); rk['no_neg'] = ranks(no_neg); rk['strip'] = ranks(strip)
        rk['strip+expand'] = [dn.rank(dn.vec([expand(dn, strip(q[fl]))])[0], q['id']) for q in qs]
        rk['rocchio'] = [dn.rank(rocchio(dn, strip(q[fl])), q['id']) for q in qs]
        rk['oracle_drop'] = [dn.rank(dn.vec([oracle(dn, strip(q[fl]), q['id'])])[0], q['id']) for q in qs]
        if hybrid:
            hb = hybrid_ranks(docs, [dict(q, _=q[fl]) for q in qs], {'raw': (lambda q: q['_'], None), 'strip': (lambda q: strip(q['_']), None),
                                                                       'strip+expand': (lambda q: expand(dn, strip(q['_'])), None)}, m)
            rk.update({f'hybrid_{k}': v for k, v in hb.items()})
        base = lambda k: [rr_of(r) for r in rk['hybrid_raw' if k.startswith('hybrid_') else 'raw']]
        res[fl] = {k: dict(**metrics(v), vs_raw=boot(base(k), [rr_of(r) for r in v])) for k, v in rk.items()}
        print(f'-- {fl} ({time.time()-t0:.0f}s)')
        for k, v in res[fl].items(): print(f'  {k:22} mrr {v["mrr"]:.3f}  r@{K} {v["recall"]:.3f}  hit1 {v["hit1"]:.3f}  d {v["vs_raw"][0]:+.3f} [{v["vs_raw"][1]:+.3f},{v["vs_raw"][2]:+.3f}]', flush=True)
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(dict(n_docs=len(docs), n_q=len(qs), res=res), indent=1))
    return res


if __name__ == '__main__':
    a = sys.argv[1:]; run(*(int(x) for x in a[:2]), hybrid='--nohybrid' not in a)
