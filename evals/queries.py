"""Known-item queries and their ground truth, in four flavours, with no LLM anywhere.

Every query is derived from one source sentence that occurs exactly once in the corpus, so the
target passage and the target section are known exactly. What varies is how much of the original
wording survives:

| flavour      | what it is                                        | what it tests                        |
|--------------|---------------------------------------------------|--------------------------------------|
| `verbatim`   | the sentence itself                               | regression only — FTS answers it     |
| `degraded`   | the rarest content words deleted                  | retrieval without the rare-term key  |
| `keyword`    | five mid-frequency content words, in order        | what users actually type             |
| `paraphrase` | content words swapped for WordNet synonyms        | whether the vector leg earns its cost|

**The honest caveat.** Three of the four flavours are lexical transformations of the target text, so
they carry an inherent bias towards the FTS leg — `paraphrase` is the only one that breaks surface
overlap on purpose, and even it preserves word order and syntax. A human-written question set would
be harder for FTS and kinder to embeddings than anything here. Read the *ordering* of the
configurations, not the absolute numbers, and weight `paraphrase` most heavily when the production
queries are real questions.
"""
import json, os, random, re, tempfile
from pathlib import Path

from .refindex import ref, nrm, _WORD, _SENT
from . import corpus as C

CACHE = Path(__file__).parent/'cache'
N_PER_GENRE = 120          # source sentences per genre; every flavour reuses the same ones (paired)
SEED = 20260803

_STOP = set('''a an the and or but if while of to in on at by for with from as is are was were be been
being it its this that these those which who whom whose what when where why how not no nor so than
then there here their they them he she his her him we us our you your i me my all any both each few
more most other some such only own same too very can will just should now shall may must upon
whether pursuant accordance referred means shall'''.split())


def _content_words(s):
    return [w for w in _WORD.findall(s) if w.lower() not in _STOP and len(w) > 2]


# ---------------------------------------------------------------- synonyms
def build_synonyms(words, out=None):
    '''Cache `{word: [synonym, ...]}` from WordNet.

    nltk is imported from a scratch directory because its import guard refuses to load `regex`
    when the interpreter's cwd is an ancestor of site-packages — which it is whenever the venv
    lives inside the repo. The cache is the artifact; nothing downstream imports nltk.
    '''
    out = out or CACHE/'synonyms.json'
    cwd = os.getcwd()
    try:
        os.chdir(tempfile.gettempdir())
        import nltk
        nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)
        from nltk.corpus import wordnet as wn
        syn = {}
        for w in words:
            lw = w.lower()
            cands = []
            for s in wn.synsets(lw):
                for l in s.lemmas():
                    n = l.name().replace('_', ' ')
                    if n.lower() != lw and ' ' not in n and n.isalpha(): cands.append(n.lower())
            # first synonym of the first synset is the most common sense — good enough, and
            # deterministic, which matters more here than being clever
            seen, keep = set(), []
            for c in cands:
                if c not in seen: seen.add(c); keep.append(c)
            if keep: syn[lw] = keep[:3]
    finally:
        os.chdir(cwd)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(syn))
    return syn


def synonyms():
    'The cached synonym map, built on first use over every content word in the corpus.'
    f = CACHE/'synonyms.json'
    if f.exists(): return json.loads(f.read_text())
    words = set()
    for g in C.GENRES:
        for w, n in ref(g).df.items():
            if n >= 3 and w not in _STOP and len(w) > 3: words.add(w)
    return build_synonyms(sorted(words))


# ---------------------------------------------------------------- flavours
def degrade(sent, df, frac=0.34):
    'Drop the rarest content words. The sentence still reads; the exact-match key is gone.'
    words = re.findall(r"\S+", sent)
    cw = [(i, w) for i, w in enumerate(words) if _content_words(w)]
    if len(cw) < 4: return sent
    rare = sorted(cw, key=lambda t: df.get(re.sub(r'\W', '', t[1]).lower(), 0))[:max(1, int(len(cw)*frac))]
    drop = {i for i, _ in rare}
    return ' '.join(w for i, w in enumerate(words) if i not in drop)


def keywordise(sent, df, k=5, lo=3, hi=0.25, n_segs=1):
    '''Five content words a user might actually type: not the corpus's own stopwords, not hapaxes.

    The band matters. Take the rarest words and the query is a fingerprint of the passage; take the
    commonest and it is a fingerprint of the corpus.'''
    cw = [w.lower() for w in _content_words(sent)]
    band = [w for w in cw if lo <= df.get(w, 0) <= hi*n_segs]
    pick = band or cw
    seen, out = set(), []
    for w in pick:
        if w not in seen: seen.add(w); out.append(w)
    if len(out) <= k: return ' '.join(out)
    step = len(out)/k        # spread across the sentence rather than taking the first five
    return ' '.join(out[int(i*step)] for i in range(k))


def paraphrase(sent, syn, rng, rate=0.9):
    'Swap content words for WordNet synonyms. Word order and syntax survive; surface overlap does not.'
    words, n = re.findall(r"\S+|\s+", sent), 0
    out = []
    for w in words:
        core = re.sub(r'^\W+|\W+$', '', w)
        lw = core.lower()
        cands = syn.get(lw) if (core and lw not in _STOP and len(core) > 2) else None
        if cands and rng.random() < rate:
            rep = cands[0]
            if core[0].isupper(): rep = rep.capitalize()
            out.append(w.replace(core, rep, 1)); n += 1
        else: out.append(w)
    return ''.join(out), n


# ---------------------------------------------------------------- assembly
# How many source sentences may come from one section, and how many content words a sentence needs
# to survive being degraded. Both are genre-specific for the same reason: arXiv PDFs yield ~4
# headings per paper, so insisting on one query per section caps the set at 40, and the astrology
# books are written in short sentences that an 8-content-word floor throws away.
SAMPLING = {
    'regulatory': dict(per_unit=1, min_cw=8),
    'arxiv':      dict(per_unit=3, min_cw=7),
    'astrology':  dict(per_unit=1, min_cw=6),
}


def _source_sentences(r, n, rng, per_unit=1, min_cw=8):
    '''`n` source sentences, spread over sections and documents.

    Capping per section is not tidiness: two sentences from the same Article make the section-level
    metric average over a correlated pair.'''
    cands = []
    for s in r.segments:
        for m in _SENT.finditer(s['content']):
            raw = m.group(0).strip()
            key = nrm(raw)
            if key not in r.uniq_sents or key not in r.sent2u: continue
            u = r.sent2u[key]
            if u not in r.big_units: continue
            if len(_content_words(raw)) < min_cw: continue
            cands.append(dict(sent=raw, key=key, unit=u, doc=s['doc_id'], page=s['page'],
                              node=s['node_id']))
    rng.shuffle(cands)
    by_unit, by_doc, out = {}, {}, []
    cap = max(2, n // max(1, len(r.doc_titles)) * 3)     # keep one document from owning the set
    for c in cands:
        if by_unit.get(c['unit'], 0) >= per_unit or by_doc.get(c['doc'], 0) >= cap: continue
        by_unit[c['unit']] = by_unit.get(c['unit'], 0) + 1
        by_doc[c['doc']] = by_doc.get(c['doc'], 0) + 1
        out.append(c)
        if len(out) >= n: break
    return out


def build(genre, n=N_PER_GENRE, refresh=False):
    'The query set for one genre: `[{flavour: text, ...}]` plus ground truth. Cached.'
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE/f'queries_{genre}.json'
    if f.exists() and not refresh: return json.loads(f.read_text())
    r, rng, syn = ref(genre), random.Random(SEED), synonyms()
    src = _source_sentences(r, n, rng, **SAMPLING[genre])
    out = []
    for c in src:
        para, nsub = paraphrase(c['sent'], syn, rng)
        kw_ = keywordise(c['sent'], r.df, n_segs=r.n_segs)
        kwpara, _ = paraphrase(kw_, syn, rng, rate=1.0)
        out.append(dict(
            key=c['key'], unit=c['unit'], doc=c['doc'],
            doc_title=r.doc_titles.get(c['doc'], ''), page=c['page'], node=c['node'],
            verbatim=c['sent'],
            degraded=degrade(c['sent'], r.df),
            keyword=kw_,
            paraphrase=para,
            kw_para=kwpara, n_sub=nsub))
    f.write_text(json.dumps(out, indent=1))
    return out


# `kw_para` is the hard case and the one to weight: five content words, every one of them swapped
# for a synonym. It is what "semantic search" is sold on, and the only flavour here where the
# lexical leg has nothing left to match.
FLAVOURS = ('verbatim', 'degraded', 'keyword', 'paraphrase', 'kw_para')


def overlap(q, key):
    'Word-level Jaccard between a query and its source sentence — how much lexical help is left.'
    a, b = {w.lower() for w in _WORD.findall(q)}, {w.lower() for w in _WORD.findall(key)}
    return len(a & b)/max(1, len(a | b))


if __name__ == '__main__':
    for g in C.GENRES:
        qs = build(g)
        print(f'== {g}: {len(qs)} queries')
        for fl in FLAVOURS:
            ov = sum(overlap(q[fl], q['key']) for q in qs)/len(qs)
            ln = sum(len(q[fl].split()) for q in qs)/len(qs)
            print(f'   {fl:<11} mean words {ln:>5.1f}   mean lexical overlap with source {ov:.2f}')
        print('   sample:')
        q = qs[0]
        for fl in FLAVOURS: print(f'     {fl:<11} {q[fl][:110]}')
