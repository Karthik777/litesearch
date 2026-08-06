"""A reference section map per genre, built with no model, no chunker and no database.

Everything downstream is scored against this. It decides what a *section* is, which section every
sentence of the corpus lives in, and which sentences are unique enough to serve as a query. The
tree comes from `litesearch.tree.build_tree`, so the sections are the ones the library would build,
but the map is keyed on **whole sentences taken from node segments**, never on chunks: ground truth
must not move when the chunk size does, and that is the main thing being varied downstream.
"""
import re
from collections import Counter

from litesearch.tree import build_tree, doc_id, heading_path
from . import corpus as C

_ws = re.compile(r'\s+')
def nrm(s): return _ws.sub(' ', (s or '').lower()).strip()

# 40 chars is the floor for a sentence to be usable ground truth. Below it, "Article 3" and "See
# Annex I" collide across documents and the mapping stops being a mapping.
MIN_SENT = 40
_SENT = re.compile(r'[^.!?\n]{%d,320}[.!?]' % MIN_SENT)

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
_TOK = re.compile(r"[A-Za-z0-9']+")
GRAM_N = 5


def grams(text, max_words=None, n=GRAM_N):
    'Word `n`-grams of a text, lowercased, in order. The unit of overlap everywhere downstream.'
    ws = [w.lower() for w in _TOK.findall(text or '')]
    if max_words: ws = ws[:max_words]
    return [' '.join(ws[i:i+n]) for i in range(max(0, len(ws)-n+1))]
_NUMERAL = re.compile(r'^([IVXLCDM]+|\d+[A-Za-z]?)$', re.I)
_STRUCT_WORD = re.compile(r'^(BOOK|TITLE|PART|ANNEX|APPENDIX|CHAPTER|SECTION|SUBTITLE|ARTICLE|RULE|'
                          r'CLAUSE|LESSON|SCHEDULE|PAGES)$', re.I)

# A section has to hold this much text before "did you find the right section" means anything.
MIN_UNIT_CHARS = 300


class RefIndex:
    '''Reference sections and the sentence → section map for one genre.

    Attributes:
      `nodes`     — `{node_id: dict(title, level, parent_id, doc_id, heading)}`
      `unit_of`   — `{node_id: unit_id}`; a unit is an Article, a CHAPTER or a paper section
      `sent2u`    — `{normalised sentence: unit_id}` for sentences with one unambiguous owner
      `uniq_sents`— sentences occurring exactly once in the corpus (safe known-item queries)
      `df`        — document frequency of each word over node segments
    '''
    def __init__(self, genre):
        self.genre = genre
        self.unit_re = C.GENRE_UNIT[genre]
        self.nodes, self.doc_titles, self.segments = {}, {}, []
        for title, pages in C.load(genre).items():
            did = doc_id(title, title)
            self.doc_titles[did] = title
            tree = build_tree(pages, title=title)
            for nd in tree:
                nid = f'{did}#{nd.seq}'
                self.nodes[nid] = dict(title=nd.title, level=nd.level, doc_id=did,
                                       parent_id=None if nd.parent is None else f'{did}#{nd.parent}',
                                       page_start=nd.page_start, page_end=nd.page_end,
                                       heading=heading_path(tree, nd, title))
                for page, seg in nd.segments:
                    self.segments.append(dict(content=seg, doc_id=did, node_id=nid, page=page))
        self._build_units()
        self._build_sentmap()
        self._build_df()
        self._build_grams()

    # ---- units -------------------------------------------------------------
    def _unit(self, nid):
        'Nearest ancestor that counts as a section for this genre; the node itself when any node does.'
        seen = set()
        while nid and nid not in seen:
            seen.add(nid)
            nd = self.nodes.get(nid)
            if not nd: return None
            if self.unit_re is None:
                if nd['level'] >= 1: return nid
            elif self.unit_re.match(nd['title'] or ''):
                return nid
            nid = nd['parent_id']
        return None

    def _build_units(self):
        self.unit_of = {nid: self._unit(nid) for nid in self.nodes}
        self.unit_chars = Counter()
        for s in self.segments:
            if (u := self.unit_of.get(s['node_id'])): self.unit_chars[u] += len(s['content'])
        self.units = sorted(self.unit_chars)
        self.big_units = {u for u, n in self.unit_chars.items() if n >= MIN_UNIT_CHARS}

    # ---- sentence -> unit --------------------------------------------------
    def _build_sentmap(self):
        'Map every corpus sentence to its unit, dropping sentences with more than one owner.'
        owner, dup, cnt = {}, set(), Counter()
        for s in self.segments:
            u = self.unit_of.get(s['node_id'])
            for m in _SENT.finditer(s['content']):
                snt = nrm(m.group(0))
                cnt[snt] += 1
                if snt in owner:
                    if owner[snt] != u: dup.add(snt)
                else: owner[snt] = u
        self.sent2u = {s: u for s, u in owner.items() if s not in dup and u}
        self.sent_seg = {}
        for s in self.segments:
            for m in _SENT.finditer(s['content']):
                self.sent_seg.setdefault(nrm(m.group(0)), s)
        self.uniq_sents = {s for s, n in cnt.items() if n == 1}

    def _build_df(self):
        self.df, self.n_segs = Counter(), len(self.segments)
        for s in self.segments:
            for w in {w.lower() for w in _WORD.findall(s['content'])}: self.df[w] += 1

    def _build_grams(self):
        '''`{word 5-gram: unit_id}`, dropping every 5-gram that occurs in more than one section.

        Sentences are the wrong unit for this map. A 256-character chunk often contains no complete
        sentence at all, so a sentence-keyed map would score fine chunking as unable to find
        anything — an artefact of the metric, not a property of the configuration. A 5-gram is
        short enough that any chunk worth returning contains several, and long enough that the ones
        which survive the uniqueness filter really do name a place in the corpus.'''
        owner, dup = {}, set()
        for s in self.segments:
            u = self.unit_of.get(s['node_id'])
            if not u: continue
            for g in grams(s['content']):
                if g in owner:
                    if owner[g] != u: dup.add(g)
                else: owner[g] = u
        self.gram2u = {g: u for g, u in owner.items() if g not in dup}

    # ---- mapping a retrieved passage back onto sections --------------------
    def units_of_text(self, text, max_units=8, max_words=900):
        'Sections covered by a retrieved passage, in order of first appearance. Chunking-agnostic.'
        out, seen = [], set()
        for g in grams(text, max_words):
            u = self.gram2u.get(g)
            if u and u not in seen:
                seen.add(u); out.append(u)
                if len(out) >= max_units: break
        return out

    def unit_title(self, u): return (self.nodes.get(u) or {}).get('title', '')

    def unit_label(self, u):
        nd = self.nodes.get(u) or {}
        return f"{self.doc_titles.get(nd.get('doc_id'), '?')[:26]} › {nd.get('title', '')[:44]}"

    # ---- heading queries ---------------------------------------------------
    def usable_heading(self, u):
        '''A unit title that could plausibly be typed as a query, else None.

        `Article 27` is not one — it names a location, not a topic, so heading queries over EU
        legislation would measure string lookup. `CHAPTER VII. ALCHEMY` is.'''
        t = self.unit_title(u)
        content = [w for w in re.split(r'[\s.:—–,-]+', t)
                   if w and not _NUMERAL.match(w) and not _STRUCT_WORD.match(w) and len(w) >= 3]
        if len(content) < 2 or t.lower().startswith('pages '): return None
        return ' '.join(content)

    def summary(self):
        lv = Counter(n['level'] for n in self.nodes.values())
        seg_chars = [len(s['content']) for s in self.segments]
        return dict(genre=self.genre, docs=len(self.doc_titles), nodes=len(self.nodes),
                    units=len(self.units), big_units=len(self.big_units), segments=len(self.segments),
                    mean_seg=round(sum(seg_chars)/max(len(seg_chars), 1)),
                    uniq_sents=len(self.uniq_sents), mapped_sents=len(self.sent2u),
                    levels=dict(sorted(lv.items())),
                    headed=sum(1 for u in self.big_units if self.usable_heading(u)))


_CACHE = {}
def ref(genre):
    'Memoised RefIndex.'
    if genre not in _CACHE: _CACHE[genre] = RefIndex(genre)
    return _CACHE[genre]


if __name__ == '__main__':
    for g in C.GENRES:
        s = ref(g).summary()
        print(f"{s['genre']:<11} {s['docs']:>3}doc {s['nodes']:>5}node {s['units']:>5}unit "
              f"{s['big_units']:>5}big {s['segments']:>5}seg({s['mean_seg']:>4}ch) "
              f"{s['mapped_sents']:>6}mapped-sent {s['headed']:>4}headed  levels={s['levels']}")
