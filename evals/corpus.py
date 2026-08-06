"""The three-genre eval corpus, parsed once and cached as `[(page, text)]` per document.

Three genres, chosen because they fail differently:

- **regulatory** — 9 EU directives/regulations (486 pages). Real hierarchy (TITLE › CHAPTER ›
  Article), formulaic language, enormous term repetition across documents.
- **arxiv** — 12 papers, cs.* and astro-ph. Markdown headings survive conversion, notation and
  equations do not, and the astro-ph slice shares vocabulary with the astrology books without
  sharing meaning.
- **astrology** — 7 Project Gutenberg books (1822–1920s). Long, archaic prose, `CHAPTER XI` lines
  instead of markup, one book with no headings at all, and tables of planetary aspects that
  survive conversion as noise.
"""
import json, re
from pathlib import Path

HERE   = Path(__file__).parent
CACHE  = HERE/'cache'
REG_DIR   = HERE.parent/'examples/pdfs'
ARXIV_DIR = HERE/'corpus/arxiv'
ASTRO_DIR = HERE/'corpus/astrology'

# A printed page of a book is ~2000 characters. The Gutenberg texts arrive as one stream, and
# feeding them in as a single page would hand the tree layer a document with no page numbers —
# not a kindness, just a different corpus from the scans these books actually exist as.
PAGE_CHARS = 2000

GENRE_UNIT = {          # what "which section answers this" means per genre
    'regulatory': re.compile(r'^\s*#*\s*Article\s+\d+', re.I),
    'arxiv':      None,                                  # any tree node: a paper's own headings
    'astrology':  re.compile(r'^\s*#*\s*(CHAPTER|BOOK|PART|SECTION)\b', re.I),
}


def paginate(text, page_chars=PAGE_CHARS):
    'Split a text stream into pseudo-pages at paragraph boundaries.'
    paras, pages, buf, n = re.split(r'\n\s*\n', text), [], [], 0
    for p in paras:
        if n and n + len(p) > page_chars:
            pages.append('\n\n'.join(buf)); buf, n = [], 0
        buf.append(p); n += len(p) + 2
    if buf: pages.append('\n\n'.join(buf))
    return list(enumerate(pages))


def _pdf_pages(path):
    from litesearch.data import pdf_parse
    return list(enumerate(pdf_parse(str(path))))


def _load_raw(genre):
    'Parse a genre from source files. Slow (PDF conversion); `load` caches the result.'
    if genre == 'mixed':
        # one store holding all three genres, because that is what a real library looks like:
        # nobody keeps their regulations, their papers and their books in three separate databases
        return {f'{g}:{k}': v for g in GENRES for k, v in load(g).items()}
    if genre == 'regulatory':
        return {p.stem: _pdf_pages(p) for p in sorted(REG_DIR.glob('*.pdf'))}
    if genre == 'arxiv':
        return {p.stem: _pdf_pages(p) for p in sorted(ARXIV_DIR.glob('*.pdf'))}
    if genre == 'astrology':
        return {p.stem: paginate(p.read_text(errors='replace')) for p in sorted(ASTRO_DIR.glob('*.txt'))}
    raise ValueError(f'unknown genre {genre!r}')


def load(genre, refresh=False):
    'Cached `{doc_key: [(page, text)]}` for one genre.'
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE/f'pages_{genre}.json'
    if f.exists() and not refresh:
        return {k: [(int(p), t) for p, t in v] for k, v in json.loads(f.read_text()).items()}
    docs = _load_raw(genre)
    f.write_text(json.dumps(docs))
    return docs


GENRES = ('regulatory', 'arxiv', 'astrology')
ALL = GENRES + ('mixed',)


def stats(genre):
    docs = load(genre)
    npg = sum(len(v) for v in docs.values())
    nch = sum(len(t) for v in docs.values() for _, t in v)
    return dict(genre=genre, docs=len(docs), pages=npg, chars=nch,
                chars_per_page=round(nch/max(npg, 1)))


if __name__ == '__main__':
    for g in GENRES:
        s = stats(g)
        print(f"{s['genre']:<12} {s['docs']:>3} docs  {s['pages']:>5} pages  "
              f"{s['chars']/1e6:>5.2f}M chars  {s['chars_per_page']:>5} chars/page")
