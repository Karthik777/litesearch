"""Fetch the three-genre eval corpus: arXiv PDFs and Project Gutenberg astrology texts.

Regulatory documents already live in `examples/pdfs` (EU legislation, shipped with the repo), so
only the other two genres are downloaded. Run once; every fetch is skipped if the file exists.
"""
import re, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).parent
ARXIV, ASTRO = HERE/'corpus/arxiv', HERE/'corpus/astrology'

# Two arXiv slices on purpose. cs.* papers are the ordinary RAG target; astro-ph shares surface
# vocabulary with the astrology books (Mars, conjunction, ascending node) without sharing meaning,
# which is the only cheap way to test whether a retriever separates genres or just matches words.
ARXIV_IDS = [
    '1706.03762',  # Attention Is All You Need
    '2005.11401',  # RAG
    '2004.04906',  # DPR
    '2112.09118',  # Contriever
    '2409.04701',  # Late chunking
    '2007.01282',  # FiD
    '2310.11511',  # Self-RAG
    '2404.16130',  # GraphRAG
    '1801.06146',  # ULMFiT (older, different typography)
    'astro-ph/0207156',  # older astro-ph, TeX-era typesetting
    '1808.07573',  # exoplanet transit timing
    '2201.06729',  # solar system dynamics
]

GUTENBERG = {
    70850: 'ptolemy_tetrabiblos',
    46963: 'sepharial_horoscope',
    70749: 'sepharial_cosmic_symbolism',
     1650: 'burgoyne_light_of_egypt',
    78150: 'mercier_astrology_in_medicine',
    78789: 'pavitt_talismans_zodiacal_gems',
    43548: 'waite_key_to_the_tarot',
}

UA = {'User-Agent': 'litesearch-eval/0.1 (+https://github.com/Karthik777/litesearch)'}


def _get(url, dest, tries=4):
    'Download with backoff; a partial file is removed rather than left to poison a later run.'
    if dest.exists() and dest.stat().st_size > 1024: return dest
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r: data = r.read()
            if len(data) < 1024: raise IOError(f'{len(data)} bytes')
            dest.write_bytes(data)
            return dest
        except Exception as e:
            print(f'  {dest.name}: {type(e).__name__} {e} (try {i+1})', flush=True)
            if dest.exists(): dest.unlink()
            time.sleep(2**i)
    return None


_GUT_START = re.compile(r'\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*', re.I|re.S)
_GUT_END   = re.compile(r'\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*', re.I|re.S)

def strip_gutenberg(txt):
    'Drop the Gutenberg licence header/footer so the corpus is the book, not the boilerplate.'
    if (m := _GUT_START.search(txt)): txt = txt[m.end():]
    if (m := _GUT_END.search(txt)):   txt = txt[:m.start()]
    return txt.strip()


def fetch_arxiv():
    ARXIV.mkdir(parents=True, exist_ok=True)
    for aid in ARXIV_IDS:
        dest = ARXIV/f"{aid.replace('/','_')}.pdf"
        if dest.exists(): print(f'  have {dest.name}'); continue
        print(f'  get {aid}', flush=True)
        if not _get(f'https://arxiv.org/pdf/{aid}', dest): print(f'  FAILED {aid}')
        time.sleep(3)   # arXiv asks for one request every few seconds


def fetch_gutenberg():
    ASTRO.mkdir(parents=True, exist_ok=True)
    for gid, name in GUTENBERG.items():
        dest = ASTRO/f'{name}.txt'
        if dest.exists(): print(f'  have {dest.name}'); continue
        raw = ASTRO/f'.raw_{gid}.txt'
        ok = None
        for url in (f'https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt',
                    f'https://www.gutenberg.org/files/{gid}/{gid}-0.txt',
                    f'https://www.gutenberg.org/ebooks/{gid}.txt.utf-8'):
            if (ok := _get(url, raw)): break
        if not ok: print(f'  FAILED {gid} {name}'); continue
        dest.write_text(strip_gutenberg(raw.read_text(errors='replace')), encoding='utf-8')
        raw.unlink()
        print(f'  {dest.name}: {dest.stat().st_size//1024} KB')
        time.sleep(2)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('all','arxiv'):     print('arXiv:');     fetch_arxiv()
    if what in ('all','gutenberg'): print('Gutenberg:'); fetch_gutenberg()
