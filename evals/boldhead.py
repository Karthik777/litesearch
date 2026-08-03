"""Promote bold-run lines to markdown headings before the tree is built.

`07_doc_eval` found that pdf-oxide marks too many lines as `#` on legislation. Papers are the
opposite failure: pdf-oxide renders a paper's section headings as **bold runs** —

    **3.1** **Encoder** **and** **Decoder** **Stacks**

— which is not a heading by any rule `build_tree` applies, so `Attention Is All You Need` gets a
four-node tree (title, abstract, references, appendix) and every numbered section disappears.

This is an eval arm, not a library change: it rewrites the pages before `add_doc` sees them, so the
cost of the idea can be measured against the same tree code that everything else in this evaluation
uses. Two guards keep it from re-creating the legislation problem — a line has to be *entirely*
bold, and a page that produces more than `MAX_PER_PAGE` of them produces none.
"""
import re

_ALL_BOLD = re.compile(r'^(?:\*\*[^*]+\*\*\s*)+$')
_NUMBERED = re.compile(r'^(\d+(?:\.\d+)*)\.?\s+(\S.*)$')
MAX_PER_PAGE = 4.0          # same spirit as tree.MAX_HEAD_DENSITY
MAX_TITLE = 70


def _debold(line): return re.sub(r'\*\*', '', line).strip()


def as_heading(line):
    'The markdown heading this bold-run line should become, or None.'
    s = line.strip()
    if not _ALL_BOLD.match(s): return None
    t = _debold(s)
    if not t or len(t) > MAX_TITLE: return None
    if (m := _NUMBERED.match(t)):
        depth = 1 + min(m.group(1).count('.') + 1, 3)      # "3" -> h2, "3.1" -> h3, "3.1.1" -> h4
        return '#'*depth + ' ' + t
    words = t.split()
    # an unnumbered candidate has to look like a title: short, and not a sentence
    if 1 <= len(words) <= 8 and t[0].isupper() and not t.endswith(('.', ',', ';', ':')):
        return '## ' + t
    return None


def promote(pages):
    'Rewrite `[(page, text)]`, turning qualifying bold-run lines into markdown headings.'
    out, promoted = [], 0
    for pg, txt in pages:
        lines, n = (txt or '').splitlines(), 0
        new = []
        for ln in lines:
            h = as_heading(ln)
            if h: new.append(h); n += 1
            else: new.append(ln)
        promoted += n
        out.append((pg, '\n'.join(new), n))
    if promoted/max(1, len(pages)) > MAX_PER_PAGE:
        return [(pg, txt) for pg, txt in pages], 0       # too dense to be structure; leave it alone
    return [(pg, t) for pg, t, _ in out], promoted


def promote_docs(docs):
    'Apply `promote` to every document of a genre. Returns (docs, {title: n_promoted}).'
    out, counts = {}, {}
    for k, pages in docs.items():
        out[k], counts[k] = promote(pages)
    return out, counts


if __name__ == '__main__':
    from . import corpus as C
    from litesearch.tree import build_tree, detect_mode
    from collections import Counter
    for g in C.GENRES:
        docs = C.load(g)
        new, counts = promote_docs(docs)
        b = sum(len(build_tree(p, title=k)) for k, p in docs.items())
        a = sum(len(build_tree(p, title=k)) for k, p in new.items())
        modes_b = Counter(detect_mode(p) for p in docs.values())
        modes_a = Counter(detect_mode(p) for p in new.values())
        print(f'{g:<12} nodes {b:>5} -> {a:>5}   promoted {sum(counts.values()):>5} lines   '
              f'modes {dict(modes_b)} -> {dict(modes_a)}')
