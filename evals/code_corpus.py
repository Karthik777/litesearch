"""Build a known-item code-retrieval corpus from the three repos' own Python.

One chunk per function or method that carries a docstring. The query is the docstring summary (the
first sentence, the intent a developer would type); the gold is that function. Docstring words that
also appear in the code make this favour lexical systems, so it is a floor for ColBERT, not a thumb
on its side. Functions whose summary is shorter than four words, or whose body is under three lines,
are dropped as too thin to retrieve on.
"""
import ast, json, re
from pathlib import Path

REPOS = {'litesearch': '/home/user/litesearch/litesearch',
         'kosha':      '/home/user/kosha/kosha',
         'vishalakshi':'/home/user/vishalakshi/vishalakshi'}
OUT = Path(__file__).parent/'cache/colbert/code_corpus.json'


def _summary(doc):
    'First sentence of a docstring, whitespace-collapsed.'
    s = ' '.join(doc.strip().split())
    m = re.split(r'(?<=[.?!])\s', s, maxsplit=1)
    return m[0] if m else s


def _defs(src, path, repo):
    'Every func/method with a docstring, as (doc_id, query, code).'
    try: tree = ast.parse(src)
    except SyntaxError: return []
    out, lines = [], src.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)): continue
        doc = ast.get_docstring(node)
        if not doc: continue
        seg = ast.get_source_segment(src, node)
        if not seg or seg.count('\n') < 3: continue
        summ = _summary(doc)
        if len(summ.split()) < 4: continue
        out.append(dict(doc_id=f'{repo}:{path.name}:{node.name}:{node.lineno}',
                        query=summ, content=seg))
    return out


def build(out=OUT):
    'Collect functions across the three repos, dedup by summary, and write the corpus + queries.'
    rows, seen = [], set()
    for repo, root in REPOS.items():
        for py in sorted(Path(root).rglob('*.py')):
            if py.name.startswith('_') and py.name != '__init__.py': pass
            for d in _defs(py.read_text(errors='replace'), py, repo):
                key = d['query'].lower()
                if key in seen: continue           # a summary that maps to two functions is not known-item
                seen.add(key); rows.append(d)
    chunks = [dict(content=r['content'], doc_id=r['doc_id']) for r in rows]
    queries = [dict(query=r['query'], doc_id=r['doc_id']) for r in rows]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(chunks=chunks, queries=queries), indent=1))
    print(f'{len(chunks)} functions across {len(REPOS)} repos -> {out}')
    return out


if __name__ == '__main__':
    build()
