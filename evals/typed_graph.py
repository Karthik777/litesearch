"""Typed (LLM-built) graph vs hybrid and the PMI graph on cross-reference bridges.

A local model (via rishi/urai, llama.cpp) extracts typed relations per chunk; the prompt is adapted
from artifact-pyramids (L1 summary, L2 typed atoms traceable to source). Following `refers_to` edges
reaches the referenced article; co-occurrence and hybrid cannot. rishi is imported inside the
extractor so the rest of `evals/` runs without it. Method and numbers in `evals/RESULTS.md`.
"""
import json, re
import numpy as np

from litesearch import database, DTYPE
from litesearch.core import rrf_all
from .build import db_path
from .encoders import enc
from .multihop import _toks

REL_VOCAB = ["refers_to","defines","excludes","applies_to","subject_to","amends","part_of"]
REL_MAP = {"cross_ref":"refers_to","references":"refers_to","cites":"refers_to","refer_to":"refers_to",
           "applies":"applies_to","subject":"subject_to","amend":"amends","partof":"part_of"}
ENT_TYPES = ["article_ref","legal_concept","actor","transaction_type","goods_or_services","place","condition"]

SYSTEM = (
"You extract a typed knowledge graph from one chunk of EU legal text, as an artifact pyramid.\n"
"L1: a one-sentence summary. L2: typed atoms grounded in THIS chunk, each traceable to it.\n"
f"entities: {{name, type}}, type in {ENT_TYPES}. relations: {{src, rel, dst}}, rel in {REL_VOCAB}.\n"
"CRITICAL: for EVERY 'Article N', 'Article N(x)' or 'point (x) of Article N' in the text, emit a "
"refers_to relation whose dst is exactly 'Article N'. Canonicalize names so a reference collapses "
"to one node. Output STRICT JSON only: "
'{"summary": str, "entities": [{"name": str, "type": str}], "relations": [{"src": str, "rel": str, "dst": str}]}')

def _artnorm(x):
    "'Article 3(1)' -> 'Article 3'."
    m = re.search(r'Article\s+(\d+)', str(x)); return f'Article {m.group(1)}' if m else None

def _norm_rel(r):
    r = (r or '').lower().strip(); return r if r in REL_VOCAB else REL_MAP.get(r)

def _parse(txt):
    "First JSON object in a model reply, tolerant of fences."
    m = re.search(r'\{.*\}', txt or '', re.S)
    for a in ([m.group(0), m.group(0).replace('\n',' ')] if m else []):
        try: return json.loads(a)
        except Exception: pass
    return {}

def rishi_chat(gguf, model_id=None):
    "chat(system,user)->str over a local GGUF via rishi/urai. Stateless per call."
    import rishi.core, urai
    chat = urai.Chat(gguf or model_id, runtime='llama', opts=dict(temp=0))
    def call(system, user):
        r = chat.oneshot(user, sp=system, max_tokens=640)
        return r.get('content','') if isinstance(r, dict) else str(r)
    return call

def extract(chunks, chat, log=print):
    "Per chunk: {id, summary, entities, relations}. `chat(system,user)->str`."
    out = []
    for i, c in enumerate(chunks):
        j = _parse(chat(SYSTEM, f"CHUNK:\n# {c.get('heading') or ''}\n{c['content'][:1400]}"))
        ents = [e for e in j.get('entities',[]) if isinstance(e,dict) and e.get('name')]
        rels = [dict(r, rel=nr) for r in j.get('relations',[])
                if isinstance(r,dict) and r.get('src') and r.get('dst') and (nr:=_norm_rel(r.get('rel')))]
        out.append(dict(id=c['id'], summary=j.get('summary',''), entities=ents, relations=rels))
        log(f"  [{i+1}/{len(chunks)}] {len(ents)}e {len(rels)}r")
    return out

def _bridges(sub, tg_by, allrows, ART, art_targets):
    "Cross-reference bridges: query = source tokens absent from the referenced article; target = its chunks."
    out = []
    for x in sub:
        t = tg_by.get(x['id']) or {}
        for a in {a for r in t.get('relations',[]) if r.get('rel')=='refers_to' and (a:=_artnorm(r.get('dst')))}:
            tgt = art_targets(a)
            if not tgt or x['id'] in tgt: continue
            tt = set().union(*[_toks(allrows[i]['content']) for i in tgt])
            anc = [w for w in sorted(_toks(x['content']), key=len, reverse=True) if w not in tt][:6]
            if len(anc) >= 3: out.append(dict(query=' '.join(anc[:4]), target=set(tgt), art=a))
    return out

def evaluate(sub, tg, genre='regulatory', doc=None):
    "hybrid vs pmi-graph vs typed-graph on the bridges. Returns rows; prints a table."
    import vruksha  # noqa: applies graph_search
    db = database(str(db_path(genre,'c512','bge-small','tree'))); e = enc('bge-small')
    rows = {r['id']: r for r in db.t.store(select='id,doc_id,node_id,heading,content')}
    doc = doc or rows[sub[0]['id']]['doc_id']
    allrows = {i:r for i,r in rows.items() if r['doc_id']==doc}
    node_of = {i:allrows[i]['node_id'] for i in allrows}
    ART = {}
    for r in allrows.values():
        if (m:=re.search(r'Article\s+(\d+)', r['heading'] or '')): ART.setdefault(f"Article {m.group(1)}", []).append(r['id'])
    art_targets = lambda a: [i for i in ART.get(a,[])[:4] if i in allrows]
    tg_by = {t['id']:t for t in tg}
    refers = {t['id']:{a for r in t.get('relations',[]) if r.get('rel')=='refers_to' and (a:=_artnorm(r.get('dst')))} for t in tg}
    art_doc = {a:[i for i in ART.get(a,[]) if i in allrows] for s in refers.values() for a in s}
    bridges = _bridges(sub, tg_by, allrows, ART, art_targets)
    qv = lambda q: np.asarray(e.qry([q])[0], dtype=DTYPE).tobytes()
    def rr(hits, target, k=50):
        tn = {node_of[t] for t in target}
        return next((1.0/(i+1) for i,h in enumerate(hits[:k]) if h.get('id') in target or h.get('node_id') in tn), 0.0)
    def hybrid(q): return db.search(q, qv(q), columns=['id','node_id'], limit=50) or []
    def pmi(q): return db.graph_search(q, qv(q), columns=['id','node_id'], limit=50, graph_w=1.0) or []
    def typed(q, seedk=8):
        base = hybrid(q); seen=set()
        leg = [dict(id=t, node_id=node_of.get(t)) for h in base[:seedk] for a in refers.get(h['id'],())
               for t in art_doc.get(a,[]) if not (t in seen or seen.add(t))]
        return rrf_all([base, leg], k=60, limit=50, id_key='id', weights=[1.0,1.0])
    out = []
    print(f"  {'method':<16}{'target MRR':>12}{'target hit':>12}  ({len(bridges)} bridges)")
    for label, fn in [('hybrid',hybrid),('pmi-graph',pmi),('typed-graph',typed)]:
        rrs = [rr(fn(b['query']), b['target']) for b in bridges]
        m, h = float(np.mean(rrs or [0])), float(np.mean([x>0 for x in rrs] or [0]))
        print(f"  {label:<16}{m:>12.4f}{h:>12.4f}"); out.append(dict(method=label, mrr=m, hit=h, n=len(bridges)))
    db.conn.close(); return out
