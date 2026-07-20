# Late Chunking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add late-chunking encoders to `FastEncode` and benchmark them against naive, full-doc, and contextual embedding on a long-context retrieval dataset scored through litesearch hybrid search.

**Architecture:** Three `FastEncode` subclasses in `nbs/03_utils.ipynb` embed a whole document once (per-token embeddings) then mean-pool per chunk span; a `chunk_spans` helper in `nbs/02_data.ipynb` supplies identical `(start,end,text)` spans to every method; an eval notebook `nbs/04_latechunk_eval.ipynb` builds one litesearch store per method and compares retrieval via `db.search` (hybrid RRF), aggregating chunk hits to parent docs.

**Tech Stack:** nbdev, ONNX Runtime, HF `tokenizers`, chonkie, numpy, litesearch (`database`, `FastEncode`), HF `datasets` (LongEmbed), `rishi` (local litert Gemma).

## Global Constraints

- Source of truth is the notebooks under `nbs/`; NEVER edit `litesearch/*.py` by hand — regenerate with `nbdev_prepare`. (Project CLAUDE.md says `nbdev_prepare`; global CLAUDE.md says `nbdev-prepare`. Use whichever resolves in this env — they are aliases of the same entry point.)
- Every exported cell is marked `#| export`; every test cell is a plain cell with `assert`/`test_eq` (nbdev runs plain cells as tests).
- Python style: fastcore idioms (`store_attr`, `L`, `ifnone`, `patch`, `delegates`), one-line docstrings, no decorative comments, no box-drawing separators, no alignment padding.
- New subclasses reuse `FastEncode` attributes only: `self.tok`, `self.sess`, `self._input_names`, `self.tti`, `self.prompt`, `self.normalize`, `self.dtype`, `self.max_seq_len`. Do not duplicate ONNX/session setup.
- Package installs via `uv add` only (never pip). Run tools via `uv run`.
- Embeddings are numpy float16 (`self.dtype`); litesearch stores embeddings as `.tobytes()`.
- Commit after each task with a `feat:`/`test:`/`chore:` prefix.

---

### Task 1: `chunk_spans` helper (shared chunker with offsets)

**Files:**
- Modify: `nbs/02_data.ipynb` (add near `chunk_markdown`, ~cell after line 108 region)
- Regenerates: `litesearch/data.py`, `litesearch/_modidx.py`

**Interfaces:**
- Consumes: chonkie `FastChunker`/`BaseChunker` (already imported in `02_data.ipynb`: `from chonkie import RecursiveChunker, FastChunker, BaseChunker`).
- Produces: `chunk_spans(text:str, chunker:BaseChunker=None) -> L` of `(start:int, end:int, text:str)` tuples. Every eval method and the late-chunk encoder consume these spans.

- [ ] **Step 1: Add the failing test cell** in `nbs/02_data.ipynb`

```python
_t = "First sentence here. Second sentence follows. Third one ends it."
_spans = chunk_spans(_t)
assert len(_spans) >= 1
for s,e,txt in _spans:
    assert _t[s:e] == txt          # offsets index back into the original text exactly
assert _spans[0][0] == 0           # first chunk starts at 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run nbdev_test --file_glob "02_data.ipynb"`
Expected: FAIL with `NameError: name 'chunk_spans' is not defined`

- [ ] **Step 3: Add the implementation cell** (marked `#| export`) directly above the test

```python
#| export
def chunk_spans(text:str,            # text to split
                chunker:BaseChunker=None
) -> L:
    'Split text into chunks, returning (start_char, end_char, text) spans into the original text.'
    r = chunker or FastChunker()
    return L(r(text)).map(lambda c: (c.start_index, c.end_index, c.text))
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run nbdev_test --file_glob "02_data.ipynb"`
Expected: PASS

- [ ] **Step 5: Export + clean + commit**

```bash
uv run nbdev_prepare
git add nbs/02_data.ipynb litesearch/data.py litesearch/_modidx.py
git commit -m "feat: chunk_spans helper returning char offsets for late chunking"
```

---

### Task 2: `LateChunkFastEncode` (single-pass late chunking)

**Files:**
- Modify: `nbs/03_utils.ipynb` (add cells after the `FastEncode` class; also add its name to the `__all__` cell)
- Regenerates: `litesearch/utils.py`, `litesearch/_modidx.py`

**Interfaces:**
- Consumes: `FastEncode` (`self.tok`, `self.sess`, `self._input_names`, `self.tti`, `self.prompt`, `self.normalize`, `self.dtype`), `chunk_spans` from Task 1.
- Produces:
  - `LateChunkFastEncode._token_embeddings(text:str) -> (token_embs:np.ndarray (seq,dim), offsets:list[(int,int)], mask:np.ndarray (seq,))`
  - `LateChunkFastEncode.encode_late_chunks(text:str, spans:list[(int,int)], prompt:str=None) -> np.ndarray (n_spans, dim)`

- [ ] **Step 1: Add the failing test cell** in `nbs/03_utils.ipynb`

```python
_lc = LateChunkFastEncode(model_dict=nomic_text_v15, max_seq_len=2048)
_doc = "Cats are small carnivorous mammals. They are kept as pets worldwide. Dogs are loyal companion animals."
_spans = [(s,e) for s,e,_ in chunk_spans(_doc)]
_embs = _lc.encode_late_chunks(_doc, _spans)
assert _embs.shape == (len(_spans), _lc.sess.get_outputs()[0].shape[-1])
import numpy as np
assert np.allclose(np.linalg.norm(_embs.astype(np.float32), axis=1), 1.0, atol=1e-2)  # normalized
# late-chunk full-span vector is close to (not equal to) the plain document embedding
_full = _lc.encode_late_chunks(_doc, [(0, len(_doc))])[0].astype(np.float32)
_plain = _lc.encode_document([_doc])[0].astype(np.float32)
assert float(_full @ _plain) > 0.9
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run nbdev_test --file_glob "03_utils.ipynb"`
Expected: FAIL with `NameError: name 'LateChunkFastEncode' is not defined`

- [ ] **Step 3: Add the implementation cell** (marked `#| export`)

```python
#| export
class LateChunkFastEncode(FastEncode):
    'Embed the whole doc once; mean-pool per chunk span so each chunk vector keeps full-doc context.'
    def _token_embeddings(self, text:str):
        'Single forward pass; returns (token_embeddings, char offsets, attention mask).'
        enc = self.tok.encode(text, add_special_tokens=True)
        ids = np.array([enc.ids], dtype=np.int64)
        msk = np.array([enc.attention_mask], dtype=np.int64)
        inp = dict(input_ids=ids)
        if 'attention_mask' in self._input_names: inp['attention_mask'] = msk
        if self.tti and 'token_type_ids' in self._input_names: inp['token_type_ids'] = np.zeros(ids.shape, dtype=np.int64)
        token_embs = self.sess.run(None, inp)[0][0]
        return token_embs, enc.offsets, msk[0]

    def encode_late_chunks(self, text:str, spans:list, prompt:str=None):
        'Pool per (start,end) char span over full-doc token embeddings.'
        prompt = prompt if prompt is not None else self.prompt.get('document', None)
        full = prompt.format(text=text) if prompt else text
        prefix_len = len(full) - len(text)
        token_embs, offsets, msk = self._token_embeddings(full)
        out = np.zeros((len(spans), token_embs.shape[-1]), dtype=np.float32)
        for i,(cs,ce) in enumerate(spans):
            cs, ce = cs+prefix_len, ce+prefix_len
            idx = [t for t,(s,e) in enumerate(offsets) if msk[t] and e>cs and s<ce]
            if idx: out[i] = token_embs[idx].mean(axis=0)
        if self.normalize: out = out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-12, None)
        return out.astype(self.dtype)
```

- [ ] **Step 4: Add `LateChunkFastEncode` to the `__all__` list** in the notebook's first export cell (the `__all__ = [...]` cell), appending the string `'LateChunkFastEncode'`.

- [ ] **Step 5: Run to verify it passes**

Run: `uv run nbdev_test --file_glob "03_utils.ipynb"`
Expected: PASS

- [ ] **Step 6: Export + clean + commit**

```bash
uv run nbdev_prepare
git add nbs/03_utils.ipynb litesearch/utils.py litesearch/_modidx.py
git commit -m "feat: LateChunkFastEncode single-pass late chunking"
```

---

### Task 3: `LongLateChunkFastEncode` (windowed, token-weighted)

**Files:**
- Modify: `nbs/03_utils.ipynb` (add after `LateChunkFastEncode`; add name to `__all__`)
- Regenerates: `litesearch/utils.py`, `litesearch/_modidx.py`

**Interfaces:**
- Consumes: `LateChunkFastEncode` (Task 2), `self.max_seq_len`.
- Produces:
  - `LongLateChunkFastEncode._make_windows(text:str, window_chars:int, overlap_chars:int) -> list[(int,int)]`
  - `LongLateChunkFastEncode.encode_long_document(text:str, spans:list[(int,int)], window_chars:int=None, overlap_chars:int=None, prompt:str=None) -> np.ndarray (n_spans, dim)`

- [ ] **Step 1: Add the failing test cell** in `nbs/03_utils.ipynb`

```python
_llc = LongLateChunkFastEncode(model_dict=nomic_text_v15, max_seq_len=256)  # small ctx forces multiple windows
_long = (" ".join(f"Sentence number {i} about topic {i%5}." for i in range(200)))
_spans = [(s,e) for s,e,_ in chunk_spans(_long)]
_wins = _llc._make_windows(_long, 400, 80)
assert len(_wins) > 1                       # long text spans several windows
assert _wins[0][0] == 0 and _wins[-1][1] == len(_long)   # covers head and tail
_embs = _llc.encode_long_document(_long, _spans, window_chars=400, overlap_chars=80)
assert _embs.shape == (len(_spans), _llc.sess.get_outputs()[0].shape[-1])
import numpy as np
assert np.all(np.linalg.norm(_embs.astype(np.float32), axis=1) > 0)   # every chunk got a vector
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run nbdev_test --file_glob "03_utils.ipynb"`
Expected: FAIL with `NameError: name 'LongLateChunkFastEncode' is not defined`

- [ ] **Step 3: Add the implementation cell** (marked `#| export`)

```python
#| export
class LongLateChunkFastEncode(LateChunkFastEncode):
    'Late chunking for docs beyond the context window via overlapping windows and token-weighted averaging.'
    def _make_windows(self, text, window_chars, overlap_chars):
        'Stepped char windows covering the whole text including the tail.'
        step = max(window_chars - overlap_chars, 1)
        starts = list(range(0, max(len(text) - overlap_chars, 1), step))
        windows = [(s, min(s + window_chars, len(text))) for s in starts]
        if windows[-1][1] < len(text): windows.append((max(len(text) - window_chars, 0), len(text)))
        return windows

    def encode_long_document(self, text, spans, window_chars=None, overlap_chars=None, prompt=None):
        'Pool each span within every overlapping window; combine by token-weighted average.'
        max_tok = (self.max_seq_len or 512) - 8
        window_chars = window_chars or int(max_tok * 3.5)
        overlap_chars = overlap_chars if overlap_chars is not None else window_chars // 5
        windows = self._make_windows(text, window_chars, overlap_chars)
        tmpl = prompt if prompt is not None else (self.prompt.get('document', None) or '{text}')
        chunk_sums, chunk_weights, dim = None, np.zeros(len(spans)), None
        for ws,we in windows:
            win_text = text[ws:we]
            full = tmpl.format(text=win_text)
            token_embs, offsets, msk = self._token_embeddings(full)
            prefix_len = len(full) - len(win_text)
            if dim is None:
                dim = token_embs.shape[-1]
                chunk_sums = np.zeros((len(spans), dim), dtype=np.float32)
            for i,(cs,ce) in enumerate(spans):
                local_cs, local_ce = cs-ws, ce-ws
                if local_ce <= 0 or local_cs >= (we-ws): continue
                local_cs = max(local_cs,0)+prefix_len
                local_ce = min(local_ce,we-ws)+prefix_len
                idx = [t for t,(s,e) in enumerate(offsets) if msk[t] and e>local_cs and s<local_ce]
                if not idx: continue
                w = len(idx)
                chunk_sums[i] += token_embs[idx].mean(axis=0) * w
                chunk_weights[i] += w
        out = np.zeros_like(chunk_sums)
        ok = chunk_weights > 0
        out[ok] = chunk_sums[ok] / chunk_weights[ok, None]
        if self.normalize: out = out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-12, None)
        return out.astype(self.dtype)
```

- [ ] **Step 4: Add `LongLateChunkFastEncode` to the `__all__` list.**

- [ ] **Step 5: Run to verify it passes**

Run: `uv run nbdev_test --file_glob "03_utils.ipynb"`
Expected: PASS

- [ ] **Step 6: Export + clean + commit**

```bash
uv run nbdev_prepare
git add nbs/03_utils.ipynb litesearch/utils.py litesearch/_modidx.py
git commit -m "feat: LongLateChunkFastEncode windowed late chunking"
```

---

### Task 4: `AutoLateChunkFastEncode` (length-routed)

**Files:**
- Modify: `nbs/03_utils.ipynb` (add after `LongLateChunkFastEncode`; add name to `__all__`)
- Regenerates: `litesearch/utils.py`, `litesearch/_modidx.py`

**Interfaces:**
- Consumes: `LongLateChunkFastEncode` (Task 3), `self.tok`, `self.max_seq_len`.
- Produces:
  - `AutoLateChunkFastEncode._count_tokens(text:str) -> int`
  - `AutoLateChunkFastEncode.encode_auto(text:str, spans:list[(int,int)], prompt:str=None, long_ratio:float=4.0, **kw) -> (np.ndarray (n_spans,dim), tier:str)` where `tier` ∈ {`'normal'`,`'long'`,`'longer'`}

- [ ] **Step 1: Add the failing test cell** in `nbs/03_utils.ipynb`

```python
_a = AutoLateChunkFastEncode(model_dict=nomic_text_v15, max_seq_len=128)  # tiny ctx to exercise all tiers
_short = "One short sentence."
_med = " ".join(["Filler sentence about things."]*60)
_huge = " ".join(["Filler sentence about things."]*800)
def _sp(t): return [(s,e) for s,e,_ in chunk_spans(t)]
assert _a.encode_auto(_short, _sp(_short))[1] == 'normal'
assert _a.encode_auto(_med, _sp(_med))[1] == 'long'
assert _a.encode_auto(_huge, _sp(_huge))[1] == 'longer'
_embs,_tier = _a.encode_auto(_med, _sp(_med))
assert _embs.shape[0] == len(_sp(_med))
# token count is truncation-free and larger than the small ctx
assert _a._count_tokens(_huge) > 128
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run nbdev_test --file_glob "03_utils.ipynb"`
Expected: FAIL with `NameError: name 'AutoLateChunkFastEncode' is not defined`

- [ ] **Step 3: Add the implementation cell** (marked `#| export`)

```python
#| export
class AutoLateChunkFastEncode(LongLateChunkFastEncode):
    'Route to single-pass / windowed / tight-windowed late chunking by document token count.'
    def _count_tokens(self, text):
        'Token count with truncation disabled (tokenizer only, no ONNX run).'
        trunc = self.tok.truncation
        self.tok.no_truncation()
        n = len(self.tok.encode(text, add_special_tokens=True).ids)
        if trunc: self.tok.enable_truncation(**{k:trunc[k] for k in ('max_length','stride','strategy','direction') if k in trunc})
        return n

    def encode_auto(self, text, spans, prompt=None, long_ratio=4.0, **kw):
        'Return (embeddings, tier); tier is normal / long / longer by token count vs context window.'
        max_tok = self.max_seq_len or 512
        n_tok = self._count_tokens(text)
        if n_tok <= max_tok - 8:
            return self.encode_late_chunks(text, spans, prompt=prompt), 'normal'
        if n_tok <= max_tok * long_ratio:
            return self.encode_long_document(text, spans, prompt=prompt, **kw), 'long'
        max_chars = int((max_tok - 8) * 3.5)
        return self.encode_long_document(text, spans, prompt=prompt,
            window_chars=kw.pop('window_chars', max_chars),
            overlap_chars=kw.pop('overlap_chars', max_chars // 8), **kw), 'longer'
```

- [ ] **Step 4: Add `AutoLateChunkFastEncode` to the `__all__` list.**

- [ ] **Step 5: Run to verify it passes**

Run: `uv run nbdev_test --file_glob "03_utils.ipynb"`
Expected: PASS

- [ ] **Step 6: Export + clean + commit**

```bash
uv run nbdev_prepare
git add nbs/03_utils.ipynb litesearch/utils.py litesearch/_modidx.py
git commit -m "feat: AutoLateChunkFastEncode length-routed late chunking"
```

---

### Task 5: Eval notebook scaffold + LongEmbed loader

**Files:**
- Create: `nbs/04_latechunk_eval.ipynb`
- Modify: `pyproject.toml` (add `datasets` dep), `settings.ini` if it lists nbs (nbdev auto-discovers; no change usually)
- Regenerates: nothing exported (this notebook is `#| eval: false`; not part of the package `__all__`)

**Interfaces:**
- Consumes: HF `datasets`.
- Produces:
  - `load_longembed(task:str, max_docs:int=200, max_queries:int=100) -> (corpus:dict[str,str], queries:dict[str,str], qrels:dict[str,set[str]])`
  - module constant `TASKS = ['narrativeqa', '2wikimqa']`

- [ ] **Step 1: Add the `datasets` dependency**

```bash
uv add datasets
```
Expected: `pyproject.toml` gains a `datasets` entry under dependencies.

- [ ] **Step 2: Create `nbs/04_latechunk_eval.ipynb`** with a frontmatter/first cell:

```python
#| default_exp latechunk_eval
```
and a second raw/markdown cell titled `# Late Chunking Accuracy Eval`, then a cell:

```python
#| eval: false
#| hide
from litesearch import database
from litesearch.utils import FastEncode, AutoLateChunkFastEncode, nomic_text_v15
from litesearch.data import chunk_spans
import numpy as np, json
from datasets import load_dataset
```

- [ ] **Step 3: Add a DISCOVERY cell** (run once to confirm the dataset schema before writing the loader). This is a real inspection step, not a placeholder:

```python
#| eval: false
_probe = load_dataset('dwzhu/LongEmbed', 'narrativeqa')
print(_probe)                      # inspect split names
for split in _probe: print(split, _probe[split].column_names, _probe[split][0].keys())
```
Record the actual split names and columns (expected: a `corpus` split with `id`/`text`, a `queries` split with `id`/`text`, and a `qrels` split with `qid`/`doc_id` — adjust the loader in Step 4 to the confirmed names if they differ).

- [ ] **Step 4: Add the failing test cell**

```python
#| eval: false
_corpus,_queries,_qrels = load_longembed('narrativeqa', max_docs=20, max_queries=5)
assert isinstance(_corpus, dict) and isinstance(_queries, dict) and isinstance(_qrels, dict)
assert len(_corpus) <= 20 and len(_queries) <= 5
# every qrel points at a doc actually in the (capped) corpus
for qid, docs in _qrels.items():
    assert qid in _queries
    assert all(d in _corpus for d in docs)
assert len(_qrels) >= 1             # at least one query has a judged, in-corpus doc
```

- [ ] **Step 5: Run to verify it fails**

Run: execute this cell in a Python kernel (`mcp__safepyrun__run_python` per project rule, or run the notebook cell directly). `nbdev_test` is NOT used here — these cells are `#| eval: false` and are excluded from CI; they need network/model downloads and are verified interactively during development.
Expected: FAIL with `NameError: name 'load_longembed' is not defined`

- [ ] **Step 6: Add the loader cell** (above the test), using the schema confirmed in Step 3:

```python
#| eval: false
def load_longembed(task, max_docs=200, max_queries=100):
    'Load a LongEmbed task capped to max_docs/max_queries; keep only queries whose judged doc survives the cap.'
    ds = load_dataset('dwzhu/LongEmbed', task)
    corpus = {str(r['id']): r['text'] for r in ds['corpus'].select(range(min(max_docs, len(ds['corpus']))))}
    queries = {str(r['id']): r['text'] for r in ds['queries']}
    qrels = {}
    for r in ds['qrels']:
        qid, did = str(r['qid']), str(r['doc_id'])
        if did in corpus and qid in queries: qrels.setdefault(qid, set()).add(did)
    queries = {q: queries[q] for q in list(qrels)[:max_queries]}
    qrels = {q: qrels[q] for q in queries}
    return corpus, queries, qrels
```

- [ ] **Step 7: Run to verify it passes**

Run: execute this cell in a Python kernel (`mcp__safepyrun__run_python` per project rule, or run the notebook cell directly) — NOT `nbdev_test` (cell is `#| eval: false`, excluded from CI).
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add nbs/04_latechunk_eval.ipynb pyproject.toml uv.lock
git commit -m "feat: LongEmbed loader for late-chunking eval"
```

---

### Task 6: Metrics (nDCG@10, Recall@k) + chunk→doc aggregation

**Files:**
- Modify: `nbs/04_latechunk_eval.ipynb`

**Interfaces:**
- Consumes: nothing external.
- Produces:
  - `agg_docs(hits:list[dict]) -> list[str]` — ranked unique `doc_id`s from litesearch hits (best rank kept), reading `doc_id` from each hit's `metadata` JSON.
  - `ndcg_at_k(ranked:list[str], relevant:set[str], k:int=10) -> float`
  - `recall_at_k(ranked:list[str], relevant:set[str], k:int) -> float`

- [ ] **Step 1: Add the failing test cell**

```python
#| eval: false
_hits = [{'metadata': json.dumps({'doc_id':'A','chunk_idx':0})},
         {'metadata': json.dumps({'doc_id':'B','chunk_idx':3})},
         {'metadata': json.dumps({'doc_id':'A','chunk_idx':1})}]
assert agg_docs(_hits) == ['A','B']                     # dedup, keep first occurrence
assert recall_at_k(['A','B','C'], {'A','C'}, 3) == 1.0
assert recall_at_k(['A','B','C'], {'A','C'}, 1) == 0.5
assert abs(ndcg_at_k(['A','B'], {'A'}, 10) - 1.0) < 1e-9   # relevant doc ranked first -> perfect
assert ndcg_at_k(['B','A'], {'A'}, 10) < 1.0               # relevant doc ranked second -> discounted
```

- [ ] **Step 2: Run to verify it fails**

Run: execute this cell in a Python kernel (`mcp__safepyrun__run_python` per project rule, or run the notebook cell directly) — NOT `nbdev_test` (cell is `#| eval: false`, excluded from CI).
Expected: FAIL with `NameError: name 'agg_docs' is not defined`

- [ ] **Step 3: Add the implementation cell**

```python
#| eval: false
def agg_docs(hits):
    'Collapse ranked chunk hits to unique parent doc_ids, keeping best rank.'
    seen, out = set(), []
    for h in hits:
        d = json.loads(h['metadata'])['doc_id']
        if d not in seen: seen.add(d); out.append(d)
    return out

def recall_at_k(ranked, relevant, k):
    'Fraction of relevant docs present in the top-k ranking.'
    if not relevant: return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)

def ndcg_at_k(ranked, relevant, k=10):
    'Binary-relevance nDCG@k.'
    dcg = sum(1.0/np.log2(i+2) for i,d in enumerate(ranked[:k]) if d in relevant)
    idcg = sum(1.0/np.log2(i+2) for i in range(min(len(relevant), k)))
    return float(dcg/idcg) if idcg else 0.0
```

- [ ] **Step 4: Run to verify it passes**

Run: execute this cell in a Python kernel (`mcp__safepyrun__run_python` per project rule, or run the notebook cell directly) — NOT `nbdev_test` (cell is `#| eval: false`, excluded from CI).
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add nbs/04_latechunk_eval.ipynb
git commit -m "feat: eval metrics and chunk-to-doc aggregation"
```

---

### Task 7: Index builders for the four methods

**Files:**
- Modify: `nbs/04_latechunk_eval.ipynb`
- Modify: `pyproject.toml` (add `rishi` dep)

**Interfaces:**
- Consumes: `database`, `FastEncode`, `AutoLateChunkFastEncode`, `chunk_spans`, `nomic_text_v15`, `rishi.core.Chat`/`resp_text`.
- Produces (each returns a searchable litesearch `Database` whose store rows carry `metadata={'doc_id','chunk_idx'}`):
  - `build_naive(corpus, enc) -> Database`
  - `build_fulldoc(corpus, enc) -> Database`
  - `build_latechunk(corpus, lc_enc) -> Database`
  - `build_contextual(corpus, enc, chat) -> Database`
  - `situate(chat, doc, chunk) -> str` (cached blurb)

- [ ] **Step 1: Add the `rishi` dependency**

```bash
uv add rishi
```

- [ ] **Step 2: Add the failing test cell** (uses a tiny 2-doc corpus so it runs fast; `rishi`/Gemma download may take minutes on first run)

```python
#| eval: false
_corpus = {'d1':'Cats are small mammals kept as pets. They purr when content.',
           'd2':'The Eiffel Tower is in Paris. It was built in 1889 for the World Fair.'}
_enc = FastEncode(model_dict=nomic_text_v15, max_seq_len=2048)
_lc  = AutoLateChunkFastEncode(model_dict=nomic_text_v15, max_seq_len=2048)
for _db in (build_naive(_corpus,_enc), build_fulldoc(_corpus,_enc), build_latechunk(_corpus,_lc)):
    _q = 'where is the eiffel tower'
    _hits = _db.search(_q, _enc.encode_query(_q).tobytes(), limit=10)
    assert agg_docs(_hits)[0] == 'd2'          # obvious relevant doc ranks first
```

- [ ] **Step 3: Run to verify it fails**

Run: execute this cell in a Python kernel (`mcp__safepyrun__run_python` per project rule, or run the notebook cell directly) — NOT `nbdev_test` (cell is `#| eval: false`, excluded from CI).
Expected: FAIL with `NameError: name 'build_naive' is not defined`

- [ ] **Step 4: Add the implementation cell**

```python
#| eval: false
def _store(corpus, rows):
    'Build an in-memory litesearch store from prepared rows.'
    db = database()
    st = db.get_store()
    st.insert_all(rows)
    return db

def build_naive(corpus, enc):
    'Chunk-then-embed each chunk independently.'
    rows = []
    for did,txt in corpus.items():
        spans = chunk_spans(txt)
        embs = enc.encode_document([t for _,_,t in spans])
        for ci,((_,_,t),e) in enumerate(zip(spans,embs)):
            rows.append({'content':t,'embedding':e.tobytes(),'metadata':json.dumps({'doc_id':did,'chunk_idx':ci})})
    return _store(corpus, rows)

def build_fulldoc(corpus, enc):
    'One embedding per whole document.'
    rows = []
    for did,txt in corpus.items():
        e = enc.encode_document([txt])[0]
        rows.append({'content':txt,'embedding':e.tobytes(),'metadata':json.dumps({'doc_id':did,'chunk_idx':0})})
    return _store(corpus, rows)

def build_latechunk(corpus, lc_enc):
    'Late chunking: embed whole doc, pool per chunk span.'
    rows = []
    for did,txt in corpus.items():
        spans = chunk_spans(txt)
        embs,_ = lc_enc.encode_auto(txt, [(s,e) for s,e,_ in spans])
        for ci,((_,_,t),e) in enumerate(zip(spans,embs)):
            rows.append({'content':t,'embedding':e.tobytes(),'metadata':json.dumps({'doc_id':did,'chunk_idx':ci})})
    return _store(corpus, rows)

_SIT_CACHE = {}
def situate(chat, doc, chunk):
    'Generate (and cache) a short doc-situating blurb for a chunk via local Gemma.'
    key = (hash(doc), hash(chunk))
    if key in _SIT_CACHE: return _SIT_CACHE[key]
    from rishi.core import resp_text
    prompt = (f"<document>\n{doc}\n</document>\nHere is a chunk from it:\n<chunk>\n{chunk}\n</chunk>\n"
              "Give a short sentence situating this chunk within the document for search. Answer with only that sentence.")
    blurb = resp_text(chat(prompt)).strip()
    _SIT_CACHE[key] = blurb
    return blurb

def build_contextual(corpus, enc, chat):
    'Contextual retrieval: prepend a Gemma-generated situating blurb to each chunk before embedding.'
    rows = []
    for did,txt in corpus.items():
        spans = chunk_spans(txt)
        texts = [f"{situate(chat, txt, t)}\n{t}" for _,_,t in spans]
        embs = enc.encode_document(texts)
        for ci,(t,e) in enumerate(zip(texts,embs)):
            rows.append({'content':t,'embedding':e.tobytes(),'metadata':json.dumps({'doc_id':did,'chunk_idx':ci})})
    return _store(corpus, rows)
```

- [ ] **Step 5: Run to verify it passes**

Run: execute this cell in a Python kernel (`mcp__safepyrun__run_python` per project rule, or run the notebook cell directly) — NOT `nbdev_test` (cell is `#| eval: false`, excluded from CI).
Expected: PASS (naive/fulldoc/latechunk asserted; `build_contextual` is defined and exercised in Task 8)

- [ ] **Step 6: Commit**

```bash
git add nbs/04_latechunk_eval.ipynb pyproject.toml uv.lock
git commit -m "feat: four index builders (naive, fulldoc, latechunk, contextual)"
```

---

### Task 8: Run comparison + results table

**Files:**
- Modify: `nbs/04_latechunk_eval.ipynb`

**Interfaces:**
- Consumes: everything from Tasks 5–7.
- Produces:
  - `evaluate(db, queries, qrels, enc) -> dict` with keys `ndcg@10`, `recall@1`, `recall@5`, `recall@10` (query-averaged).
  - `run_comparison(tasks=TASKS, max_docs=..., max_queries=...) -> dict[method -> dict[metric -> float]]` and a printed table.

- [ ] **Step 1: Add the failing test cell**

```python
#| eval: false
_db = build_naive(_corpus, _enc)
_r = evaluate(_db, {'q1':'where is the eiffel tower'}, {'q1':{'d2'}}, _enc)
assert set(_r) == {'ndcg@10','recall@1','recall@5','recall@10'}
assert 0.0 <= _r['ndcg@10'] <= 1.0 and _r['recall@1'] == 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: execute this cell in a Python kernel (`mcp__safepyrun__run_python` per project rule, or run the notebook cell directly) — NOT `nbdev_test` (cell is `#| eval: false`, excluded from CI).
Expected: FAIL with `NameError: name 'evaluate' is not defined`

- [ ] **Step 3: Add the implementation cell**

```python
#| eval: false
def evaluate(db, queries, qrels, enc):
    'Query-averaged nDCG@10 and Recall@{1,5,10} over a built store.'
    ndcg=r1=r5=r10=0.0; n=len(queries)
    for qid,qtext in queries.items():
        ranked = agg_docs(db.search(qtext, enc.encode_query(qtext).tobytes(), limit=100))
        rel = qrels[qid]
        ndcg += ndcg_at_k(ranked, rel, 10)
        r1 += recall_at_k(ranked, rel, 1)
        r5 += recall_at_k(ranked, rel, 5)
        r10 += recall_at_k(ranked, rel, 10)
    return {'ndcg@10':ndcg/n,'recall@1':r1/n,'recall@5':r5/n,'recall@10':r10/n}

def run_comparison(tasks=TASKS, max_docs=200, max_queries=100, use_contextual=True):
    'Build all methods per task, evaluate, and print a method x metric table (averaged over tasks).'
    enc = FastEncode(model_dict=nomic_text_v15, max_seq_len=8192)
    lc  = AutoLateChunkFastEncode(model_dict=nomic_text_v15, max_seq_len=8192)
    chat = None
    if use_contextual:
        from rishi.core import Chat
        chat = Chat(cache_dir='.cache/litertlm')
    methods = {'naive':lambda c: build_naive(c,enc),
               'fulldoc':lambda c: build_fulldoc(c,enc),
               'latechunk':lambda c: build_latechunk(c,lc)}
    if use_contextual: methods['contextual'] = lambda c: build_contextual(c,enc,chat)
    agg = {m:{'ndcg@10':0,'recall@1':0,'recall@5':0,'recall@10':0} for m in methods}
    for task in tasks:
        corpus,queries,qrels = load_longembed(task, max_docs, max_queries)
        for m,build in methods.items():
            res = evaluate(build(corpus), queries, qrels, enc)
            for k in agg[m]: agg[m][k] += res[k]/len(tasks)
    hdr = f"{'method':<12}" + "".join(f"{k:>10}" for k in ('ndcg@10','recall@1','recall@5','recall@10'))
    print(hdr); print('-'*len(hdr))
    for m,r in agg.items(): print(f"{m:<12}" + "".join(f"{r[k]:>10.4f}" for k in ('ndcg@10','recall@1','recall@5','recall@10')))
    return agg
```

- [ ] **Step 4: Run to verify it passes**

Run: execute this cell in a Python kernel (`mcp__safepyrun__run_python` per project rule, or run the notebook cell directly) — NOT `nbdev_test` (cell is `#| eval: false`, excluded from CI).
Expected: PASS

- [ ] **Step 5: Add a final run cell** (not asserted — this is the actual experiment; slow, downloads models/data):

```python
#| eval: false
results = run_comparison(max_docs=100, max_queries=50)
results
```

- [ ] **Step 6: Commit**

```bash
git add nbs/04_latechunk_eval.ipynb
git commit -m "feat: run late-chunking comparison and print results table"
```

---

## Self-Review

**Spec coverage:**
- Encoders (LateChunk / LongLateChunk / AutoLateChunk) → Tasks 2, 3, 4. ✓
- `chunk_spans` helper → Task 1. ✓
- Notebook test cells → each encoder/eval task has assert cells. ✓
- LongEmbed loader, NarrativeQA + 2WikiMQA, doc cap → Task 5 (`TASKS`, `max_docs`). ✓
- Four methods sharing chunk_spans → Task 7. ✓
- rishi contextual baseline w/ cache → Task 7 (`situate` + `_SIT_CACHE`). ✓
- litesearch hybrid RRF scoring → Task 8 (`db.search` default). ✓
- chunk→doc max-rank aggregation → Task 6 (`agg_docs`). ✓
- nDCG@10 + Recall@{1,5,10} table → Tasks 6, 8. ✓
- `uv add rishi` / `datasets` → Tasks 5, 7. ✓

**Placeholder scan:** No TBD/TODO. The one inspection step (Task 5 Step 3) is a real, executable discovery cell with a command and expected output, not a deferred decision.

**Type consistency:** `chunk_spans` returns `(start,end,text)`; encoders take `(start,end)` spans (builders strip the text) — consistent. `agg_docs` reads `metadata` JSON with `doc_id`; all builders write that key. `encode_auto` returns `(embs, tier)`; `build_latechunk` unpacks both. `evaluate` metric keys match `run_comparison` aggregation keys.

**Execution model:** Tasks 1–4 add real exported code with exported tests; verify them with `nbdev_test`/`nbdev_prepare` (they run in CI). Every cell in the eval notebook (`nbs/04_latechunk_eval.ipynb`) is `#| eval: false`, so `nbdev_prepare` never executes it — this keeps model/dataset downloads out of CI. The executor therefore verifies Tasks 5–8 by running each cell directly in a Python kernel (`mcp__safepyrun__run_python` per the project rule) and confirming no `AssertionError`, rather than via `nbdev_test`. The final `run_comparison(...)` cell is the actual experiment and is run once, manually.
