"""The encoders under test, wrapped so every store in the eval is float16 and comparable.

Five, spanning three orders of magnitude of cost per chunk:

| name          | params | dim | ctx  | what it is                                        |
|---------------|--------|-----|------|---------------------------------------------------|
| `potion-32M`  | 32M*   | 512 |  ∞   | model2vec static vectors — a lookup table, no ONNX |
| `bge-small`   | 33M    | 384 |  512 | the ordinary small ONNX retriever                  |
| `jina-v2-sm`  | 33M    | 512 | 8192 | small *and* long-context — the late-chunking model  |
| `egemma-300m` | 300M   | 768 | 2048 | a real quality ceiling on CPU                       |
| `nomic-v1.5`  | 137M   | 768 | 8192 | long-context quality ceiling                        |

*static: 32M is the distilled vocabulary table, not a forward pass — there is no attention, so
there is also no context of any kind. It cannot do late chunking, and that is the point of having it
here: it is the floor that everything else has to beat to justify its latency.

**A dtype trap worth naming.** `model2vec` returns float32 and `evals.onnx.FastEncode` returns float16, while
`Database.search` and `vec_search` both default to `dtype=np.float16`. Store float32 bytes and
search them as float16 and every distance is computed over reinterpreted bytes at twice the
dimension — no error is raised, and the vector leg silently returns noise. Everything here is
normalised to float16 on the way out for exactly that reason.
"""
import time
import numpy as np
from fastcore.all import AttrDict

from litesearch.utils import static_embedder
from .onnx import FastEncode, AutoLateChunkFastEncode, nomic_text_v15, embedding_gemma

DT = np.float16

_plain = AttrDict(document='{text}', query='{text}')
bge_small_md  = AttrDict(model='onnx-community/bge-small-en-v1.5-ONNX', onnx_path='onnx/model.onnx',
                         prompt=_plain, tti=True)
jina_v2_sm_md = AttrDict(model='jinaai/jina-embeddings-v2-small-en', onnx_path='model.onnx',
                         prompt=_plain, tti=True)


class Enc:
    'One encoder, normalised: `doc`/`qry` return float16, `late` exists only if the model has context.'
    def __init__(self, name, dim, max_seq, doc, qry, late=None, kind='onnx', note=''):
        self.name, self.dim, self.max_seq, self.kind, self.note = name, dim, max_seq, kind, note
        self._doc, self._qry, self.late = doc, qry, late

    def doc(self, texts):
        v = self._doc(list(texts))
        return np.asarray(v, dtype=DT)

    def qry(self, texts):
        v = self._qry(list(texts))
        return np.asarray(v, dtype=DT)

    def __repr__(self): return f'<Enc {self.name} dim={self.dim} ctx={self.max_seq} {self.kind}>'


def _onnx(name, md, dim, max_seq, late=True, note=''):
    fe = FastEncode(model_dict=md, max_seq_len=max_seq, batch_size=32, dtype=DT)
    assert fe.sess is not None, f'{name}: ONNX session failed to load'
    lc = AutoLateChunkFastEncode(model_dict=md, max_seq_len=max_seq, dtype=DT) if late else None
    return Enc(name, dim, max_seq, fe.encode_document, fe.encode_query, lc, 'onnx', note)

def _static(name, repo, dim, note=''):
    sm = static_embedder(repo)
    f = lambda ts: sm.encode(list(ts))
    return Enc(name, dim, 10**9, f, f, None, 'static', note)


BUILDERS = {
    'potion-32M':  lambda: _static('potion-32M', 'minishlab/potion-retrieval-32M', 512,
                                   'static lookup table; no context, no late chunking'),
    'bge-small':   lambda: _onnx('bge-small', bge_small_md, 384, 512,
                                 note='the default small retriever'),
    'jina-v2-sm':  lambda: _onnx('jina-v2-sm', jina_v2_sm_md, 512, 8192,
                                 note='small and long-context; the late-chunking model'),
    'egemma-300m': lambda: _onnx('egemma-300m', embedding_gemma, 768, 2048,
                                 note='quality ceiling, 10x the cost'),
    'nomic-v1.5':  lambda: _onnx('nomic-v1.5', nomic_text_v15, 768, 8192,
                                 note='long-context quality ceiling'),
}

_LOADED = {}
def enc(name):
    'Memoised encoder — ONNX sessions are expensive to build and thread-safe to reuse.'
    if name not in _LOADED: _LOADED[name] = BUILDERS[name]()
    return _LOADED[name]


def throughput(name, n=64, chars=512):
    'Chunks encoded per second on this machine, at a realistic chunk length.'
    e = enc(name)
    txt = ['the influence of the planet upon the native is described as follows. ' * (chars//68)][0]
    e.doc([txt[:chars]])                                   # warm up
    t = time.time(); v = e.doc([txt[:chars]]*n); el = time.time()-t
    return dict(encoder=name, dim=int(v.shape[1]), ctx=e.max_seq, kind=e.kind,
                chunks_per_s=round(n/el, 1), dtype=str(v.dtype))


if __name__ == '__main__':
    for n in BUILDERS:
        try:
            r = throughput(n)
            print(f"{r['encoder']:<13} dim={r['dim']:<4} ctx={r['ctx']:<6} {r['kind']:<7} "
                  f"{r['chunks_per_s']:>7} chunks/s  {r['dtype']}")
        except Exception as ex:
            print(f'{n:<13} FAILED {type(ex).__name__}: {ex}')
