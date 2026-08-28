"""Late chunking, kept here because the measurement rejected it and `evals/` is the only caller.

Embedding a chunk inside its whole document rather than on its own is a good idea that this corpus
does not reward: -0.033 to -0.053 weighted MRR against plain chunk embedding. `evals/RESULTS.md`
has the method. The code stays so the number can be reproduced. Not shipped in the wheel.
"""

import numpy as np
from fastcore.all import L
from litesearch.utils import FastEncode

class LateChunkFastEncode(FastEncode):
	'Embed the whole doc once; mean-pool per chunk span so each chunk vector keeps full-doc context.'
	def _token_embeddings(self, text:str):
		'Single forward pass; returns (token_embeddings, char offsets, attention mask with special tokens zeroed).'
		enc = self.tok.encode(text, add_special_tokens=True)
		ids = np.array([enc.ids], dtype=np.int64)
		msk = np.array([enc.attention_mask], dtype=np.int64)
		inp = dict(input_ids=ids)
		if 'attention_mask' in self._input_names: inp['attention_mask'] = msk
		if self.tti and 'token_type_ids' in self._input_names: inp['token_type_ids'] = np.zeros(ids.shape, dtype=np.int64)
		token_embs = self.sess.run(None, inp)[0][0]
		pool_msk = msk[0] * (1 - np.array(enc.special_tokens_mask, dtype=np.int64))
		return token_embs, enc.offsets, pool_msk

	def encode_late_chunks(self, text:str, spans:list, prompt:str=None):
		'Pool per (start,end) char span over full-doc token embeddings. Truncates past max_seq_len; use encode_auto for long docs.'
		prompt = prompt if prompt is not None else self.prompt.get('document', None)
		full = prompt.format(text=text) if prompt else text
		prefix_len = len(full) - len(text)
		token_embs, offsets, msk = self._token_embeddings(full)
		out = np.zeros((len(spans), token_embs.shape[-1]), dtype=np.float32)
		empty = 0
		for i,(cs,ce) in enumerate(spans):
			cs, ce = cs+prefix_len, ce+prefix_len
			idx = [t for t,(s,e) in enumerate(offsets) if msk[t] and e>cs and s<ce]
			if idx: out[i] = token_embs[idx].mean(axis=0)
			else: empty += 1
		if empty: warnings.warn(f'{empty}/{len(spans)} spans got no tokens (doc likely exceeds max_seq_len); use encode_auto/encode_long_document for long docs')
		if self.normalize: out = out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-12, None)
		return out.astype(self.dtype)

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

class AutoLateChunkFastEncode(LongLateChunkFastEncode):
    'Route to single-pass / windowed / tight-windowed late chunking by document token count.'
    def _count_tokens(self, text):
        'Token count with truncation disabled (tokenizer only, no ONNX run).'
        trunc = self.tok.truncation
        self.tok.no_truncation()
        try:
            return len(self.tok.encode(text, add_special_tokens=True).ids)
        finally:
            if trunc: self.tok.enable_truncation(**{k:trunc[k] for k in ('max_length','stride','strategy','direction') if k in trunc})

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
