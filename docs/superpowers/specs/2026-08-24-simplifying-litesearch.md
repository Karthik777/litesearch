# Simplifying litesearch

Date: 2026-08-24
Repos: `litesearch`, and whatever comes out of it; `fossick` and `kosha` for the shared forks

## The finding

litesearch is 3,953 lines of definitions across ten modules, exports 167 names, and assembles
`Database` by monkeypatching 29 methods onto it from five of those modules. `Index`, the
documented front door, has 10 public members. `Database` has 66.

Two thirds of the bulk is in three files, and two of them are not about search:

| module | def-lines | what it is |
|----|----|----|
| `graph.py` | 855 | an entity graph whose retrieval leg is measured **negative** on ordinary queries |
| `sanskrit.py` | 807 | a Sanskrit philology library, of which 68 lines are load-bearing for every store |
| `tree.py` | 704 | the document tree. This is the thing that earns its keep |
| `core.py` | 429 | the database, the hybrid search, RRF |
| `data.py` | 415 | PDF, OCR, chunking, code parsing, file profiles, FTS query preprocessing |
| `utils.py` | 365 | encoders: ONNX, image, multimodal, and late chunking |
| `quality.py` | 240 | the noise features, moved in from vishalakshi today |
| `api.py` | 98 | `Index` |
| `cli.py` + `postfix.py` | 40 | |

The README's own ladder says which parts are measured wins and which are not. Read against the
line counts, the library spends most of itself below the line:

| change | Δ weighted MRR | lines it costs |
|----|----|----|
| `pre()` on the FTS leg | **+0.016 → +0.093** | 37 |
| 512-char chunks | **+0.06 → +0.12** | a constant |
| cross-encoder rerank | **+0.026 → +0.077** | ~15 |
| Sanskrit FTS tokenizer | **1.000 Devanagari→verse recall** | 68 |
| document tree, for ranking | −0.052 → +0.011 (kept for `toc`/`read`/`sections`) | 704 |
| late chunking | −0.033 → −0.053 | 94, and out of `__all__` |
| entity graph leg | **−0.070 → −0.160** | 555 |

## Duplicate calls and methods

**Three implementations of Reciprocal Rank Fusion, and `rrf_all` is exactly the other two.**
Verified by running them on the same input:

```
rrf_merge(fts, vec, k, limit)          == rrf_all([fts, vec], k, limit)              True
_wrrf(fts, vec, wf, wv, k, limit)      == rrf_all([fts, vec], k, limit, [wf, wv])    True
```

Identical orderings and identical scores to 1e-12. `rrf_merge` (core.py:428) is called once, by
`Database.search`. `_wrrf` (tree.py:669) is called once, by `doc_search`. `rrf_all` (graph.py:881)
is called seven times inside `graph_search`, five of them the same fallback line. Two of the three
should go, and `rrf_all` belongs in `core` beside the thing it fuses, not in the graph module.

**Two retrieval paths that return different shapes, and the documented one is the weaker.**

```
Index.search  → Database.search(rrf=True)  → rrf_merge          no span merge, key `heading`
Index.sections→ Database.sections → doc_search → search(rrf=False) → _wrrf + merge_spans, key `breadcrumb`
```

`doc_search` sets `breadcrumb` to a copy of `heading`, so the same string ships under two names.
vishalakshi — the caller the README nominates — overrides `Index.search` to call `db.doc_search`
instead, because it wants the span merging. When the flagship caller bypasses the front door, the
front door is wrong.

**`Database.search` is two functions wearing one name.** `rrf=True` returns a ranked list;
`rrf=False` returns `dict(fts=…, vec=…)`. Every internal caller passes `rrf=False`, i.e. uses it
as "run the two legs and give them to me". That is a different function: `_legs(q, emb, …)`.

**Two k-NN implementations and two clustering implementations over the same vectors.**
`quality._knn` (blocked matmul, HNSW past 30k) and `graph._knn_clusters` (over a usearch index);
`quality._centroids` (k-means++ on the sphere) and `graph._usearch_clusters` / `_cluster_groups`.
This is partly my doing: I moved `quality` in today without checking what `graph` already had.

**Seven names forked across litesearch, fossick and kosha, six already drifted:**

| name | in | identical |
|----|----|----|
| `repo_root` | litesearch, fossick, kosha | same code, three copies |
| `mv_skill_md` | litesearch, fossick, kosha | drifted |
| `clean_md` | litesearch, fossick | drifted — fossick strips a "Press enter or click to view" line litesearch keeps |
| `ocr_parse`, `fix_layout`, `orphan_vals` | litesearch, fossick | drifted |
| `scrambled_layout` | litesearch, fossick | identical |

A PDF cleaned through litesearch keeps junk that the same PDF cleaned through fossick does not.
That is what a fork costs.

## Two unused runtime dependencies

`pyproject.toml` declares `notebook` and `pandas`. Neither is imported anywhere in `litesearch/`;
the only occurrences of either word are a comment in `graph.py` and the string `'notebook'` as a
file-kind label in `tree.py`.

`notebook` pulls **93 distributions**, including all of JupyterLab, nbconvert, pyzmq, tornado and
ipykernel, into the runtime environment of a library whose pitch is "no server, no new
infrastructure". This is the highest-value line in this document and it is a two-line diff.

## Packages warranted

**sanskrit → its own package.** 807 lines, of which 68 are the FTS5 tokenizer the README says is
"on for every store, not only Sanskrit ones" and is "the single largest measured retrieval win in
the repository". Those 68 lines stay in litesearch. The other ~740 — metre and prosody (155),
verse chunking and citation (91), GRETIL/TEI/VR/DCS parsers (143), vidyut lemmas and
Monier-Williams glosses (145), `by_lemma` / `by_meter` (26) — are a philology library that happens
to register a chunker. Same shape as rahasya and varga: measured, self-contained, wanted by people
who do not want the rest.

**graph → its own package.** 855 lines, split four ways:

- 555, the graph proper: `build_graph`, `resolve_entities`, `cooccur_edges`, `graph_search`, PMI
  edges, union-find resolution, personalised PageRank. Measured −0.070 to −0.160 on ordinary
  queries, opt-in by name, `context(graph=False)` by default. One caller: vishalakshi's `connect`.
- 188, clustering and topics: `topic_nodes`, `clusters`, `peers`, `ctfidf_labels`. **Used by
  default**, and by vishalakshi's `map` and `topic_tree`. Stays.
- 76, entity extraction: `code_entities`, `prose_windows`, `text_entities`. varga imports
  `text_entities`. Stays, or moves with the graph and varga follows.
- 22 + 14: `hash_embed` is an encoder living in the graph module — it belongs in `utils`. `rrf_all`
  belongs in `core`.

**The heavy encoders → an extra, not a package.** `FastEncode` (141), `FastEncodeImage` and
`FastEncodeMultimodal` (122) already sit behind an `onnx` extra for their runtime but ship in the
wheel. `static_embedder` — the default, and the one `evals/` says wins — is six lines. Move the
ONNX family behind `litesearch[onnx]` properly.

**Late chunking → delete it.** 94 lines, three subclasses, measured −0.033 to −0.053, already out
of `__all__`. Keeping unexported code that measurement rejected is how a module gets to 365 lines.
The eval notebook that produced the number stays.

**The PDF helpers → one home, shared with fossick.** `clean_md`, `fix_layout`, `ocr_parse`,
`orphan_vals`, `scrambled_layout` are one job forked twice. Both packages already depend on
`pdf-oxide` and `liteparse` and neither depends on the other, so this wants either a small shared
package or fossick as the owner with litesearch importing. `pdf_parse` itself does **not** merge:
it returns `[(page, text)]` for the tree and fossick's `pdf2md` returns a string, and vishalakshi
deliberately uses litesearch's because the chunk sizes are measured against it.

## Where that leaves it

| | now | after |
|----|----|----|
| `graph.py` | 855 | 188 (topics and clusters) |
| `sanskrit.py` | 807 | 68 (the tokenizer) |
| `utils.py` | 365 | ~150 |
| `data.py` | 415 | ~280 |
| `tree.py`, `core.py`, `quality.py`, `api.py`, rest | 1,511 | ~1,480 (RRF and `_legs` cleanup) |
| **total** | **3,953** | **~2,170** |

167 exported names becomes roughly 90. The 29 patched `Database` methods become about 20, from
three modules instead of five.

## What `Index` should become

This is the part worth arguing about, and I have not decided it. `Index` is 98 lines wrapping a
`Database` that has 66 public members, and it exposes 7 of them. The README tells a caller to
"drop to `database()` when it stops fitting", which means going from a 10-name API to a 66-name
one with no ladder in between. Two ways out:

1. **`Index` grows a `where=`.** It is what vishalakshi needed and had to override all four
   retrieval methods to get — and `where=` already reaches `Database.search` through
   `doc_search`, so `Index` only has to stop swallowing it. Smallest change, keeps the two-route
   story.
2. **`Index` becomes the only route** and `database()` becomes internal. Bigger, and it means
   deciding which of the 66 `Database` members are API and which are plumbing — which is the same
   measured/policy/seam split that drove the vishalakshi work.

## Order to do it in

The first three need no decision and no release coordination:

1. Drop `notebook` and `pandas` from `pyproject.toml`. Two lines, 93 distributions.
2. Collapse the three RRFs onto `rrf_all`, move it to `core`, keep `rrf_merge` and `_wrrf` as
   deprecated one-line aliases for one release.
3. Delete late chunking.

Then the splits, each a library release before litesearch can drop its copy:

4. `sanskrit` out, tokenizer retained.
5. `graph` out, topics and clusters retained.
6. The PDF helpers to one home; fossick and kosha drop their forks of `repo_root` and
   `mv_skill_md` at the same time.

And last, because it changes what callers write:

7. `Index.where=`, and the `Database.search` / `_legs` split.

## Loose ends

- `nbs/04_latechunk_eval.ipynb`, `07_doc_eval.ipynb` and `10_sanskrit_eval.ipynb` declare
  `default_exp` for modules that do not exist and have no `#| export` cells. The directives are
  vestigial.
- Five eval notebooks live in `nbs/` and are in the docs build and the test run, while a separate
  3,543-line `evals/` directory holds the rest. vishalakshi keeps `evals/` out of both. Pick one.
- `nbs/` has duplicate number prefixes: two `04_`, two `07_`, two `09_`.
- `litesearch/SKILL.md` ships in the package directory; kosha does the same. Not a problem, worth
  knowing when the splits move files.
