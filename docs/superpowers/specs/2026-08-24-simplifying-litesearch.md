# Simplifying litesearch

Date: 2026-08-24
Repos: `litesearch`, and whatever comes out of it. `fossick` and `kosha` for the shared forks.

## The finding

litesearch is 3,953 lines of definitions across ten modules. It exports 167 names and assembles
`Database` by monkeypatching 29 methods onto it from five of those modules. `Index`, the
documented front door, has 10 public members. `Database` has 66.

Two thirds of the bulk sits in three files, and two of them are not about search.

| module | def-lines | what it is |
|----|----|----|
| `graph.py` | 855 | an entity graph whose retrieval leg measures negative on ordinary queries |
| `sanskrit.py` | 807 | a Sanskrit philology library. 68 lines of it are load-bearing for every store |
| `tree.py` | 704 | the document tree. This is the thing that earns its keep |
| `core.py` | 429 | the database, the hybrid search, RRF |
| `data.py` | 415 | PDF, OCR, chunking, code parsing, file profiles, FTS query preprocessing |
| `utils.py` | 365 | encoders: ONNX, image, multimodal, and late chunking |
| `quality.py` | 240 | the noise features, moved in from vishalakshi |
| `api.py` | 98 | `Index` |
| `cli.py` + `postfix.py` | 40 | |

The README's own ladder says which parts are measured wins. Read against the line counts, the
library spends most of itself below the line.

| change | delta weighted MRR | lines it costs |
|----|----|----|
| `pre()` on the FTS leg | +0.016 to +0.093 | 37 |
| 512-char chunks | +0.06 to +0.12 | a constant |
| cross-encoder rerank | +0.026 to +0.077 | about 15 |
| Sanskrit FTS tokenizer | 1.000 Devanagari to verse recall | 68 |
| document tree, for ranking | -0.052 to +0.011, kept for `toc`, `read` and `sections` | 704 |
| late chunking | -0.033 to -0.053 | 94, and out of `__all__` |
| entity graph leg | -0.070 to -0.160 | 555 |

## Duplicate calls and methods

Three implementations of Reciprocal Rank Fusion, and `rrf_all` is exactly the other two. Verified
by running them on the same input.

```
rrf_merge(fts, vec, k, limit)          == rrf_all([fts, vec], k, limit)              True
_wrrf(fts, vec, wf, wv, k, limit)      == rrf_all([fts, vec], k, limit, [wf, wv])    True
```

Identical orderings and identical scores to 1e-12. `rrf_merge` at core.py:428 has one caller,
`Database.search`. `_wrrf` at tree.py:669 has one caller, `doc_search`. `rrf_all` at graph.py:881
is called seven times inside `graph_search`, five of them the same fallback line. Two of the three
should go, and `rrf_all` belongs in `core` beside the thing it fuses rather than in the graph
module.

Two retrieval paths returned different shapes, and the documented one was the weaker.

```
Index.search  -> Database.search(rrf=True)  -> rrf_merge   no span merge, key `heading`
Index.sections-> sections -> doc_search -> search(rrf=False) -> _wrrf + merge_spans, key `breadcrumb`
```

`doc_search` sets `breadcrumb` to a copy of `heading`, so the same string travels under two names.
vishalakshi, the caller the README nominates, overrode `Index.search` to call `db.doc_search`
instead, because it wanted the span merging. When the flagship caller bypasses the front door, the
front door is wrong. Fixed on this branch: `Index.search` is `doc_search` now.

`Database.search` is two functions wearing one name. `rrf=True` returns a ranked list. `rrf=False`
returns `dict(fts=…, vec=…)`. Every internal caller passes `rrf=False`, using it as "run the two
legs and hand them to me", which is a different function and wants a different name.

Two k-NN implementations and two clustering implementations run over the same vectors.
`quality._knn` uses a blocked matmul with HNSW past 30k rows. `graph._knn_clusters` walks a
usearch index. `quality._centroids` is k-means++ on the sphere. `graph._usearch_clusters` and
`_cluster_groups` cluster the same store a different way. This one is partly my doing: I moved
`quality` in without checking what `graph` already had.

Seven names are forked across litesearch, fossick and kosha. Six have drifted.

| name | in | state |
|----|----|----|
| `repo_root` | litesearch, fossick, kosha | same code, three copies |
| `mv_skill_md` | litesearch, fossick, kosha | drifted |
| `clean_md` | litesearch, fossick | drifted. fossick strips a "Press enter or click to view" line that litesearch keeps |
| `ocr_parse`, `fix_layout`, `orphan_vals` | litesearch, fossick | drifted |
| `scrambled_layout` | litesearch, fossick | identical |

A PDF cleaned through litesearch keeps junk that the same PDF cleaned through fossick does not.
That is what a fork costs.

## Two unused runtime dependencies

`pyproject.toml` declared `notebook` and `pandas`. Neither was imported anywhere in `litesearch/`.
The only occurrences of either word were a comment in `graph.py` and the string `'notebook'` as a
file-kind label in `tree.py`.

`notebook` pulls 93 distributions, including all of JupyterLab, nbconvert, pyzmq, tornado and
ipykernel, into the runtime of a library whose pitch is "no server, no new infrastructure". Both
are gone on this branch.

## Packages warranted

sanskrit should be its own package. 807 lines, of which 68 are the FTS5 tokenizer. The README calls
that tokenizer "on for every store, not only Sanskrit ones" and "the single largest measured
retrieval win in the repository", so those 68 lines stay. The other 740 are a philology library
that happens to register a chunker.

| part | lines |
|----|----|
| metre and prosody | 155 |
| verse chunking and citation | 91 |
| GRETIL, TEI, VR and DCS parsers | 143 |
| vidyut lemmas and Monier-Williams glosses | 145 |
| `by_lemma` and `by_meter` | 26 |

Same shape as rahasya and varga. Measured, self-contained, wanted by people who do not want the
rest.

graph should be its own package. 855 lines split four ways.

- 555 are the graph proper: `build_graph`, `resolve_entities`, `cooccur_edges`, `graph_search`, PMI
  edges, union-find resolution, personalised PageRank. Measured -0.070 to -0.160 on ordinary
  queries, opt-in by name, and `context(graph=False)` by default. One caller: vishalakshi's
  `connect`.
- 188 are clustering and topics: `topic_nodes`, `clusters`, `peers`, `ctfidf_labels`. These run by
  default, and vishalakshi's `map` and `topic_tree` use them. They stay.
- 76 are entity extraction: `code_entities`, `prose_windows`, `text_entities`. varga imports
  `text_entities`. It stays, or it moves with the graph and varga follows.
- 22 and 14 are misplaced. `hash_embed` is an encoder living in the graph module and belongs in
  `utils`. `rrf_all` belongs in `core`.

The heavy encoders want the wheel to stop carrying them. `FastEncode` at 141 lines, plus
`FastEncodeImage` and `FastEncodeMultimodal` at 122, are already lazy at runtime but ship in every
install. `static_embedder`, the default and the one `evals/` says wins, is six lines.

Late chunking should be deleted. 94 lines, three subclasses, measured -0.033 to -0.053, and already
out of `__all__`. Keeping code that measurement rejected is how a module reaches 365 lines. The
eval notebook that produced the number stays.

The PDF helpers want one home shared with fossick. `clean_md`, `fix_layout`, `ocr_parse`,
`orphan_vals` and `scrambled_layout` are one job forked twice. Both packages already depend on
`pdf-oxide` and `liteparse`, and neither depends on the other, so this wants either a small shared
package or fossick as the owner with litesearch importing.

`pdf_parse` itself does not merge. It returns `[(page, text)]` for the tree, and fossick's `pdf2md`
returns a string. vishalakshi uses litesearch's on purpose. The chunk sizes are measured against
it.

## Where that leaves it

| | now | after |
|----|----|----|
| `graph.py` | 855 | 188, the topics and clusters |
| `sanskrit.py` | 807 | 68, the tokenizer |
| `utils.py` | 365 | about 150 |
| `data.py` | 415 | about 280 |
| `tree.py`, `core.py`, `quality.py`, `api.py`, the rest | 1,511 | about 1,480 after the RRF and `_legs` cleanup |
| total | 3,953 | about 2,170 |

167 exported names becomes roughly 90. The 29 patched `Database` methods become about 20, from
three modules instead of five.

## What `Index` should become

Half of this is done. `Index.search` is `doc_search` now, and `search`, `sections` and `context`
take a `where`, which is what vishalakshi needed and had to override all four retrieval methods to
get. `read` takes a `store` for the same reason.

The open half is whether `database()` stays a public route. The README tells a caller to "drop to
`database()` when it stops fitting". That means going from a 10-name API to a 66-name one, with no
ladder in between. Making `Index` the only route means deciding which of the 66 `Database` members
are API and which are plumbing. That is the same measured-policy-seam split that drove the
vishalakshi work.

## Order to do it in

Three of these need no decision and no release coordination. Two are done.

1. Drop `notebook` and `pandas`. Two lines, 93 distributions. Done.
2. Remove the five extras and point the messages at the packages. Done.
3. Collapse the three RRFs onto `rrf_all` and move it to `core`. Keep `rrf_merge` and `_wrrf` as
   deprecated one-line aliases for one release.
4. Delete late chunking.

Then the splits, each needing a library release before litesearch can drop its copy.

5. sanskrit out, tokenizer retained.
6. graph out, topics and clusters retained.
7. The PDF helpers to one home. fossick and kosha drop their forks of `repo_root` and
   `mv_skill_md` at the same time.

And last, because it changes what callers write: the `Database.search` and `_legs` split, and the
question of whether `database()` stays public.

## Loose ends

`nbs/04_latechunk_eval.ipynb`, `07_doc_eval.ipynb` and `10_sanskrit_eval.ipynb` declare
`default_exp` for modules that do not exist and have no `#| export` cells. The directives are
vestigial.

Five eval notebooks live in `nbs/` and run in the docs build and the test run, while a separate
3,543-line `evals/` directory holds the rest. vishalakshi keeps `evals/` out of both. Pick one.

`nbs/` has duplicate number prefixes: two `04_`, two `07_`, two `09_`.
