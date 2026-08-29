# Working in this repo

nbdev. The notebooks under `nbs/` are the source; `litesearch/*.py` is generated. Edit the
notebook, run `nbdev_export`, never edit the `.py`. `README.md` comes from `nbs/index.ipynb`
through `nbdev_readme`. CI runs `nbdev_export` and fails on a diff.

## Dependencies point one way

ganapati, vruksha and kosha import litesearch. litesearch names none of them, in any dependency
group, dev included: a dev-group entry is still a cycle at install time. `evals/` may use one,
imported inside the function that needs it so the rest of `evals/` runs without it.

`Database.context(graph=True)` raises `ImportError` naming vruksha rather than `AttributeError`.
That is the shape any future seam should take.

## The two routes

`Index` decides the encoder, dtype, chunk size, retrieval strategy and tree from `evals/`.
`database()` is the same object underneath, reachable as `Index.db`. A change that makes `Index`
harder to use so `database()` can be simpler has the tradeoff backwards.

## Every default is measured

`evals/RESULTS.md` holds the number behind each one, and `python -m evals.decide` reproduces
them. Changing a default means changing the number first. A feature that measures negative stays
off by name, or leaves.

## No extras

`rerank=True` wants flashrank and `FastEncode` wants onnxruntime. Each is imported where it is
used and raises saying what to install. Do not add an optional-dependency group.

## The tokenizer chain is load-bearing

`_FTS_TOKENIZE` puts `sanskrit` inside porter, not around it. Outside, the fold is computed on
porter's output and an ASCII query stems to something the index does not hold. A store's
tokenizer is fixed when its table is created, so this cannot be decided per corpus.

## Prose in notebooks

Short. Lead with what the code does. Numbers instead of adjectives. No em dashes, no bold inside
a paragraph, no rhetorical questions. A rationale longer than three sentences belongs in a
docstring or in `evals/`.

## Docstrings and comments

One line. A second sentence only for a measured number or a footgun. Inline comments in a `def`
signature are nbdev docments and become the API parameter table. A comment that restates the line
under it goes.

## Evals

`evals/` is not shipped in the wheel. Results go in `evals/RESULTS.md` with the method. Paired
bootstrap over queries, CI spanning zero reported as no difference.
