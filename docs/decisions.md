# Decisions

This page lists the decisions that a maintainer must know. Each entry gives the decision, the reason, the version or date, and the place in the code.
The rules of daily work (dependencies, the two routes, measured defaults, no extras, the tokenizer chain, prose, evals) are in [CLAUDE.md](../CLAUDE.md). This page does not repeat them.
The numbers behind each default are in [evals/RESULTS.md](../evals/RESULTS.md) and the 0.1.23 entry of the [CHANGELOG](../CHANGELOG.md). [rag_tiers.md](rag_tiers.md) gives the tiers that `nbs/08_tier_eval.ipynb` links to.
The old notes (consolidation, graph and ingestion performance, Sanskrit design, the late-chunking plan, the simplification spec) are in the git history under `docs/`.

## Scope

- **litesearch owns retrieval for all the owner's apps.** Code that works the same on a corpus of PDFs lives here. Code that knows a function or a symbol lives in kosha. A caller with a gap sends a patch here. It does not keep a wrapper.
  Why: one search engine, easy upkeep. (2 Aug 2026, 1 Nov 2026) The consumers are kosha, vishalakshi, ganapati and vruksha.
- **The entity graph left.** `Database.context(graph=)` and `graph_search` are gone. The typed graph is vruksha's.
  Why: the graph leg gave -0.07 to -0.16 MRR at 5 to 8 times the latency. (0.1.35) `evals/RESULTS.md`.
- **Late chunking left the package.** It gave -0.03 to -0.05, and -0.18 on the vector leg, at 3 times the index time. It stays in `evals/latechunk.py` so the question can be asked again. (0.1.23, Aug 2026)
- **Sanskrit philology left; the tokenizer stayed.** Metres, scansion and lemmas are in ganapati, which registers its profiles through `load_plugins`. litesearch keeps the fold and the tokenizer. (0.1.38) `litesearch/sanskrit.py`, `load_plugins` in `litesearch/data.py`.

## Index defaults

- **`Index` is the front door.** Static `potion-multilingual-128M`, float16, 512-character chunks, `pre()` on the FTS leg, an HNSW leg and a document tree.
  Why: chunk size and `pre()` gave the largest gains (+0.06 to +0.14 and up to +0.33). The encoder changed little (0.018 to 0.046 spread) at 1,700 times less index cost. The tree is a wash for ranking but gives `toc`, `read` and `sections`. (0.1.23, 0.1.24) `litesearch/api.py`, `static_embedder` in `litesearch/utils.py`.
- **`rerank=True` is the one choice for the caller.** It fetches 30 candidates first (`RERANK_FANOUT`), because a rerank of 10 only reorders 10. It gave +0.026 to +0.077 at about 10 times the latency. (0.1.23) `litesearch/core.py`.
- **Static models load with laghu**, which maps the model from disk. Private memory fell from 1,373 MB to 80 MB. No model2vec fallback. (0.1.54) `static_embedder`.
- **A store is stamped with its encoder.** Another encoder raises `StoreMismatch`. (0.1.35) `litesearch/core.py`.
- **One text is stored one time.** A file's source is its resolved path. A `sources` table maps each source to its document. Unchanged text is skipped and an edited source replaces its chunks. (0.1.50) `litesearch/tree.py`.
- **The default walk skips generated folders** (`_proc`, `_docs`, `_site`, `node_modules`, `__pycache__`), so an nbdev repo is not indexed three times. (0.1.25) `skip_folder_re` in `litesearch/data.py`.

## FTS and Sanskrit

- **The fold is a token in the index, not a copy in `metadata`.** `कृष्ण` and `kṛṣṇa` both index as `krsna`. It costs 1.03x disk against 1.51x for a stored fold, and plain `db.search` works.
  The cost: a store made with this chain cannot be read by a connection that has not registered the tokenizer. (7 Aug 2026) `_FTS_TOKENIZE` in `litesearch/core.py`.
- **`sanskrit` sits inside porter.** See CLAUDE.md. Test it with a locative (`-e`) word and an exact quoted query, because a prefix query hides the fault.
- **`roman_fold` gives a third token**, so a plain-English spelling (`Bheeshma`) finds `भीष्म`. It adds 14.7 % to the index on English. (0.1.36) `litesearch/sanskrit.py`.
- **adhara is the native tokenizer and vector ranker when it loads.** `SANSKRIT_TOKENIZE` is `'adhara'` then. The Python tokenizer is still registered as `sanskrit`, so old tables work. If adhara does not load, search runs in Python with a warning. No Windows wheel. (0.1.61) `_load_adhara` in `litesearch/core.py`, `litesearch/sanskrit.py`.
- **A facet goes in the `metadata` JSON, not a new column.** FTS indexes `metadata`, so a facet is a search term and a `where` filter with no schema change. Join a multi-part term with `_` so it stays one token.

## Vectors and usearch

- **usearch is probed before it loads.** A subprocess builds a small index first. The verdict is cached in `cache_dir()/usearch.json` by CPU, version and file size. `LITESEARCH_USEARCH=0|1` overrides it.
  Why: the extension crashed Python on some GitHub AMD runners, and numkong wheels do not import on macOS 14. (0.1.47, 0.1.48) `litesearch/postfix.py`.
- **When usearch is off, `ExactIndex` searches with numpy.** Same calls, exact results. (0.1.47) `litesearch/core.py`.
- **The HNSW sidecar is written to a temporary file and renamed.** No process loads a part-written index. A missing, stale or torn sidecar is built again from the table. (0.1.52, 0.1.58) `get_index`.
- **A filtered vector search uses adhara on all cores.** A `where` that most rows pass went from 518 ms to 118 ms on 181,290 chunks. (0.1.61) `_adhara_ids` in `litesearch/core.py`.

## Writes and threads

- **Writes commit in batches of 1,000 rows.** Earlier batches stay if a later one fails. One commit per insert was 6.7 times slower. (0.1.19, 0.1.59) `upsert_all`, `write_txn` in `litesearch/core.py`.
- **`add_doc` updates only its own ANN keys; `add_dir` rebuilds one time.** A rebuild per document made ingest grow as n^1.72. (0.1.19)
- **Parse in processes, not threads,** and only for heavy files (`PARSE_HEAVY`) at `MIN_PARALLEL_FILES` (64) or more. Threads were slower than serial (0.76x). Processes gave 3.3x. HTML is not heavy. (0.1.19, 0.1.45) `litesearch/data.py`, `litesearch/tree.py`.
- **Each connection sets its own busy timeout (30 s) and a 64 MiB WAL limit**, and truncates the WAL after a bulk write. (0.1.59) `database()`.
- **Statements run under a connection lock**, and `get_index` does not take `db_lock`. A caller (Leela) can hold `db_lock` while workers read. `shared=True` wraps one connection for many threads. (0.1.40, 0.1.49, 0.1.60) `Conn` in `litesearch/core.py`.
- **Sharding is not worth it for speed.** After the fixes, one database at 400 documents beat 8 shards. Ask again only when the HNSW index no longer fits in RAM. (0.1.19) `bench_shards` in `evals/ingest_bench.py`.
- **`import litesearch` sets `ORT_DISABLE_TELEMETRY=1`.** The onnxruntime telemetry thread could abort the process at exit. (0.1.61) `litesearch/postfix.py`.
