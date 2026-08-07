# Sanskrit: design notes

Working notes on the two decisions in `litesearch.sanskrit` that are easy to get wrong, written
after comparing this implementation against an independent one (branch
`claude/sanskrit-verse-chunking-graph-tuav90`) that solved the same problem differently. Both
approaches work; they fail differently, and the differences are the useful part.

## 1. Making FTS scheme-agnostic: index-time fold vs. stored fold

The problem is that one verse circulates in Devanagari, IAST, SLP1 and Harvard-Kyoto, and a reader
types whichever they know. Both implementations answer it by reducing everything to one ASCII
skeleton — `कृष्ण` and `kṛṣṇa` both become `krsna` — and they differ only in *where the skeleton
lives*.

| | this branch: colocated token | the other branch: fold in `metadata` |
|---|---|---|
| where the fold lives | emitted by the FTS5 tokenizer, index only | a copy of the text, in the store's `metadata` column |
| disk cost | **1.03x** | **1.51x** |
| works with plain `db.search` | yes | no — the caller must route through `sanskrit_query` |
| database readable without litesearch | **no** | yes |
| retrofits an existing store | no — tokenizer is fixed at table creation | yes |

The disk figures are measured over the same 2,400 Devanagari rows, VACUUMed, against a baseline
store with no cross-script support at all.

Neither column is the winner. The colocated token is nearly free and needs no cooperation from
callers, which is why it is the default here. But it buys that by putting a Python function in the
index path, and the cost is exact and worth stating plainly:

```
$ sqlite3 store.db "select * from store_fts where store_fts match 'krsna'"
Error: no such tokenizer: sanskrit
```

A store built with the `sanskrit` chain cannot be **read or written** by any connection that has
not registered the tokenizer — not `sqlite3`, not a GUI browser, not a reader in another language,
not a future litesearch that dropped the module. The stored fold has no such property: it is
ordinary text in an ordinary column, and every SQLite client in the world can read it.

That is a real argument for the other design, and the honest summary is that this is a durability
/ convenience trade, not a quality one. If a corpus has to outlive this library, store the fold.

## 2. A synthesized token must be emitted *inside* every transform the query also gets

This is the rule the first version here got wrong, and it is worth stating generally because
nothing about it is specific to Sanskrit.

The chain is `porter simplify casefold 1 sanskrit unicodewords`, read outermost-first. `sanskrit`
originally sat on the **outside**, which looks harmless — it only adds tokens — but it means the
fold is computed on porter's *output*, and porter's output depends on the script:

| | indexing `धर्मक्षेत्रे` | querying `dharmaksetre` |
|---|---|---|
| porter sees | Devanagari — passes through untouched | ASCII — stems to `dharmaksetr` |
| fold emitted | `dharmaksetre` | (already ASCII, unchanged) |
| token stored / sought | `dharmaksetre` | `dharmaksetr` |

The two never meet. Moving `sanskrit` inside porter makes the fold just another token that porter
then stems, so both sides land on the same string. Measured on ordinary Devanagari vocabulary,
3 of 14 words were unreachable by their own ASCII spelling before the move and 0 after.

Two things about how this survived review are worth remembering:

- **The test used a word porter happens not to stem.** `srimata` has no suffix porter recognises,
  so the original assertion passed. The words that fail are the ones ending in a stemmable suffix —
  and the Sanskrit locative singular is `-e`, which porter strips, so the failure was systematic
  rather than exotic. The regression test now uses locatives for exactly this reason.
- **Prefix search hid it.** `fts_search` appends `*` to each term, and `dharmaksetr*` does match
  the indexed `dharmaksetre`. The bug is only visible on an exact, unwildcarded query — which is
  why the test now asserts on `"dharmaksetre"` in quotes.

## 3. Scansion, if it is ever added here

The other branch implements classical scansion — syllable weights, the eight gaṇas, 20 metres
derived from their recipes — and uses metre as its verse/prose classifier. It is the most
interesting thing in that branch and it mostly works: śārdūlavikrīḍita, mandākrāntā and anuṣṭubh
are all identified correctly from either script. Two defects were found in it, and anyone
implementing scansion here should expect both, because they are properties of the problem:

- **Do not strip whitespace before syllabifying.** Matching vowels longest-first across a word
  boundary fuses `a` + `i` into the diphthong `ai`: `jīva iti` scans as `jī-vai-ti`, three
  syllables instead of four. One lost syllable makes the pāda the wrong length and the verse
  fails to match any metre. Word boundaries have to survive into the syllabifier.
- **A syllable count is not a metre.** Any 32-syllable unit was labelled `anuṣṭubh` with no check
  on the cadence — 32 repetitions of the syllable `ka` classify as anuṣṭubh. Since the classifier
  then feeds a verse/prose decision, unmetred prose of the right length is filed as verse. The
  count is a necessary condition, not a sufficient one.

## 4. Format coverage is a separate axis from linguistic depth

The other branch reads one input shape well — GRETIL plain text, where the citation is printed
between daṇḍas as `// Mn_1.1 //` — and strips tags from everything else. That is a reasonable
scope choice, but it means structured formats degrade *silently* rather than loudly:

| fixture | units found | citations recovered |
|---|---|---|
| `isopanisad_excerpt.htm` (GRETIL plain text) | 10 | 9 |
| `manu_tei_excerpt.xml` (TEI) | 12 | **0** |
| `lalita_excerpt.xml` (vedicreader) | 10 | **0** |

TEI keeps the citation in `<lg xml:id="Manu_1.1">` — an attribute, which tag-stripping deletes
before anything can read it — and the whole `<teiHeader>` is indexed as body text. The lesson is
that a reader per format is not redundant with a good segmenter: no amount of cleverness about
daṇḍas recovers an address that was thrown away at parse time.

The same shape appears in their CoNLL-U loader, which is correct for real 10-column CoNLL-U and
silently indexes the wrong columns for the ID-less tabular export DCS also publishes — morphological
features land in the `lemmas` field, and every verse gets the same citation, which collapses an
entire text onto one graph node. A loader that assumes a format should assert the format.

## Worth borrowing

Not implemented here — recorded because they are good ideas that this branch does not have:

- **The citation as tree structure.** `ViP_1,1.1` is a complete address (aṃśa 1, adhyāya 1, verse 1).
  Where prose infers a hierarchy from headings, a referenced text simply states one, for every leaf.
- **The colophon as the node summary.** `iti ... prathamo 'dhyāyaḥ` is a one-line statement of what
  the section was, written by the tradition. It beats any extractive summary of the first 300
  characters.
- **The verse as a graph entity**, so `parallel` / `quotes` / `follows` fit the existing
  entity-to-entity edges table with no schema change. A śloka is routinely not the property of the
  text you found it in, and "this verse is also in the Mahābhārata" is a finding, not an artefact.
- **A `lemma_fn` seam.** Sandhi means the surface form is often not what anyone types. Indexing
  validated lemmas beside the surface text matters more in Sanskrit than stemming does in English.
