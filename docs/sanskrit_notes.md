# Sanskrit: design notes

Working notes on the decisions in `litesearch.sanskrit` that are easy to get wrong, written after
comparing this implementation against an independent one (branch
`claude/sanskrit-verse-chunking-graph-tuav90`) that solved the same problem differently. Both
approaches work; they fail differently, and the differences are the useful part.

## 0. Where the facets live, and why that is the whole integration story

Everything this module computes about a chunk — its metre, its gaṇa signature, the cadence variant
of its śloka — goes into the store's existing `metadata` column as JSON. Nothing gets a column of
its own and no schema changes.

That is not a shortcut, it is the integration. `get_store` indexes `metadata` for FTS *beside*
`content`, so a facet written there is (a) matched by ordinary `db.search`, (b) filterable by any
`where` clause, and (c) visible to every downstream caller without that caller learning anything
about Sanskrit. [vishalakshi](https://github.com/vedicreader/vishalakshi) threads `where` straight
through `find`, `sections` and `context` down to `doc_search`, so
`v.find(q, where="metadata like '%mandākrāntā%'")` works today against a vault built with this
module — no vishalakshi change, no new API, no migration.

Two small choices make the facets behave as *search terms* rather than as a blob:

- **Metre names are stored in Devanagari-friendly IAST** (`anuṣṭubh`, `śārdūlavikrīḍita`). The
  folding tokenizer covers the metadata column too, so `anustubh` typed on an ASCII keyboard
  reaches them — the same mechanism that makes the text itself scheme-agnostic.
- **The gaṇa signature is joined with `_`**, not spaces or hyphens: `ma_bha_na_ta_ta_ga_ga`. The
  tokenizer chain treats `_` as a word joiner (it is why `fts_search` survives as one token), so
  the signature stays a single searchable term instead of seven meaningless ones. *Find every verse
  shaped like this one* is then a query, not a scan.

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

## 3. Scansion: the four things that decide whether a verse scans

Scansion is now implemented here — `syllables`, `scan`, `ganas`, `detect_meter`, and `verse_meta`,
which puts metre and gaṇa signature into every chunk's `metadata`. The comparison branch got there
first and its catalogue was right; four separate details decide whether a real verse actually
scans, and each of them silently costs one syllable, which is enough to make the pāda the wrong
length and the verse match nothing.

- **Word boundaries must survive into the syllabifier.** Matching vowels longest-first over the
  whitespace-stripped string fuses `a` + `i` into the diphthong `ai`: `jīva iti` scans as
  `jī-vai-ti`, three syllables instead of four. Consonant clusters are the opposite case and *do*
  cross a boundary — `tat sarvam` closes its first syllable on `t`+`s` — so the fence goes around
  vowel matching only.
- **A trailing consonant at the end of the text closes its syllable.** Two consonants are needed
  mid-text because a single one is the onset of the next syllable; at the end there is no next
  syllable, so one is enough. This is what makes `gam` guru — and it is how the gaṇa table checks
  out against the `yamātārājabhānasalagam` mnemonic, which is the test that caught it here.
- **A syllable count is not a metre.** Labelling any 32-syllable unit `anuṣṭubh` means 32
  repetitions of `ka` are an anuṣṭubh, and so is prose of the right length. The even pādas carry
  the strict cadence — 5 laghu, 6 guru, 7 laghu — and checking it is what makes the name mean
  something. The odd pādas take pathyā or one of four licensed vipulā shapes, which is worth
  reporting rather than collapsing.
- **The citation does not scan.** `|| Manu_1.1 ||` contributes `ma` + `nu`, so every verse in a
  GRETIL file comes out 34 syllables instead of 32 and matches nothing at all. On the Manu fixture
  this was the difference between 0 verses identified and all of them. Glosses and headings go the
  same way.

One design choice worth stating: metre is **not** used as a verse/prose classifier here. The other
branch does, and it is a tempting reuse, but the failure mode is that a passage of bhāṣya gets
filed as śloka on the strength of its length. `detect_meter` returning `name=None` is a statement
about the catalogue, not about whether the text is verse.

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
