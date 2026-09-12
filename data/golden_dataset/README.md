# Golden evaluation dataset

This directory is the ground truth the nightly Ragas evaluation run scores
retrieval and generation against. It has three files, corresponding to three
different ways a real RAG system over SEC filings gets tested, and three
different ways ground truth was established for each.

## `numeric_questions.json` (40 questions)

XBRL-grounded numeric questions -- one to four per company, covering total
revenue, net income, operating income, diluted EPS, and total assets.

Ground truth comes directly from SEC's own structured XBRL company-facts
API, not from reading the filing text. `scripts/build_golden_dataset.py`
pulls each fact, restricts it to a datapoint whose accession number matches
a filing actually in this project's corpus *and* whose period matches that
filing's own primary reporting period (an early version of this script
picked up prior-year comparative figures embedded in the same XBRL payload
-- see `docs/adr/` history / commit log for that fix), then spot-checks that
the number appears verbatim in the filing's own parsed text before including
it. All 40 questions passed that check.

## `narrative_questions.json` (10 narrative + 5 cross-document = 15 questions)

Narrative questions ask about deal terms, segment definitions, and MD&A
commentary that XBRL doesn't tag -- e.g. the exact consideration Boeing paid
for Spirit AeroSystems, how Johnson & Johnson's Kenvue separation was
structured, what Microsoft's Intelligent Cloud segment comprises. Ground
truth for these was located by grepping the actual parsed filing text for
the relevant terms (not by asking a model to summarize), and
`scripts/build_narrative_questions.py` re-verifies every claimed
number/phrase against that same source text at generation time.

Cross-document questions require comparing figures that live in two
different filings -- two different quarters of the same company, or the
same fiscal-year metric across two different companies. Their reference
answers are computed arithmetically from the already-verbatim-verified rows
in `numeric_questions.json`, so the comparison math is only ever built on
numbers that already passed the numeric harness's own verification.

## `unanswerable_questions.json` (11 questions)

Plausible-sounding questions this corpus genuinely cannot answer, each
labeled with *why*: a company not in the corpus (Tesla, Amazon), a fiscal
period outside the corpus's 2025-2026 filing window, a forward-looking
projection (10-Ks report history, not forecasts), a document type this
corpus doesn't include (executive compensation lives in a DEF 14A proxy
statement, not a 10-K/10-Q), a segment a company divested before any filing
in this corpus was even written (J&J's Consumer Health/Kenvue), or a metric
companies simply don't disclose at that granularity (market share,
country-level headcount).

These exist because a RAG system that always answers is more dangerous than
one that sometimes doesn't -- the evaluation harness scores refusal
behavior on this set separately from answer quality on the other two, and a
system that fabricates a confident-sounding number for
"Tesla's revenue" should fail that check even if its retrieval and
generation both score well elsewhere.

## Composition

| File | Count | What it tests |
|---|---|---|
| `numeric_questions.json` | 40 | Single-fact numeric retrieval, XBRL-verified |
| `narrative_questions.json` (narrative) | 10 | Prose/MD&A comprehension, single document |
| `narrative_questions.json` (cross_document) | 5 | Synthesis across two filings |
| `unanswerable_questions.json` | 11 | Refusal / hallucination resistance |
| **Total** | **66** | |
