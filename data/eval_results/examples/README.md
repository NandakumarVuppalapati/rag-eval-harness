# Example evaluation runs

These two files are the first full runs of the 66-question golden dataset
through the real pipeline (real Pinecone retrieval, real Claude Haiku
generation) for each embedding model, kept as committed examples of actual
output rather than regenerated on every nightly run. Routine runs land in
`data/eval_results/*.json` (gitignored) and Postgres, not here.

Both were first run with `--skip-ragas` (see `scripts/run_evaluation.py`'s
module docstring): the Ragas judge (OpenAI `gpt-4o-mini`) was rate-limited
at the time, and retrieval + generation don't need to be re-run just to
wait out an unrelated provider's quota. `scripts/score_ragas.py` has since
backfilled both files in place -- `ragas_pending` is now `false` and the
real scores are below.

## What they already show, without Ragas

Running these surfaced a real bug: `refused` was originally computed as an
exact match against the refusal sentence, but Claude Haiku routinely
appends an explanation after it despite being asked not to. Every genuine
refusal was therefore being scored as a hallucination -- caught because the
refusal rate on the (deliberately unanswerable) 11-question slice came back
0% when the answers themselves clearly said otherwise. Fixed in
`generation/generator.py` to a prefix match (see its module docstring and
`tests/test_generator.py`); the `refused` flags in these two files were
corrected in place from the original answer text, no re-generation needed.

With that fixed, both runs correctly refuse all 11 unanswerable questions
(refusal_rate 1.0). But a meaningful share of *answerable* questions were
also refused -- not because the model failed, but because retrieval didn't
surface the passage that actually contains the answer:

| Embedding model | Answerable questions refused |
|---|---|
| `voyage-finance-2` | 21 / 55 |
| `text-embedding-3-small` | 25 / 55 |

This is retrieval failing quietly, exactly the failure mode this project
exists to catch -- and it's the first real evidence for this project's
central comparison: the domain-tuned embedding model retrieves noticeably
better than the general-purpose baseline for these financial questions,
though neither is close to reliable yet. Both numbers are large enough to
be a real finding, not noise, and are left as-is rather than tuned away,
per this project's own stated purpose.

## Now scored with Ragas

Backfilling hit a second real bug along the way: ragas 0.4+'s
`EvaluationResult` only implements `__getitem__(key: str)`, with no
`keys()`/`__iter__`, so the original `dict(ragas_result)` used to build the
aggregate raised `KeyError: 0` immediately -- a code path nothing had
exercised end-to-end before, since every earlier attempt hit the OpenAI
rate limit first. Fixed in `harness.py`'s `score_answerable` by computing
each metric's mean directly from the per-question scores instead (see its
docstring and the regression test in `tests/test_harness.py`).

| Metric | `voyage-finance-2` | `text-embedding-3-small` |
|---|---|---|
| Faithfulness | 0.884 | 0.826 |
| Answer relevancy | 0.467 | 0.437 |
| Context precision | 0.360 | 0.470 |
| Context recall | 0.514 | 0.567 |
| Refusal rate (unanswerable slice) | 1.0 | 1.0 |

These are computed over all 55 answerable questions, including the ones
each model refused (a refusal scores near-zero on faithfulness and answer
relevancy, since it doesn't attempt the question). Read next to the
refusal counts above, the result is more nuanced than a single winner:
`voyage-finance-2` refuses fewer answerable questions, and when it does
answer, is judged more faithful to its retrieved context and more relevant
to the question -- but on the two metrics that score retrieval quality
directly, context precision and recall, the general-purpose baseline
scores slightly higher. That isn't necessarily a contradiction: the two
models refuse different questions, so they aren't being scored on the same
subset of "hard" cases, and their precision/recall numbers aren't directly
comparable on a level playing field. Flagging that honestly, rather than
picking whichever framing tells the cleaner story, is the point of building
an evaluation harness in the first place.
