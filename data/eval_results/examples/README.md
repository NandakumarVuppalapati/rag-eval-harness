# Example evaluation runs

These two files are the first full runs of the 66-question golden dataset
through the real pipeline (real Pinecone retrieval, real Claude Haiku
generation) for each embedding model, kept as committed examples of actual
output rather than regenerated on every nightly run. Routine runs land in
`data/eval_results/*.json` (gitignored) and Postgres, not here.

Both were run with `--skip-ragas` (see `scripts/run_evaluation.py`'s module
docstring): the Ragas judge (OpenAI `gpt-4o-mini`) was rate-limited at the
time, and retrieval + generation don't need to be re-run just to wait out an
unrelated provider's quota. Their `ragas` field is empty and `ragas_pending`
is `true` until `scripts/score_ragas.py` backfills it.

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
