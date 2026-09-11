# RAG Evaluation & Observability Harness

Production RAG systems rarely have a good answer to the question *"how do you know when it's wrong?"* Retrieval can quietly start missing the right passages. Generation can hallucinate fluently, without throwing a single error. Neither shows up in a normal API status code.

This project is a harness that continuously interrogates a real RAG system — instead of just running it — to answer that question with evidence instead of guesswork.

## What it does

A RAG pipeline answers financial-research questions by retrieving passages from a small corpus of public SEC filings (10-K / 10-Q) and generating grounded answers over them. Separately, a golden-question evaluation dataset (hand-curated questions with known-correct answers and source passages) is run against that pipeline on a schedule, scored with [Ragas](https://github.com/explodinggraph/ragas) on retrieval precision/recall, faithfulness, and answer relevancy, and the results are persisted so quality can be tracked over time and regressions can be flagged automatically — before a user notices them.

## Why these design choices

- **Domain-specific embeddings, benchmarked against a general-purpose baseline.** The primary embedding model is Voyage AI's `voyage-finance-2`, trained specifically on financial text; `text-embedding-3-small` runs alongside it as a documented baseline so the retrieval-quality delta between domain-tuned and general-purpose embeddings is measured, not assumed.
- **Cross-family LLM judging.** Generation uses Claude Haiku; the Ragas evaluation judge uses a different model family (OpenAI) to avoid the self-preference bias that comes from a model grading its own answers.
- **Real infrastructure, cost-bounded deliberately.** Every API call in this project is real (no mocked demo data) — but model choice, dataset size, and caching are all tuned to keep this genuinely cheap to run, documented in [`docs/adr/`](docs/adr/).

Full architecture, setup, and rationale are being written up as the project is built — see `docs/adr/` for the decision log as it grows.

## Status

This project is under active development. See the commit history for progress.

## License

MIT — see [LICENSE](LICENSE).
