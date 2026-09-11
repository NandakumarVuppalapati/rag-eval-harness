# ADR 0002: Dual embedding indexes -- domain-tuned primary, general-purpose baseline

## Status

Accepted

## Context

Choosing an embedding model for a domain-specific RAG system is usually
done by assumption ("a finance-tuned model should retrieve better on
financial text") rather than by measurement. This project's whole
premise is not trusting RAG behavior without evidence, so the embedding
choice should be held to the same standard as everything else it
evaluates.

Voyage AI publishes `voyage-finance-2`, an embedding model trained
specifically on financial text, and general-purpose alternatives like
OpenAI's `text-embedding-3-small` are the default choice most RAG
tutorials reach for. They also have different output dimensionality
(1024 vs 1536), which matters mechanically: a Pinecone index has a
fixed dimension, so serving both from one index isn't an option.

## Decision

Build two complete, parallel Pinecone indexes from the identical chunk
set: `rag-eval-harness-voyage` (voyage-finance-2, primary) and
`rag-eval-harness-openai` (text-embedding-3-small, baseline). Every
chunk is embedded and upserted into both, with identical chunk_ids and
metadata, so retrieval quality between the two is a controlled
comparison -- same chunks, same metadata, only the embedding model
differs.

The Ragas evaluation harness (see docs/adr/0003, once written) runs
the same golden-question set against both indexes, so the retrieval
precision/recall delta between domain-tuned and general-purpose
embeddings is a real measured number in this project's results, not an
assumption in its README.

## Consequences

- Roughly double the embedding cost and Pinecone storage versus a
  single-index design -- negligible in absolute terms (a few dollars
  total for ~11,900 chunks at either model's per-token price) but worth
  naming as a real, deliberate tradeoff.
- Any future embedding model swap follows the same pattern: add a
  third index rather than replace one, so historical comparisons stay
  valid.
- If the ablation shows no meaningful difference between the two
  models on this corpus, that is itself a useful, reportable result --
  the point of measuring is being willing to find either answer.
