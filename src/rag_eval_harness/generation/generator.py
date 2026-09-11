"""Generate grounded answers from retrieved context using Claude Haiku.

The system prompt is deliberately strict: answer only from the provided
excerpts, and say so explicitly when the answer isn't in them. That
refusal behavior is itself one of the things the golden dataset tests
(see the unanswerable-question category) -- a model that quietly
hallucinates a plausible number when the context doesn't contain one is
exactly the failure mode this whole project exists to catch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import anthropic

from rag_eval_harness.observability.pricing import estimate_cost_usd
from rag_eval_harness.retrieval.retriever import RetrievedChunk

GENERATION_MODEL = "claude-haiku-4-5"

REFUSAL_TEXT = "I cannot answer this based on the provided context."

_SYSTEM_PROMPT = f"""You are a financial research assistant answering questions about \
public companies using ONLY the excerpts provided below, drawn from their SEC filings \
(10-K annual reports and 10-Q quarterly reports).

Rules:
1. Answer strictly from the provided excerpts. Do not use outside knowledge, even if \
you are confident about the real-world answer.
2. If the excerpts do not contain enough information to answer the question, respond \
with exactly this sentence and nothing else: "{REFUSAL_TEXT}"
3. When you do answer, cite the company ticker and filing (form + report date) each \
fact came from, e.g. "(AAPL 10-K, 2025-09-27)".
4. Be precise with numbers -- do not round or approximate figures that appear exactly \
in the excerpts.
"""


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    refused: bool


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks):
        blocks.append(
            f"[Excerpt {i+1} -- {c.ticker} {c.form}, period {c.report_date}]\n{c.text}"
        )
    return "\n\n".join(blocks)


class Generator:
    def __init__(self, client: anthropic.Anthropic, model: str = GENERATION_MODEL) -> None:
        self._client = client
        self._model = model

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> GenerationResult:
        context_block = _format_context(chunks)
        user_message = (
            f"Excerpts:\n\n{context_block}\n\n---\n\nQuestion: {question}"
        )

        start = time.monotonic()
        response = self._client.messages.create(
            model=self._model,
            max_tokens=500,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        latency_ms = (time.monotonic() - start) * 1000

        answer = response.content[0].text.strip()
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = estimate_cost_usd(self._model, input_tokens, output_tokens)

        return GenerationResult(
            answer=answer,
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            refused=(answer.strip() == REFUSAL_TEXT),
        )
