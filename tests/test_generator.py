"""Tests for refusal detection in Generator.generate.

Regression coverage for a real bug: refusal was originally detected with
an exact-string match against REFUSAL_TEXT, but Claude Haiku routinely
appends an explanation after the required refusal sentence despite the
system prompt asking for "nothing else". That made every correct refusal
in a real evaluation run score as a hallucination (0% refusal rate on the
unanswerable golden-dataset slice, when the model was in fact refusing
every single one) -- see the docstring in generator.py for how this was
caught.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from rag_eval_harness.generation.generator import REFUSAL_TEXT, Generator


def _fake_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
    )


def _generator_with_response(text: str) -> Generator:
    client = MagicMock()
    client.messages.create.return_value = _fake_response(text)
    return Generator(client)


def test_exact_refusal_text_is_detected():
    generator = _generator_with_response(REFUSAL_TEXT)
    result = generator.generate("What was Tesla's revenue?", chunks=[])
    assert result.refused is True


def test_refusal_with_trailing_explanation_is_still_detected():
    # This is the shape Claude Haiku actually produces in practice: the
    # required sentence, followed by an explanation it wasn't asked for.
    text = (
        f"{REFUSAL_TEXT}\n\nThe excerpts provided are from Johnson & Johnson's "
        "SEC filings and do not contain any information about Tesla."
    )
    generator = _generator_with_response(text)
    result = generator.generate("What was Tesla's revenue?", chunks=[])
    assert result.refused is True


def test_grounded_answer_is_not_flagged_as_refusal():
    text = "Apple's total net sales were $416,161 million. (AAPL 10-K, 2025-09-27)"
    generator = _generator_with_response(text)
    result = generator.generate("What were Apple's net sales?", chunks=[])
    assert result.refused is False


def test_answer_that_merely_mentions_the_refusal_phrase_midway_is_not_a_refusal():
    # Only a genuine *prefix* match should count -- a grounded answer that
    # happens to reference the same words partway through should not.
    text = (
        "The filing does not say whether the company cannot answer this "
        "based on the provided context in other years; for fiscal 2025, "
        "net sales were $416,161 million."
    )
    generator = _generator_with_response(text)
    result = generator.generate("What were Apple's net sales?", chunks=[])
    assert result.refused is False
