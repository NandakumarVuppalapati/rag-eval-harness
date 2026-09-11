"""Approximate per-token pricing for cost tracking.

These are maintained by hand and will drift from list price over time --
they exist so every query and every evaluation run carries a rough cost
estimate for the observability layer (docs/adr/0004), not to be an
invoice-accurate ledger. Update the constants below if a provider
changes pricing; nothing else in the codebase needs to change.

All figures are USD per 1,000,000 tokens.
"""

from __future__ import annotations

PRICING_USD_PER_MILLION_TOKENS = {
    # Generation models
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    # Embedding models
    "voyage-finance-2": {"input": 0.06, "output": 0.0},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    rates = PRICING_USD_PER_MILLION_TOKENS.get(model)
    if rates is None:
        return 0.0
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
