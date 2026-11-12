"""Model IDs and prices (USD per million tokens, prompts under 100K tokens).

Checked November 2026 against published pricing. Verify against Anthropic's docs before
quoting figures publicly.
"""
from __future__ import annotations

MODELS = {
    "haiku": "claude-haiku-5-5",
    "sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
}

PRICES = {  # (input, output) per million tokens
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    inp, out = PRICES[model]
    return (input_tokens * inp + output_tokens * out) / 1_000_000
