"""Hallucination judge: given the source prompt and a response, decide whether the response
contains any claim not supported by the source."""
from __future__ import annotations

from .llm import LLM, Call

JUDGE_VERSION = "v2"  # v2: structured outputs instead of forced tool use

SYSTEM = """You are a strict fact checker for retrieval augmented generation.
You receive the SOURCE (the instructions and reference material a model was given) and the
RESPONSE the model produced. Decide whether the RESPONSE contains any hallucination:

- Conflict: a claim that contradicts the source (wrong date, number, name, relation).
- Baseless information: a claim the source does not support, even if it might be true in the world.

Paraphrase, summarising and reasonable wording changes are NOT hallucinations.
Saying the source does not contain the answer is NOT a hallucination.
Quote each hallucinated span exactly as it appears in the RESPONSE. Keep spans short: the key phrase, not whole paragraphs."""

TOOL = {
    "name": "verdict",
    "description": "Record the hallucination verdict.",
    "input_schema": {
        "type": "object",
        "properties": {
            "hallucinated": {"type": "boolean"},
            "spans": {"type": "array", "items": {"type": "string"},
                      "description": "Exact hallucinated text from the RESPONSE (max 5 spans, each under 30 words), empty if none."},
            "reason": {"type": "string", "description": "One sentence."},
        },
        "required": ["hallucinated", "spans", "reason"],
    },
}


async def judge(llm: LLM, model: str, source_prompt: str, response: str) -> Call:
    user = f"<SOURCE>\n{source_prompt}\n</SOURCE>\n\n<RESPONSE>\n{response}\n</RESPONSE>"
    return await llm.run(model=model, system=SYSTEM, user=user, max_tokens=1024,
                         tool=TOOL, tag=f"judge-{JUDGE_VERSION}")
