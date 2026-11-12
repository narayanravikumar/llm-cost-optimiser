"""Turn the Pareto front into one recommendation for the client's priorities, and explain it
in plain English. The explanation may only use numbers from the facts it is given; any other
number triggers one retry, then the explanation is withheld (the project is about hallucination,
so the recommender must not hallucinate either).

Usage:
    uv run python -m src.costopt.recommend --min-quality 0.80
    uv run python -m src.costopt.recommend --min-quality 0.60 --w-cost 0.7 --w-quality 0.2 --w-latency 0.1
    uv run python -m src.costopt.recommend --min-quality 0.80 --no-explain
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re

from .moo import normalise, pareto
from .pipeline import BASELINE, done_configs
from .pricing import MODELS

LABELS = {
    "model": {"haiku": "Haiku (smallest model)", "sonnet": "Sonnet (mid model)", "opus": "Opus (largest model)"},
    "prompt": {"plain": "plain prompt", "grounded": "grounded prompt (answer only from the material)",
               "extract": "extract-then-answer prompt"},
    "context": {"full": "all reference material", "60": "60% of the reference material",
                "35": "35% of the reference material"},
    "effort": {"low": "low reasoning effort", "high": "high reasoning effort"},
}


def describe(row: dict) -> str:
    c = row["config"]
    return (f"{LABELS['model'][c['model']]}, {LABELS['prompt'][c['prompt']]}, "
            f"{LABELS['context'][c['context']]}, {c['max_tokens']} token answer limit, "
            f"{LABELS['effort'][c['effort']]}")


def recommend(min_quality: float, weights: dict[str, float], max_p95: float | None = None,
              max_cost: float | None = None, rows: list[dict] | None = None,
              base: dict | None = None) -> dict:
    """Best Pareto optimal configuration for the client's limits and priority weights.
    `rows`/`base` default to the local evaluation log; the web app passes its exported copy."""
    if rows is None:
        rows = list(done_configs().values())
        base = done_configs()[BASELINE.id]
    front = pareto(rows, base)
    ok = [r for r in front if r["quality"] >= min_quality
          and (max_p95 is None or r["latency_p95"] <= max_p95)
          and (max_cost is None or r["cost_per_1k"] <= max_cost)]
    if not ok:
        return {"status": "no_option", "baseline": base, "front_size": len(front)}
    total = sum(weights.values())
    w = [weights["quality"] / total, weights["cost"] / total, weights["latency"] / total]
    score = lambda r: sum(wi * xi for wi, xi in zip(w, normalise(r)))
    ranked = sorted(ok, key=score, reverse=True)
    return {"status": "ok", "choice": ranked[0], "runner_up": ranked[1] if len(ranked) > 1 else None,
            "baseline": base, "options": len(ok), "front_size": len(front)}


def facts(rec: dict, min_quality: float, weights: dict) -> dict:
    def card(r):
        return {"setup": describe(r), "quality_pct": round(100 * r["quality"]),
                "cost_per_1000_requests_usd": round(r["cost_per_1k"], 2),
                "p95_latency_seconds": round(r["latency_p95"], 1),
                "complete_answers_pct": round(100 * r["complete_rate"])}
    b, c = rec["baseline"], rec["choice"]
    cb, cc = card(b), card(c)  # changes are computed from the rounded figures so they add up
    out = {"client_requirements": {"minimum_quality_pct": round(100 * min_quality),
                                   "priority_weights": weights},
           "current_setup": cb, "recommended": cc,
           "change_vs_current": {
               "cost_change_pct": round(100 * (cc["cost_per_1000_requests_usd"]
                                               / cb["cost_per_1000_requests_usd"] - 1)),
               "quality_change_points": cc["quality_pct"] - cb["quality_pct"],
               "p95_latency_change_seconds": round(cc["p95_latency_seconds"]
                                                   - cb["p95_latency_seconds"], 1) + 0.0},
           "options_meeting_requirements": rec["options"], "evaluated_on_requests": c["n"],
           # 95% margin of error on a quality figure measured on n requests (normal approximation)
           "quality_margin_of_error_points": round(100 * 1.96 * (c["quality"] * (1 - c["quality"])
                                                                   / c["n"]) ** 0.5)}
    if rec["runner_up"]:
        out["next_best_alternative"] = card(rec["runner_up"])
    return out


SYSTEM = """You advise a client on how to configure their AI answering system.
Write 3 to 5 sentences of plain English for a non-technical manager:
what to switch to, what they gain, what they give up, and the next best alternative if given.
Use ONLY numbers that appear in the FACTS. Do not round differently, estimate or add numbers.
Mention that results come from a test of the stated number of requests.
If the quality change is smaller than the quality margin of error, say the quality is about the
same rather than claiming it rose or fell."""

_NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


def allowed_numbers(d) -> set[str]:
    s = json.dumps(d)
    nums = set(_NUM.findall(s)) | {"1000"}  # "per 1,000 requests" is a unit, not a claim
    return nums | {n.lstrip("-") for n in nums}


def unsupported(text: str, f: dict) -> list[str]:
    ok = allowed_numbers(f)
    return [n for n in _NUM.findall(text.replace(",", "")) if n not in ok and n.lstrip("-") not in ok]


async def explain(f: dict) -> str:
    from .llm import LLM

    llm = LLM()
    user = f"FACTS:\n{json.dumps(f, indent=2)}"
    for attempt in range(2):
        call = await llm.run(model=MODELS["sonnet"], system=SYSTEM, user=user, max_tokens=1024,
                             effort="low", tag=f"explain-v2-{attempt}")
        bad = unsupported(call.output, f)
        if not bad:
            return call.output.strip()
        user += f"\n\nYour previous draft used numbers not in the FACTS: {bad}. Use only FACTS numbers."
    return f"[Explanation withheld: it used numbers not in the facts: {bad}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-quality", type=float, default=0.80)
    ap.add_argument("--w-quality", type=float, default=0.4)
    ap.add_argument("--w-cost", type=float, default=0.4)
    ap.add_argument("--w-latency", type=float, default=0.2)
    ap.add_argument("--max-p95", type=float)
    ap.add_argument("--max-cost", type=float)
    ap.add_argument("--no-explain", action="store_true")
    a = ap.parse_args()
    weights = {"quality": a.w_quality, "cost": a.w_cost, "latency": a.w_latency}
    rec = recommend(a.min_quality, weights, a.max_p95, a.max_cost)
    if rec["status"] != "ok":
        print(f"No configuration on the Pareto front ({rec['front_size']}) meets these requirements.")
        return
    f = facts(rec, a.min_quality, weights)
    print(json.dumps(f, indent=2))
    if not a.no_explain:
        print("\n" + asyncio.run(explain(f)))


if __name__ == "__main__":
    main()
