"""Calibrate the judge against RAGTruth human labels, for each candidate judge model.

Usage:
    uv run python -m src.costopt.calibrate_judge --dry-run        # estimate cost, no API calls
    uv run python -m src.costopt.calibrate_judge --models haiku sonnet
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from .data import judge_calibration_set
from .judge import JUDGE_VERSION, SYSTEM
from .metrics import binary_report
from .pricing import MODELS, cost_usd

OUT = Path(__file__).resolve().parents[2] / "runs" / "judge_calibration.json"


def estimate(models: list[str], n: int) -> None:
    rows = judge_calibration_set(n)
    chars = sum(len(SYSTEM) + len(r.source.prompt) + len(r.response) for r in rows)
    in_tok = chars / 3.8 + 300 * len(rows)  # rough chars per token, plus tool schema
    out_tok = 120 * len(rows)
    for m in models:
        mid = MODELS[m]
        print(f"{m:7s} ~{in_tok/1e3:.0f}k in, ~{out_tok/1e3:.0f}k out  ->  ~${cost_usd(mid, int(in_tok), out_tok):.2f}")


async def calibrate(models: list[str], n: int) -> None:
    from .judge import judge
    from .llm import LLM

    rows = judge_calibration_set(n)
    llm = LLM()
    results = {"judge_version": JUDGE_VERSION, "n": len(rows), "models": {}}
    for m in models:
        mid = MODELS[m]
        calls = await asyncio.gather(*(judge(llm, mid, r.source.prompt, r.response) for r in rows))
        y_true = [r.hallucinated for r in rows]
        y_pred = [bool(c.output["hallucinated"]) for c in calls]
        rep = binary_report(y_true, y_pred)
        by_task = {}
        for task in ("QA", "Summary"):
            idx = [i for i, r in enumerate(rows) if r.source.task_type == task]
            by_task[task] = binary_report([y_true[i] for i in idx], [y_pred[i] for i in idx])
        lat = sorted(c.latency_s for c in calls)
        rep.update({
            "cost_usd": sum(c.cost for c in calls),
            "cost_per_1k_judgements": 1000 * sum(c.cost for c in calls) / len(calls),
            "latency_p50_s": statistics.median(lat),
            "latency_p95_s": lat[int(0.95 * (len(lat) - 1))],
            "by_task": by_task,
        })
        results["models"][m] = rep
        errors = [{"id": r.id, "task": r.source.task_type, "human": r.hallucinated,
                   "judge": c.output["hallucinated"], "reason": c.output["reason"],
                   "human_spans": r.spans, "judge_spans": c.output["spans"]}
                  for r, c in zip(rows, calls) if r.hallucinated != bool(c.output["hallucinated"])]
        (OUT.parent / f"judge_errors_{m}.json").write_text(json.dumps(errors, indent=2))
        print(f"{m:7s} F1 {rep['f1']:.2f}  kappa {rep['kappa']:.2f}  "
              f"P {rep['precision']:.2f}  R {rep['recall']:.2f}  "
              f"${rep['cost_per_1k_judgements']:.2f}/1k  p95 {rep['latency_p95_s']:.1f}s")
    OUT.write_text(json.dumps(results, indent=2))
    print(f"saved {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["haiku", "sonnet", "opus"], choices=list(MODELS))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    estimate(a.models, a.n) if a.dry_run else asyncio.run(calibrate(a.models, a.n))
