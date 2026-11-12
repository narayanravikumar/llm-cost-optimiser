"""Random search baseline: the same 6 starting configurations as qEHVI, then random picks,
tracking hypervolume after every paid evaluation.

Usage:
    uv run python -m src.costopt.search_random --dry-run
    uv run python -m src.costopt.search_random --budget 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

from .llm import LLM
from .moo import hypervolume, initial_design, pareto
from .pipeline import BASELINE, all_configs, done_configs, estimate, evaluate_and_log, show

OUT = Path(__file__).resolve().parents[2] / "runs" / "random.jsonl"


def plan(budget: int, seed: int = 1):
    init = initial_design()
    rest = [c for c in all_configs() if c not in init]
    return init + random.Random(seed).sample(rest, budget - len(init))


async def run(budget: int) -> None:
    llm = LLM()
    baseline = await evaluate_and_log(BASELINE, llm)
    rows = []
    OUT.write_text("")
    for step, cfg in enumerate(plan(budget), 1):
        row = await evaluate_and_log(cfg, llm)
        rows.append(row)
        hv = hypervolume(rows, baseline)
        with OUT.open("a") as f:
            f.write(json.dumps({"step": step, "config_id": cfg.id, "hv": hv}) + "\n")
        print(f"[{step:2d}/{budget}] hv {hv:.4f}  ", end="")
        show(row)
    front = pareto(rows, baseline)
    print(f"\nPareto front ({len(front)} configurations):")
    for r in sorted(front, key=lambda r: r["cost_per_1k"]):
        show(r)


def dry_run(budget: int) -> None:
    done = done_configs()
    total = 0.0
    for cfg in plan(budget):
        cost = 0.0 if cfg.id in done else estimate(cfg, quiet=True)
        total += cost
        print(f"{cfg.id:32s} {'cached' if cfg.id in done else f'~${cost:.2f}'}")
    print(f"\nTotal for {budget} configurations: ~${total:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    dry_run(a.budget) if a.dry_run else asyncio.run(run(a.budget))
