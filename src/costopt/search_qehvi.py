"""Bayesian multi objective search with qLogEHVI (BoTorch).

Starts from the same 6 configurations as random search, then for each step:
  1. fit one Gaussian process per objective (quality, cost, latency; all normalised, higher = better)
  2. score every configuration not yet evaluated by expected hypervolume improvement
  3. pay to evaluate the best one
The space has only 108 configurations, so the acquisition is maximised by scoring all of them
(ADR 0006) rather than with a continuous optimiser.

Usage:
    uv run python -m src.costopt.search_qehvi --budget 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import torch
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from gpytorch.mlls import SumMarginalLogLikelihood

from .llm import LLM
from .moo import feasible, hypervolume, initial_design, normalise, pareto
from .pipeline import BASELINE, SPACE, Config, all_configs, evaluate_and_log, show

OUT = Path(__file__).resolve().parents[2] / "runs" / "qehvi.jsonl"
torch.set_default_dtype(torch.double)


def encode(cfg: Config) -> list[float]:
    """9 numbers: one hot model (3), one hot prompt (3), context fraction, long cap, high effort."""
    x = [float(cfg.model == m) for m in SPACE["model"]]
    x += [float(cfg.prompt == p) for p in SPACE["prompt"]]
    x += [1.0 if cfg.context == "full" else int(cfg.context) / 100]
    x += [float(cfg.max_tokens == 512), float(cfg.effort == "high")]
    return x


def suggest(rows: list[dict], configs: list[Config], baseline: dict, candidates: list[Config]):
    X = torch.tensor([encode(c) for c in configs])
    Y = torch.tensor([normalise(r) for r in rows])
    gps = [SingleTaskGP(X, Y[:, i : i + 1], outcome_transform=Standardize(m=1)) for i in range(3)]
    model = ModelListGP(*gps)
    fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))

    # Only feasible points define the current front; infeasible ones sit at the reference point.
    mask = torch.tensor([feasible(r, baseline) for r in rows])
    Y_front = torch.where(mask.unsqueeze(-1), Y, torch.zeros_like(Y))
    ref = torch.zeros(3)
    acqf = qLogExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref.tolist(),
        partitioning=FastNondominatedPartitioning(ref_point=ref, Y=Y_front),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([256]), seed=0),
    )
    Xc = torch.tensor([encode(c) for c in candidates])
    with torch.no_grad():
        scores = acqf(Xc.unsqueeze(1))
    best = int(scores.argmax())
    return candidates[best], float(scores[best])


async def run(budget: int) -> None:
    torch.manual_seed(0)
    llm = LLM()
    baseline = await evaluate_and_log(BASELINE, llm)
    configs, rows = [], []
    OUT.write_text("")

    async def pay(step: int, cfg: Config, how: str, acq: float | None = None):
        row = await evaluate_and_log(cfg, llm)
        configs.append(cfg)
        rows.append(row)
        hv = hypervolume(rows, baseline)
        with OUT.open("a") as f:
            f.write(json.dumps({"step": step, "config_id": cfg.id, "hv": hv, "how": how,
                                "log_ehvi": acq}) + "\n")
        print(f"[{step:2d}/{budget}] {how:6s} hv {hv:.4f}  ", end="")
        show(row)

    for step, cfg in enumerate(initial_design(), 1):
        await pay(step, cfg, "init")
    for step in range(len(configs) + 1, budget + 1):
        candidates = [c for c in all_configs() if c not in configs]
        cfg, acq = suggest(rows, configs, baseline, candidates)
        await pay(step, cfg, "qEHVI", acq)

    front = pareto(rows, baseline)
    print(f"\nPareto front ({len(front)} configurations):")
    for r in sorted(front, key=lambda r: r["cost_per_1k"]):
        show(r)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=20)
    a = ap.parse_args()
    asyncio.run(run(a.budget))
