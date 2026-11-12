"""Compare random search and qEHVI: hypervolume per paid evaluation, final fronts, and the
recommendation against the client's baseline.

Usage:
    uv run python -m src.costopt.compare      # writes docs/hv_comparison.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .moo import feasible, pareto
from .pipeline import BASELINE, done_configs, show

ROOT = Path(__file__).resolve().parents[2]


def curve(name: str) -> list[float]:
    path = ROOT / "runs" / f"{name}.jsonl"
    return [json.loads(l)["hv"] for l in path.read_text().splitlines()] if path.exists() else []


def main() -> None:
    rnd, bo = curve("random"), curve("qehvi")
    fig, ax = plt.subplots(figsize=(7, 4))
    for ys, label, style in ((rnd, "Random search", "--"), (bo, "qEHVI (Bayesian)", "-")):
        if ys:
            ax.plot(range(1, len(ys) + 1), ys, style, marker="o", ms=3, label=label)
    ax.axvline(6.5, color="grey", lw=0.8)
    ax.text(6.7, ax.get_ylim()[1] * 0.05, "shared start | search begins", fontsize=8, color="grey")
    ax.set_xlabel("Paid evaluations (configurations)")
    ax.set_ylabel("Hypervolume (higher is better)")
    ax.set_title("Finding the cost, quality and latency trade-off")
    ax.legend()
    fig.tight_layout()
    out = ROOT / "docs" / "hv_comparison.png"
    fig.savefig(out, dpi=200)
    print(f"saved {out}")
    for name, ys in (("random", rnd), ("qEHVI", bo)):
        if ys:
            hit = next((i + 1 for i, v in enumerate(ys) if v >= 0.95 * max(rnd + bo)), None)
            when = f"after {hit} evaluations" if hit else "never"
            print(f"{name:7s} final hv {ys[-1]:.4f}; reached 95% of the best hv found: {when}")

    rows = list(done_configs().values())
    base = done_configs()[BASELINE.id]
    front = pareto(rows, base)
    print(f"\nCombined Pareto front ({len(front)}):")
    for r in sorted(front, key=lambda r: r["cost_per_1k"]):
        show(r)
    ok = [r for r in rows if feasible(r, base) and r["quality"] >= base["quality"] - 0.02]
    best = min(ok, key=lambda r: r["cost_per_1k"])
    print("\nBaseline:")
    show(base)
    print("Cheapest configuration within 2 points of baseline quality:")
    show(best)
    print(f"-> {1 - best['cost_per_1k'] / base['cost_per_1k']:.0%} cheaper, "
          f"p95 {best['latency_p95'] - base['latency_p95']:+.1f}s, "
          f"quality {best['quality'] - base['quality']:+.2f}")


if __name__ == "__main__":
    main()
