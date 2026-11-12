"""Export everything the web page needs into web/results.json (committed, so the deployed app
needs no API key and costs nothing per visitor).

Usage:
    uv run python -m src.costopt.export
"""
from __future__ import annotations

import json
from pathlib import Path

from .moo import COST_RANGE, LATENCY_CEIL, QUALITY_FLOOR, REFUSAL_SLACK, feasible, pareto
from .pipeline import BASELINE, done_configs
from .recommend import describe

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "web" / "results.json"
KEEP = ("config", "config_id", "n", "quality", "complete_rate", "faithfulness", "truncated_rate",
        "cost_per_1k", "latency_p50", "latency_p95", "refusal_rate", "avg_output_tokens")


def curve(name: str) -> list[dict]:
    path = ROOT / "runs" / f"{name}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def main() -> None:
    rows = list(done_configs().values())
    base = done_configs()[BASELINE.id]
    front = {r["config_id"] for r in pareto(rows, base)}
    data = {
        "baseline_id": BASELINE.id,
        "normalisation": {"quality_floor": QUALITY_FLOOR, "cost_range": COST_RANGE,
                          "latency_ceiling": LATENCY_CEIL, "refusal_slack": REFUSAL_SLACK},
        "rows": [{**{k: r[k] for k in KEEP}, "label": describe(r), "on_front": r["config_id"] in front,
                  "feasible": feasible(r, base)} for r in rows],
        "search": {"random": curve("random"), "qehvi": curve("qehvi")},
    }
    OUT.write_text(json.dumps(data, indent=1))
    print(f"{len(rows)} configurations ({len(front)} on the Pareto front) -> {OUT}")


if __name__ == "__main__":
    main()
