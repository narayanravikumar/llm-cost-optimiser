"""Shared multi objective helpers: normalisation, feasibility, Pareto front, hypervolume,
and the initial design both search methods start from (so the comparison is fair)."""
from __future__ import annotations

import math
import random

from .pipeline import Config, all_configs

# Fixed bounds so random search and qEHVI are scored against the same reference point.
# Every objective is mapped to [0, 1] where higher is better; the reference point is the origin.
QUALITY_FLOOR = 0.50        # below this a configuration earns no hypervolume
COST_RANGE = (0.10, 12.0)   # $ per 1k requests, log scale (spans Haiku/short to Opus/full)
LATENCY_CEIL = 15.0         # p95 seconds
REFUSAL_SLACK = 0.10        # ADR 0004: refusal rate may exceed the baseline by at most this


def normalise(row: dict) -> tuple[float, float, float]:
    f = (row["quality"] - QUALITY_FLOOR) / (1 - QUALITY_FLOOR)
    lo, hi = map(math.log10, COST_RANGE)
    c = (hi - math.log10(max(row["cost_per_1k"], 1e-9))) / (hi - lo)
    l = 1 - row["latency_p95"] / LATENCY_CEIL
    return tuple(min(1.0, max(0.0, v)) for v in (f, c, l))


def feasible(row: dict, baseline: dict) -> bool:
    """Usable at all (quality above the floor) and not buying faithfulness with refusals."""
    return (row["quality"] > QUALITY_FLOOR
            and row["refusal_rate"] <= baseline["refusal_rate"] + REFUSAL_SLACK)


def dominates(a, b) -> bool:
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def pareto(rows: list[dict], baseline: dict) -> list[dict]:
    ok = [r for r in rows if feasible(r, baseline)]
    pts = [normalise(r) for r in ok]
    return [r for r, p in zip(ok, pts) if not any(dominates(q, p) for q in pts if q is not p)]


def _area2d(pts: list[tuple[float, float]]) -> float:
    area, best_y = 0.0, 0.0
    for x, y in sorted(pts, reverse=True):  # sweep x from high to low
        if y > best_y:
            area += x * (y - best_y)
            best_y = y
    return area


def hypervolume(rows: list[dict], baseline: dict) -> float:
    """Exact 3D hypervolume (maximisation, reference point at the origin) by slicing on the
    third objective. Fine for the few dozen points this project produces."""
    pts = [normalise(r) for r in rows if feasible(r, baseline)]
    if not pts:
        return 0.0
    zs = sorted({p[2] for p in pts}, reverse=True) + [0.0]
    hv = 0.0
    for z_hi, z_lo in zip(zs, zs[1:]):
        hv += _area2d([(p[0], p[1]) for p in pts if p[2] >= z_hi]) * (z_hi - z_lo)
    return hv


def initial_design(n: int = 6, seed: int = 0) -> list[Config]:
    return random.Random(seed).sample(all_configs(), n)
