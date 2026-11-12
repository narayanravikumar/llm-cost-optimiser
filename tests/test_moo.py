"""Checks on the multi objective maths the whole study depends on."""
import random

from src.costopt import moo

BASE = {"refusal_rate": 0.03}


def row(q, cost, lat, refusal=0.0):
    return {"quality": q, "cost_per_1k": cost, "latency_p95": lat, "refusal_rate": refusal}


def test_perfect_point_fills_the_unit_cube():
    assert abs(moo.hypervolume([row(1.0, 0.10, 0.0)], BASE) - 1.0) < 1e-9


def test_hypervolume_matches_monte_carlo():
    pts = [row(0.85, 1.0, 7.5), row(0.95, 5.0, 4.0), row(0.80, 0.3, 2.0), row(0.84, 1.2, 8.0)]
    exact = moo.hypervolume(pts, BASE)
    norm = [moo.normalise(p) for p in pts]
    rng, n = random.Random(0), 100_000
    hits = sum(any(all(u[i] <= p[i] for i in range(3)) for p in norm)
               for u in ([rng.random() for _ in range(3)] for _ in range(n)))
    assert abs(exact - hits / n) < 0.01


def test_infeasible_points_add_nothing():
    assert moo.hypervolume([row(0.99, 0.1, 1.0, refusal=0.5)], BASE) == 0.0  # too many refusals
    assert moo.hypervolume([row(0.40, 0.1, 1.0)], BASE) == 0.0               # below quality floor


def test_pareto_drops_dominated_points():
    a, b = row(0.9, 1.0, 3.0), row(0.8, 2.0, 4.0)  # a is better on every objective
    assert moo.pareto([a, b], BASE) == [a]
