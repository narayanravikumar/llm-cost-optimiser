# 0002: Bayesian multi objective optimisation (qEHVI) over NSGA-II

Status: accepted

## Context
Each configuration evaluation means running ~60 requests through a generator and a judge.
That costs money and minutes. Evolutionary methods like NSGA-II need thousands of evaluations.

## Decision
Use Gaussian process surrogates with q Expected Hypervolume Improvement (via Ax / BoTorch),
which is designed for expensive black box problems with a budget of tens to hundreds of evaluations.
Compare against random search under the same budget, measured by hypervolume.

## Consequences
- Mixed categorical and numeric search space. Implemented directly in BoTorch over a fully
  enumerated discrete space rather than through Ax (see ADR 0006).
- Noisy objectives (latency, judge variance) are handled by the GP noise term; repeated evaluation
  of the best configurations is a follow up.

Cost line: random search at equal budget is the control; the claim to prove is higher hypervolume
for the same spend.
