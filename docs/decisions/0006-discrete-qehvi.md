# 0006: qLogEHVI over the full discrete space, one GP per objective

Status: accepted

## Context
Each evaluation costs real money (up to ~$0.75 for one configuration), so the search must find
the trade-off front in as few evaluations as possible. This is an expensive black box,
multi objective problem: the textbook case for Bayesian optimisation with hypervolume based
acquisition.

## Decision
- **Surrogate:** one Gaussian process per objective (quality, cost, latency, all normalised so
  higher is better), via BoTorch ModelListGP with standardised outcomes.
- **Encoding:** 9 numbers per configuration: one hot model, one hot prompt, context fraction,
  long output cap flag, high effort flag.
- **Acquisition:** qLogExpectedHypervolumeImprovement (log form for stable gradients and scores
  when improvements are tiny), 256 quasi Monte Carlo samples.
- **Maximisation:** the space has only 108 configurations, so every unevaluated one is scored
  and the best is chosen. No continuous relaxation or rounding is needed.
- **Feasibility:** infeasible points (quality below 0.5, or refusals above baseline + 10 points)
  are placed at the reference point so they never define the front, but the GPs still learn from
  them.
- **Fair comparison:** qEHVI and random search share the same 6 starting configurations and the
  same budget of 20 evaluations.

## Not done (follow ups)
- Refusal rate modelled as an explicit outcome constraint.
- Batch suggestions (q > 1) to run several configurations in parallel.

Cost line: one suggestion takes ~2s of CPU; the search costs whatever the configurations it
chooses cost, at most ~$0.75 each.
