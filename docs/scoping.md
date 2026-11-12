# Scoping: LLM pipeline cost optimiser

## The client's ask
"Our AI bill is too high. Cut it."

## The real problem
Cost, answer quality and latency move together. Swapping to a cheaper model, trimming context or
shortening prompts each saves money, but some of those changes quietly increase hallucinations or
slow responses. Nobody can see the trade-off, so cuts are made blind and reversed after complaints.

## Objectives (three, competing)
| Objective | Direction | How it is measured |
|---|---|---|
| Faithfulness | maximise | % of answers with no hallucinated claim, scored by a judge calibrated against human labels (RAGTruth) |
| Cost | minimise | $ per 1,000 requests, from token counts and provider prices |
| Latency | minimise | p95 seconds per request |

## Decision variables (the pipeline configuration)
- Generator model (small, mid, large tier)
- Prompt strategy: plain, grounded (answer only from context), extract then answer
- Context kept: 100%, 60%, 35% of the reference material (most relevant sentences)
- Max output tokens, effort (see ADR 0004)

## Constraints
- Every evaluation costs real money, so the optimiser gets a budget of about 40 configurations.
- Faithfulness may not fall below a floor the client sets.

## Out of scope
- Fine tuning or self hosting models
- Changing the client's retrieval system
- Prompt caching at the provider level (noted as a follow up)

## What the client gets
1. A Pareto front: every configuration where you cannot cut cost without losing quality or speed.
2. A recommended configuration for their stated priorities, with a plain English explanation.
3. Evidence: the Bayesian optimiser finds the front in fewer paid evaluations than random search.

## Success metric
Same faithfulness as the current setup (within 2 points) at materially lower cost, found within the
evaluation budget.
