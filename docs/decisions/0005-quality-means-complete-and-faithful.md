# 0005: Quality objective = complete AND faithful, not faithfulness alone

Status: accepted (supersedes the faithfulness objective in the scoping doc)

## Context
The first random search put two Haiku configurations on the Pareto front with faithfulness 1.00
at 35% context. Inspection showed the metric was being gamed:
- 213 Haiku answers stopped at exactly the 256 token cap, mid sentence. A cut off answer
  contains no false claims, so the judge scored it faithful.
- In "extract then answer", 30 of 59 long outputs never reached the answer section. The runner
  fell back to the raw text, which was quotes copied from the source: "100% faithful".

## Root cause: thinking tokens eat the output budget
With effort enabled, hidden thinking is billed as output and counts against max_tokens.
| Model | Capped answers: billed output tokens | of which visible text | Ends mid sentence |
|---|---|---|---|
| Haiku | ~339 | ~28 | 97% |
| Opus | ~338 | ~95 | 98% |
| Sonnet | ~265 | ~168 | 89% |
Even uncapped Haiku answers spend about half their billed output tokens on thinking.

## Decision
- An answer counts only if it is complete: it did not hit max_tokens, and for "extract" it
  contains a closed <answer> section. Incomplete answers are failures and are not judged.
- Quality = share of requests with a complete answer that the judge finds faithful.
- Faithfulness among complete answers is still reported, but is not the objective.
- Configurations below quality 0.5 are infeasible (no usable product), as are those whose refusal
  rate exceeds the baseline by more than 10 points. Refusal rate is now over all requests.

## Consequences
- Baseline (Sonnet, plain, full context, 512, high effort): quality 0.82, 90% complete.
- Tight token caps with thinking on are broken configurations; the optimiser has to learn that.

Cost line: re-scoring all 21 runs cost $0 (every model call was cached).
