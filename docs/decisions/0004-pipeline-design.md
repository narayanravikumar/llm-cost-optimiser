# 0004: Pipeline configuration space and how each run is scored

Status: accepted

## Configuration space (3 x 3 x 3 x 2 x 2 = 108 configurations)
| Variable | Values | Why |
|---|---|---|
| Model | Haiku, Sonnet, Opus 5.5 | The biggest cost lever |
| Prompt | plain, grounded, extract then answer | Grounding instructions are free; extract then answer costs output tokens |
| Context kept | 100%, 60%, 35% of the reference material | Input tokens are most of the cost |
| Max output tokens | 256, 512 | Caps cost and latency |
| Effort | low, high | Replaces temperature, which the current API no longer accepts |

## Choices
- **Relative context budgets.** Eval set QA passages average ~330 tokens and articles ~660, so
  fixed budgets of 1,500 or 750 tokens would almost never trim. Percentages make the variable
  matter (average prompt: 687 tokens at 100%, 434 at 60%, 272 at 35%).
- **Trimming** keeps the sentences most relevant to the question (QA) or most central to the
  article (Summary), in original order. Deliberately simple: the point is the trade-off, not the
  best retriever.
- **The judge sees the full original source**, not the trimmed context. A claim that is true in
  the source is not a hallucination just because it was trimmed away.
- **Refusal rate is a guard rail, not an objective.** Trimming context can make a pipeline
  "faithful" by refusing to answer. Configurations whose refusal rate is more than 10 points above
  the baseline are treated as infeasible.
- **Cost objective counts generation only.** Judging is evaluation overhead the client would not pay
  in production; it is logged separately as eval cost.
- **Baseline (the client's current setup):** Sonnet, plain prompt, full context, 512 tokens,
  high effort.

Cost line: evaluating one configuration on 60 sources costs from about $0.02 (Haiku, 35% context)
to about $0.57 (Opus, extract, full context). The baseline costs about $0.28.
