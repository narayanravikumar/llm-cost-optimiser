# 0001: Use RAGTruth as the evaluation data and judge ground truth

Status: accepted

## Context
The quality objective needs a judge we can trust. A judge is only trustworthy if it is checked
against human labels.

## Decision
Use RAGTruth (QA and Summary tasks, test split). It has ~18k responses from six models with
human span level hallucination labels. The judge is calibrated against these labels before it
scores anything in the optimiser. A small set of my own labels is used as a sanity check.

## Consequences
- Data2txt is excluded to keep the first version focused.
- Labels come from 2023 era models, so absolute hallucination rates will differ from current models.

Cost line: no direct cost. Saves hand labelling hundreds of examples.
