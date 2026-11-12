# 0003: Haiku 5.5 as the hallucination judge

Status: accepted

## Context
The optimiser's quality objective (faithfulness) is scored by an LLM judge. The judge must agree
with human labels and be cheap enough to run on every candidate configuration.

## Evidence
Calibration on 200 RAGTruth test responses (balanced: 100 hallucinated, 100 clean, half QA,
half Summary). Judge v2: structured outputs, short quoted spans.

| Judge | F1 | Cohen's kappa | Precision | Recall | False alarms | Misses | $ per 1k judgements | p95 latency |
|---|---|---|---|---|---|---|---|---|
| Haiku 5.5 | 0.83 | 0.63 | 0.76 | 0.92 | 29 | 8 | $0.35 | 3.9s |
| Sonnet 5.5 | 0.79 | 0.50 | 0.68 | 0.95 | 45 | 5 | $4.48 | 4.2s |
| Opus 5.5 | 0.79 | 0.51 | 0.69 | 0.93 | 42 | 7 | $9.97 | 6.1s |

Kappa is lower on Summary than QA for every judge (Haiku 0.58 vs 0.68).

## Decision
Use Haiku 5.5. It has the highest agreement with human labels, is about 28x cheaper than Opus
and the fastest.

## What the errors show
- Larger judges are stricter, not worse at reading: they flag more, so they lose precision
  against the human labels.
- 25 responses are flagged by all three judges but labelled clean by humans. On inspection many
  contain real unsupported claims (a year not in the source, effects attributed to the wrong
  cause). RAGTruth labels miss some hallucinations, so measured precision understates the judge.
- Opus-only flags are a mix of real catches (a name meaning taken from the wrong passage, a
  reversed cause) and pedantry (an unsupported "frequent").
- Only 3 hallucinated responses are missed by all three judges.

## Follow up
Hand label the 25 shared disagreements and report kappa against corrected labels.

Cost line: calibrating all three judges cost about $3. Scoring one candidate configuration
(60 answers) with Haiku costs about $0.02.
