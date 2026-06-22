# Baseline before agent — 2026-06-22

Captured immediately before building the LangGraph agentic layer, so
post-agent runs can be compared against this baseline (before/after evidence).

## Aggregate metrics

| metric        | k=3    | k=10   |
|---------------|--------|--------|
| hit_rate      | 100%   | 100%   |
| faithfulness  | 88%    | 84%    |
| relevancy     | 84%    | 80%    |

Test set: 25 questions (10 control easy + 15 hard compositional).
Judge: gpt-4.1-mini, binary 0/1 (known self-preference bias).

## Known failure modes (what the agent must fix)

| Question | k that failed | Failure type      | What the model did wrong                                  |
|----------|---------------|-------------------|-----------------------------------------------------------|
| h01      | k=10          | composition       | "Large = 24 nodes on F64" (correct: 8)                    |
| h03      | k=3, k=10     | composition       | Right answer (F32), muddled reasoning conflated with F64  |
| h04      | k=3, k=10     | confabulation     | k=10 invented "Optimize Write" name; missed 50% compression |
| h05      | k=3, k=10     | confabulation     | k=10 invented Runtime 2.0 / Spark 4.1 / Delta 4.1         |
| h06      | k=3           | confabulation     | Invented SQL syntax `RETAIN 168 HOURS` (real = Spark flag)|
| h14      | k=10          | confabulation     | Computed F128 limits despite F128 not being in docs       |

## Diagnosis

Two distinct problems, not one:

1. **Confabulation** — model invents plausible-sounding facts despite system
   prompt instructing "use only the provided context." Dominant on h04, h05,
   h06, h14.
2. **Composition / multi-fact omission** — model retrieves correct chunks but
   combines them wrongly or skips required facts. h01, h03, h04 (50% miss).

Retrieval is NOT the bottleneck (hit_rate 100% on every question, including
hard multi-hop). Agent value goes into **forcing grounded answering**, not
into more retrieval.

## Agent design implication

Primary agent job:
- **Citation-forced verification** — require model to quote the source
  sentence for each claim; refuse if it cannot.
- **Claim decomposition + coverage** — split multi-fact questions into
  atomic claims, verify each independently.

Re-retrieval loop is secondary (retrieval already strong).

## Target after agent

Move faithfulness from 84-88% toward 95%+; relevancy from 80-84% toward
90%+; preserve hit_rate at 100%; expect latency to roughly double from the
extra verification calls.
