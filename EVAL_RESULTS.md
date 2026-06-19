# Evaluation Results — RAG AI Tutor

Tracks retrieval + generation quality over time, comparing this project's
**custom LLM-as-judge** (`src/evals.py`) against the **RAGAS** standard framework
(`src/ragas_eval.py`). This is an experiment log: each run is appended with its
full configuration so numbers stay interpretable as the system evolves.

> **Why two evaluators?** The custom judge was built first (binary 0/1 per answer,
> averaged). RAGAS scores at the *claim* level and adds retrieval-quality metrics
> (precision/recall) the custom keyword hit-rate only approximated. Running both
> cross-checks them: where they agree, confidence is high; where they diverge, the
> divergence is itself a finding (see Findings).

---

## Run log

| Run | Date | Dataset | Questions | Judge model | Embeddings | k values | What changed |
|-----|------|---------|-----------|-------------|------------|----------|--------------|
| R1  | 2026-06-19 | Microsoft Fabric docs test set v1 | 30 | gpt-4.1-mini (temp 0) | text-embedding-3-small (1536d) | 3, 10 | First RAGAS integration; baseline vs custom evals. Numbers from committed CSVs; ±0.03 run-to-run noise observed (see limitations). |

---

## Run R1 — 2026-06-19

**Config:** RAGAS 0.4.3 (classic `ragas.metrics` API) · Azure OpenAI judge `gpt-4.1-mini` @ temp 0 · same retrieval pipeline as the live app (`src/rag.py`) · 30-question Fabric-docs test set · Azure AI Search (HNSW).

### RAGAS metrics

Numbers below are from the committed CSVs (`data/eval/ragas_scores_k{3,10}.csv`).
They carry a run-to-run noise of roughly ±0.03 — see Known limitations #2.

| Metric | k=3 | k=10 | Direction k=3→k=10 |
|--------|-----|------|--------------------|
| faithfulness | 0.954 | 0.968 | ↑ better grounding |
| answer_relevancy | 0.899 | 0.894 | ↓ slight dilution |
| context_precision (w/ reference) | 0.897 | 0.777 | ↓↓ more noise |
| context_recall | 0.844 | 0.883 | ↑ slightly |

### RAGAS vs custom LLM-as-judge

| Metric | k | RAGAS | Custom | Gap | Interpretation |
|--------|---|-------|--------|-----|----------------|
| faithfulness | 3 | 0.954 | 1.000 | −0.046 | Custom binary judge scored *perfect*; RAGAS's per-claim scoring exposed small grounding gaps the coarse 0/1 metric hid. |
| answer_relevancy | 3 | 0.899 | 0.867 | +0.032 | Close agreement; both methods converge. |
| faithfulness | 10 | 0.968 | 1.000 | −0.032 | Same pattern — more context improves grounding, narrowing the divergence. |
| answer_relevancy | 10 | 0.894 | 0.933 | −0.039 | Mild disagreement; within noise. |

---

## Findings

1. **k=3 is the better operating point for this corpus.**
   Raising k from 3→10 trades **precision for faithfulness**: faithfulness rises
   (0.954→0.968) but context_precision falls sharply (0.897→0.777). Recall improves
   only modestly (0.844→0.883), so the extra 7 chunks add more noise than coverage.
   **Decision: keep k=3 in production.** (The precision drop is the consistent
   signal across both runs; faithfulness/recall vary within the ±0.03 noise band.)

2. **The custom judge was over-lenient on faithfulness.**
   It reported a perfect 1.000 at both k values; RAGAS — scoring each claim in the
   answer independently — found 0.954 (k=3) / 0.968 (k=10). The custom metric's
   whole-answer binary verdict gave no partial credit, so a mostly-grounded answer
   with one weak claim still scored 1. This is the central reason RAGAS was added:
   it measures what the custom keyword approach structurally could not.

3. **answer_relevancy agrees across both methods** (within ~0.04), which validates
   the custom judge on the metric where the two are directly comparable.

4. **q03 nuance (partial-credit illustration).** Q: "How many workspace roles does
   Fabric support?" — system answered "four workspace roles" (correct); reference
   added "(workload-level RBAC)". A binary metric scores this 1 or 0; RAGAS can
   reflect the missing sub-detail as a fractional recall — exactly the granularity
   the upgrade was meant to capture.

---

## Known limitations

| # | Limitation | Impact |
|---|-----------|--------|
| 1 | **Self-preference bias** — judge (`gpt-4.1-mini`) is the same model family that generated the answers | Both custom and RAGAS scores may be inflated; affects absolute values, not the relative k=3 vs k=10 comparison |
| 2 | **Run-to-run noise (~±0.03)** — the judge returns 1 generation per judgment, not the 3 RAGAS requests (`LLM returned 1 generation` warnings), so each score is a single noisy draw rather than an average | Two runs of the *same* config differed by up to 0.034 (context_recall k=3: 0.878 vs 0.844). Treat all numbers as ±0.03; trust the *direction* of large gaps (e.g. precision k=3 vs k=10), not the third decimal. Fixable by configuring `n=3` on the judge deployment. |
| 3 | Single dataset (Fabric docs), 30 questions | Small N; results are directional, not statistically tight |
| 4 | RAGAS classic API (0.4.3) emits deprecation warnings | Cosmetic; pinned deliberately (newer collections API needs a different LLM interface) |

---

## Roadmap (future runs append above as R2, R3, …)

| Planned | Expected effect on these numbers |
|---------|----------------------------------|
| Stronger independent judge (gpt-4.1, not -mini) | Reduces self-preference bias → likely lowers absolute scores, raises trust |
| Phoenix (Arize) tracing via OpenTelemetry | Per-question drill-down, no score change |
| Better PDF extraction (tables/headings) | Should raise context_precision + recall |
| Re-test after agentic (LangGraph) layer | New baseline for the agentic answer path |

---

## Reproducing

```bash
# 1. Build the eval dataset (main venv — needs Azure Search SDK)
.venv\Scripts\python -m src.ragas_prep

# 2. Score with RAGAS (isolated venv — see requirements-eval.txt)
.venv-eval\Scripts\python src\ragas_eval.py

# Per-question scores are written to data/eval/ragas_scores_k{3,10}.csv
```