"""Stage 2 of the RAGAS evaluation: score the prepared dataset with RAGAS.

Reads the datasets produced by src/ragas_prep.py and scores them with RAGAS's
standard RAG metrics, then prints the aggregate numbers next to this project's
own custom-eval numbers so the two can be compared.

Metrics:
  - faithfulness        : is the answer grounded in the retrieved context?
  - answer_relevancy    : does the answer actually address the question?
  - context_precision   : are the retrieved chunks relevant (ranked well)?
  - context_recall      : did retrieval cover what the reference answer needs?

The first two mirror the custom judge in src/evals.py; the last two are
retrieval-quality metrics the custom evals only approximated with a keyword
hit-rate, so RAGAS adds genuinely new signal here.

This script runs in the ISOLATED .venv-eval (see requirements-eval.txt for why).
It does not import the project's `src` package or the Azure Search SDK — it only
reads the prepared JSON and calls Azure OpenAI as the judge. Run from project root:

    .venv-eval\\Scripts\\python src\\ragas_eval.py --k 3 --limit 3 --strict   # smoke test
    .venv-eval\\Scripts\\python src\\ragas_eval.py                            # full, both k

Per-question scores are written to data/eval/ragas_scores_k{k}.csv so results
survive the process (RAGAS only returns them in-memory).

NOTE: this uses the classic `ragas.metrics` API. In ragas 0.4.3 it emits a
deprecation warning (the redesigned `ragas.metrics.collections` metrics require
ragas's instructor-LLM interface rather than the langchain wrappers used here).
The classic path integrates cleanly with Azure OpenAI via `evaluate()` and is
stable through the 0.x line, so it is the deliberate choice; the warnings are
silenced below.
"""
import argparse
import json
import os
import warnings

from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

from ragas import EvaluationDataset, SingleTurnSample, RunConfig, evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

DATASET_TEMPLATE = "data/eval/ragas_dataset_k{k}.json"
SCORES_TEMPLATE = "data/eval/ragas_scores_k{k}.csv"
K_VALUES = [3, 10]

# Custom-eval numbers from src/evals.py on the 30-question set, for side-by-side
# context. Update these if you re-run the custom evals. RAGAS scores 0-1; the
# custom judge is binary 0/1 averaged, so both are on the same 0-1 scale.
CUSTOM_REFERENCE = {
    3: {"faithfulness": 1.00, "answer_relevancy": 0.867},
    10: {"faithfulness": 1.00, "answer_relevancy": 0.933},
}


def build_judge():
    """Build the Azure OpenAI judge LLM + embeddings, wrapped for RAGAS.

    Same deployments the pipeline uses (gpt-4.1-mini judge, text-embedding-3-small).
    temperature=0 on the judge for reproducible scoring. Self-preference bias is a
    known limitation: the judge model is the same family that wrote the answers.
    """
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_KEY"]
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    chat_deployment = os.getenv("CHAT_DEPLOYMENT", "gpt-4.1-mini")
    embed_deployment = os.getenv("EMBED_DEPLOYMENT", "text-embedding-3-small")

    llm = AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_deployment=chat_deployment,
        temperature=0,
    )
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_deployment=embed_deployment,
    )
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


def load_dataset(k: int, limit: int | None) -> EvaluationDataset:
    with open(DATASET_TEMPLATE.format(k=k), encoding="utf-8") as f:
        rows = json.load(f)
    if limit:
        rows = rows[:limit]
    samples = [
        SingleTurnSample(
            user_input=r["user_input"],
            response=r["response"],
            retrieved_contexts=r["retrieved_contexts"],
            reference=r["reference"],
        )
        for r in rows
    ]
    return EvaluationDataset(samples=samples)


def score_k(k: int, llm, embeddings, limit: int | None, strict: bool) -> dict:
    dataset = load_dataset(k, limit)
    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]
    # max_workers kept low because the Azure free trial rate-limits aggressively;
    # high retry/wait lets RAGAS ride out 429s rather than failing the run.
    run_config = RunConfig(max_workers=4, max_retries=15, max_wait=60, timeout=180)

    print(f"\n--- Scoring k={k} ({len(dataset)} samples) ---")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
        raise_exceptions=strict,
        show_progress=True,
    )

    df = result.to_pandas()

    # Persist per-question scores — RAGAS only returns them in-memory, so without
    # this they vanish on exit. CSV: git-diffable, auditable, the standard eval
    # artifact. Skipped during --limit smoke tests so partial runs don't overwrite
    # a full result file.
    if not limit:
        scores_path = SCORES_TEMPLATE.format(k=k)
        df.to_csv(scores_path, index=False)
        print(f"Saved per-question scores -> {scores_path}")

    means = {col: float(df[col].mean()) for col in df.select_dtypes("number").columns}
    return means


def print_comparison(k: int, ragas_means: dict) -> None:
    print(f"\n===== RAGAS results @ k={k} =====")
    for metric, value in ragas_means.items():
        print(f"  {metric:<32} {value:.3f}")

    ref = CUSTOM_REFERENCE.get(k, {})
    if ref:
        print(f"\n  vs custom evals (src/evals.py) @ k={k}:")
        print(f"  {'metric':<22}{'RAGAS':>8}{'custom':>9}")
        pairs = [("faithfulness", "faithfulness"), ("answer_relevancy", "answer_relevancy")]
        for ragas_key, custom_key in pairs:
            r = next((v for c, v in ragas_means.items() if ragas_key in c), None)
            c = ref.get(custom_key)
            if r is not None and c is not None:
                print(f"  {custom_key:<22}{r:>8.3f}{c:>9.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the RAGAS dataset.")
    parser.add_argument("--k", type=int, choices=K_VALUES, help="Score only this k (default: both).")
    parser.add_argument("--limit", type=int, default=None, help="Score only first N samples (smoke test).")
    parser.add_argument("--strict", action="store_true", help="Raise on first error (use for smoke tests).")
    args = parser.parse_args()

    load_dotenv()
    llm, embeddings = build_judge()

    ks = [args.k] if args.k else K_VALUES
    for k in ks:
        means = score_k(k, llm, embeddings, args.limit, args.strict)
        print_comparison(k, means)


if __name__ == "__main__":
    main()