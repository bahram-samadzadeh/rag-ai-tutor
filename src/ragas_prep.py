"""Stage 1 of the RAGAS evaluation: build the dataset RAGAS will score.

RAGAS needs four things per question — the question, the generated answer, the
retrieved context chunks, and a reference answer. This module produces exactly
that by running the *real* pipeline (the same get_clients/embed_query/retrieve/
generate_answer functions the live app uses), then writes the result to JSON.

Splitting "run the pipeline" (here) from "score with RAGAS" (src/ragas_eval.py)
mirrors the rest of this project's decoupled-stage design and has a concrete
payoff: RAGAS scoring can be re-run and tuned against this JSON without paying
for another round of Azure retrieval + generation calls.

This stage runs in the MAIN .venv (it needs the Azure Search SDK). The scoring
stage runs in the isolated .venv-eval. Run from the project root:

    .venv\\Scripts\\python -m src.ragas_prep            # full 30-question set
    .venv\\Scripts\\python -m src.ragas_prep --limit 3  # smoke test first
"""
import argparse
import json

from .rag import get_clients, embed_query, retrieve, generate_answer

TEST_SET_PATH = "data/eval/test_set.json"
OUTPUT_TEMPLATE = "data/eval/ragas_dataset_k{k}.json"

# Same retrieval depths the custom evals use, so the RAGAS numbers can be
# compared to the custom numbers at matching k rather than across different k.
K_VALUES = [3, 10]


def load_test_set(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_dataset(openai_client, search_client, test_set: list[dict], k: int) -> list[dict]:
    """Run retrieval + answer generation at depth k and shape it for RAGAS.

    `retrieved_contexts` is kept as a LIST of per-chunk strings (not the joined
    block) because RAGAS scores context precision/recall per retrieved chunk.
    The answer, however, is generated from the joined context exactly as the
    live app does, so the response we score is the real system's response.
    """
    samples = []
    for item in test_set:
        question = item["question"]

        q_vec = embed_query(openai_client, question)
        chunks = retrieve(search_client, q_vec, question, k=k)
        contexts = [c["content"] for c in chunks]

        context_block = "\n\n".join(contexts)
        answer = generate_answer(openai_client, question, context_block) or ""

        samples.append(
            {
                "id": item["id"],
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": item["reference_answer"],
            }
        )
        print(f"  k={k} {item['id']}: {len(contexts)} contexts, answer {len(answer)} chars")

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the RAGAS evaluation dataset.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N questions (smoke test before the full run).",
    )
    args = parser.parse_args()

    test_set = load_test_set(TEST_SET_PATH)
    if args.limit:
        test_set = test_set[: args.limit]

    # Same clients and pipeline functions the live app uses.
    openai_client, search_client = get_clients()

    for k in K_VALUES:
        print(f"\n--- Building dataset at k={k} ({len(test_set)} questions) ---")
        samples = build_dataset(openai_client, search_client, test_set, k)
        out_path = OUTPUT_TEMPLATE.format(k=k)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(samples)} samples -> {out_path}")


if __name__ == "__main__":
    main()
