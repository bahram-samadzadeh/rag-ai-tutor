"""Evaluate the RAG pipeline against a labelled test set.

Runs the evaluation at multiple retrieval depths (k) so we can see both:
  - retrieval hit rate at a strict k (does the right chunk rank near the top?),
  - how answer quality changes with more vs less retrieved context.

For each question and each k it:
  1. retrieves the top-k chunks (same retrieval the live app uses) and records
     whether the expected keywords appear (retrieval hit rate),
  2. generates an answer from those chunks via the shared generate_answer,
  3. uses an LLM-as-judge to score faithfulness and relevancy as binary 0/1.

All pipeline stages are imported from `rag` so the evaluation measures the
real system rather than a re-implementation.

Run:
    python -m src.evals
"""
import json

from openai import AzureOpenAI

from .config import settings
from .rag import get_clients, embed_query, retrieve, generate_answer

TEST_SET_PATH = "data/eval/test_set.json"

# Retrieval depths to evaluate. A strict k (3) tests ranking precision and is
# discriminating; a generous k (10) is what the app uses for answering.
K_VALUES = [3, 10]

JUDGE_PROMPT = (
    "You are a strict evaluator of RAG answers. Given a question, the retrieved "
    "context, the generated answer, and a reference answer, score the generated "
    "answer on two metrics. Each score is BINARY: exactly 0 or 1.\n"
    "- faithfulness: 1 if every claim in the answer is supported by the retrieved "
    "context, 0 if any claim is unsupported or hallucinated.\n"
    "- relevancy: 1 if the answer addresses the question and matches the reference "
    "answer's meaning, 0 otherwise.\n"
    "Return ONLY a JSON object: {\"faithfulness\": <0 or 1>, \"relevancy\": <0 or 1>}."
)


def load_test_set(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieval_hit(chunks: list[dict], keywords: list[str]) -> bool:
    """A hit if any expected keyword appears in any retrieved chunk's content."""
    joined = " ".join(c["content"].lower() for c in chunks)
    return any(kw.lower() in joined for kw in keywords)


def judge(client: AzureOpenAI, question: str, context: str, answer: str, reference: str) -> dict:
    """LLM-as-judge: score faithfulness and relevancy as binary 0/1."""
    resp = client.chat.completions.create(
        model=settings.ENRICH_DEPLOYMENT,  # cheap model is fine for judging
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Retrieved context:\n{context}\n\n"
                    f"Generated answer: {answer}\n\n"
                    f"Reference answer: {reference}"
                ),
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def evaluate_at_k(openai_client, search_client, test_set, k):
    """Run the full eval at a single retrieval depth k; return aggregate rates."""
    hits = faithfulness_total = relevancy_total = 0.0
    n = len(test_set)

    for item in test_set:
        question = item["question"]

        q_vec = embed_query(openai_client, question)
        chunks = retrieve(search_client, q_vec, question, k=k)
        context = "\n\n".join(c["content"] for c in chunks)

        hit = retrieval_hit(chunks, item["retrieval_keywords"])
        hits += int(hit)

        answer = generate_answer(openai_client, question, context)
        scores = judge(openai_client, question, context, answer, item["reference_answer"])
        faithfulness_total += float(scores.get("faithfulness", 0))
        relevancy_total += float(scores.get("relevancy", 0))

        print(
            f"  k={k} {item['id']}: hit={hit}  "
            f"faith={scores.get('faithfulness')}  rel={scores.get('relevancy')}"
        )

    return {
        "hit_rate": hits / n,
        "faithfulness": faithfulness_total / n,
        "relevancy": relevancy_total / n,
        "n": n,
    }


def main() -> None:
    test_set = load_test_set(TEST_SET_PATH)
    openai_client, search_client = get_clients()

    results = {}
    for k in K_VALUES:
        print(f"\n--- Evaluating at k={k} ---")
        results[k] = evaluate_at_k(openai_client, search_client, test_set, k)

    print("\n=== Aggregate comparison ===")
    print(f"{'metric':<18}" + "".join(f"k={k:<8}" for k in K_VALUES))
    for metric in ("hit_rate", "faithfulness", "relevancy"):
        row = f"{metric:<18}"
        for k in K_VALUES:
            row += f"{results[k][metric]:.2%}".ljust(10)
        print(row)


if __name__ == "__main__":
    main()