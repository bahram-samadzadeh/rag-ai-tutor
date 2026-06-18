"""Evaluate the RAG pipeline against a labelled test set.

For each question in the test set this:
  1. runs the same retrieval the live app uses, to measure retrieval hit rate
     (did the retrieved chunks contain the expected keywords?),
  2. generates an answer via the same generate_answer function the app uses,
     then
  3. uses an LLM-as-judge to score faithfulness (is the answer grounded in the
     retrieved context?) and answer relevancy (does it address the question?),
     each scored BINARY (0 or 1) and averaged into a rate.

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

# The judge scores each metric BINARY (0 or 1); averaging gives a clean rate.
# Binary judgments are more reproducible than fuzzy 0.0-1.0 scores because LLMs
# are far more consistent at yes/no decisions than at fine-grained magnitudes.
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
    """A hit if any expected keyword appears in any retrieved chunk's content.

    Keyword match (not exact chunk id) is robust to the answer text being split
    across chunk boundaries, which is common with overlapping chunks.
    """
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


def main() -> None:
    test_set = load_test_set(TEST_SET_PATH)

    # Same clients and pipeline functions the live app uses.
    openai_client, search_client = get_clients()

    hits = 0
    faithfulness_total = 0.0
    relevancy_total = 0.0
    n = len(test_set)

    for item in test_set:
        question = item["question"]

        # Retrieve via the shared pipeline functions (real system behaviour).
        q_vec = embed_query(openai_client, question)
        chunks = retrieve(search_client, q_vec, question)
        context = "\n\n".join(c["content"] for c in chunks)

        # Retrieval hit rate.
        hit = retrieval_hit(chunks, item["retrieval_keywords"])
        hits += int(hit)

        # Generate the answer with the same function the app uses.
        answer = generate_answer(openai_client, question, context)

        # Judge faithfulness + relevancy (binary).
        scores = judge(openai_client, question, context, answer, item["reference_answer"])
        faithfulness_total += float(scores.get("faithfulness", 0))
        relevancy_total += float(scores.get("relevancy", 0))

        print(
            f"{item['id']}: hit={hit}  "
            f"faith={scores.get('faithfulness')}  rel={scores.get('relevancy')}"
        )

    print("\n=== Aggregate baseline ===")
    print(f"Retrieval hit rate: {hits / n:.2%}  ({hits}/{n})")
    print(f"Faithfulness rate:  {faithfulness_total / n:.2%}  ({int(faithfulness_total)}/{n})")
    print(f"Relevancy rate:     {relevancy_total / n:.2%}  ({int(relevancy_total)}/{n})")


if __name__ == "__main__":
    main()
