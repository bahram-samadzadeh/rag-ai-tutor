"""Evaluate the agent (graph.py) against the same 25-Q hard set as the baseline.

Runs the full LangGraph agent per question, then scores the FINAL answer with
the exact same LLM-as-judge used in evals.py — so faithfulness/relevancy are
directly comparable to BASELINE_PRE_AGENT.md. Also reports agent-specific
signals the baseline has no concept of: abstention rate and mean confidence,
which is where the anti-confabulation behavior shows up.

The agent runs at its operating point (k=3), so this is a single-k evaluation
compared against the baseline's k=3 column.

Run:
    python -m src.agent_evals
"""
from .config import settings
from .rag import get_clients
from .graph import build_graph
from .evals import load_test_set, retrieval_hit, judge, TEST_SET_PATH

# Baseline k=3 numbers from BASELINE_PRE_AGENT.md, printed alongside for contrast.
BASELINE_K3 = {"hit_rate": 1.00, "faithfulness": 0.88, "relevancy": 0.84}


def main() -> None:
    test_set = load_test_set(TEST_SET_PATH)
    openai_client, search_client = get_clients()
    app = build_graph(openai_client, search_client)

    n = len(test_set)
    hits = faith_total = rel_total = 0.0
    refused_count = conf_total = 0.0

    for item in test_set:
        final = app.invoke({"question": item["question"]})

        chunks = final.get("retrieved_chunks", [])
        context = "\n\n".join(c["content"] for c in chunks)
        answer = final.get("answer", "")

        hit = retrieval_hit(chunks, item["retrieval_keywords"])
        hits += int(hit)

        scores = judge(openai_client, item["question"], context, answer,
                       item["reference_answer"])
        faith_total += float(scores.get("faithfulness", 0))
        rel_total += float(scores.get("relevancy", 0))

        refused = bool(final.get("refused", False))
        refused_count += int(refused)
        conf_total += float(final.get("confidence", 0.0))

        print(f"  {item['id']}: hit={hit}  faith={scores.get('faithfulness')}  "
              f"rel={scores.get('relevancy')}  refused={refused}  "
              f"conf={final.get('confidence', 0.0):.2f}")

    agent = {
        "hit_rate": hits / n,
        "faithfulness": faith_total / n,
        "relevancy": rel_total / n,
    }

    print("\n=== Agent vs baseline (k=3) ===")
    print(f"{'metric':<15}{'baseline':<12}{'agent':<12}{'delta':<10}")
    for m in ("hit_rate", "faithfulness", "relevancy"):
        b, a = BASELINE_K3[m], agent[m]
        print(f"{m:<15}{b:<12.2%}{a:<12.2%}{a - b:+.2%}")

    print(f"\nabstention_rate: {refused_count / n:.2%}")
    print(f"mean_confidence: {conf_total / n:.2f}")
    print(f"n: {n}")


if __name__ == "__main__":
    main()