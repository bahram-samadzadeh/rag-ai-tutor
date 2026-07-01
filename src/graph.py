"""LangGraph orchestrator for the self-correcting RAG agent.

Wires the existing pure nodes into a graph with a bounded correction loop:

    rewrite -> answer -> verify -> [correct]* -> finalize

The primary LLM answers once; verify_answer decomposes that answer into atomic
claims and flags unsupported ones; the correct node re-retrieves for each
unsupported claim and re-checks it, promoting any that are now grounded. Claims
still unsupported after MAX_RETRIES are reported as gaps by finalize rather than
asserted — the agent's guarantee against confabulation.

Clients are built once (get_clients) and closed over by the nodes, so no node
constructs its own — the project's dependency-injection rule.
"""
from langgraph.graph import StateGraph, END

from .config import settings
from .state import AgentState, AnswerSchema
from .rag import get_clients, embed_query, retrieve, generate_answer
from .query_rewriter import rewrite_query
from .verify_answer import verify_answer, check_claim
from .finalize import finalize_answer

# Operating retrieval depth — empirically justified (k=10 overloads context,
# RAGAS context_precision 0.777 vs 0.897 at k=3).
RETRIEVE_K = 3
# Loop cap: persistent failure abstains rather than retrying forever.
MAX_RETRIES = 2


def build_graph(openai_client, search_client):
    """Compile the agent graph with clients injected into every node."""

    def node_rewrite(state: AgentState) -> dict:
        """Reformulate the raw question into a retrieval-optimized query."""
        return {"rewritten_query": rewrite_query(openai_client, state["question"])}

    def node_answer(state: AgentState) -> dict:
        """Retrieve (hybrid + semantic rerank) and generate the first answer."""
        query = state["rewritten_query"]
        vector = embed_query(openai_client, query)
        chunks = retrieve(search_client, vector, query, k=RETRIEVE_K)
        context = "\n\n".join(c["content"] for c in chunks)
        answer = generate_answer(openai_client, state["question"], context)
        return {"retrieved_chunks": chunks, "answer": answer}

    def node_verify(state: AgentState) -> dict:
        """Split the answer into claims; separate grounded from unsupported."""
        result = verify_answer(openai_client, state["answer"], state["retrieved_chunks"])
        grounded = [c["claim"] for c in result["claims"] if c["supported"]]
        return {
            "grounded_facts": grounded,
            "unsupported": result["unsupported"],
            "retry_count": 0,
        }

    def node_correct(state: AgentState) -> dict:
        """Re-retrieve per unsupported claim and re-check; promote the grounded."""
        grounded = list(state["grounded_facts"])
        still_unsupported = []
        for claim in state["unsupported"]:
            query = rewrite_query(openai_client, claim)
            vector = embed_query(openai_client, query)
            chunks = retrieve(search_client, vector, query, k=RETRIEVE_K)
            if check_claim(openai_client, claim, chunks):
                grounded.append(claim)
            else:
                still_unsupported.append(claim)
        return {
            "grounded_facts": grounded,
            "unsupported": still_unsupported,
            "retry_count": state["retry_count"] + 1,
        }

    def node_finalize(state: AgentState) -> dict:
        """Synthesize the final answer, admitting any still-missing facts."""
        not_found = state["unsupported"]
        answer = finalize_answer(
            openai_client, state["question"], state["grounded_facts"], not_found
        )
        total = len(state["grounded_facts"]) + len(not_found)
        return {
            "answer": answer,
            "not_found": not_found,
            "confidence": (len(state["grounded_facts"]) / total) if total else 0.0,
            "refused": not state["grounded_facts"],
        }

    def route_after_verify(state: AgentState) -> str:
        """Enter the correction loop only if something is unsupported."""
        return "finalize" if not state["unsupported"] else "correct"

    def route_after_correct(state: AgentState) -> str:
        """Loop until everything is grounded or the retry cap is reached."""
        if not state["unsupported"] or state["retry_count"] >= MAX_RETRIES:
            return "finalize"
        return "correct"

    graph = StateGraph(AgentState)
    graph.add_node("rewrite", node_rewrite)
    graph.add_node("answer", node_answer)
    graph.add_node("verify", node_verify)
    graph.add_node("correct", node_correct)
    graph.add_node("finalize", node_finalize)

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "answer")
    graph.add_edge("answer", "verify")
    graph.add_conditional_edges("verify", route_after_verify,
                                {"correct": "correct", "finalize": "finalize"})
    graph.add_conditional_edges("correct", route_after_correct,
                                {"correct": "correct", "finalize": "finalize"})
    graph.add_edge("finalize", END)
    return graph.compile()


def run_agent(question: str) -> AnswerSchema:
    """Build clients once, run the agent, return the validated answer."""
    openai_client, search_client = get_clients()
    app = build_graph(openai_client, search_client)
    final = app.invoke({"question": question})

    citations = sorted(
        {c.get("source") for c in final.get("retrieved_chunks", []) if c.get("source")}
    )
    return AnswerSchema(
        answer=final["answer"],
        citations=citations,
        confidence=final.get("confidence", 0.0),
        refused=final.get("refused", False),
    )


if __name__ == "__main__":
    q = "What should I do when the connection is dropped before saving changes in Microsoft Fabric?"
    result = run_agent(q)
    print(f"Q: {q}\n")
    print(f"A: {result.answer}\n")
    print(f"confidence={result.confidence:.2f}  refused={result.refused}")
    print(f"citations={result.citations}")