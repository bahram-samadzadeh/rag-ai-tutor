"""Finalize node: synthesize the trustworthy final answer after verification.

By the time we reach here the self-correction loop has classified every atomic
claim as either grounded (supported by retrieved chunks) or not-found (still
unsupported after re-retrieval, capped). This node hands BOTH lists to the
primary LLM and asks it to write the user-facing answer that states only what
is grounded and explicitly admits the gaps — never papering over a not-found
fact with a confident guess. That honesty is the whole point of the agent:
turning "confidently wrong" into "correct, and clear about what it doesn't know".

Pure node: client injected, no I/O beyond the LLM call, no import-time side
effects. Produces the final answer string for AnswerSchema.answer.
"""
from openai import AzureOpenAI

from .config import settings

# Private to this module — one owner, stays local.
_SYSTEM_PROMPT = (
    "You write the final answer using ONLY the verified facts provided. Inputs:\n"
    "- QUESTION: the user's question.\n"
    "- GROUNDED FACTS: the ONLY things you may assert as true.\n"
    "- NOT FOUND: aspects the documentation did NOT support.\n"
    "\n"
    "If NOT FOUND is empty: answer normally using only grounded facts. No flag.\n"
    "If NOT FOUND has items: begin with a single line starting '⚠️ ' that names "
    "what the documentation does not confirm, THEN answer with the grounded "
    "facts, THEN restate the gap plainly.\n"
    "\n"
    "Hard rules:\n"
    "- Never add facts, qualifiers, or reassurance beyond GROUNDED FACTS. "
    "Banned hedges: 'generally', 'typically', 'usually', 'should be'.\n"
    "- Never speculate about NOT FOUND items ('if it could, then...' is forbidden).\n"
    "- Never give an unqualified 'Yes' if any required part is in NOT FOUND."
)


def finalize_answer(
    openai_client: AzureOpenAI,
    question: str,
    grounded_facts: list[str],
    not_found: list[str],
) -> str:
    """Compose the final answer from verified facts, admitting any gaps.

    Args:
        openai_client: injected primary client — built once upstream.
        question: the user's original question.
        grounded_facts: atomic claims confirmed by retrieval.
        not_found: claims still unsupported after the capped re-retrieval loop.

    Returns:
        The final answer string. Falls back to a plain abstention message if the
        model returns nothing, so the pipeline always yields a usable answer.
    """
    grounded_block = "\n".join(f"- {f}" for f in grounded_facts) or "(none)"
    not_found_block = "\n".join(f"- {f}" for f in not_found) or "(none)"
    user_content = (
        f"QUESTION:\n{question}\n\n"
        f"GROUNDED FACTS:\n{grounded_block}\n\n"
        f"NOT FOUND:\n{not_found_block}"
    )

    response = openai_client.chat.completions.create(
        model=settings.CHAT_DEPLOYMENT,
        temperature=0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    answer = (response.choices[0].message.content or "").strip()
    return answer or (
        "I don't have enough grounded information in the documentation to "
        "answer this reliably."
    )