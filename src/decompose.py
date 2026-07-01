"""Decompose node: break a question into atomic sub-questions.

Why this exists: a single complex question ("what happens if X and then Y
during Z?") often requires retrieving and verifying MULTIPLE separate facts.
Retrieving on the raw question alone tends to fetch chunks for only the
dominant clause, silently dropping the others — the "multi-fact omission"
failure mode measured in evals.py. Splitting first gives every clause its
own retrieval + verification pass downstream.

Pure node: client injected, no I/O beyond the LLM call, no import-time
side effects. Fills AgentState["sub_questions"].
"""
from openai import AzureOpenAI

from .config import settings

# Private to this module — one owner, so it stays local (same convention as
# query_rewriter.py's _SYSTEM_PROMPT).
_SYSTEM_PROMPT = (
    "Break the user's question into the smallest set of atomic sub-questions "
    "needed to answer it completely. If the question is already atomic, "
    "return it unchanged as the only item. Each sub-question must be "
    "self-contained (resolve pronouns/references). Return ONLY a JSON array "
    "of strings, nothing else."
)


def decompose(openai_client: AzureOpenAI, question: str) -> list[str]:
    """Return atomic sub-questions for `question` (min length 1).

    Args:
        openai_client: injected client — built once upstream, not here.
        question: the raw user question.

    Returns:
        List of sub-question strings. Falls back to [question] on any
        parse failure, so a bad decomposition never drops the question
        entirely — the graph still has something to retrieve on.
    """
    import json

    response = openai_client.chat.completions.create(
        model=settings.CHAT_DEPLOYMENT,
        temperature=0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed) and parsed:
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    return [question]