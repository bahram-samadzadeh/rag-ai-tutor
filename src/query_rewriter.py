"""Query rewriting node logic: reformulate a raw question into a better
retrieval query.

Users ask in conversational, paraphrased language ("what do I do if it drops
before saving?") — full of filler and light on the exact terms the index was
built from ("AutoSave", "connection loss", "Microsoft Fabric"). Both halves of
hybrid search suffer: BM25 misses the keywords, and the embedding drifts toward
the chatty phrasing. Rewriting the query into canonical, technical terms lifts
recall before a single chunk is fetched.

This module owns ONLY the rewrite logic and its own prompt. It builds no
clients and runs nothing at import — the LLM client is injected, so the same
function is reusable and unit-testable with a fake client. The graph wraps this
to map over `sub_questions`, producing `rewritten_queries`.
"""
from openai import AzureOpenAI

from .config import settings

# Private to this module: a rewrite prompt has one owner, so it lives here
# rather than in a shared file. Folds three intents into one pass —
# denoise (drop filler), clarify (resolve vague references), and technical
# expansion (surface exact product/feature terms) — instead of separate modes.
_SYSTEM_PROMPT = (
    "You rewrite a user's question into a single, precise search query for a "
    "technical documentation index. Remove conversational filler, resolve "
    "vague references, and use exact technical terms and product names where "
    "implied. Preserve the original intent and any constraints. Return ONLY "
    "the rewritten query as plain text — no preamble, quotes, or explanation."
)


def rewrite_query(openai_client: AzureOpenAI, query: str) -> str:
    """Return a retrieval-optimized rewrite of a single query.

    Args:
        openai_client: injected Azure OpenAI client (not built here — so this
            node stays pure and testable, matching the rest of the pipeline).
        query: one question to reformulate (e.g. a sub-question from decompose).

    Returns:
        The rewritten query string. Falls back to the original query if the
        model returns nothing, so a bad rewrite can never blank out retrieval.
    """
    response = openai_client.chat.completions.create(
        model=settings.CHAT_DEPLOYMENT,
        # temperature=0: reformulation is a deterministic transform, not a
        # creative task — the same question should always rewrite the same way.
        temperature=0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    )
    rewritten = (response.choices[0].message.content or "").strip()
    return rewritten or query