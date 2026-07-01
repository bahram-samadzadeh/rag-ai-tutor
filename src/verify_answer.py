"""Answer-verification node: decompose a generated answer into atomic claims
and check each one against the retrieved chunks.

This is the heart of the self-correction loop. The primary LLM has already
produced an answer from the retrieved context; this node asks a reasoning model
to (1) break that answer into the smallest verifiable factual claims — the same
move RAGAS faithfulness makes — and (2) judge, per claim, whether the chunks
actually support it. Unsupported claims are exactly the confabulations the eval
set measures; surfacing them lets the graph re-retrieve for just those claims
instead of trusting the whole answer.

It does NOT rewrite the answer or retrieve anything itself. It only reports
grounding, so the graph can decide whether to loop (re-retrieve) or stop.

Pure node: client injected, no I/O beyond the LLM call, no import-time side
effects. Feeds the coverage/retry logic in graph.py.
"""
from openai import AzureOpenAI

from .config import settings

# Private to this module — one owner, stays local (same convention as the
# other nodes' _SYSTEM_PROMPT).
_SYSTEM_PROMPT = (
    "You are a strict grounding checker for a retrieval-augmented answer.\n"
    "\n"
    "You are given CONTEXT (retrieved documentation chunks) and an ANSWER "
    "generated from it. Do two things:\n"
    "1. Decompose the ANSWER into the smallest possible atomic factual claims "
    "— one verifiable fact each, no compound statements.\n"
    "2. For EACH claim, decide whether the CONTEXT directly supports it. A "
    "claim is 'supported' only if the context states or clearly entails it. "
    "If the context is silent, vague, or contradicts it, mark it 'unsupported'. "
    "Do not use outside knowledge — judge only against the given context.\n"
    "\n"
    "Return ONLY a JSON object of this exact shape:\n"
    '{"claims": [{"claim": <string>, "supported": <true|false>}]}\n'
    "nothing else."
)


def verify_answer(
    openai_client: AzureOpenAI,
    answer: str,
    chunks: list[dict],
) -> dict:
    """Decompose `answer` into atomic claims and check each against `chunks`.

    Args:
        openai_client: injected reasoning client — built once upstream.
        answer: the primary LLM's generated answer to verify.
        chunks: retrieved chunks (each a dict with a 'content' key), used as
            the sole grounding context.

    Returns:
        A dict:
          {
            "claims": [{"claim": str, "supported": bool}, ...],
            "unsupported": [str, ...],   # claims with no grounding
            "all_supported": bool,       # True => loop can stop
          }
        On any parse failure, returns all_supported=False with an empty claim
        list, so a broken check is treated as "not yet verified" (fail-safe:
        the graph won't wrongly declare an unverified answer grounded).
    """
    import json

    context = "\n\n".join(c.get("content", "") for c in chunks)
    user_content = f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"

    response = openai_client.chat.completions.create(
        model=settings.CHAT_DEPLOYMENT,
        temperature=0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()

    try:
        parsed = json.loads(raw)
        claims = parsed.get("claims", [])
        if isinstance(claims, list):
            clean = [
                {"claim": str(c["claim"]), "supported": bool(c["supported"])}
                for c in claims
                if isinstance(c, dict) and "claim" in c and "supported" in c
            ]
            unsupported = [c["claim"] for c in clean if not c["supported"]]
            return {
                "claims": clean,
                "unsupported": unsupported,
                # all_supported only if we actually found claims and none failed
                "all_supported": bool(clean) and not unsupported,
            }
    except (json.JSONDecodeError, TypeError, KeyError):
        pass

    # Fail-safe: unparseable => treat as unverified, never as "all good".
    return {"claims": [], "unsupported": [], "all_supported": False}