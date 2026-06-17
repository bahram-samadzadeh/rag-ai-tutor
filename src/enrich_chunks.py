"""Enrich chunks with LLM-generated topic + summary metadata.

Reads the chunks produced by the chunking stage, asks the LLM for a short
topic and one-sentence summary per chunk, prepends them into an `embed_text`
field (so the metadata shapes the embedding vector), and writes the enriched
chunks for the indexing stage.

Paths are derived from the source name so multiple datasets never collide.
Processing is batched, concurrency-capped, retried with backoff, and
checkpointed so a crash can resume without repeating work.

Run:
    python -m src.enrich_chunks
"""
import os
import json
import asyncio

from openai import AsyncAzureOpenAI, RateLimitError, APIError

from .config import settings

# ---------------------------------------------------------------------------
# Source + derived paths (must match the chunking stage's naming)
# ---------------------------------------------------------------------------
SOURCE = "fabric-data-engineering"
INPUT_PATH = f"data/processed/{SOURCE}_chunks.json"
OUTPUT_PATH = f"data/processed/{SOURCE}_enriched.json"

ENRICH_DEPLOYMENT = settings.ENRICH_DEPLOYMENT   # e.g. "gpt-4.1-mini"
CONCURRENCY = 2
MAX_RETRIES = 5
BATCH = 50

client = AsyncAzureOpenAI(
    azure_endpoint=settings.AOAI_ENDPOINT,
    api_key=settings.AOAI_KEY,
    api_version=settings.AOAI_API_VERSION,
)

SYSTEM_PROMPT = (
    "You label technical documentation chunks. "
    "Return ONLY a JSON object with keys 'topic' and 'summary'. "
    "'topic' is a short heading (max 6 words). "
    "'summary' is one sentence describing the chunk. No extra text."
)


def load_chunks(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint(path: str) -> dict:
    # Resume support: already-enriched ids are skipped on rerun.
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return {c["id"]: c for c in json.load(f)}

def save_checkpoint(path: str, enriched_map: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(enriched_map.values()), f, ensure_ascii=False, indent=2)


async def enrich_one(chunk: dict, sem: asyncio.Semaphore) -> dict | None:
    async with sem:                                  # cap concurrency
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.chat.completions.create(
                    model=ENRICH_DEPLOYMENT,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": chunk["content"]},
                    ],
                    temperature=0,                   # deterministic labels
                    response_format={"type": "json_object"},
                )
                data = json.loads(resp.choices[0].message.content)
                topic = data.get("topic", "").strip()
                summary = data.get("summary", "").strip()

                # Prepend metadata to the text that will be embedded.
                embed_text = f"[Topic: {topic}] [Summary: {summary}]\n\n{chunk['content']}"

                # Carry all original fields, add the new ones.
                return {
                    **chunk,
                    "topic": topic,
                    "summary": summary,
                    "embed_text": embed_text,
                }
            except (RateLimitError, APIError) as e:
                wait = 2 ** (attempt + 1)            # 2s, 4s, 8s ...
                print(f"  {e.__class__.__name__} on {chunk['id']}, retry in {wait}s")
                await asyncio.sleep(wait)
        print(f"  FAILED after {MAX_RETRIES} retries: {chunk['id']}")
        return None


async def main() -> None:
    chunks = load_chunks(INPUT_PATH)
    enriched = load_checkpoint(OUTPUT_PATH)

    todo = [c for c in chunks if c["id"] not in enriched]
    print(f"{len(chunks)} chunks total, {len(enriched)} done, {len(todo)} to process")

    sem = asyncio.Semaphore(CONCURRENCY)

    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        results = await asyncio.gather(*(enrich_one(c, sem) for c in batch))
        for r in results:
            if r:
                enriched[r["id"]] = r
        save_checkpoint(OUTPUT_PATH, enriched)        # checkpoint after each batch
        print(f"  checkpoint: {len(enriched)}/{len(chunks)} saved")

    print(f"Done. Enriched chunks written to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
