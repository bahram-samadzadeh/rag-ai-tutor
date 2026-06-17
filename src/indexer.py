"""Embed chunks and upload them to the Azure AI Search index.

Reads the source text, chunks it, embeds each chunk with the Azure
OpenAI embedding deployment, and uploads the results (text + vector +
metadata) to the search index in batches.
"""
import json

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI

from .config import settings

# Source + derived path (must match the enrichment stage's naming).
SOURCE = "fabric-data-engineering"
ENRICHED_PATH = f"data/processed/{SOURCE}_enriched.json"

def embed_texts(client: AzureOpenAI, texts: list[str]) -> list[list[float]]:
    """Embed a list of texts in a single API call.

    Batching multiple texts per request is far faster and cheaper than
    one call per chunk, and stays within the model's input limits for
    chunks of this size.
    """
    response = client.embeddings.create(
        model=settings.EMBED_DEPLOYMENT,
        input=texts,
    )
    return [item.embedding for item in response.data]

def load_enriched_chunks(path: str) -> list[dict]:
    """Load the enriched chunks JSON produced by the enrichment step."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
 
 


def build_and_upload(enriched_path: str) -> None:
    """Embed enriched chunks and upload them to the search index."""
    openai_client = AzureOpenAI(
        azure_endpoint=settings.AOAI_ENDPOINT,
        api_key=settings.AOAI_KEY,
        api_version=settings.AOAI_API_VERSION,
    )
    search_client = SearchClient(
        endpoint=settings.SEARCH_ENDPOINT,
        index_name=settings.INDEX_NAME,
        credential=AzureKeyCredential(settings.SEARCH_KEY),
    )
    chunks = load_enriched_chunks(enriched_path)
    print(f"Loaded {len(chunks)} enriched chunks.")

    # Embed in batches to limit request size and respect rate limits.
    batch_size = 50
    documents = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embed_texts(openai_client, [c["embed_text"] for c in batch])
        for chunk, vector in zip(batch, vectors):
            documents.append(
                {
                    "id": chunk["id"],
                    "content": chunk["content"],
                    "topic": chunk["topic"],
                    "summary": chunk["summary"], 
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                    "content_vector": vector,
                }
            )
        print(f"Embedded {min(start + batch_size, len(chunks))}/{len(chunks)}")

    # Upload in batches of 500 — Azure AI Search caps batch size and large
    # single uploads can exceed the request size limit.
    upload_batch = 500
    total_succeeded = 0
    for start in range(0, len(documents), upload_batch):
        batch = documents[start : start + upload_batch]
        result = search_client.upload_documents(documents=batch)
        total_succeeded += sum(1 for r in result if r.succeeded)
        print(f"Uploaded {min(start + upload_batch, len(documents))}/{len(documents)}")
    print(f"Uploaded {total_succeeded}/{len(documents)} documents to '{settings.INDEX_NAME}'.")


if __name__ == "__main__":
    build_and_upload(ENRICHED_PATH)
