"""Embed chunks and upload them to the Azure AI Search index.

Reads the source text, chunks it, embeds each chunk with the Azure
OpenAI embedding deployment, and uploads the results (text + vector +
metadata) to the search index in batches.
"""
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI

from .config import settings
from .chunking import chunk_text


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


def build_and_upload(text: str, source: str) -> None:
    """Chunk, embed, and upload the given text to the search index."""
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

    chunks = chunk_text(
        text,
        source=source,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    print(f"Chunked into {len(chunks)} pieces.")

    # Embed in batches to limit request size and respect rate limits.
    batch_size = 50
    documents = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embed_texts(openai_client, [c["content"] for c in batch])
        for chunk, vector in zip(batch, vectors):
            documents.append(
                {
                    "id": chunk["id"],
                    "content": chunk["content"],
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                    "content_vector": vector,
                }
            )
        print(f"Embedded {min(start + batch_size, len(chunks))}/{len(chunks)}")

    # Upload to the index. Search accepts up to 1000 docs per batch;
    result = search_client.upload_documents(documents=documents)
    succeeded = sum(1 for r in result if r.succeeded)
    print(f"Uploaded {succeeded}/{len(documents)} documents to '{settings.INDEX_NAME}'.")


if __name__ == "__main__":
    with open("data/fabric-data-engineering.txt", "r", encoding="utf-8") as f:
        full_text = f.read()

    # Test on the last 200k characters before scaling to the full document.
    build_and_upload(full_text, source="fabric-data-engineering")
