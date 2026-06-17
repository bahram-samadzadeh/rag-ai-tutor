"""Query the RAG pipeline: retrieve relevant chunks and generate an answer.

Embeds the user's question, runs a hybrid (vector + keyword) search against
the index to find the most relevant chunks, then passes those chunks to the
chat model as grounding context to generate a final answer.
"""
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI

from .config import settings


# Instructs the model to answer only from retrieved context, which is
# the core guardrail that makes a RAG answer grounded rather than invented.
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using only the "
    "provided context. If the answer is not in the context, say you "
    "don't know rather than guessing."
)

# Toggle to print retrieved chunks for inspection/debugging.
DEBUG_RETRIEVAL = True


def retrieve(
    search_client: SearchClient,
    query_vector: list[float],
    query_text: str,
    k: int = 10,
) -> list[dict]:
    """Return the top-k most relevant chunks using hybrid search.

    Passing both `search_text` (keyword/BM25) and `vector_queries` makes this
    a hybrid query: semantic similarity catches paraphrases, while keyword
    matching catches exact terms (e.g. "AutoSave", "Save failures") that a
    pure vector search can rank too low.
    """
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=k,
        fields="content_vector",
    )
    results = search_client.search(
        search_text=query_text,
        vector_queries=[vector_query],
        select=["content", "topic", "summary", "source", "chunk_index"],
        top=k,
    )
    return list(results)


def answer_question(question: str) -> str:
    """Run the full RAG flow for a single question."""
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

    # 1. Embed the question with the same model used for the chunks, so
    #    question and chunks live in the same vector space.
    embed_response = openai_client.embeddings.create(
        model=settings.EMBED_DEPLOYMENT,
        input=[question],
    )
    query_vector = embed_response.data[0].embedding

    # 2. Retrieve the most relevant chunks (hybrid search).
    chunks = retrieve(search_client, query_vector, question)

    if DEBUG_RETRIEVAL:
        print("\n--- Retrieved chunks ---")
        for i, c in enumerate(chunks):
            preview = c["content"][:200].replace("\n", " ")
            print(f"[{i + 1}] chunk_index={c['chunk_index']}  topic={c.get('topic')!r}")
            print(f"     {preview}...\n")
        print("--- end ---\n")

    context = "\n\n".join(c["content"] for c in chunks)

    # 3. Generate an answer grounded in the retrieved context.
    chat_response = openai_client.chat.completions.create(
        model=settings.CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return chat_response.choices[0].message.content


if __name__ == "__main__":
    question = "What should i do when the connection is dropped before saving the changes in Microsoft Fabric?"
    print(f"Q: {question}\n")
    print(f"A: {answer_question(question)}")