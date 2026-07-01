"""Query the RAG pipeline: retrieve relevant chunks and generate an answer.

Embeds the user's question, runs a hybrid (vector + keyword) search against
the index to find the most relevant chunks, then passes those chunks to the
chat model as grounding context to generate a final answer.

The individual stages (client setup, embedding, retrieval, answer generation)
are exposed as small reusable functions so other modules — notably the eval
harness and, going forward, graph.py's orchestrator — can drive the exact
same pipeline rather than re-implementing it.
"""
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI

from .config import settings
# Tracing setup must import before clients are constructed so the OpenAI SDK
# is patched before get_clients() builds the AzureOpenAI client. See call below.
from .tracing import setup_tracing, traced_retrieval, traced_ask

# Activate Phoenix at module load — runs once, before any client is created.
# Idempotent (guarded in tracing.py), so re-imports by the eval harness are safe.
setup_tracing()


# Instructs the model to answer only from retrieved context, which is
# the core guardrail that makes a RAG answer grounded rather than invented.
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using only the "
    "provided context. If the answer is not in the context, say you "
    "don't know rather than guessing."
)

# Toggle to print retrieved chunks for inspection/debugging.
DEBUG_RETRIEVAL = False


def get_clients() -> tuple[AzureOpenAI, SearchClient]:
    """Construct the OpenAI and Search clients from settings.

    Exposed so callers (app, evals, graph.py) build clients once and reuse
    them across many questions instead of reconstructing per call.
    """
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
    return openai_client, search_client


def embed_query(openai_client: AzureOpenAI, question: str) -> list[float]:
    """Embed a question with the same model used for the chunks, so question
    and chunks live in the same vector space."""
    response = openai_client.embeddings.create(
        model=settings.EMBED_DEPLOYMENT,
        input=[question],
    )
    return response.data[0].embedding


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
    # Manual span: Azure Search is not an OpenAI call, so OpenInference can't
    # see it. Without this the trace shows the answer but not WHICH chunks fed
    # it — the half of the pipeline that actually breaks.
    with traced_retrieval(query_text, k) as span:
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
            # Semantic reranker (cross-encoder) re-scores the hybrid candidates by
            # reading query + content together, then we keep the top `k` reranked.
            query_type="semantic",
            semantic_configuration_name="rag-semantic-config",
        )
        chunks = list(results)
        # Record what was retrieved so the span is inspectable in the Phoenix UI.
        span.set_attribute("retrieved_count", len(chunks))
        span.set_attribute(
            "retrieved_chunk_indices",
            str([c.get("chunk_index") for c in chunks]),
        )
        return chunks


def generate_answer(openai_client: AzureOpenAI, question: str, context: str) -> str:
    """Generate an answer grounded in the retrieved context.

    Single source of truth for answer generation: both the live app and the
    eval harness call this, so a change here is reflected in measurements.
    """
    chat_response = openai_client.chat.completions.create(
        model=settings.CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return chat_response.choices[0].message.content


def answer_question(
    question: str,
    openai_client: AzureOpenAI,
    search_client: SearchClient,
) -> str:
    """Run the linear RAG flow for a single question (embed → retrieve → answer).

    Clients are passed in rather than constructed here. This makes the
    function a pure, injectable node — graph.py will build clients ONCE at
    startup and hand them to every node (this one included), instead of each
    node reconstructing its own. Same reason your other functions already
    take clients as arguments.
    """
    # Parent span: nests embed + retrieve + generate under one tree, so each
    # question renders as a single collapsible trace in the Phoenix UI.
    with traced_ask(question):
        query_vector = embed_query(openai_client, question)
        chunks = retrieve(search_client, query_vector, question)

        if DEBUG_RETRIEVAL:
            print("\n--- Retrieved chunks ---")
            for i, c in enumerate(chunks):
                preview = c["content"][:200].replace("\n", " ")
                print(f"[{i + 1}] chunk_index={c['chunk_index']}  topic={c.get('topic')!r}")
                print(f"     {preview}...\n")
            print("--- end ---\n")

        context = "\n\n".join(c["content"] for c in chunks)
        return generate_answer(openai_client, question, context)


if __name__ == "__main__":
    # Clients built ONCE here, then injected — same pattern graph.py will use.
    openai_client, search_client = get_clients()

    question = "What should i do when the connection is dropped before saving the changes in Microsoft Fabric?"
    print(f"Q: {question}\n")
    print(f"A: {answer_question(question, openai_client, search_client)}")