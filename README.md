# AI Tutor — RAG over a Technical Documentation Corpus

A retrieval-augmented generation (RAG) pipeline serving grounded answers over a technical documentation corpus, intended as an AI study assistant. The stack covers chunking, metadata enrichment, embedding into a vector store, semantic retrieval, reranking, LLM answer synthesis, guardrails, and an agentic layer for adaptive tutoring. Quality is measured by an eval harness over a custom dataset (faithfulness, answer relevancy, retrieval hit rate). Each component is benchmarked against a baseline so optimizations are metric-justified and latency/cost trade-offs stay explicit.

## Architecture (target)

```
PDF source
   │
   ├─ 1. Text extraction                                          [implemented]
   │
   ├─ 2. Chunking            (LangChain RecursiveCharacterTextSplitter)   [implemented]
   │
   ├─ 3. Enrichment          (LLM-generated topic + summary metadata)     [planned]
   │
   ├─ 4. Embedding + upload   (Azure OpenAI embeddings → Azure AI Search, HNSW)   [planned]
   │
   ├─ 5. Retrieval + answer   (vector / hybrid search → LLM answer synthesis)     [planned]
   │
   ├─ 6. Reranking            (Azure semantic reranker → external reranker)        [planned]
   │
   ├─ 7. Guardrails           (Azure AI Content Safety)                            [planned]
   │
   ├─ 8. Agentic layer        (adaptive tutoring loop — LangGraph)                 [planned]
   │
   └─ 9. Evaluation           (custom dataset: faithfulness, relevancy, hit rate)  [planned]
```

> Azure setup (Azure OpenAI + Azure AI Search resources) is provisioned. Stages 1–2 are implemented; stages 3–9 are planned.

## Tech stack

| Layer | Technology |
|-------|------------|
| Language | Python |
| Chunking | LangChain (`RecursiveCharacterTextSplitter`) |
| Embeddings + generation | Azure OpenAI |
| Vector store / search | Azure AI Search (HNSW, vector + keyword + hybrid) — provisioned, not yet indexed |
| Reranking (planned) | Azure semantic reranker → Cohere / MonoT5 |
| Guardrails (planned) | Azure AI Content Safety |
| Agentic layer (planned) | LangGraph |
| Evaluation (planned) | LLM-as-judge harness over a custom dataset |

## Status

### Implemented
- **PDF text extraction** — extracts raw text from the source documentation.
- **Chunking** — splits extracted text into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`, with deterministic chunk IDs (MD5) for idempotent processing.
- **Azure environment** — Azure OpenAI and Azure AI Search resources provisioned and configured.

### Roadmap
- **Chunk enrichment** — LLM-generated topic + summary per chunk, prepended before embedding to improve retrieval (throttled, resumable batch processing).
- **Embedding + indexing** — embed chunks via Azure OpenAI and upload to Azure AI Search with metadata.
- **Retrieval + answer generation** — vector/hybrid retrieval feeding an LLM for grounded answers.
- **Reranking** — Azure semantic reranker baseline, then an external reranker (Cohere / MonoT5), compared via evals.
- **Evaluation suite** — held-out custom dataset scored for faithfulness, answer relevancy, and retrieval hit rate.
- **Guardrails** — input/output safety via Azure AI Content Safety.
- **Guided practice mode** — adaptive question/grade/retrieve loop (LangGraph) on top of ad-hoc Q&A.

## Design notes
- **Build → measure → improve.** Each enhancement (enrichment, reranking) is added against a baseline and validated with evals rather than assumed to help.
- **Preprocessing runs locally**, calling Azure services via API; indexing and serving run against Azure. Suitable for this corpus size without a full cloud data pipeline.
- **Cost-aware model routing (planned):** a lightweight model for enrichment, a stronger model for answer generation.

## Getting started

> Setup instructions will expand as the pipeline grows. Current scope covers extraction and chunking.

```bash
# clone
git clone <repo-url>
cd <repo>

# install dependencies
pip install -r requirements.txt
```

Azure credentials (endpoint + key for Azure OpenAI and Azure AI Search) are read from environment variables and are **not** committed to the repository.

## License

To be determined.
