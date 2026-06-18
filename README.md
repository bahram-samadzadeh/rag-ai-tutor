# AI Tutor — RAG over a Technical Documentation Corpus

A retrieval-augmented generation (RAG) pipeline serving grounded answers over a technical documentation corpus, intended as an AI study assistant. The stack covers chunking, metadata enrichment, embedding into a vector store, hybrid retrieval, cross-encoder reranking, LLM answer synthesis, and an eval harness — with an agentic tutoring layer planned on top. Quality is measured by a custom eval harness (faithfulness, answer relevancy, retrieval hit rate). Each component is benchmarked against a baseline so optimizations are metric-justified and latency/cost trade-offs stay explicit.

## Results

Measured on a custom evaluation set with a binary LLM-as-judge (faithfulness, relevancy) and keyword-based retrieval hit-rate. Metrics are reported at **k=3**, a strict setting that tests ranking precision.

Adding the Azure semantic reranker (a cross-encoder) on top of hybrid retrieval, measured on the initial 22-question set:

| Pipeline | Relevancy @k=3 | Hit-rate @k=3 |
|----------|---------------|---------------|
| Hybrid retrieval (vector + BM25) | 73% | 91% |
| + Semantic reranking | **95%** | **100%** |

The eval set was then expanded to 30 questions with harder paraphrased and multi-hop cases to keep it discriminating. On the expanded set the reranked pipeline scores **86.7% relevancy @k=3** (93.3% @k=10), with faithfulness at 100% throughout. Each improvement was validated against the prior baseline rather than assumed.

## Architecture

```
PDF source
   │
   ├─ 1. Text extraction                                                      [implemented]
   │
   ├─ 2. Chunking            (LangChain RecursiveCharacterTextSplitter)        [implemented]
   │
   ├─ 3. Enrichment          (LLM-generated topic + summary, prepended)        [implemented]
   │
   ├─ 4. Embedding + index    (Azure OpenAI embeddings → Azure AI Search, HNSW) [implemented]
   │
   ├─ 5. Retrieval + answer   (hybrid vector + BM25 → LLM answer synthesis)     [implemented]
   │
   ├─ 6. Reranking            (Azure semantic reranker, cross-encoder)          [implemented]
   │
   ├─ 7. Evaluation           (custom set: faithfulness, relevancy, hit rate)   [implemented]
   │
   ├─ 8. Guardrails           (Azure AI Content Safety)                         [planned]
   │
   └─ 9. Agentic layer        (adaptive tutoring loop — LangGraph)              [planned]
```

## Pipeline

The pipeline runs as a sequence of stages, each its own module reading and writing intermediate JSON so stages stay independent, inspectable, and re-runnable:

```
data/raw/{source}.txt
  → chunking.py        → data/processed/{source}_chunks.json
  → enrich_chunks.py   → data/processed/{source}_enriched.json
  → indexer.py (+ create_index.py schema) → Azure AI Search
  → rag.py (query time) → evals.py (measurement)
```

## Tech stack

| Layer | Technology |
|-------|------------|
| Language | Python |
| Chunking | LangChain (`RecursiveCharacterTextSplitter`) |
| Enrichment | Azure OpenAI (gpt-4.1-mini) — topic + summary metadata |
| Embeddings | Azure OpenAI (text-embedding-3-small, 1536-dim) |
| Vector store / search | Azure AI Search (HNSW; hybrid vector + BM25) |
| Reranking | Azure semantic reranker (cross-encoder) |
| Answer generation | Azure OpenAI (gpt-4.1-mini) |
| Evaluation | Custom LLM-as-judge harness (binary faithfulness / relevancy + hit rate) |
| Agentic layer (planned) | LangGraph |
| Guardrails (planned) | Azure AI Content Safety |

## Status

### Implemented
- **Text extraction & chunking** — recursive splitting with deterministic MD5 chunk IDs for idempotent processing.
- **Chunk enrichment** — an LLM generates a topic + one-sentence summary per chunk, prepended to the text before embedding so the metadata shapes the vector. Runs async with concurrency capping, exponential backoff on rate limits, and checkpointing for crash-safe resume.
- **Embedding & indexing** — chunks embedded and uploaded to Azure AI Search with batched, retry-guarded uploads.
- **Hybrid retrieval** — vector + BM25 keyword search (RRF-fused), with the topic/summary metadata available for filtering.
- **Semantic reranking** — Azure's cross-encoder re-scores the top hybrid candidates by reading query + content together.
- **Evaluation harness** — a labelled question set scored at multiple retrieval depths (k=3 and k=10) for retrieval hit-rate and LLM-judged faithfulness/relevancy; imports the live pipeline so it measures the real system.

### Roadmap
- **Guardrails** — input/output safety via Azure AI Content Safety.
- **Standardized evals + tracing** — RAGAS for standardized metrics; Phoenix (OpenInference) for per-request tracing (query → retrieved chunks → prompt → answer → latency/tokens/cost).
- **Stronger judge** — an independent, stronger judge model to reduce self-preference bias.
- **Agentic tutoring** — an adaptive question/grade/retrieve loop (LangGraph) on top of ad-hoc Q&A.
- **Richer ingestion** — table extraction and diagram captioning for fuller document coverage.

## Design notes
- **Build → measure → improve.** Each enhancement (enrichment, reranking) is added against a baseline and validated with the eval harness rather than assumed to help.
- **Decoupled pipeline stages.** Each stage reads and writes JSON, so a failure in one stage doesn't lose earlier work and any stage can be re-run or inspected in isolation.
- **Single source of truth for evals.** The eval harness imports the same retrieval and answer-generation functions the app uses, so measurements reflect real behaviour rather than a re-implementation.
- **Cost-aware model routing.** A lightweight model handles bulk enrichment; answer generation and judging are configured separately so each can be upgraded independently.

## Getting started

```bash
# clone
git clone <repo-url>
cd <repo>

# install dependencies
pip install -r requirements.txt
```

Set Azure credentials (endpoints + keys for Azure OpenAI and Azure AI Search) as environment variables; they are read from a local `.env` and are **not** committed.

Run the pipeline end to end:

```bash
python -m src.chunking         # raw text → chunks
python -m src.enrich_chunks    # add topic + summary metadata
python -m src.create_index     # define the search index + semantic config
python -m src.indexer          # embed + upload
python -m src.rag              # ask a question
python -m src.evals            # measure the pipeline
```

## License

To be determined.