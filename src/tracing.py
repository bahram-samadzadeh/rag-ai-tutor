"""Observability setup for the RAG pipeline.

Isolated here (rather than inline in rag.py) so instrumentation is a separable
concern: rag.py imports one function and stays focused on retrieval/generation.
Phoenix + OpenInference auto-traces the OpenAI SDK; Azure AI Search is NOT an
OpenAI call, so the retrieve step is traced manually via the `traced_retrieval`
context manager below.
"""

from contextlib import contextmanager

from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace

# Module-level guard: register() spins up the embedded collector and patches the
# OpenAI SDK. Calling it twice creates duplicate providers and split traces, so
# we make setup idempotent — safe to call from rag.py on every import.
_TRACER_PROVIDER = None


def setup_tracing(project_name: str = "rag-ai-tutor"):
    """Boot Phoenix and instrument the OpenAI SDK.

    Must run BEFORE the first Azure OpenAI client call: the instrumentor patches
    the SDK's request path, and a client created before patching is invisible to
    it. In practice that means calling this at the top of rag.py, before
    get_clients(). register() launches the embedded UI on localhost:6006 — do
    NOT also run `phoenix serve`, that produces a second collector.
    """
    global _TRACER_PROVIDER
    if _TRACER_PROVIDER is not None:
        return _TRACER_PROVIDER

    _TRACER_PROVIDER = register(project_name=project_name)
    OpenAIInstrumentor().instrument(tracer_provider=_TRACER_PROVIDER)
    return _TRACER_PROVIDER


# Tracer is fetched lazily (not at import) so it resolves to the provider that
# setup_tracing() registered, not the default no-op one.
def _tracer():
    return trace.get_tracer(__name__)


@contextmanager
def traced_retrieval(query: str, k: int):
    """Manual span around the Azure AI Search call.

    Why manual: OpenAIInstrumentor only sees OpenAI SDK calls (embeddings, chat).
    The retrieval step is a separate Azure Search client — without this span the
    trace shows the generated answer but not WHICH chunks fed it, which is the
    half that actually breaks. Set retrieved IDs on the span after the call:

        with traced_retrieval(query, k) as span:
            results = search_client.search(...)
            chunks = [...]
            span.set_attribute("retrieved_ids", str([c["id"] for c in chunks]))
            return chunks
    """
    with _tracer().start_as_current_span("retrieve") as span:
        span.set_attribute("query", query)
        span.set_attribute("top_k", k)
        yield span


@contextmanager
def traced_ask(question: str):
    """Parent span wrapping one full ask() — retrieval + generation.

    Makes each question render as ONE collapsible trace tree in the Phoenix UI
    (retrieve span + the auto-traced embedding/chat spans nest under it) instead
    of scattered top-level spans. This is the demo-quality view: open a question,
    see the whole chain. Wrap your ask() body in this.
    """
    with _tracer().start_as_current_span("ask") as span:
        span.set_attribute("question", question)
        yield span