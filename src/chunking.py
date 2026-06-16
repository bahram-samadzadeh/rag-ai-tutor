# src/chunking.py
"""Recursive text chunking for the RAG pipeline.

Splits extracted document text into overlapping chunks suitable for
embedding and storage in Azure AI Search. Uses LangChain's
RecursiveCharacterTextSplitter so that natural boundaries (paragraphs,
then lines, then sentences) are preserved as far as possible.
"""
import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[dict]:
    """Split text into overlapping chunks with stable, Azure-safe IDs.
    
    Args:
        text: Full extracted document text.
        source: Source identifier (e.g. filename) for traceability.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters shared between consecutive chunks to
            preserve context across boundaries.

    Returns:
        List of dicts, each with keys: id, content, source, chunk_index.
    """
    # Separators are tried in order, strongest boundary first:
    # paragraph -> line -> sentence -> word -> character (fallback).
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # split_text applies the recursive logic and returns plain strings.
    chunks = splitter.split_text(text)

    result = []
    for i, chunk in enumerate(chunks):
        # Deterministic ID: re-running overwrites instead of duplicating.
        # md5 hash guarantees an Azure AI Search-legal key (no spaces/dots).
        raw_id = f"{source}-{i}"
        chunk_id = hashlib.md5(raw_id.encode()).hexdigest()
        result.append({"id": chunk_id, "content": chunk, "source": source, "chunk_index": i})

    return result