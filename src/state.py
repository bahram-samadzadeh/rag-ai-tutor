"""Data contracts for the agentic RAG graph."""
from typing import TypedDict

from pydantic import BaseModel, Field


class AgentState(TypedDict, total=False):
    question: str
    rewritten_query: str
    retrieved_chunks: list[dict]
    grounded_facts: list[str]
    unsupported: list[str]
    not_found: list[str]
    retry_count: int
    confidence: float
    refused: bool
    answer: str


class AnswerSchema(BaseModel):
    answer: str = Field(description="Grounded answer, or an abstention message.")
    citations: list[str] = Field(default_factory=list)
    confidence: float
    refused: bool