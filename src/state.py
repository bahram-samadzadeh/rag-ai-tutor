"""Data contracts for the agentic RAG graph."""
from typing import Optional, TypedDict

from pydantic import BaseModel, Field


class AgentState(TypedDict, total=False):
    question: str
    sub_questions: list[str]
    bridge_entities: list[str]
    required_relationships: list[str]
    rewritten_queries: list[str]
    retrieved_chunks: list[dict]
    coverage_ok: bool
    reasoning: str
    citations: list[str]
    verified_relationships: list[str]
    missing_relationships: list[str]
    critical_constraint: Optional[str]
    constraint_verified: bool
    refused: bool
    confidence: float
    answer: str
    retry_count: int


class AnswerSchema(BaseModel):
    answer: str = Field(description="Grounded answer, or an abstention message.")
    citations: list[str] = Field(default_factory=list)
    confidence: float
    refused: bool
    required_relationships: list[str] = Field(default_factory=list)
    verified_relationships: list[str] = Field(default_factory=list)
    missing_relationships: list[str] = Field(default_factory=list)
    critical_constraint: Optional[str] = None
    constraint_verified: bool = False