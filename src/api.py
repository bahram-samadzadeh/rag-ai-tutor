from fastapi import FastAPI
from pydantic import BaseModel

from .graph import run_agent
from .guardrails import get_safety_client, is_safe

app = FastAPI()
safety_client = get_safety_client()


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask(q: Question):
    if not is_safe(safety_client, q.question):
        return {"answer": "This question was flagged as unsafe and cannot be processed.", "citations": [], "confidence": 0.0, "refused": True}
    return run_agent(q.question)