from fastapi import FastAPI
from pydantic import BaseModel

from .graph import run_agent

app = FastAPI()


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask(q: Question):
    return run_agent(q.question)