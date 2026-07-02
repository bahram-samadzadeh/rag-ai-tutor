from fastapi.testclient import TestClient

import src.api as api
from src.state import AnswerSchema

client = TestClient(api.app)


def test_unsafe_question_is_refused(monkeypatch):
    monkeypatch.setattr(api, "is_safe", lambda c, t: False)
    r = client.post("/ask", json={"question": "something harmful"})
    assert r.status_code == 200
    assert r.json()["refused"] is True


def test_safe_question_returns_answer(monkeypatch):
    monkeypatch.setattr(api, "is_safe", lambda c, t: True)
    monkeypatch.setattr(
        api, "run_agent",
        lambda q: AnswerSchema(answer="ok", citations=["doc1"], confidence=0.9, refused=False),
    )
    r = client.post("/ask", json={"question": "What is Fabric?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "ok"
    assert body["refused"] is False


def test_missing_question_field_is_rejected():
    r = client.post("/ask", json={})
    assert r.status_code == 422
