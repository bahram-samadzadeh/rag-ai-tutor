import pytest

from src.state import AnswerSchema


def test_answer_schema_valid():
    a = AnswerSchema(
        answer="Fabric is an analytics platform.",
        citations=["chunk-1"],
        confidence=0.9,
        refused=False,
    )
    assert a.refused is False
    assert a.citations == ["chunk-1"]


def test_answer_schema_defaults_empty_citations():
    a = AnswerSchema(answer="x", confidence=0.5, refused=False)
    assert a.citations == []


def test_answer_schema_rejects_non_numeric_confidence():
    with pytest.raises(Exception):
        AnswerSchema(
            answer="x",
            citations=[],
            confidence="banana",  # cannot become a float
            refused=False,
        )