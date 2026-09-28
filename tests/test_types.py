import pytest

from deskmind_brain.types import Question, choice_confidence, from_answer, normalize, score_value, to_answer

CHOICE = Question(type="choice", instructions="team?", criteria={"billing": "b", "technical": "t", "sales": "s"})
SCORE = Question(type="score", instructions="how angry?", criteria=["calm", "annoyed", "angry"])
NOUL = Question(type="noul", instructions="urgent?")


def test_options():
    assert CHOICE.options() == ["billing", "technical", "sales"]
    assert SCORE.options() == ["0", "1", "2"]
    assert NOUL.options() == ["false", "true"]


def test_criteria_validation():
    with pytest.raises(ValueError):
        Question(type="choice", instructions="x", criteria=["a", "b"])
    with pytest.raises(ValueError):
        Question(type="score", instructions="x", criteria=["only one"])


def test_choice_confidence_matches_published_formula():
    # Published Opus answer on evals.typesafe.ai: max prob 0.4 over 8 options -> 0.3142857...
    dist = {str(i): p for i, p in enumerate([0.03, 0.01, 0.4, 0.23, 0.01, 0.3, 0.005, 0.015])}
    assert choice_confidence(dist) == pytest.approx(0.3142857142857143)
    assert choice_confidence({"a": 0.5, "b": 0.5}) == pytest.approx(0.0)
    assert choice_confidence({"a": 1.0, "b": 0.0}) == pytest.approx(1.0)


def test_normalize_fills_missing_and_renormalizes():
    assert normalize({"a": 2.0}, ["a", "b"]) == {"a": 1.0, "b": 0.0}
    assert normalize({}, ["a", "b"]) == {"a": 0.5, "b": 0.5}


def test_answer_round_trip():
    for q, dist in (
        (CHOICE, {"billing": 0.2, "technical": 0.7, "sales": 0.1}),
        (SCORE, {"0": 0.1, "1": 0.3, "2": 0.6}),
        (NOUL, {"false": 0.25, "true": 0.75}),
    ):
        answer = to_answer(q, dist)
        assert answer["type"] == q.type
        assert from_answer(q, answer) == pytest.approx(dist)


def test_score_answer_fields():
    answer = to_answer(SCORE, {"0": 0.1, "1": 0.3, "2": 0.6})
    assert answer["score"] == pytest.approx(1.5)
    assert answer["legend"] == {"0": "calm", "1": "annoyed", "2": "angry"}
    assert score_value(answer["probabilities"]) == pytest.approx(1.5)


def test_hard_answers_become_one_hot():
    assert from_answer(CHOICE, {"type": "choice", "choice": "sales"}) == {"billing": 0.0, "technical": 0.0, "sales": 1.0}
    assert from_answer(SCORE, {"type": "score", "score": 1.4}) == {"0": 0.0, "1": 1.0, "2": 0.0}
