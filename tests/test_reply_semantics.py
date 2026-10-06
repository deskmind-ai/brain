"""Reply semantics a client can check (deskmind#36 item 2): every offered option has a probability, they sum to 1,
the choice is the argmax with ties going to the first option in order, the confidence formula is fixed, and
questions a two-stage server did not score say so (scored: false) instead of passing a uniform guess as an answer."""

import pytest

from deskmind_brain.eval.calibrate import apply
from deskmind_brain.eval.data import EvalItem
from deskmind_brain.eval.local_logits import LocalLogitsPredictor
from deskmind_brain.router import route
from deskmind_brain.types import Question, choice_confidence, to_answer

OPS = {"CLICK": "click", "TYPE_TEXT": "type", "DONE": "done"}


def agent_item():
    goal = {"goal": "save it", "rules": ["r"]}
    return EvalItem(id="x", suite="t", group="g", state={"page": {"title": "w"}}, references={}, questions={
        "operation": Question(type="choice", instructions=goal, criteria=OPS),
        "click_target": Question(type="choice", instructions={**goal, "operation": "CLICK"},
                                 criteria={"1": {"element": "[1] Save"}, "2": {"element": "[2] Cancel"}}),
        "type_text_target": Question(type="choice", instructions={**goal, "operation": "TYPE_TEXT"},
                                     criteria={"3": {"element": "[3] Name"}, "4": {"element": "[4] Notes"}}),
        "type_text_value": Question(type="choice", instructions={"goal": "save it", "rules": "pick"},
                                    criteria={"1": {"value": "a"}, "2": {"value": "b"}}),
    })


class Stub(LocalLogitsPredictor):
    def __init__(self, preference, two_stage=True, terminal_heads=True):
        self.preference, self.two_stage, self.terminal_heads, self.device = preference, two_stage, terminal_heads, "cpu"

    def _prompt_ids(self, context, block):   # no tokenizer: characters stand in for tokens
        return [ord(c) for c in context + block]

    def _score(self, state_text, prompts):
        out = []
        for p in prompts:
            w = {o: self.preference.get(o, 1.0) for o in p.options}
            out.append({o: v / sum(w.values()) for o, v in w.items()})
        return out


def test_two_stage_marks_the_heads_it_did_not_score():
    pred = Stub({"CLICK": 9.0, "1": 5.0}).predict(agent_item())
    a = pred.answers
    assert a["operation"]["choice"] == "CLICK" and "scored" not in a["operation"]
    assert "scored" not in a["click_target"] and a["click_target"]["choice"] == "1"
    for qid in ("type_text_target", "type_text_value"):  # not CLICK's heads: placeholders, said so
        assert a[qid]["scored"] is False
        assert set(a[qid]["probabilities"].values()) == {0.5}   # still uniform, for clients that predate the flag


def test_one_pass_and_terminal_steps_score_everything():
    for stub in (Stub({"CLICK": 9.0}, two_stage=False), Stub({"DONE": 9.0})):
        assert not any(a.get("scored") is False for a in stub.predict(agent_item()).answers.values())


def test_a_second_predict_starts_clean():
    stub = Stub({"CLICK": 9.0})
    stub.predict(agent_item())
    stub.preference = {"DONE": 9.0}
    assert not any(a.get("scored") is False for a in stub.predict(agent_item()).answers.values())


def test_calibration_leaves_placeholders_alone():
    item = agent_item()
    pred = Stub({"CLICK": 9.0}).predict(item)
    out = apply(item, pred, {"target": 2.0, "operation": 2.0, "value": 2.0, "choice": 2.0})
    assert out.answers["type_text_value"]["scored"] is False


def test_the_router_keeps_the_flag_of_the_tier_it_takes():
    item = agent_item()
    fast = Stub({"CLICK": 1.2}).predict(item).answers          # unsure: escalates
    strong = Stub({"TYPE_TEXT": 9.0, "3": 9.0, "2": 9.0}).predict(item).answers
    body = {"state": item.state, "questions": {k: q.model_dump() for k, q in item.questions.items()}}
    answers, rec = route(body, fast, lambda _b: strong, threshold=0.96)
    assert rec["by"] == "strong"
    assert answers["click_target"]["scored"] is False                      # strong scored TYPE_TEXT's heads only
    assert "scored" not in answers["type_text_target"] and "scored" not in answers["type_text_value"]


def test_probabilities_cover_the_options_and_sum_to_one():
    q = Question(type="choice", instructions="?", criteria={"a": "", "b": "", "c": ""})
    ans = to_answer(q, {"a": 2.0, "z": 5.0})                # an unknown key is dropped, a missing one is 0
    assert list(ans["probabilities"]) == ["a", "b", "c"]
    assert sum(ans["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)


def test_ties_go_to_the_first_option_in_order():
    q = Question(type="choice", instructions="?", criteria={"b": "", "a": "", "c": ""})
    assert to_answer(q, {"b": 0.4, "a": 0.4, "c": 0.2})["choice"] == "b"
    q2 = Question(type="choice", instructions="?", criteria=[{"key": "a", "description": ""}, {"key": "b", "description": ""}])
    assert to_answer(q2, {"a": 0.5, "b": 0.5})["choice"] == "a"


def test_confidence_formula():
    dist = {"a": 0.6, "b": 0.3, "c": 0.1}
    assert choice_confidence(dist) == pytest.approx((3 * 0.6 - 1) / 2)
