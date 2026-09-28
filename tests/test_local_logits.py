"""Prompt construction and tournament logic, with the model call stubbed out (no torch needed)."""

import pytest

from deskmind_brain.eval.data import EvalItem, Reference
from deskmind_brain.eval.local_logits import FINALISTS_PER_CHUNK, ROUND_SIZE, LocalLogitsPredictor, question_block
from deskmind_brain.types import Question, from_answer


def test_question_blocks():
    text, labels = question_block(Question(type="noul", instructions="urgent?", criteria={"true": "now"}), ["false", "true"])
    assert labels == ["No", "Yes"] and "Yes means: now" in text
    score = Question(type="score", instructions="anger?", criteria=["calm", "angry"])
    text, labels = question_block(score, score.options())
    assert labels == ["0", "1"] and "1. angry" in text
    choice = Question(type="choice", instructions="team?", criteria={"billing": "money", "tech": "bugs"})
    text, labels = question_block(choice, ["tech"])
    assert labels == ["A"] and "A. tech: bugs" in text and "billing" not in text


class StubPredictor(LocalLogitsPredictor):
    """Scores each option by a fixed preference; records the rounds it was asked."""

    def __init__(self, preference):
        self.preference = preference
        self.rounds = []
        self.device = "cpu"

    def _score(self, state_text, prompts):
        self.rounds.append([p.options for p in prompts])
        out = []
        for p in prompts:
            weights = {o: self.preference.get(o, 1.0) for o in p.options}
            total = sum(weights.values())
            out.append({o: w / total for o, w in weights.items()})
        return out


def test_tournament_for_large_choice():
    options = {f"intent_{i}": "" for i in range(60)}
    item = EvalItem(
        id="x", suite="t", group="g", state="hi",
        questions={"intent": Question(type="choice", instructions="?", criteria=options),
                   "ok": Question(type="noul", instructions="?")},
        references={"intent": Reference(probs={o: float(o == "intent_42") for o in options})},
    )
    stub = StubPredictor({"intent_42": 50.0, "intent_7": 10.0})
    pred = stub.predict(item)
    # Round 1: three chunks of <= 26 plus the noul; round 2: one final over the finalists.
    assert [len(r) for r in stub.rounds] == [4, 1]
    assert len(stub.rounds[1][0]) == 3 * FINALISTS_PER_CHUNK <= ROUND_SIZE
    dist = from_answer(item.questions["intent"], pred.answers["intent"])
    assert max(dist, key=dist.get) == "intent_42"
    assert all(p > 0 for p in dist.values())
    assert sum(dist.values()) == pytest.approx(1.0)


def test_two_level_tournament_covers_all_options():
    options = {f"o{i}": "" for i in range(255)}
    item = EvalItem(
        id="x", suite="t", group="g", state="hi",
        questions={"q": Question(type="choice", instructions="?", criteria=options)},
        references={"q": Reference(probs={o: float(o == "o200") for o in options})},
    )
    stub = StubPredictor({"o200": 100.0})
    dist = from_answer(item.questions["q"], stub.predict(item).answers["q"])
    assert len(stub.rounds) == 3  # 10 chunks -> 30 finalists -> 2 chunks -> final
    assert max(dist, key=dist.get) == "o200"
    assert all(p > 0 for p in dist.values()) and sum(dist.values()) == pytest.approx(1.0)


def test_merge_tournament_keeps_chunk_ratios():
    from deskmind_brain.eval.logits_base import merge_tournament

    chunks = [{"a": 0.6, "b": 0.3, "c": 0.1}, {"d": 0.5, "e": 0.5}]
    final = {"a": 0.8, "d": 0.2}
    dist = merge_tournament(chunks, final)
    assert dist["b"] / dist["a"] == pytest.approx(0.3 / 0.6)
    assert dist["e"] / dist["d"] == pytest.approx(1.0)
    assert dist["a"] / dist["d"] == pytest.approx(0.8 / 0.2)
    assert sum(dist.values()) == pytest.approx(1.0)


def test_hoist_shared_moves_common_instruction_fields_only():
    from deskmind_brain.eval.logits_base import hoist_shared, render_context
    from deskmind_brain.types import Question

    rules = "long shared rules"
    qs = {
        "op": Question(type="choice", instructions={"goal": "g", "rules": rules, "operation": "pick"}, criteria={"A": "a", "B": "b"}),
        "tgt": Question(type="choice", instructions={"goal": "g", "rules": rules, "operation": "CLICK"}, criteria={"1": "x", "2": "y"}),
    }
    shared, stripped = hoist_shared(qs)
    assert shared == {"goal": "g", "rules": rules}
    assert stripped["tgt"].instructions == {"operation": "CLICK"}
    assert rules in render_context({"s": 1}, shared)
    plain = {"q": Question(type="noul", instructions="text"), "r": Question(type="noul", instructions="other")}
    assert hoist_shared(plain) == ({}, plain)
    assert render_context("x") == "<state>\nx\n</state>\n\n"


def test_round_size_52_scores_up_to_52_options_in_one_round():
    options = {f"o{i}": "" for i in range(40)}
    item = EvalItem(id="x", suite="t", group="g", state="hi",
                    questions={"q": Question(type="choice", instructions="?", criteria=options)}, references={})
    stub = StubPredictor({"o39": 100.0})
    stub.round_size = 52
    stub.predict(item)
    assert [len(r) for r in stub.rounds] == [1]
    text, labels = question_block(item.questions["q"], list(options))
    assert labels[26] == "a" and labels[-1] == "n"
