import math

import pytest

from deskmind_brain.eval.data import EvalItem, Prediction, Reference
from deskmind_brain.eval.metrics import aurc, ece, report, score_pair, score_run, selective_accuracy, summarize
from deskmind_brain.eval.predictors import ReferencePredictor, UniformPredictor
from deskmind_brain.types import Question, to_answer


def make_item() -> EvalItem:
    return EvalItem(
        id="i1",
        suite="t",
        group="g",
        state="My card was charged twice.",
        questions={
            "team": Question(type="choice", instructions="team?", criteria={"billing": "", "technical": "", "sales": ""}),
            "urgent": Question(type="noul", instructions="urgent?"),
            "anger": Question(type="score", instructions="anger?", criteria=["calm", "annoyed", "angry"]),
        },
        references={
            "team": Reference(probs={"billing": 0.8, "technical": 0.15, "sales": 0.05}),
            "urgent": Reference(probs={"false": 0.9, "true": 0.1}),
            "anger": Reference(probs={"0": 0.0, "1": 1.0, "2": 0.0}, soft=False),
        },
    )


def test_reference_predictor_is_perfect():
    item = make_item()
    results = score_run([item], {item.id: ReferencePredictor().predict(item)})
    assert all(r.answered and r.correct == 1.0 for r in results)
    assert all(r.kl == pytest.approx(0.0, abs=1e-9) and r.tv == pytest.approx(0.0) for r in results)


def test_uniform_gets_chance_accuracy_via_tie_credit():
    item = make_item()
    results = {r.qid: r for r in score_run([item], {item.id: UniformPredictor().predict(item)})}
    assert results["team"].correct == pytest.approx(1 / 3)
    assert results["urgent"].correct == pytest.approx(1 / 2)
    assert results["team"].nll == pytest.approx(math.log(3))


def test_pair_metrics_by_hand():
    item = make_item()
    q = item.questions["team"]
    pred = Prediction("i1", {"team": to_answer(q, {"billing": 0.5, "technical": 0.5 - 1e-9, "sales": 1e-9})})
    r = score_pair(item, "team", pred)
    assert r.correct == 1.0
    assert r.confidence == pytest.approx(0.5)
    assert r.nll == pytest.approx(math.log(2), rel=1e-6)
    assert r.brier == pytest.approx(0.25 + 0.25, rel=1e-6)
    assert r.tv == pytest.approx(0.5 * (0.3 + 0.35 + 0.05), rel=1e-6)


def test_missing_and_errored_predictions_are_unanswered():
    item = make_item()
    assert not score_pair(item, "team", None).answered
    assert not score_pair(item, "team", Prediction("i1", {}, error="boom")).answered
    rep = report(score_run([item], {}))
    assert rep["overall"]["coverage"] == 0.0


def test_score_mae_is_normalized():
    item = make_item()
    q = item.questions["anger"]
    r = score_pair(item, "anger", Prediction("i1", {"anger": to_answer(q, {"0": 0.0, "1": 0.0, "2": 1.0})}))
    assert r.score_abs_err == pytest.approx(0.5)


class _R:
    label = None

    def __init__(self, confidence, correct):
        self.confidence, self.correct = confidence, correct


def test_balanced_accuracy_ignores_class_imbalance():
    from deskmind_brain.eval.metrics import balanced_accuracy

    item = make_item()
    q = item.questions["urgent"]
    # reference says false; a model that always answers false is right here but has no discrimination
    always_false = [score_pair(item, "urgent", Prediction("i1", {"urgent": to_answer(q, {"false": 1.0, "true": 0.0})}))]
    item2 = make_item()
    item2.references["urgent"] = Reference(probs={"false": 0.1, "true": 0.9})
    always_false += [score_pair(item2, "urgent", Prediction("i1", {"urgent": to_answer(q, {"false": 1.0, "true": 0.0})}))]
    assert summarize(always_false)["accuracy"] == pytest.approx(0.5)
    assert balanced_accuracy(always_false) == pytest.approx(0.5)
    assert math.isnan(balanced_accuracy(always_false[:1]))  # single class -> undefined


def test_calibration_summaries():
    perfect = [_R(1.0, 1.0)] * 4
    assert ece(perfect) == pytest.approx(0.0)
    assert aurc(perfect) == pytest.approx(0.0)
    overconfident = [_R(0.95, 0.0)] * 2 + [_R(0.95, 1.0)] * 2
    assert ece(overconfident) == pytest.approx(0.45)
    ranked = [_R(0.9, 1.0), _R(0.8, 1.0), _R(0.7, 0.0), _R(0.6, 0.0)]
    assert selective_accuracy(ranked, 0.5) == 1.0
    assert aurc(ranked) == pytest.approx((0 + 0 + 1 / 3 + 2 / 4) / 4)


def test_train_kl_ignores_padded_candidates():
    """Padded candidate slots must not turn the KL term into NaN."""
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[1.0, 0.5, float("-inf")]])
    teacher = torch.tensor([[0.7, 0.3, 0.0]])
    valid = torch.tensor([[1.0, 1.0, 0.0]])
    log_probs = torch.log_softmax(logits, dim=-1)
    keep = valid > 0
    cross = torch.where(keep, teacher * log_probs, torch.zeros_like(teacher)).sum(-1)
    entropy = torch.where(keep, teacher * teacher.clamp_min(1e-9).log(), torch.zeros_like(teacher)).sum(-1)
    kl = -cross + entropy
    assert torch.isfinite(kl).all() and kl.item() > 0


def test_temperature_fit_sharpens_underconfident_predictions():
    from deskmind_brain.eval.calibrate import apply, fit, rescale

    assert rescale({"a": 0.6, "b": 0.4}, 1.0) == {"a": 0.6, "b": 0.4}
    sharp = rescale({"a": 0.6, "b": 0.4}, 0.5)
    assert sharp["a"] == pytest.approx(0.36 / 0.52)
    items, preds = [], {}
    for i in range(40):  # always right, but only 60% confident
        it = make_item()
        it.id = f"i{i}"
        items.append(it)
        q = it.questions["urgent"]
        preds[it.id] = Prediction(it.id, {"urgent": to_answer(q, {"false": 0.6, "true": 0.4})})
    temps = fit(items, preds)
    assert temps["noul"] < 0.5
    calibrated = apply(items[0], preds["i0"], temps)
    assert calibrated.answers["urgent"]["noul"] < 0.2


def test_token_batches_respect_budget():
    from deskmind_brain.train.train import Example, token_batches

    ex = [Example(input_ids=[0] * n, label_ids=[1], teacher=[1.0], gold=0) for n in (1000, 1000, 2500, 900, 900, 900)]
    batches = token_batches(ex, max_batch=2, token_budget=3072)
    assert [len(b) for b in batches] == [2, 1, 2, 1]
    assert all(max(len(e.input_ids) for e in b) * len(b) <= 3072 for b in batches)
