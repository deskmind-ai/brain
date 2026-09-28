"""Post-hoc temperature scaling, one temperature per question kind.

A distribution p is rescaled as p_i^(1/T) / sum_j p_j^(1/T), which equals dividing the answer logits by T. T is
fitted by minimizing NLL against the reference distributions on a calibration set that must be disjoint from the
eval set; `crossfit` reports the effect on one suite with 2-fold fitting.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from deskmind_brain.eval.data import EvalItem, Prediction
from deskmind_brain.types import Question, from_answer, to_answer

GRID = [round(0.2 + 0.05 * i, 2) for i in range(57)]  # 0.20 .. 3.00


def kind(question: Question) -> str:
    """Calibration bucket: long option lists (element pickers) behave differently from short ones."""
    if question.type == "choice":
        return "choice>10" if len(question.options()) > 10 else "choice"
    return question.type


def rescale(dist: dict[str, float], temperature: float) -> dict[str, float]:
    if temperature == 1.0:
        return dist
    logs = {k: math.log(max(v, 1e-12)) / temperature for k, v in dist.items()}
    top = max(logs.values())
    w = {k: math.exp(v - top) for k, v in logs.items()}
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def apply(item: EvalItem, pred: Prediction, temps: dict[str, float]) -> Prediction:
    if pred.error:
        return pred
    answers = {}
    for qid, answer in pred.answers.items():
        q = item.questions.get(qid)
        t = temps.get(kind(q), 1.0) if q else 1.0
        answers[qid] = answer if q is None or t == 1.0 else to_answer(q, rescale(from_answer(q, answer), t))
    return Prediction(pred.item_id, answers, latency_s=pred.latency_s, cost_usd=pred.cost_usd, usage=pred.usage)


def _pairs(items: list[EvalItem], preds: dict[str, Prediction]):
    for it in items:
        p = preds.get(it.id)
        if not p or p.error:
            continue
        for qid, ref in it.references.items():
            if qid in p.answers:
                q = it.questions[qid]
                yield kind(q), from_answer(q, p.answers[qid]), ref.probs


def fit(items: list[EvalItem], preds: dict[str, Prediction]) -> dict[str, float]:
    by_kind: dict[str, list] = {}
    for k, dist, ref in _pairs(items, preds):
        by_kind.setdefault(k, []).append((dist, ref))

    def nll(pairs, t):
        total = 0.0
        for dist, ref in pairs:
            q = rescale(dist, t)
            total -= sum(r * math.log(max(q.get(o, 0.0), 1e-12)) for o, r in ref.items() if r > 0)
        return total / len(pairs)

    return {k: min(GRID, key=lambda t: nll(pairs, t)) for k, pairs in by_kind.items() if len(pairs) >= 20}


def crossfit(items: list[EvalItem], preds: dict[str, Prediction]) -> tuple[dict[str, Prediction], list[dict]]:
    """2-fold: fit on even items, apply to odd ones and vice versa. Returns calibrated predictions and both fits."""
    folds = [items[0::2], items[1::2]]
    out, fits = {}, []
    for i in range(2):
        temps = fit(folds[i], preds)
        fits.append(temps)
        for it in folds[1 - i]:
            if it.id in preds:
                out[it.id] = apply(it, preds[it.id], temps)
    return out, fits


class CalibratedPredictor:
    """Wraps any predictor and rescales its answers with fitted temperatures (a JSON file {kind: T})."""

    def __init__(self, inner, temps_path: str):
        self.inner = inner
        self.temps = json.loads(Path(temps_path).read_text())
        self.name = f"{inner.name}~{Path(temps_path).stem}"

    def predict(self, item: EvalItem) -> Prediction:
        return apply(item, self.inner.predict(item), self.temps)
