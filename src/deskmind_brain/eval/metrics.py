"""Per-question scoring against reference distributions, plus aggregation.

Hard metrics use the reference argmax as the label: accuracy, NLL, Brier, ECE, AURC, selective accuracy.
Soft metrics compare the whole distribution with the reference mean: cross-entropy, KL, total variation.
Top-label probability is the confidence signal for ECE/AURC so all three question types share one scale.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

from deskmind_brain.eval.data import EvalItem, Prediction
from deskmind_brain.types import Distribution, from_answer, score_value

EPS = 1e-6
ECE_BINS = 10


@dataclass
class PairResult:
    item_id: str
    group: str
    qid: str
    qtype: str
    k: int
    answered: bool
    label: str | None = None  # reference argmax, for class-balanced accuracy
    correct: float = 0.0  # 1 if the predicted argmax is the reference label; ties get fractional credit
    confidence: float = 0.0
    nll: float = 0.0
    brier: float = 0.0
    cross_entropy: float = 0.0
    kl: float = 0.0
    tv: float = 0.0
    score_abs_err: float | None = None  # |E_pred - E_ref| / (K - 1), score questions only


def k_bucket(k: int) -> str:
    if k <= 2:
        return "K=2"
    if k <= 5:
        return "K=3-5"
    if k <= 10:
        return "K=6-10"
    if k <= 50:
        return "K=11-50"
    return "K>50"


def _argmax(dist: Distribution, options: list[str]) -> str:
    # Reference ties resolve to the first option in declaration order.
    return max(options, key=lambda o: dist[o])


def _hit(pred: Distribution, options: list[str], label: str) -> float:
    """Expected accuracy of taking the predicted argmax with random tie-breaking."""
    top = max(pred[o] for o in options)
    tied = [o for o in options if pred[o] >= top - 1e-12]
    return 1.0 / len(tied) if label in tied else 0.0


def score_pair(item: EvalItem, qid: str, pred: Prediction | None) -> PairResult:
    question = item.questions[qid]
    options = question.options()
    ref = item.references[qid].probs
    base = PairResult(item.id, item.group, qid, question.type, len(options), answered=False)
    answer = pred.answers.get(qid) if pred and not pred.error else None
    if answer is None:
        return base
    try:
        p = from_answer(question, answer)
    except (KeyError, TypeError, ValueError):
        return base

    label = _argmax(ref, options)
    clipped = {o: min(max(p[o], EPS), 1.0) for o in options}
    ref_entropy = -sum(ref[o] * math.log(ref[o]) for o in options if ref[o] > 0)
    cross_entropy = -sum(ref[o] * math.log(clipped[o]) for o in options)
    result = PairResult(
        item.id,
        item.group,
        qid,
        question.type,
        len(options),
        answered=True,
        label=label,
        correct=_hit(p, options, label),
        confidence=max(p.values()),
        nll=-math.log(clipped[label]),
        brier=sum((p[o] - (1.0 if o == label else 0.0)) ** 2 for o in options),
        cross_entropy=cross_entropy,
        kl=max(cross_entropy - ref_entropy, 0.0),
        tv=0.5 * sum(abs(p[o] - ref[o]) for o in options),
    )
    if question.type == "score":
        result.score_abs_err = abs(score_value(p) - score_value(ref)) / (len(options) - 1)
    return result


def score_run(items: Iterable[EvalItem], preds: dict[str, Prediction]) -> list[PairResult]:
    return [score_pair(it, qid, preds.get(it.id)) for it in items for qid in it.references]


def ece(pairs: list[PairResult], bins: int = ECE_BINS) -> float:
    if not pairs:
        return float("nan")
    buckets: dict[int, list[PairResult]] = defaultdict(list)
    for r in pairs:
        buckets[min(int(r.confidence * bins), bins - 1)].append(r)
    return sum(
        len(b) / len(pairs) * abs(sum(r.correct for r in b) / len(b) - sum(r.confidence for r in b) / len(b))
        for b in buckets.values()
    )


def aurc(pairs: list[PairResult]) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    if not pairs:
        return float("nan")
    ranked = sorted(pairs, key=lambda r: -r.confidence)
    errors, total = 0.0, 0.0
    for i, r in enumerate(ranked, 1):
        errors += 1.0 - r.correct
        total += errors / i
    return total / len(ranked)


def selective_accuracy(pairs: list[PairResult], coverage: float) -> float:
    if not pairs:
        return float("nan")
    n = max(1, round(len(pairs) * coverage))
    kept = sorted(pairs, key=lambda r: -r.confidence)[:n]
    return sum(r.correct for r in kept) / n


def coverage_at_risk(pairs: list[PairResult], max_risk: float) -> float:
    """Largest share of pairs a router can hand to the model, most confident first, keeping their error rate within
    max_risk. This is the routing question itself; a fixed threshold (p >= 0.8) means different risks on different
    models and question types, so shares at one threshold are not comparable."""
    if not pairs:
        return float("nan")
    errors, best = 0.0, 0
    for i, r in enumerate(sorted(pairs, key=lambda r: -r.confidence), 1):
        errors += 1.0 - r.correct
        if errors <= max_risk * i:
            best = i
    return best / len(pairs)


def balanced_accuracy(pairs: list[PairResult]) -> float:
    """Mean of the per-reference-class accuracies: unaffected by a dominant class (0.5 = no discrimination)."""
    per: dict[str, list[float]] = defaultdict(list)
    for r in pairs:
        if r.label is not None:
            per[r.label].append(r.correct)
    if len(per) < 2:
        return float("nan")
    return sum(sum(v) / len(v) for v in per.values()) / len(per)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(results: list[PairResult]) -> dict[str, float]:
    answered = [r for r in results if r.answered]
    score_errs = [r.score_abs_err for r in answered if r.score_abs_err is not None]
    return {
        "pairs": len(results),
        "coverage": len(answered) / len(results) if results else float("nan"),
        "accuracy": _mean([r.correct for r in answered]),
        "balanced_accuracy": balanced_accuracy(answered),
        "nll": _mean([r.nll for r in answered]),
        "brier": _mean([r.brier for r in answered]),
        "ece": ece(answered),
        "aurc": aurc(answered),
        "sel_acc@80": selective_accuracy(answered, 0.8),
        "cov@risk5": coverage_at_risk(answered, 0.05),
        "cov@risk10": coverage_at_risk(answered, 0.10),
        "cross_entropy": _mean([r.cross_entropy for r in answered]),
        "kl": _mean([r.kl for r in answered]),
        "tv": _mean([r.tv for r in answered]),
        "score_mae": _mean(score_errs),
    }


def breakdown(results: list[PairResult], key: Callable[[PairResult], str]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[PairResult]] = defaultdict(list)
    for r in results:
        groups[key(r)].append(r)
    return {name: summarize(rs) for name, rs in sorted(groups.items())}


def report(results: list[PairResult]) -> dict[str, object]:
    by_group = breakdown(results, lambda r: r.group)
    macro = {
        m: _mean([g[m] for g in by_group.values() if not math.isnan(g[m])])
        for m in ("accuracy", "balanced_accuracy", "nll", "brier", "ece", "kl", "tv")
    }
    return {
        "overall": summarize(results),
        "macro_by_group": macro,
        "by_group": by_group,
        "by_type": breakdown(results, lambda r: r.qtype),
        "by_k": breakdown(results, lambda r: k_bucket(r.k)),
    }
