"""Attach a teacher's answer distribution to every training question.

Runs the same logit-reading predictor used for evaluation, so the student is trained on exactly the
distribution the teacher would return at inference time. Predictions are written incrementally and resumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deskmind_brain.eval.data import load_items, load_predictions, write_jsonl
from deskmind_brain.eval.predictors import make_predictor
from deskmind_brain.eval.runner import run_predictions
from deskmind_brain.train.data import OUT_DIR
from deskmind_brain.types import from_answer


def check_teacher(teacher: str) -> None:
    """The hosted System One API is never a teacher: its outputs must not become training labels.

    `systemone:` specs are allowed only with an explicit base URL that is not the hosted API (e.g. a local
    deskmind-brain-serve tier distilling into a smaller one)."""
    spec = teacher.lower()
    hosted = "systemone" in spec and ("@" not in spec or "typesafe" in spec)
    if hosted or "jev" in spec:
        raise SystemExit(f"refusing teacher {teacher!r}: hosted System One outputs are not used as training labels")


def label(teacher: str = "local:Qwen/Qwen3.5-4B", data_dir: Path = OUT_DIR, limit: int | None = None) -> dict:
    check_teacher(teacher)
    items = load_items(data_dir)
    preds_path = data_dir / f"teacher_{teacher.split(':')[-1].replace('/', '_')}.jsonl"
    preds = run_predictions(make_predictor(teacher), items, preds_path, limit=limit)
    return attach(data_dir, preds, teacher)


def _argmax(dist: dict[str, float]) -> str:
    return max(dist, key=dist.get)


def attach(data_dir: Path, preds: dict, teacher: str, fill_gold: float | None = None, require_agree: bool = False) -> dict:
    """Write items_labeled.jsonl: every item with a teacher prediction, its distributions in meta.teacher_probs.

    fill_gold: items without a prediction get the gold label smoothed to this much mass instead of being dropped.
    require_agree: drop items where the teacher's top answer contradicts a reference (filters bad synthetic data).
    """
    items = load_items(data_dir)
    labeled, missing, filled, disagreed = [], 0, 0, 0
    for item in items:
        pred = preds.get(item.id)
        if not pred or pred.error:
            if fill_gold is None:
                missing += 1
                continue
            teacher_probs = {}
            for qid, ref in item.references.items():
                gold, rest = _argmax(ref.probs), (1 - fill_gold) / max(len(ref.probs) - 1, 1)
                teacher_probs[qid] = {o: fill_gold if o == gold else rest for o in ref.probs}
            item.meta = {**item.meta, "teacher": "gold", "teacher_probs": teacher_probs}
            labeled.append(item)
            filled += 1
            continue
        teacher_probs = {}
        for qid, question in item.questions.items():
            if qid in pred.answers:
                teacher_probs[qid] = from_answer(question, pred.answers[qid])
        if require_agree and any(
            qid in teacher_probs and _argmax(teacher_probs[qid]) != _argmax(ref.probs) for qid, ref in item.references.items()
        ):
            disagreed += 1
            continue
        item.meta = {**item.meta, "teacher": teacher, "teacher_probs": teacher_probs}
        labeled.append(item)
    write_jsonl(data_dir / "items_labeled.jsonl", (it.to_json() for it in labeled))
    return {"labeled": len(labeled), "missing": missing, "gold_filled": filled, "disagreed": disagreed,
            "out": str(data_dir / "items_labeled.jsonl")}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="local:Qwen/Qwen3.5-4B")
    p.add_argument("--data-dir", type=Path, default=OUT_DIR)
    p.add_argument("--from-predictions", type=Path, help="attach an existing predictions JSONL (e.g. from your own teacher)")
    p.add_argument("--fill-gold", type=float, help="use the smoothed gold label for items the teacher did not label")
    p.add_argument("--require-agree", action="store_true", help="drop items whose teacher answer contradicts the label")
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    check_teacher(args.teacher)
    if args.from_predictions:
        result = attach(args.data_dir, load_predictions(args.from_predictions), args.teacher, args.fill_gold, args.require_agree)
    else:
        result = label(args.teacher, args.data_dir, limit=args.limit)
    print(json.dumps(result, indent=2))
