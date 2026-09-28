from __future__ import annotations

import sys
from pathlib import Path

from deskmind_brain.eval.data import EvalItem, Prediction, load_predictions, write_jsonl
from deskmind_brain.eval.predictors import Predictor


def run_predictions(
    predictor: Predictor,
    items: list[EvalItem],
    out_path: Path,
    resume: bool = True,
    limit: int | None = None,
) -> dict[str, Prediction]:
    """Predict every item, appending to `out_path` as we go. Items that already succeeded are skipped."""
    done = load_predictions(out_path) if resume and out_path.exists() else {}
    if not resume and out_path.exists():
        out_path.unlink()
    todo = [it for it in items if it.id not in done or done[it.id].error]
    if limit is not None:
        todo = todo[:limit]
    for i, item in enumerate(todo, 1):
        try:
            pred = predictor.predict(item)
        except Exception as exc:  # recorded per item so one failure does not stop the run
            pred = Prediction(item_id=item.id, answers={}, error=f"{type(exc).__name__}: {exc}")
        write_jsonl(out_path, [pred.to_json()], append=True)
        done[item.id] = pred
        status = "error" if pred.error else "ok"
        print(f"[{i}/{len(todo)}] {item.id} {status}", file=sys.stderr)
    return done
