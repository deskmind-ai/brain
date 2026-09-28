"""Recovery slice: after the environment rejects an action, does the policy change its answer?

An earlier 0.8B checkpoint did not: probed on a desktop task it kept the rejected target at ~0.9 across three
rejections, where a stronger model moves to the right element on the first retry. This measures that directly, on states whose
history ends in a failed step ("!! <reason>" in the history):

  change rate  — the policy's chosen target differs from the one the history says was just rejected
  recovery acc — of those, how many pick the gold target (the unmet precondition the reason names)
  stuck rate   — it repeats the rejected action

    uv run python scripts/recovery_eval.py --requests path/to/requests.jsonl \
        --chooser brain-0.8b=http://127.0.0.1:8794 --chooser brain-4b=http://127.0.0.1:8793
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path


def ask(base_url: str, body: dict, api_key: str, model: str, timeout_s: float = 120) -> dict:
    payload = {**body, "model": model}
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/systemone", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout_s).read())


def rejected_label(state: dict) -> str | None:
    """The label of the action the environment just rejected, if the last history entry failed."""
    history = state.get("recent_actions") or []
    if not history:
        return None
    last = history[-1]
    if last.get("ok") is False or last.get("page_changed") is False:
        return last.get("action")
    return None


def label_of(questions: dict, operation: str, target: str | None) -> str | None:
    head = questions.get((operation or "").lower() + "_target") or {}
    element = (head.get("criteria") or {}).get(target, {}).get("element")
    return element.split("] ", 1)[-1] if isinstance(element, str) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=Path, required=True, help="jsonl of {state, questions, references} requests")
    ap.add_argument("--chooser", action="append", required=True, help="name=base_url")
    ap.add_argument("--model", default="deskmind-brain-local")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path, default=Path("runs/recovery"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.requests.read_text().splitlines()]
    rows = [r for r in rows if rejected_label(r["state"])][: args.limit]
    print(f"{len(rows)} states whose last action was rejected")
    args.out.mkdir(parents=True, exist_ok=True)

    for spec in args.chooser:
        name, _, base = spec.partition("=")
        key = os.environ.get("SYSTEMONE_API_KEY", "local") if "typesafe.ai" in base else "local"
        model = "jev-latest" if "typesafe.ai" in base else args.model
        changed = correct = stuck = errors = 0
        records = []
        for row in rows:
            body = {"state": row["state"], "questions": row["questions"]}
            try:
                answers = ask(base, body, key, model)["answers"]
            except Exception as e:  # noqa: BLE001 — a failed call is reported, not fatal
                errors += 1
                continue
            operation = answers.get("operation", {}).get("choice")
            target = (answers.get((operation or "").lower() + "_target") or {}).get("choice")
            picked = label_of(row["questions"], operation, target)
            rejected = rejected_label(row["state"])
            gold_op = (row.get("references", {}).get("operation", {}).get("probs") or {})
            gold_op = max(gold_op, key=gold_op.get) if gold_op else None
            gold_head = row.get("references", {}).get((gold_op or "").lower() + "_target", {}).get("probs") or {}
            gold_target = max(gold_head, key=gold_head.get) if gold_head else None
            moved = picked is None or picked != rejected
            changed += moved
            stuck += not moved
            hit = operation == gold_op and (gold_target is None or target == gold_target)
            correct += hit
            records.append({"id": row.get("id"), "rejected": rejected, "picked": picked, "operation": operation,
                            "gold_op": gold_op, "gold_target": gold_target, "changed": moved, "correct": hit})
        n = max(len(rows) - errors, 1)
        summary = {"chooser": name, "states": len(rows), "errors": errors, "change_rate": round(changed / n, 3),
                   "stuck_rate": round(stuck / n, 3), "recovery_acc": round(correct / n, 3)}
        (args.out / f"{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
