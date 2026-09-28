"""Completion slice: does the policy say DONE exactly when the goal is visibly met, and confidently enough to act on?

On the desktop harness (DeskMind Hands) earlier checkpoints rarely ranked DONE first after the goal was met, and
when they did it was with low confidence — under the 0.5 floor a router needs. Others showed the opposite error on a
travel site (DONE before searching). This scores both directions on held-out browser states
built as contrasting pairs (done_* and not_done_* scenarios):

  done recall      gold DONE states where the policy's first choice is DONE
  done@0.5         ... and its DONE probability is at least 0.5 (what a router can act on)
  false done       gold not-DONE states where the policy's first choice is DONE
  fix accuracy     gold not-DONE states where the policy picks the gold operation and target

    uv run python scripts/done_eval.py --requests path/to/requests.jsonl --chooser brain-0.8b=http://127.0.0.1:8794
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


def ask(base_url: str, body: dict, timeout_s: float = 120) -> dict:
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/systemone",
                                 data=json.dumps({**body, "model": "deskmind-brain-local"}).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout_s).read())


def gold_of(row: dict) -> tuple[str | None, str | None]:
    refs = row.get("references") or {}
    ops = (refs.get("operation") or {}).get("probs") or {}
    op = max(ops, key=ops.get) if ops else None
    head = (refs.get((op or "").lower() + "_target") or {}).get("probs") or {}
    return op, (max(head, key=head.get) if head else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=Path, required=True)
    ap.add_argument("--chooser", action="append", required=True, help="name=base_url")
    ap.add_argument("--out", type=Path, default=Path("runs/done"))
    args = ap.parse_args()
    rows = [json.loads(l) for l in args.requests.read_text().splitlines() if l.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    for spec in args.chooser:
        name, _, base = spec.partition("=")
        pos = pos_hit = pos_conf = neg = neg_false = neg_fix = errors = 0
        records = []
        for row in rows:
            gold_op, gold_target = gold_of(row)
            try:
                answers = ask(base, {"state": row["state"], "questions": row["questions"]})["answers"]
            except Exception:  # noqa: BLE001 — reported, not fatal
                errors += 1
                continue
            op_answer = answers.get("operation") or {}
            op = op_answer.get("choice")
            p_done = (op_answer.get("probabilities") or {}).get("DONE", 0.0)
            target = (answers.get((op or "").lower() + "_target") or {}).get("choice")
            if gold_op == "DONE":
                pos += 1
                pos_hit += op == "DONE"
                pos_conf += op == "DONE" and p_done >= 0.5
            else:
                neg += 1
                neg_false += op == "DONE"
                neg_fix += op == gold_op and (gold_target is None or target == gold_target)
            records.append({"id": row.get("id"), "scenario": (row.get("meta") or {}).get("scenario"), "gold_op": gold_op,
                            "op": op, "p_done": round(p_done, 3), "target": target, "gold_target": gold_target})
        summary = {"chooser": name, "states": len(rows), "errors": errors,
                   "done_recall": round(pos_hit / max(pos, 1), 3), "done_at_0.5": round(pos_conf / max(pos, 1), 3),
                   "false_done": round(neg_false / max(neg, 1), 3), "fix_accuracy": round(neg_fix / max(neg, 1), 3),
                   "positives": pos, "negatives": neg}
        (args.out / f"{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
