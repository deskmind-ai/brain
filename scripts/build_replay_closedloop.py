"""Build the closed-loop replay set (fixtures/replay/v2/closedloop.jsonl.gz) and the version 2 manifest.

Source: one run of hands' gym on a second machine (M2 Max, macOS 27.2, released Peekaboo 4.7.0, hands main with
hands#6 and #7), 2026-10-06. G18b as the app runs it (0.8B -> 4B, threshold 0.96, two-stage) drove 75 held-out gym
tasks by itself (`tools/gym/run.py --split 2 --beta 0`; music, mail and settings, 25 each), so every state is one the
model reached on its own. Every step is labelled by the gym's oracle, which reads the app's true state: the gold is
independent of the model. One row the oracle could not label (after an irreversible wrong step) is left out.

What makes the set different from v1's gym set: the states come from the model's own trajectory, and each fixture
keeps what the model chose there (meta.closed_loop: its choices, confidence, routing and whether the task passed), so a
gate's verdict on a replayed answer can be compared with what happened on the desktop. 74 of the 75 tasks passed: this
is a regression set, not a hard one. The one that failed, gym-mail-邮筒-s0071, is a missed DONE with an irreversible
write (steps 4-6: the right message was deleted, then another with the same subject).

    uv run python scripts/build_replay_closedloop.py --rows path/to/{music,mail,settings}.rows.jsonl \\
        --routing path/to/routing.jsonl

Rows are hands' gym rows (one per model call: state, questions, answers, label). The routing log is brain's
--routing-log for the same run; its last len(rows) lines are matched to the rows in order and checked by operation.
The output is frozen like v1: the manifest records the count and sha256, and the loader refuses a changed file.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
from collections import Counter
from pathlib import Path

from deskmind_brain.eval.data import EvalItem, Reference
from deskmind_brain.eval.replay import GATES_VERSION, categories_of, session_of, sha256_file, split_of, step_key
from deskmind_brain.types import Question

ROOT = Path(__file__).resolve().parent.parent
V1, V2 = ROOT / "fixtures" / "replay" / "v1", ROOT / "fixtures" / "replay" / "v2"
SET, RUN_LABEL = "closedloop", "closedloop.g18b-20261006"


def item_of(row: dict, routing: dict | None) -> EvalItem | None:
    """One gym row as a fixture; None when the oracle gave no label."""
    label = row.get("label")
    if not label or row.get("label_source") != "oracle":
        return None
    questions = {qid: Question.model_validate(q) for qid, q in row["questions"].items()}
    refs = {}
    for qid, ans in label.items():
        choice, opts = str(ans["choice"]), questions[qid].options()
        if choice not in opts:
            raise ValueError(f"{row['task']} step {row['step']}: label {qid}={choice!r} is not among the options")
        refs[qid] = Reference(probs={o: 1.0 if o == choice else 0.0 for o in opts}, n_sets=1, soft=False)
    answers = row.get("answers") or {}
    item = EvalItem(
        id=f"hands/{RUN_LABEL}/{row['task']}/{row['step']}", suite="hands_gym_closed_loop", group="desktop",
        state=row["state"], questions=questions, references=refs,
        meta={"source": "hands_gym_closed_loop", "task_id": row["task"], "step": row["step"], "app": row.get("app"),
              "lang": row.get("lang"), "seed": row.get("seed"), "template_id": row.get("template_id"),
              "gym_split": row.get("split"), "split_version": row.get("split_version"),
              "label_source": "oracle", "oracle_version": row.get("oracle_version"), "oracle_note": row.get("note"),
              "gold_op": str(label["operation"]["choice"]),
              "closed_loop": {
                  "planner": "G18b router (brain-0.8b -> brain-4b g18b-q8, threshold 0.96, two-stage)",
                  "choices": {qid: (a or {}).get("choice") for qid, a in answers.items()},
                  "confidence": (answers.get("operation") or {}).get("confidence"),
                  "routing": routing, "task_passed": (row.get("run_result") or {}).get("passed")}})
    session = session_of(item.id)
    item.meta["replay"] = {"set": SET, "session": session, "split": split_of(session), "categories": categories_of(item)}
    return item


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", type=Path, required=True)
    ap.add_argument("--routing", type=Path, help="brain --routing-log of the same run")
    args = ap.parse_args()
    rows = [json.loads(line) for p in args.rows for line in p.open(encoding="utf-8") if line.strip()]
    routing: list[dict | None] = [None] * len(rows)
    if args.routing:
        log = [json.loads(line) for line in args.routing.open() if line.strip()][-len(rows):]
        ops = [((r.get("answers") or {}).get("operation") or {}).get("choice") for r in rows]
        if len(log) != len(rows) or any(entry["final_op"] != op for entry, op in zip(log, ops)):
            raise SystemExit("the routing log does not line up with the rows (count or operation differs)")
        routing = [{"by": "strong" if e["escalated"] else "fast", "reason": e["reason"], "fast_op": e["fast_op"],
                    "fast_conf": e["conf"], "total_s": e["total_s"]} for e in log]
    items = sorted(filter(None, (item_of(r, rt) for r, rt in zip(rows, routing))), key=lambda it: step_key(it.id))

    V2.mkdir(parents=True, exist_ok=True)
    out = V2 / f"{SET}.jsonl.gz"
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:   # mtime=0: the same bytes every time
        for it in items:
            gz.write((json.dumps(it.to_json(), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    out.write_bytes(buf.getvalue())

    v1 = json.loads((V1 / "manifest.json").read_text())
    sets = [{**s, "path": s["path"] if s["visibility"] == "private" else f"../v1/{s['path']}"} for s in v1["sets"]]
    sets.append({"name": SET, "path": out.name, "visibility": "public",
                 "source": "hands tools/gym/run.py --split 2 --beta 0, G18b router driving 75 held-out gym tasks "
                           "(music, mail, settings); gym oracle labels; second machine, 2026-10-06 (deskmind#18)",
                 "license": "Apache-2.0", "count": len(items), "sha256": sha256_file(out),
                 "note": "states from the model's own trajectory; meta.closed_loop keeps what it chose and whether the "
                         "task passed (74/75): a regression set. gym-mail-邮筒-s0071 is the missed DONE"})
    manifest = {**{k: v for k, v in v1.items() if k != "sets"}, "version": "2", "gates_version": GATES_VERSION,
                "note": "version 1's five sets, unchanged and read from ../v1 (the private two from "
                        "$DESKMIND_REPLAY_PRIVATE as before), plus closedloop. Version 1's manifest and baseline stay "
                        "as they are; a baseline over this manifest has to be run before it can gate anything.",
                "sets": sets}
    (V2 / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    splits = Counter(it.meta["replay"]["split"] for it in items)
    print(f"{len(items)} fixtures ({len(rows) - len(items)} unlabelled rows left out) -> {out}")
    print("split", dict(splits), "| sessions", len({it.meta['replay']['session'] for it in items}),
          "| apps", dict(Counter(it.meta["app"] for it in items)))


if __name__ == "__main__":
    main()
