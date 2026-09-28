"""Offline desktop check on held-out hands tasks: does the model pick the oracle's operation, and its target?

The held-out suite is whole tasks (two per family) that `import_hands_dagger.py --holdout-per-family` kept out of
training: every state the student visited on them, labelled by the corrected oracle. Scored per family, because the
families a checkpoint never passes are the ones the next training round has to move.

    python scripts/desktop_holdout_eval.py path/to/predictions.jsonl [more runs...]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

SUITE = Path("data/eval/desktop_holdout/items.jsonl")


def top(d: dict) -> str | None:
    return max(d, key=d.get) if d else None


def main() -> None:
    args = sys.argv[1:]
    suite = SUITE
    if args[:1] == ["--suite"]:  # e.g. data/eval/desktop_undo_probe/items.jsonl: states whose right move is NOT undo
        suite, args = Path(args[1]), args[2:]
    items = {json.loads(l)["id"]: json.loads(l) for l in suite.read_text().splitlines() if l.strip()}
    for run in args:
        preds = {json.loads(l)["item_id"]: json.loads(l) for l in Path(run).read_text().splitlines() if l.strip()}
        fam = defaultdict(lambda: [0, 0, 0])  # states, operation right, operation + target right
        for iid, it in items.items():
            f = it["meta"]["task_id"].rsplit("-", 1)[0].removeprefix("T-")
            answers = (preds.get(iid) or {}).get("answers") or {}
            gold_op = top(it["references"]["operation"]["probs"])
            op = (answers.get("operation") or {}).get("choice")
            heads = [q for q in it["references"] if q != "operation"]
            full = op == gold_op and all((answers.get(q) or {}).get("choice") == top(it["references"][q]["probs"]) for q in heads)
            for key in (f, "ALL"):
                fam[key][0] += 1
                fam[key][1] += op == gold_op
                fam[key][2] += full
        cells = "  ".join(f"{k} {v[1]}/{v[0]}" for k, v in sorted(fam.items()) if k != "ALL")
        a = fam["ALL"]
        print(f"{Path(run).stem}: operation {a[1] / a[0]:.3f}  operation+target {a[2] / a[0]:.3f}  ({a[0]} states)\n  by family (operation): {cells}")


if __name__ == "__main__":
    main()
