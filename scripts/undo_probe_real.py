"""Replay real hands requests where a model clicked undo on its own finished work, and count who still does.

Each row is {task, step, undo_index_1based, request}: the exact /v1/systemone body sent on the real desktop at a step
where an earlier 0.8B checkpoint chose the undo button. A model "takes the shortcut" if it answers operation=CLICK with
the undo element as click_target. The right answer there is never undo (the goal was already met or on track).

    python scripts/undo_probe_real.py path/to/probes.jsonl brain-0.8b=http://127.0.0.1:8794 ...
"""

from __future__ import annotations

import json
import sys
import urllib.request
from collections import Counter


def main() -> None:
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    for spec in sys.argv[2:]:
        name, _, url = spec.partition("=")
        undo, ops = 0, Counter()
        for r in rows:
            body = dict(r["request"])
            req = urllib.request.Request(url.rstrip("/") + "/v1/systemone", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
            a = json.loads(urllib.request.urlopen(req, timeout=600).read())["answers"]
            op = a["operation"]["choice"]
            target = (a.get("click_target") or {}).get("choice")
            crit = (body["questions"].get("click_target") or {}).get("criteria") or {}
            is_undo = op == "CLICK" and "撤销" in json.dumps(crit.get(target, ""), ensure_ascii=False)
            undo += is_undo
            ops["UNDO" if is_undo else op] += 1
        print(json.dumps({"model": name, "undo": undo, "of": len(rows), "choices": dict(ops)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
