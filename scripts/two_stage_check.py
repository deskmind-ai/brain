"""Two-stage vs one-pass on real agent requests: same answers where it matters, how much faster.

Replays full planner requests (all ~10 questions, exactly as hands sent them) from a DAgger file against two servers (repeat-request cache off) of
the same checkpoint, one started plainly and one with --two-stage. Two-stage only answers the operation and the heads
that operation needs; those must agree with the one-pass answers (argmax), and the per-request latency should drop.

    python scripts/two_stage_check.py --rows path/to/dagger_rows.jsonl --n 40 \\
        --one http://127.0.0.1:8797 --two http://127.0.0.1:8798
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.request

from deskmind_brain.eval.logits_base import heads_for


def ask(url: str, body: dict) -> tuple[dict, float]:
    req = urllib.request.Request(url.rstrip("/") + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    t = time.perf_counter()
    out = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return out["answers"], time.perf_counter() - t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--one", required=True)
    ap.add_argument("--two", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.rows) if l.strip()]
    rows = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
    t1, t2, op_agree, head_agree, heads = [], [], 0, 0, 0
    for i, r in enumerate(rows):
        body = {"state": r["state"], "questions": r["questions"], "model": "local"}
        a1, s1 = ask(args.one, body)
        a2, s2 = ask(args.two, body)
        t1.append(s1)
        t2.append(s2)
        op = a1["operation"]["choice"]
        op_agree += op == a2["operation"]["choice"]
        for h in heads_for(op, r["questions"]):
            if len((r["questions"][h].get("criteria") or {})) > 1:
                heads += 1
                head_agree += a1[h].get("choice") == a2[h].get("choice")
    print(json.dumps({"requests": len(rows), "operation_agree": op_agree, "needed_heads": heads, "heads_agree": head_agree,
                      "one_pass_s": {"p50": round(statistics.median(t1), 2), "mean": round(statistics.mean(t1), 2)},
                      "two_stage_s": {"p50": round(statistics.median(t2), 2), "mean": round(statistics.mean(t2), 2)}}))


if __name__ == "__main__":
    main()
