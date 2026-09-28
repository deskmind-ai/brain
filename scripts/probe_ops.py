"""Replay request bodies against /v1/systemone servers and tabulate the chosen operation against the expected one.

Rows are either real probes ({task, step, request, ...}: expected op from --expect, e.g. "not:DONE") or training-style
items ({state, questions, references, meta}: expected = meta.gold_op), sampled by --n per meta.ce_kind.

    python scripts/probe_ops.py path/to/probes.jsonl --expect not:DONE brain-4b=http://127.0.0.1:8793
    python scripts/probe_ops.py path/to/counterexamples/items_labeled.jsonl --n 25 brain-4b=http://127.0.0.1:8793
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import urllib.request


def ask(url: str, body: dict) -> dict:
    req = urllib.request.Request(url.rstrip("/") + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())["answers"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("servers", nargs="+", help="name=url")
    ap.add_argument("--expect", help="expected op for real probes: OP or not:OP")
    ap.add_argument("--n", type=int, default=0, help="items per kind (training-style rows)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.path) if l.strip()]
    cases = []
    if "request" in rows[0]:
        for r in rows:
            body = r["request"] if isinstance(r["request"], dict) else json.loads(r["request"])
            cases.append((f"{r.get('task')}#{r.get('step')}", body, args.expect))
    else:
        by = collections.defaultdict(list)
        for r in rows:
            by[r["meta"].get("ce_kind") or r["meta"].get("g11_kind", "item")].append(r)
        rng = random.Random(args.seed)
        for kind, rs in sorted(by.items()):
            for r in rng.sample(rs, min(args.n or len(rs), len(rs))):
                cases.append((kind, {"state": r["state"], "questions": r["questions"]}, r["meta"]["gold_op"]))
    for spec in args.servers:
        name, _, url = spec.partition("=")
        ok, tot, ops = collections.Counter(), collections.Counter(), collections.defaultdict(collections.Counter)
        for key, body, exp in cases:
            op = ask(url, body)["operation"]["choice"]
            group = key.split("#")[0]
            good = (op != exp[4:]) if exp and exp.startswith("not:") else (op == exp)
            ok[group] += good
            tot[group] += 1
            ops[group][op] += 1
        print(f"== {name}: {sum(ok.values())}/{sum(tot.values())} as expected")
        for g in tot:
            print(f"  {g:<28} {ok[g]}/{tot[g]}  {dict(ops[g])}")


if __name__ == "__main__":
    main()
