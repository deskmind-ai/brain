"""Replay logged hands planner requests against a /v1/systemone server and report latency (and routing, if any).

Takes hands trace.jsonl files (records {"t":"request","n":..,"body":{..}}, written with HANDS_LOG_REQUESTS=1) or probe
files ({"request": {...}}), sends each body as-is, and prints p50/mean latency, the operation mix, and for a router
the share and reasons of escalated steps.

    python scripts/replay_requests.py http://127.0.0.1:8796 path/to/hands/runs/<run>/*/trace.jsonl --n 60
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.request
from collections import Counter


def bodies(paths: list[str]) -> list[dict]:
    out = []
    for p in paths:
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("t") == "request":
                out.append(r["body"])
            elif "request" in r:
                out.append(r["request"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--n", type=int, default=0, help="sample this many (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    bs = bodies(args.paths)
    if args.n and len(bs) > args.n:
        bs = random.Random(args.seed).sample(bs, args.n)
    lat, ops, by, reasons = [], Counter(), Counter(), Counter()
    for b in bs:
        req = urllib.request.Request(args.url.rstrip("/") + "/v1/systemone", data=json.dumps(b).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
        t = time.perf_counter()
        out = json.loads(urllib.request.urlopen(req, timeout=600).read())
        lat.append(time.perf_counter() - t)
        ops[(out["answers"].get("operation") or {}).get("choice")] += 1
        if "routing" in out:
            by[out["routing"]["by"]] += 1
            reasons[out["routing"]["reason"]] += 1
    print(json.dumps({"requests": len(bs), "p50_s": round(statistics.median(lat), 2), "mean_s": round(statistics.mean(lat), 2),
                      "ops": dict(ops), "by": dict(by), "reasons": dict(reasons)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
