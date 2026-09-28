"""Replay real /v1/systemone request bodies against several servers, one server at a time: mean/p50/p95 latency and
agreement of every answered head with the first server (the reference).

    python scripts/latency_agree.py ref=http://127.0.0.1:8793 q8=http://127.0.0.1:8794 --probes path/to/probes.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request


def ask(url: str, body: dict) -> tuple[dict, float]:
    req = urllib.request.Request(url.rstrip("/") + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    t = time.time()
    out = json.loads(urllib.request.urlopen(req, timeout=600).read())["answers"]
    return out, time.time() - t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("servers", nargs="+")
    ap.add_argument("--probes", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    bodies = []
    for p in args.probes:
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                b = r["request"] if isinstance(r["request"], dict) else json.loads(r["request"])
                bodies.append(b)
    if args.limit:
        bodies = bodies[: args.limit]
    ref = None
    for spec in args.servers:
        name, _, url = spec.partition("=")
        ask(url, bodies[0])  # warm-up
        answers, times = [], []
        for b in bodies:
            a, dt = ask(url, b)
            answers.append(a)
            times.append(dt)
        times.sort()
        line = f"{name}: n={len(bodies)} mean {statistics.mean(times):.2f}s p50 {times[len(times)//2]:.2f}s p95 {times[int(len(times)*0.95)]:.2f}s"
        if ref is None:
            ref = answers
        else:
            op_same = sum(a["operation"]["choice"] == r["operation"]["choice"] for a, r in zip(answers, ref))
            op = [r["operation"]["choice"] for r in ref]
            heads_same = sum(all((a.get(h) or {}).get("choice") == (r.get(h) or {}).get("choice") for h in r if h != "operation")
                             for a, r in zip(answers, ref))
            line += f" | operation agrees {op_same}/{len(ref)}, all heads agree {heads_same}/{len(ref)}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
