"""Replay /v1/systemone requests and report latency, two ways.

Logged requests against a running server: hands trace.jsonl files (records {"t":"request","n":..,"body":{..}},
written with HANDS_LOG_REQUESTS=1) or probe files ({"request": {...}}), each body sent as-is; prints p50/mean latency,
the operation mix, and for a router the share and reasons of escalated steps.

    python scripts/replay_requests.py http://127.0.0.1:8796 path/to/hands/runs/<run>/*/trace.jsonl --n 60

The correctness-gated benchmark (deskmind-ai/deskmind#16): labelled fixtures from a manifest, both tiers loaded in
this process as the Mac app runs them, cold and warm, with stage timings, checkpoint hits and the gates of
deskmind_brain/eval/replay.py. Writes <out>/{cold,warm}.jsonl, summary.json and report.md. See docs/replay.md.

    uv run --extra mlx python scripts/replay_requests.py --manifest fixtures/replay/v1/manifest.json \\
        --fast mlx:models/brain-0.8b --strong mlx:models/brain-4b --out runs/replay/baseline
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


def replay_http(args) -> None:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?")
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--n", type=int, default=0, help="sample this many (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", help="run the fixture benchmark instead (in-process, see module docstring)")
    ap.add_argument("--fast", help="fast tier predictor spec, e.g. mlx:<dir of brain-0.8b>")
    ap.add_argument("--strong", help="strong tier predictor spec, e.g. mlx:<dir of brain-4b g18b-q8>")
    ap.add_argument("--threshold", type=float, help="routing threshold (default: the fast tier's deskmind.json, else 0.96 "
                    "as the app passes it)")
    ap.add_argument("--modes", nargs="+", default=["cold", "warm"], choices=["cold", "warm"])
    ap.add_argument("--split", choices=["dev", "holdout"], help="only this split (default: both)")
    ap.add_argument("--limit", type=int, help="first N fixtures in replay order (smoke runs)")
    ap.add_argument("--no-reference", action="store_true", help="skip the strong-tier-alone pass for head agreement")
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--out", default="runs/replay/baseline")
    args = ap.parse_args()
    if not args.manifest:
        if not args.url or not args.paths:
            ap.error("give a server URL and request files, or --manifest")
        replay_http(args)
        return

    from deskmind_brain.eval.replay_run import run
    from deskmind_brain.serve import default_threshold

    if not args.fast:
        ap.error("--manifest needs --fast (and --strong for the router)")
    threshold = args.threshold if args.threshold is not None else default_threshold(args.fast, fallback=0.96)
    summary = run(args.manifest, args.fast, args.strong, threshold, args.modes, args.out, reference=not args.no_reference,
                  split=args.split, limit=args.limit, label=args.label)
    print(json.dumps({m: s["latency"]["all"] | {"gates": {k: f"{v['n']}/{v['of']}" for k, v in s["gates"]["all"].items()}}
                      for m, s in summary["modes"].items()}, ensure_ascii=False, indent=1, default=str))
    print(f"report: {args.out}/report.md")


if __name__ == "__main__":
    main()
