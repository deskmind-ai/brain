"""Option-order sensitivity: does a choice answer change when the same options are listed in another order?

For each choice question with >= 3 options, the request is sent three times -- options as given, reversed, and
shuffled -- and the argmax (by option key, so it is order-independent) is compared. A well-calibrated decision model
should show no flips.
Our letters follow option order, so a model that prefers early letters shows up here as flips.

    python scripts/order_sensitivity.py --suite path/to/suite --n 50 --server brain-4b=http://127.0.0.1:8793
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request


def ask(url: str, state, questions: dict) -> dict:
    req = urllib.request.Request(url.rstrip("/") + "/v1/systemone",
                                 data=json.dumps({"state": state, "questions": questions, "model": "local"}).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())["answers"]


def reorder(q: dict, order: list[str]) -> dict:
    return {**q, "criteria": {k: q["criteria"][k] for k in order}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--server", action="append", required=True, help="name=base_url")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in open(f"{args.suite}/items.jsonl") if l.strip()]
    rows = [r for r in rows if any(q["type"] == "choice" and isinstance(q.get("criteria"), dict) and len(q["criteria"]) >= 3
                                   for q in r["questions"].values())]
    rows = rng.sample(rows, min(args.n, len(rows)))
    for spec in args.server:
        name, _, url = spec.partition("=")
        compared = flips = 0
        first_pos_hits = [0, 0]  # how often the chosen option sits in the first slot: original, reversed
        for r in rows:
            qs = {k: q for k, q in r["questions"].items() if q["type"] == "choice" and isinstance(q.get("criteria"), dict)
                  and len(q["criteria"]) >= 3}
            keys = {k: list(q["criteria"]) for k, q in qs.items()}
            base = ask(url, r["state"], qs)
            rev = ask(url, r["state"], {k: reorder(q, keys[k][::-1]) for k, q in qs.items()})
            shuf_orders = {k: rng.sample(keys[k], len(keys[k])) for k in qs}
            shuf = ask(url, r["state"], {k: reorder(q, shuf_orders[k]) for k, q in qs.items()})
            for k in qs:
                a, b, c = base[k]["choice"], rev[k]["choice"], shuf[k]["choice"]
                compared += 2
                flips += (a != b) + (a != c)
                first_pos_hits[0] += a == keys[k][0]
                first_pos_hits[1] += b == keys[k][::-1][0]
        n_q = compared // 2
        print(json.dumps({"server": name, "suite": args.suite.rsplit("/", 1)[-1], "questions": n_q, "comparisons": compared,
                          "flips": flips, "flip_rate": round(flips / max(1, compared), 3),
                          "picked_first_slot": {"original": round(first_pos_hits[0] / max(1, n_q), 3),
                                                "reversed": round(first_pos_hits[1] / max(1, n_q), 3)}}))


if __name__ == "__main__":
    main()
