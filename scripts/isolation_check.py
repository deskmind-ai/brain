"""Question isolation: an answer must not depend on which other questions share the request.

Each question is meant to be an independent read of the state. A backend that batches questions over a shared prefix
can leak between them through padding, a shared recurrent state or a mis-sliced cache; Qwen3.5's
Gated DeltaNet layers make the recurrent case real. For each request this compares every original question's
distribution across three variants of the same request:

  base      the questions as given
  decoy     the same, plus one unrelated noul question
  reversed  the same questions in reverse order

and reports the largest probability change and how many argmax answers flipped. A clean backend shows changes at
numerical noise (bf16: ~1e-2) and no flips.

    uv run python scripts/isolation_check.py --suite path/to/suite --n 40 --server brain-0.8b=http://127.0.0.1:8794
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path

DECOY = {"type": "noul", "instructions": "Does the state mention a weather forecast for Tuesday?"}


def ask(base_url: str, state, questions: dict, timeout_s: float = 300) -> dict:
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/systemone",
                                 data=json.dumps({"state": state, "questions": questions, "model": "local"}).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout_s).read())["answers"]


def dist(answer: dict) -> dict[str, float]:
    if "noul" in answer and "probabilities" not in answer:
        return {"true": answer["noul"], "false": 1 - answer["noul"]}
    return {str(k): v for k, v in (answer.get("probabilities") or {}).items()}


def top(d: dict[str, float]) -> str | None:
    return max(d, key=d.get) if d else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=Path, required=True)
    ap.add_argument("--server", action="append", required=True, help="name=base_url")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = [json.loads(l) for l in (args.suite / "items.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if len(r["questions"]) >= 2]
    rows = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
    for spec in args.server:
        name, _, base = spec.partition("=")
        worst, flips, compared, worst_id = 0.0, 0, 0, None
        for row in rows:
            qs = row["questions"]
            base_ans = ask(base, row["state"], qs)
            variants = {"decoy": ask(base, row["state"], {**qs, "zz_isolation_decoy": DECOY}),
                        "reversed": ask(base, row["state"], dict(reversed(list(qs.items()))))}
            for qid in qs:
                p0 = dist(base_ans[qid])
                for answers in variants.values():
                    p1 = dist(answers[qid])
                    delta = max((abs(p0[k] - p1.get(k, 0.0)) for k in p0), default=0.0)
                    compared += 1
                    flips += top(p0) != top(p1)
                    if delta > worst:
                        worst, worst_id = delta, f"{row['id']}:{qid}"
        print(json.dumps({"server": name, "requests": len(rows), "comparisons": compared, "max_prob_delta": round(worst, 4),
                          "argmax_flips": flips, "worst": worst_id}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
