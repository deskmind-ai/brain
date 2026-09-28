"""Contamination check: does any training item share text with a benchmark's public items?

Every public item (state + question text + option criteria) is cut into word 13-grams (CJK characters count as
words). Training items are streamed and each of their 13-grams is looked up in that set, so memory stays small.
For each public item this reports how many of its 13-grams appear anywhere in training, and the single training item
that shares the most. It also flags training questions whose option set equals a public item's option set.

    python scripts/contamination_check.py --bench path/to/jevbench/datasets/public \
        --train data/train/g14_ambiguity data/train/jevfast_v2_g10 ...
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

N = 13
TOKEN = re.compile(r"[㐀-鿿]|[a-z0-9]+(?:['’.][a-z0-9]+)*")


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def ngrams(toks: list[str]) -> set[int]:
    return {hash(" ".join(toks[i : i + N])) for i in range(len(toks) - N + 1)}


def flatten(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(flatten(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(flatten(v) for v in value)
    return "" if value is None else str(value)


def bench_items(bench: Path) -> list[dict]:
    items = []
    for f in sorted(bench.glob("*.jsonl")):
        for line in f.open(encoding="utf-8"):
            row = json.loads(line)
            q = row.get("question") or {}
            text = " ".join([flatten(row.get("state")), flatten(q.get("instructions")), flatten(q.get("criteria"))])
            items.append({"id": row["id"], "grams": ngrams(tokens(text)), "labels": frozenset(row.get("labels") or [])})
    return items


def train_rows(dirs: list[Path]):
    for d in dirs:
        f = d / "items_labeled.jsonl" if d.is_dir() else d
        for line in f.open(encoding="utf-8"):
            row = json.loads(line)
            yield d.name, row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, required=True, help="directory of public *.jsonl items")
    ap.add_argument("--train", type=Path, nargs="+", required=True, help="training dirs (items_labeled.jsonl) or files")
    ap.add_argument("--show", type=int, default=15)
    args = ap.parse_args()

    items = bench_items(args.bench)
    owner: dict[int, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        for g in it["grams"]:
            owner[g].append(i)
    label_sets = defaultdict(list)
    for i, it in enumerate(items):
        if len(it["labels"]) >= 2:
            label_sets[it["labels"]].append(i)
    # generic option sets (yes/no, the same set on many public items) say nothing about contamination
    label_sets = {k: v for k, v in label_sets.items() if len(v) <= 2 and len(k) >= 3}

    hit_grams: list[set[int]] = [set() for _ in items]
    best: list[tuple[int, str]] = [(0, "") for _ in items]
    label_hits: list[tuple[str, str]] = []
    n_train = 0
    for src, row in train_rows(args.train):
        n_train += 1
        text = " ".join([flatten(row.get("state")), flatten(row.get("questions"))])
        shared = Counter()
        for g in ngrams(tokens(text)):
            for i in owner.get(g, ()):
                hit_grams[i].add(g)
                shared[i] += 1
        for i, c in shared.items():
            if c > best[i][0]:
                best[i] = (c, f"{src}:{row.get('id')}")
        for q in (row.get("questions") or {}).values():
            crit = q.get("criteria")
            opts = frozenset(crit) if isinstance(crit, dict) else frozenset(
                o.get("value", o) if isinstance(o, dict) else o for o in (q.get("options") or []))
            for i in label_sets.get(opts, ()):
                label_hits.append((items[i]["id"], f"{src}:{row.get('id')}"))

    rows = sorted(
        ((len(hit_grams[i]) / max(1, len(it["grams"])), len(hit_grams[i]), len(it["grams"]), best[i], it["id"])
         for i, it in enumerate(items)),
        reverse=True,
    )
    flagged = [r for r in rows if r[0] >= 0.05 or r[3][0] >= 20]
    print(f"public items: {len(items)}; training items scanned: {n_train}; n-gram size {N}")
    print(f"items with any shared {N}-gram: {sum(1 for r in rows if r[1])}; flagged (>=5% or >=20 in one item): {len(flagged)}")
    print("top public items by share of their 13-grams seen in training:")
    for frac, hit, total, (c, where), iid in rows[: args.show]:
        print(f"  {iid:40s} {hit:5d}/{total:<5d} ({frac:6.1%})  max in one training item: {c:4d}  {where}")
    print(f"training questions with an option set identical to a public item's: {len(label_hits)}")
    for iid, where in label_hits[: args.show]:
        print(f"  {iid}  <-  {where}")


if __name__ == "__main__":
    main()
