"""Accuracy of served models on labelled items (items_labeled.jsonl), per labelled head and per item kind.

Each item's request (state + questions, exactly as stored) is sent to every server, and the served answer for each
labelled head is compared with the label's argmax. An item counts as correct only when every labelled head is right.
Kinds come from meta.ce_kind, or meta.gold_op when there is none.

    python scripts/item_accuracy.py data/eval/gym_music_v1_heldout/items_labeled.jsonl \\
        g14=http://127.0.0.1:8811 g15=http://127.0.0.1:8812 [--limit 200]
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor


def ask(url: str, body: dict) -> dict:
    req = urllib.request.Request(f"{url}/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())["answers"]


OP_HEADS = {"CLICK": {"click_target"}, "OPEN": {"open_target"}, "TYPE_TEXT": {"type_text_target", "type_text_value"},
            "APPEND_TEXT": {"append_text_target", "type_text_value"}, "REPLACE_TEXT": {"replace_text_target", "replace_from", "type_text_value"},
            "SELECT": {"select_target"}, "SCROLL": {"scroll_target"}, "KEY": {"key_target"}, "FOCUS_WINDOW": {"focus_window_target"}}


def gold(item: dict) -> dict[str, set[str]]:
    """Every answer with the top probability counts as right (soft labels can accept several)."""
    out = {}
    for h, r in item["references"].items():
        top = max(r["probs"].values())
        out[h] = {k for k, v in r["probs"].items() if v >= top - 1e-9}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("servers", nargs="+", help="name=url")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    items = [json.loads(l) for l in open(args.items)][: args.limit]
    for spec in args.servers:
        name, url = spec.split("=", 1)

        def one(it):
            body = {"state": it["state"], "questions": it["questions"], "model": name}
            try:
                ans = ask(url, body)
            except Exception as e:  # noqa: BLE001
                return it, None, str(e)
            return it, ans, None

        by_kind = defaultdict(lambda: [0, 0])
        heads = defaultdict(lambda: [0, 0])
        errors = 0
        confusions = defaultdict(int)
        with ThreadPoolExecutor(args.workers) as pool:
            for it, ans, err in pool.map(one, items):
                if ans is None:
                    errors += 1
                    if errors == 1:
                        print("   first error:", err[:300])
                    continue
                g = gold(it)
                op = (ans.get("operation") or {}).get("choice")
                ok = op in g["operation"]
                heads["operation"][0] += ok
                heads["operation"][1] += 1
                if not ok:
                    confusions[f"{'/'.join(sorted(g['operation']))}->{op}"] += 1
                other_op_heads = set().union(*(OP_HEADS.get(o, set()) for o in g["operation"])) - OP_HEADS.get(op, set())
                for h, want in g.items():
                    if h == "operation" or h in other_op_heads:
                        continue  # a head that belongs to an accepted alternative the model did not choose
                    got = (ans.get(h) or {}).get("choice")
                    heads[h][0] += got in want
                    heads[h][1] += 1
                    ok &= got in want
                kind = it.get("meta", {}).get("ce_kind") or it.get("meta", {}).get("gold_op") or "?"
                by_kind[kind][0] += ok
                by_kind[kind][1] += 1
        total = [sum(v[0] for v in by_kind.values()), sum(v[1] for v in by_kind.values())]
        print(f"== {name}: {total[0]}/{total[1]} items fully right" + (f", {errors} errors" if errors else ""))
        print("   heads: " + ", ".join(f"{h} {a}/{n}" for h, (a, n) in sorted(heads.items())))
        for k, (a, n) in sorted(by_kind.items()):
            print(f"   {k:34s} {a}/{n}")
        if confusions:
            print("   op errors: " + ", ".join(f"{k} ×{v}" for k, v in sorted(confusions.items(), key=lambda x: -x[1])[:8]))


if __name__ == "__main__":
    main()
