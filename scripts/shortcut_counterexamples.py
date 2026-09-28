"""Shortcut counterexamples from the desktop DAgger states, one family per failure seen on the real-desktop diagnostic suite.

- corrupt_done  (label: keep editing). A saved TextEdit state the oracle marked DONE, with its document changed so the
  goal is no longer met: a stray punctuation mark after a value (e.g. "Status: final，"), a value put back to what
  the goal changes it from, a line an append goal asked for removed, or two rows swapped under an ordering goal.
  "saved ✓" stays in effects_so_far -- that line is what the escalation tier trusted.
- sibling_done  (label: DONE). A finished state with a near-namesake document open in another window (the Q4 save met
  the goal, then the 4B switched to Q3 and overwrote it). The same sibling is also put into unfinished states with
  their original labels, so a second window does not by itself mean DONE.
- ask           (label: ASK). A copy step whose source now holds two records that fit the goal's description, the
  value not given by the goal (a model guessed one of two orders). The desktop data had no ASK label at all.

Only the operation is labelled on counterexamples (the heads of the fixing step are not determined by the edit); the
other questions stay, unlabelled, so the prompt has the shape of a real request (see import_hands_dagger.to_item).
The diagnostic tasks are probes, never sources: everything here comes from the training families.

    python scripts/shortcut_counterexamples.py data/train/desktop/items_labeled.jsonl data/train/counterexamples
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import random
import re
from pathlib import Path

ASK = ("Ask the user one question and wait. The right move when the goal is ambiguous -- two records match what it names, "
       "or the value to enter is not determined by anything visible. Guessing is worse than asking, and asking is not "
       "being blocked.")
FOCUS_WINDOW = ("Switch to another window of this application. Only the focused window's contents are observable, so "
                "reading one document and writing another means switching between them.")
TEXTEDIT_CHROME = "关闭按钮\n全屏幕按钮\n最小化按钮\n{body}\n{name}\n组\n滚动区\n0.1851851791143417\n0.615384578704834\n保存\n新建纯文本文稿"
REVERT_RE = re.compile(r"从\s*([^\s，。,]+?)\s*(?:改成|改为|调到|调整为|变成|换成)\s*([^\s，。,]+)")
FROM_TO_RE = re.compile(r"from\s+([^\s,.]+)\s+to\s+([^\s,.]+)", re.I)
APPEND_RE = re.compile(r"追加|添加|加一行|加上|append|add a line|insert", re.I)
ORDER_RE = re.compile(r"顺序|从少到多|从小到大|从多到少|从大到小|排序|升序|降序|ascending|descending|in order", re.I)
SET_RE = re.compile(r"(?:改成|改为|设为|设成|调到|→|\bto)\s*([A-Za-z0-9\u4e00-\u9fff\-]+)")
PUNCT = ["，", "。", ",", ".", ";", "；"]
WORDS = ["draft", "final", "review", "approved", "pending", "high", "medium", "low", "Alice", "Bob", "Carol", "Erin",
         "张伟", "李娜", "杜艳", "王芳", "进行中", "已完成"]


def other_value(y: str, rng: random.Random) -> str:
    if re.fullmatch(r"[\d.]+", y):
        return re.sub(r"\d", lambda m: str(rng.randint(0, 9)), y, count=max(1, len(y) // 2))
    return rng.choice([w for w in WORDS if w != y])


def one_hot(criteria: dict, gold: str) -> tuple[dict, dict]:
    keys = list(criteria)
    rest = 0.05 / max(len(keys) - 1, 1)
    teacher = {k: (0.95 if k == gold else rest) for k in keys}
    ref = {"probs": {k: float(k == gold) for k in keys}, "n_sets": 1, "soft": False}
    return teacher, ref


def op_only(row: dict, gold: str, kind: str, extra_criteria: dict | None = None) -> dict:
    new = copy.deepcopy(row)
    q = new["questions"]["operation"]
    for k, v in (extra_criteria or {}).items():
        q["criteria"].setdefault(k, v)
    teacher, ref = one_hot(q["criteria"], gold)
    new["references"] = {"operation": ref}
    new["meta"] = {**row["meta"], "gold_op": gold, "source": "shortcut_counter", "ce_kind": kind, "teacher": "rule",
                   "teacher_probs": {"operation": teacher}}
    new["id"] = f"{row['id']}/ce-{kind}"
    return new


def text_area(state: dict) -> dict | None:
    for e in state.get("elements", []):
        if e.get("role") == "textArea" and isinstance(e.get("current_value"), str) and e["current_value"].strip():
            return e
    return None


def rewrite_line(state: dict, area: dict, old_line: str, new_line: str | None) -> None:
    """Replace one line of the document everywhere it is shown (field value, label, page text, typed history)."""
    def sub(s: str) -> str:
        if not isinstance(s, str) or old_line not in s:
            return s
        if new_line is None:
            return s.replace(old_line + "\n", "", 1) if old_line + "\n" in s else s.replace(old_line, "", 1)
        return s.replace(old_line, new_line, 1)

    area["current_value"] = sub(area["current_value"])
    area["label"] = sub(area.get("label", ""))
    state["page"]["text"] = sub(state["page"].get("text", ""))
    for a in state.get("recent_actions", []):
        for k in ("text", "action"):
            a[k] = sub(a.get(k))


def swap_lines(state: dict, area: dict, a: str, b: str) -> None:
    tmp = "\x00SWAP\x00"
    rewrite_line(state, area, a, tmp)
    rewrite_line(state, area, b, a)
    rewrite_line(state, area, tmp, b)


def fix_op(criteria: dict, prefer: list[str]) -> str:
    for op in prefer:
        if op in criteria:
            return op
    return "CLICK"


def corrupt_done(row: dict, rng: random.Random) -> dict | None:
    q = row["questions"]["operation"]
    goal = q["instructions"]["goal"]
    new = copy.deepcopy(row)
    area = text_area(new["state"])
    if area is None:
        return None
    lines = [ln for ln in area["current_value"].split("\n") if ln.strip()]
    kinds = []
    reverts = [(x, y) for x, y in REVERT_RE.findall(goal) + FROM_TO_RE.findall(goal) if y in area["current_value"] and x not in area["current_value"]]
    if reverts:
        kinds += ["revert"] * 2
    sets = [y for y in SET_RE.findall(goal) if y in area["current_value"]]
    if sets:
        kinds += ["wrong"] * 2
    if APPEND_RE.search(goal) and len(lines) >= 2:
        kinds.append("drop")
    if ORDER_RE.search(goal) and len(lines) >= 3:
        kinds.append("order")
    value_lines = [ln for ln in lines if any(tok in ln for tok in re.findall(r"[A-Za-z0-9][\w\-.:]{2,}", goal))]
    if value_lines:
        kinds.append("punct")
    if not kinds:
        return None
    kind = rng.choice(kinds)
    if kind == "revert":
        x, y = rng.choice(reverts)
        line = next(ln for ln in lines if y in ln)
        rewrite_line(new["state"], area, line, line.replace(y, x, 1))
        gold = fix_op(q["criteria"], ["REPLACE_TEXT", "TYPE_TEXT"])
    elif kind == "wrong":
        y = rng.choice(sets)
        line = next(ln for ln in lines if y in ln)
        z = other_value(y, rng)
        if z == y:
            return None
        rewrite_line(new["state"], area, line, line.replace(y, z, 1))
        gold = fix_op(q["criteria"], ["REPLACE_TEXT", "TYPE_TEXT"])
    elif kind == "drop":
        rewrite_line(new["state"], area, lines[-1], None)
        gold = fix_op(q["criteria"], ["APPEND_TEXT", "TYPE_TEXT"])
    elif kind == "order":
        i = rng.randrange(1, len(lines) - 1)
        if lines[i] == lines[i + 1]:
            return None
        swap_lines(new["state"], area, lines[i], lines[i + 1])
        gold = fix_op(q["criteria"], ["TYPE_TEXT", "REPLACE_TEXT"])
    else:
        line = rng.choice(value_lines)
        rewrite_line(new["state"], area, line, line + rng.choice(PUNCT))
        gold = fix_op(q["criteria"], ["REPLACE_TEXT", "TYPE_TEXT"])
    if area["current_value"] == text_area(row["state"])["current_value"]:
        return None
    return op_only(new, gold, f"corrupt_{kind}")


def sibling_name(name: str, rng: random.Random) -> str | None:
    digits = [m.start() for m in re.finditer(r"\d", name)]
    if digits:
        i = rng.choice(digits)
        d = int(name[i])
        return name[:i] + str((d + rng.choice([1, -1, 2])) % 10) + name[i + 1:]
    stem, dot, ext = name.rpartition(".")
    if not dot:
        return None
    return f"{stem}{rng.choice(['-old', '-副本', '_v2', ' copy'])}.{ext}"


def add_sibling(row: dict, rng: random.Random) -> dict | None:
    """The same kind of document under a one-character-off name, open in another window, left as it was."""
    new = copy.deepcopy(row)
    state = new["state"]
    area = text_area(state)
    name = state.get("page", {}).get("title") or ""
    sib = sibling_name(name, rng) if area else None
    if not sib or sib == name:
        return None
    body = area["current_value"].rstrip("\n")
    for x, y in [("final", "draft"), ("approved", "pending"), ("done", "open"), ("high", "low"), ("完成", "进行中")]:
        if y in body or x in body:
            body = body.replace(x, y) if x in body else body.replace(y, x)
            break
    else:
        body = re.sub(r"\d+", lambda m: str(int(m.group()) + rng.randint(1, 900)), body, count=1)
    others = list(state.get("read_in_other_windows") or [])
    others.insert(rng.randint(0, len(others)), {"window": sib, "text": TEXTEDIT_CHROME.format(body=body, name=sib)})
    state["read_in_other_windows"] = others
    q = new["questions"]["operation"]
    q["criteria"].setdefault("FOCUS_WINDOW", FOCUS_WINDOW)
    if rng.random() < 0.7:
        # a diagnostic task named the other file in the goal ("Q3 那份一个字都不要动") and the 4B went to look at it
        clause = rng.choice([f"{sib} 那份一个字都不要动。", f"注意别动 {sib}。", f"两份文件都开着，{sib} 保持原样。",
                             f"Leave {sib} exactly as it is.", f"Do not touch {sib}; it is open too."])
        goal = q["instructions"]["goal"].rstrip()
        q["instructions"]["goal"] = goal + ("" if goal.endswith(("。", ".", "\n")) else "。") + clause
        for other in new["questions"].values():
            if other is not q and isinstance(other.get("instructions"), dict) and "goal" in other["instructions"]:
                other["instructions"]["goal"] = q["instructions"]["goal"]
    new["id"] = f"{row['id']}/ce-sibling"
    tp = new["meta"].get("teacher_probs", {}).get("operation")
    if tp is not None and "FOCUS_WINDOW" not in tp:
        tp["FOCUS_WINDOW"] = min(tp.values())
        s = sum(tp.values())
        new["meta"]["teacher_probs"]["operation"] = {k: v / s for k, v in tp.items()}
    ref = new["references"].get("operation")
    if ref and "FOCUS_WINDOW" not in ref["probs"]:
        ref["probs"]["FOCUS_WINDOW"] = 0.0
    new["meta"] = {**new["meta"], "ce_kind": "sibling_" + ("done" if row["meta"]["gold_op"] == "DONE" else "keep")}
    return new


def make_ask(row: dict, rng: random.Random) -> dict | None:
    q = row["questions"]
    ref = (row.get("references") or {}).get("type_text_value")
    if "type_text_value" not in q or not ref:
        return None
    gold_idx = max(ref["probs"], key=ref["probs"].get)
    value = q["type_text_value"]["criteria"].get(gold_idx, {}).get("value")
    goal = q["operation"]["instructions"]["goal"]
    if not value or len(value) < 3 or value in goal:
        return None
    new = copy.deepcopy(row)
    for w in new["state"].get("read_in_other_windows") or []:
        lines = w["text"].split("\n")
        hit = [i for i, ln in enumerate(lines) if value in ln]
        if len(hit) != 1:
            continue
        i = hit[0]
        twin_value = re.sub(r"\d", lambda m: str(rng.randint(0, 9)), value)
        if twin_value == value:
            continue
        lines.insert(i + rng.choice([0, 1]), lines[i].replace(value, twin_value))
        w["text"] = "\n".join(lines)
        return op_only(new, "ASK", "ask", {"ASK": ASK})
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--corrupt", type=int, default=400)
    ap.add_argument("--sibling", type=int, default=250, help="per side: finished (DONE) and unfinished (label kept)")
    ap.add_argument("--ask", type=int, default=200)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = [json.loads(ln) for ln in open(args.src)]
    rng.shuffle(rows)
    textedit = [r for r in rows if text_area(r["state"])]
    done = [r for r in textedit if r["meta"].get("gold_op") == "DONE"]
    todo = [r for r in textedit if r["meta"].get("gold_op") not in ("DONE", "FOCUS_WINDOW")]
    copy_steps = [r for r in rows if r["meta"].get("gold_op") in ("TYPE_TEXT", "APPEND_TEXT") and r["state"].get("read_in_other_windows")]

    def take(pool, fn, n):
        out = []
        for r in pool:
            if len(out) >= n:
                break
            x = fn(r, rng)
            if x:
                out.append(x)
        return out

    out = (take(done, corrupt_done, args.corrupt)
           + take([r for r in done if "FOCUS_WINDOW" not in r["questions"]["operation"]["criteria"]], add_sibling, args.sibling)
           + take(todo, add_sibling, args.sibling)
           + take(copy_steps, make_ask, args.ask))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "items_labeled.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"pools: done {len(done)}, unfinished {len(todo)}, copy steps {len(copy_steps)}")
    print(collections.Counter((r["meta"]["ce_kind"], r["meta"]["gold_op"]) for r in out).most_common())


if __name__ == "__main__":
    main()
