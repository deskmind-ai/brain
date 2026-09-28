"""Turn hands' desktop DAgger rows into deskmind-brain training items, with name/value augmentation.

hands (the desktop harness, DeskMind Hands) runs a checkpoint as the student on real macOS train tasks and labels
every visited state from the task's effect oracle. Each row carries the state, the questions exactly as the planner
saw them, the student's own answers, and `label` (one-hot answers for the operation and the heads that operation needs).

Rows become items with only the labelled questions, hard labels smoothed to 0.95 as elsewhere. Three filters:
- `label_source` must be "oracle", "human" or "claude" (claude_pending / claude_rejected rows carry no label);
- with --oracle-version, every row must carry that `oracle_version` (an earlier oracle version had three bugs: moves to
  the wrong place, rename collisions and stray side effects had all been labelled DONE or progress);
- the harness's chrome filter is applied to older rows: window buttons and Finder column headers are no longer
  offered to the planner, so they leave `elements` and every option list; a row labelled with one is dropped;
- a row that still carries the macOS login name anywhere is dropped (newer harness versions redact it at projection time).

Undo counterexamples (--undo-counterexamples): a later harness version added a projected undo button, and in the
first batch with it a large share of the states showing it are labelled "click undo"; older rows never show it. An
earlier checkpoint learned the shortcut -- button present, click it -- and on the real desktop undid its own finished
work with high confidence, every time. So for older rows whose last effect was correct (label DONE or plain progress, never a repair rename), a copy gets the undo
button that would reverse that effect, with the original label kept: the model has to read what the button undoes.
Real "click undo" rows get no renaming variants, which lowers their weight relative to everything else.

Augmentation: nearly all typed values in browser runs come from the goal, and the desktop tasks are built
the same way, so a model can learn file names and numbers instead of the decision. Each variant renames, consistently
across goal, state and options, every file stem, integer of 3+ digits, ID like T-1234 and free-standing Chinese personal name. Labels point
at option keys, and type_text_value / replace_from options are renamed together with the state, so they stay valid.

    python scripts/import_hands_dagger.py path/to/hands/data/dagger_rows.jsonl \\
        --out data/train/desktop --variants 4 --oracle-version 2
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

CHROME = {"关闭按钮", "最小化按钮", "全屏幕按钮", "缩放按钮", "名称", "修改日期", "大小", "种类"}
FILE_RE = re.compile(r"(?<![\w.-])([A-Za-z0-9_\-一-鿿]{2,40})\.(txt|log|csv|md|json|tsv)\b")
INT_RE = re.compile(r"(?<![\d.])\d{3,9}(?![\d.])")
ID_RE = re.compile(r"\b([A-Z]{1,4})-(\d{2,6})\b")
SURNAMES = "王李张刘陈杨黄赵吴周徐孙马朱胡郭何林罗高郑梁谢宋唐许韩冯邓曹彭曾肖田董袁潘蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦方白邹孟熊秦邱江尹薛段雷侯龙史陶黎贺顾毛郝龚邵万钱严武戴莫孔汤"
# A Chinese personal name standing on its own: after an arrow, colon, list comma or line start, before a delimiter.
NAME_RE = re.compile(rf"(?:(?<=[→:：、，,\s>|])|^)([{SURNAMES}][\u4e00-\u9fff]{{1,2}})(?=[\s，,。、；;|)）]|$)", re.M)
GIVEN = "芳洋伟敏静磊婷强军杰娜涛明超秀兰霞平刚桂英华玲丽勇艳斌博宇琳浩然欣怡子轩思远雨萱晨阳"
STEMS = ["notes", "report", "ledger", "summary", "plan", "draft", "budget", "roster", "inventory", "minutes",
         "backlog", "agenda", "release", "journal", "archive", "tasks", "orders", "contacts", "metrics", "q3-review",
         "会议纪要", "周报", "预算表", "名单", "库存", "待办", "发布记录", "日志", "客户", "项目计划"]


def smooth(options: list[str], gold: str) -> dict[str, float]:
    if len(options) == 1:
        return {gold: 1.0}
    rest = 0.05 / (len(options) - 1)
    return {o: (0.95 if o == gold else rest) for o in options}


def element_label(value) -> str:
    text = value.get("element", "") if isinstance(value, dict) else str(value or "")
    return re.sub(r"^\[\d+\]\s*", "", text).strip()


def drop_chrome(row: dict) -> dict | None:
    state = row["state"]
    chrome_idx = {e["index"] for e in state.get("elements", []) if e.get("label", "").strip() in CHROME}
    state["elements"] = [e for e in state.get("elements", []) if e["index"] not in chrome_idx]
    for qid, q in row["questions"].items():
        crit = q.get("criteria")
        if not isinstance(crit, dict):
            continue
        # Only element-target questions key their options by element index; type_text_value / replace_from /
        # select options are numbered independently, so an index match there is a coincidence, not chrome.
        gone = {k for k, v in crit.items()
                if isinstance(v, dict) and "element" in v and (k in chrome_idx or element_label(v) in CHROME)}
        if not gone:
            continue
        if (row["label"].get(qid) or {}).get("choice") in gone:
            return None
        q["criteria"] = {k: v for k, v in crit.items() if k not in gone}
    return row


UNDO_RULES = [
    (re.compile(r"^renamed '(.+?)' to '(.+?)'"), lambda m, title: f"撤销上一步：把「{m.group(2)}」改回「{m.group(1)}」"),
    (re.compile(r"^created (?:folder )?'(.+?)'"), lambda m, title: f"撤销上一步：删除刚建的文件夹「{m.group(1)}」"),
    (re.compile(r"^moved '(.+?)' to '(.+?)'"), lambda m, title: f"撤销上一步：把「{m.group(1)}」移回「{title}」" if title else None),
]
TEXT_FAMILIES = {"edit_fields", "find_line", "roundtrip", "write_exact", "wrong_target"}
SAFE_LABELS = {"DONE", "OPEN", "SELECT", "TYPE_TEXT", "REPLACE_TEXT", "FOCUS_WINDOW", "APPEND_TEXT"}


def is_undo_click(row: dict) -> bool:
    ans = (row.get("label") or {}).get("click_target") or {}
    crit = (row["questions"].get("click_target") or {}).get("criteria") or {}
    return "撤销上一步" in json.dumps(crit.get(ans.get("choice"), ""), ensure_ascii=False)


REVERSE_RULES = [  # effect string hands records when the undo button reverses an effect
    (re.compile(r"^renamed '(.+?)' to '(.+?)'"), lambda m, title: f"renamed '{m.group(2)}' to '{m.group(1)}' (undo) ✓"),
    (re.compile(r"^moved '(.+?)' to '(.+?)'"), lambda m, title: f"moved '{m.group(1)}' to '{title}' (undo) ✓" if title else None),
]


def undo_counterexample(row: dict, rng: random.Random | None = None, in_loop: bool = False) -> dict | None:
    """A copy of a correct-progress row with the undo button that would reverse its last (correct) effect.

    The button goes at a random index 2-4: harness versions put it at 2 or one further down (when a synthetic view
    dropdown ranks first), and a fixed position taught an earlier checkpoint nothing that transferred. With in_loop the history
    also shows that effect undone once and redone -- most real undo failures were in such a loop -- and the
    label is still the original one: redoing is not a reason to undo again."""
    state = row["state"]
    if any(e.get("id") == "syn:undo" for e in state.get("elements", [])):
        return None
    op = ((row.get("label") or {}).get("operation") or {}).get("choice")
    effects = state.get("effects_so_far") or []
    if not effects:
        return None
    if op == "RENAME":  # progress unless it re-renames the file the last effect produced (that is a repair)
        head = next((q for q in (row.get("label") or {}) if q.startswith("rename") and q.endswith("target")), None)
        chosen = ((row["label"].get(head) or {}).get("choice")) if head else None
        target = json.dumps(((row["questions"].get(head) or {}).get("criteria") or {}).get(chosen, ""), ensure_ascii=False)
        last = re.match(r"^renamed '.+?' to '(.+?)'", effects[-1])
        if chosen is None or (last and last.group(1) in target):
            return None
    elif op not in SAFE_LABELS:
        return None
    title = ((state.get("page") or {}).get("title") or "").strip()
    label = next((f(m, title) for rx, f in UNDO_RULES if (m := rx.match(effects[-1]))), None)
    if not label:
        return None
    reverse = None
    if in_loop:
        reverse = next((f(m, title) for rx, f in REVERSE_RULES if (m := rx.match(effects[-1]))), None)
        if not reverse:
            return None
    pos = (rng or random).choice([2, 3, 4])
    pos = min(pos, len(state.get("elements", [])) + 1)
    out = shift_elements(json.loads(json.dumps(row, ensure_ascii=False)), at=pos)
    out["state"]["elements"].insert(pos - 1, {"index": str(pos), "id": "syn:undo", "role": "button", "label": label,
                                              "operations": ["CLICK"]})
    ct = out["questions"].get("click_target")
    if ct and isinstance(ct.get("criteria"), dict):
        items = list(ct["criteria"].items())
        entry = (str(pos), {"element": f"[{pos}] {label}", "role": "button", "current_value": ""})
        before = [kv for kv in items if KEY_RE.match(kv[0]) and int(KEY_RE.match(kv[0]).group(1)) < pos]
        after = [kv for kv in items if kv not in before]
        ct["criteria"] = dict(before + [entry] + after)
    if in_loop:
        st = out["state"]
        recent = st.get("recent_actions") or []
        redo = dict(recent[-1]) if recent else {"action": "(redo)", "kind": "click", "text": None, "page_changed": True}
        st["recent_actions"] = recent + [{"action": label, "kind": "click", "text": None, "page_changed": True}, redo]
        st["effects_so_far"] = effects + [reverse, effects[-1]]
    return out


KEY_RE = re.compile(r"^(\d+)(:\d+)?$")
TAG_RE = re.compile(r"^\[(\d+)(:\d+)?\]")


def shift_elements(row: dict, at: int) -> dict:
    """Renumber every element index >= `at` by +1 -- in the element list, in element-keyed options ("7", "7:2") and
    their "[7] ..." labels, and in the labels' chosen keys -- to make room for a new element at index `at`."""
    def bump(n: str) -> str:
        return str(int(n) + 1) if int(n) >= at else n

    def key(k: str) -> str:
        m = KEY_RE.match(k)
        return bump(m.group(1)) + (m.group(2) or "") if m else k

    for e in row["state"].get("elements", []):
        if str(e.get("index", "")).isdigit():
            e["index"] = bump(e["index"])
    element_qs = set()
    for qid, q in row["questions"].items():
        crit = q.get("criteria")
        if not isinstance(crit, dict) or not any(isinstance(v, dict) and "element" in v for v in crit.values()):
            continue
        element_qs.add(qid)
        new = {}
        for k, v in crit.items():
            if isinstance(v, dict) and isinstance(v.get("element"), str):
                v = {**v, "element": TAG_RE.sub(lambda m: f"[{bump(m.group(1))}{m.group(2) or ''}]", v["element"])}
            new[key(k)] = v
        q["criteria"] = new
    for qid, ans in (row.get("label") or {}).items():
        if qid in element_qs and isinstance(ans, dict) and ans.get("choice") is not None:
            ans["choice"] = key(str(ans["choice"]))
            if isinstance(ans.get("probabilities"), dict):
                ans["probabilities"] = {key(k): v for k, v in ans["probabilities"].items()}
    return row


def fix_newfolder_retype(row: dict) -> bool:
    """Relabel "type the new folder's name again" to "click 新建文件夹" when the name was just typed.

    The oracle labels TYPE_TEXT with the same name into the same box right after a successful fill (the projection
    shows the box empty -- current_value None -- after a fill, so the oracle re-reads the name as unset). Such rows
    taught an earlier 4B checkpoint to retype until the harness hid the box; on the real-desktop diagnostic suite it
    never clicked 新建文件夹."""
    recent = row["state"].get("recent_actions") or []
    label = row.get("label") or {}
    if not recent or recent[-1].get("kind") != "fill" or recent[-1].get("action") != "新建文件夹的名称":
        return False
    if (label.get("operation") or {}).get("choice") != "TYPE_TEXT":
        return False
    value_q = (row["questions"].get("type_text_value") or {}).get("criteria") or {}
    chosen = value_q.get(str((label.get("type_text_value") or {}).get("choice")), {})
    if not isinstance(chosen, dict) or chosen.get("value") != recent[-1].get("text"):
        return False
    button = next((e for e in row["state"].get("elements", []) if e.get("role") == "button" and e.get("label") == "新建文件夹"), None)
    click_q = (row["questions"].get("click_target") or {}).get("criteria") or {}
    if button is None or str(button.get("index")) not in click_q:
        return False
    row["label"] = {"operation": {"choice": "CLICK"}, "click_target": {"choice": str(button["index"])}}
    # newer harness versions show the pending name as the box's current_value; keep half the rows in that newer shape
    if int(hashlib.sha256(json.dumps(row["state"], sort_keys=True, ensure_ascii=False).encode()).hexdigest(), 16) % 2:
        for e in row["state"].get("elements", []):
            if e.get("label") == "新建文件夹的名称" and e.get("current_value") in (None, ""):
                e["current_value"] = recent[-1].get("text")
    return True


def to_item(row: dict, rid: str) -> dict | None:
    questions, references, teacher = {}, {}, {}
    for qid, ans in row["label"].items():
        q = row["questions"].get(qid)
        gold = (ans or {}).get("choice")
        if q is None or gold is None or not isinstance(q.get("criteria"), dict) or gold not in q["criteria"]:
            continue
        options = list(q["criteria"])
        questions[qid] = q
        references[qid] = {"probs": {o: float(o == gold) for o in options}, "n_sets": 1, "soft": False}
        teacher[qid] = smooth(options, gold)
    if "operation" not in questions:
        return None
    # Keep the unlabeled heads too: they are never trained on, but with prompt format 2 the goal and rules shared by
    # >= 2 questions move into the context, as they do in every real request. Keeping labeled questions only left
    # every DONE row (which labels nothing but the operation) in a prompt shape no other row and no request has --
    # a format cue the models could read DONE from (an earlier 4B checkpoint answered DONE on a state asked alone,
    # FOCUS_WINDOW on the same state asked with its heads).
    questions = {**{qid: q for qid, q in row["questions"].items() if isinstance(q, dict)}, **questions}
    return {"id": rid, "suite": "hands_dagger", "group": row.get("family", "desktop"), "state": row["state"],
            "questions": questions, "references": references,
            "meta": {"source": "hands_dagger", "task_id": row.get("task_id"), "step": row.get("step"),
                     "suite_version": row.get("suite_version"), "label_source": row.get("label_source"),
                     "gold_op": questions and row["label"]["operation"]["choice"], "teacher": "gold",
                     "teacher_probs": teacher}}


def strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, list):
        for x in obj:
            yield from strings(x)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from strings(v)


def variant(item: dict, rng: random.Random) -> dict:
    """Rename file stems, 3+ digit integers, IDs and personal names consistently across the whole item."""
    blob = "\n".join(strings({"state": item["state"], "questions": item["questions"]}))  # raw text, not JSON-escaped
    mapping: dict[str, str] = {}
    stems = {m.group(1) for m in FILE_RE.finditer(blob)}
    pool = [s for s in STEMS if s not in stems]
    rng.shuffle(pool)
    for stem, new in zip(sorted(stems, key=len, reverse=True), pool):
        mapping[stem] = new
    for m in set(INT_RE.findall(blob)):
        mapping[m] = str(rng.randint(10 ** (len(m) - 1), 10 ** len(m) - 1))
    for name in {m.group(1) for m in NAME_RE.finditer(blob)}:
        new = rng.choice(SURNAMES) + "".join(rng.choice(GIVEN) for _ in range(len(name) - 1))
        if new != name and new not in blob:
            mapping[name] = new
    for prefix, num in set(ID_RE.findall(blob)):
        mapping[f"{prefix}-{num}"] = f"{rng.choice(['T', 'INV', 'PR', 'OPS', 'CX'])}-{rng.randint(10 ** (len(num) - 1), 10 ** len(num) - 1)}"
    if not mapping:
        return None
    pattern = re.compile("|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)))

    def sub(obj):
        if isinstance(obj, str):
            return pattern.sub(lambda m: mapping[m.group(0)], obj)
        if isinstance(obj, list):
            return [sub(x) for x in obj]
        if isinstance(obj, dict):  # keys are option ids / field names: leave them
            return {k: sub(v) for k, v in obj.items()}
        return obj

    out = json.loads(json.dumps(item, ensure_ascii=False))
    out["state"], out["questions"] = sub(item["state"]), sub(item["questions"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/train/desktop"))
    ap.add_argument("--variants", type=int, default=4, help="augmented copies per real row (0 = none)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--oracle-version", help="refuse any row whose oracle_version differs")
    ap.add_argument("--text-variants", type=int, help="variants per row for TextEdit families (default: --variants); "
                    "desktop data is ~90% Finder, and the text families are where the real-desktop failures are")
    ap.add_argument("--undo-counterexamples", action="store_true",
                    help="add copies of correct-progress rows showing the undo button that would reverse them")
    ap.add_argument("--holdout-per-family", type=int, default=0,
                    help="hold out N whole tasks per family (all their states, no variants) as an offline eval suite")
    ap.add_argument("--holdout-out", type=Path, default=Path("data/eval/desktop_holdout"))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    login = getpass.getuser()
    held: set[str] = set()
    if args.holdout_per_family:
        by_family: dict[str, set[str]] = {}
        for path in args.inputs:
            for line in path.read_text().splitlines():
                if line.strip():
                    tid = json.loads(line).get("task_id") or ""
                    by_family.setdefault(tid.rsplit("-", 1)[0], set()).add(tid)
        pick = random.Random(args.seed + 1)
        for fam in sorted(by_family):
            held.update(pick.sample(sorted(by_family[fam]), min(args.holdout_per_family, len(by_family[fam]))))
    holdout: list[dict] = []
    stats, items, seen = Counter(), [], set()
    for path in args.inputs:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            stats["rows"] += 1
            if args.oracle_version and str(row.get("oracle_version")) != args.oracle_version:
                raise SystemExit(f"{path}: row with oracle_version={row.get('oracle_version')!r}, expected {args.oracle_version}")
            if row.get("label_source") not in ("oracle", "human", "claude") or not row.get("label"):
                stats["unlabelled"] += 1
                continue
            if re.search(rf"\b{re.escape(login)}\b", json.dumps(row["state"], ensure_ascii=False) + json.dumps(row["questions"], ensure_ascii=False)):
                stats["login_name"] += 1
                continue
            if fix_newfolder_retype(row):
                stats["newfolder_retype_relabelled"] += 1
            row = drop_chrome(row)
            if row is None:
                stats["chrome_label"] += 1
                continue
            key = hashlib.sha256(json.dumps([row["state"], row["label"]], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if key in seen:
                stats["duplicate"] += 1
                continue
            seen.add(key)
            item = to_item(row, f"hands/{path.stem}/{row.get('run_id')}/{row.get('step')}/{stats['rows']}")
            if item is None:
                stats["no_operation_label"] += 1
                continue
            ces = []
            if args.undo_counterexamples:
                for tag, loop in (("undo_ce", False), ("undo_ce_loop", True)):
                    ce_row = undo_counterexample(row, rng, in_loop=loop)
                    ce_item = to_item(ce_row, f"{item['id']}/{tag}") if ce_row is not None else None
                    if ce_item is not None:
                        ce_item["meta"]["undo_counterexample"] = True
                        ce_item["meta"]["undo_in_loop"] = loop
                        ces.append(ce_item)
            if item["meta"]["task_id"] in held:  # whole task out: its states and never any variant
                holdout.append(item)
                stats["holdout"] += 1
                holdout.extend(ces)  # also an offline probe for the undo shortcut
                stats["holdout_undo_ce"] += len(ces)
                continue
            items.append(item)
            stats["real"] += 1
            stats["op:" + item["meta"]["gold_op"]] += 1
            stats["src:" + row["label_source"]] += 1
            n_variants = args.variants
            family = (item["meta"].get("task_id") or "").removeprefix("T-").rsplit("-", 1)[0]
            if args.text_variants is not None and family in TEXT_FAMILIES:
                n_variants = args.text_variants
            if args.undo_counterexamples and is_undo_click(row):
                n_variants = 0  # keep the real undo clicks, but stop multiplying them
                stats["undo_click_unmultiplied"] += 1
            for ce in ces:
                items.append(ce)
                stats["undo_counterexample_loop" if ce["meta"]["undo_in_loop"] else "undo_counterexample"] += 1
                aug = variant(ce, rng)
                if aug is not None:
                    aug["id"] += "/aug0"
                    aug["meta"]["augmented"] = True
                    items.append(aug)
                    stats["undo_counterexample_aug"] += 1
            for v in range(n_variants):
                aug = variant(item, rng)
                if aug is None:
                    break
                aug["id"] += f"/aug{v}"
                aug["meta"]["augmented"] = True
                items.append(aug)
                stats["augmented"] += 1
    if holdout:
        args.holdout_out.mkdir(parents=True, exist_ok=True)
        for it in holdout:
            it["suite"] = "desktop_holdout"
        (args.holdout_out / "items.jsonl").write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in holdout) + "\n")
        stats["holdout_tasks"] = len(held)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "items_labeled.jsonl").write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n")
    print(json.dumps(dict(sorted(stats.items())), ensure_ascii=False))


if __name__ == "__main__":
    main()
