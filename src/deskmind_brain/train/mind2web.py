"""Mind2Web (osunlp/Mind2Web train split) -> element-target training items in a compact page format.

Each action step becomes one Choice question: the page is rendered compactly (`[ref] role
"name"` per actionable element, visible text in between), the goal is the task plus the most recent actions, and
the options are the actionable elements. Real pages hold hundreds of candidates, so each item keeps a contiguous
window of 10-90 elements around the target (roughly one screen, matching the eval's p50 30 / max 98 options) at a
random offset, so the target's position carries no signal.

Public data only; run where the dataset was downloaded (e.g. a CUDA machine).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from deskmind_brain.eval.data import EvalItem, Reference, write_jsonl
from deskmind_brain.types import Question

SUITE = "mind2web"
MAX_NAME = 60
MAX_TEXT = 80
MAX_PAGE_CHARS = 4500  # ~1.2k tokens; text lines are dropped first when a page runs over
GENERIC_ROLES = {"container", "listitem", "label"}
INPUT_ROLES = {"checkbox": "checkbox", "radio": "radio", "submit": "button", "button": "button", "reset": "button",
               "search": "searchbox", "image": "button"}


def _clip(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def role_of(el) -> str:
    tag = el.tag.lower() if isinstance(el.tag, str) else ""
    explicit = (el.get("role") or "").strip().lower()
    if explicit:
        return explicit
    if tag == "a":
        return "anchor"
    if tag == "input":
        return INPUT_ROLES.get((el.get("type") or "text").lower(), "textbox")
    return {"button": "button", "select": "combobox", "textarea": "textbox", "img": "image", "li": "listitem",
            "label": "label", "option": "option", "svg": "image"}.get(tag, "container")


def name_of(el) -> str:
    for attr in ("aria_label", "aria-label", "placeholder", "alt", "title", "value", "name"):
        if el.get(attr):
            return _clip(el.get(attr), MAX_NAME)
    return _clip(" ".join(t for t in el.itertext()), MAX_NAME)


def element_kind(el) -> str:
    """How a browser agent acts on it: fill text inputs, select from dropdowns, click everything else."""
    tag = el.tag.lower() if isinstance(el.tag, str) else ""
    if tag == "select":
        return "select"
    if tag == "textarea" or tag == "input" and (el.get("type") or "text").lower() in ("text", "search", "email", "tel", "url", "number", "password"):
        return "fill"
    return "click"


def page_elements(html: str, candidate_ids: set[str], detail: dict | None = None):
    """Document-order stream of ("el", node_id, role, name) and ("text", text) entries.

    With `detail`, also records per kept node: kind (click/fill/select), current value and dropdown options.
    """
    from lxml import html as lxml_html

    root = lxml_html.fromstring(html)
    out = []
    for event, el in _walk(root):
        if event == "start":
            node_id = el.get("backend_node_id")
            if node_id in candidate_ids:
                role, name = role_of(el), name_of(el)
                # generic wrappers around other candidates duplicate their children's text; keep the inner ones
                wrapper = role in GENERIC_ROLES and any(
                    d.get("backend_node_id") in candidate_ids for d in el.iterdescendants() if isinstance(d.tag, str)
                )
                last = next((e for e in reversed(out) if e[0] == "el"), None)
                duplicate = last is not None and (last[2], last[3]) == (role, name)
                if (name or role not in GENERIC_ROLES) and not wrapper and not duplicate:
                    out.append(("el", node_id, role, name))
                    if detail is not None:
                        kind = element_kind(el)
                        detail[node_id] = {
                            "kind": kind,
                            "value": el.get("value", "") if kind != "click" else "",
                            "options": [_clip(" ".join(o.itertext()), MAX_NAME) for o in el.iter("option")] if kind == "select" else [],
                        }
            if el.tag == "text" and el.text and el.text.strip():
                out.append(("text", _clip(el.text, MAX_TEXT)))
    return out


def _walk(el):
    yield "start", el
    for child in el:
        if isinstance(child.tag, str):
            yield from _walk(child)


def render_window(stream, target_id: str, rng: random.Random) -> tuple[str, dict[str, str], str] | None:
    els = [i for i, e in enumerate(stream) if e[0] == "el"]
    pos = [k for k, i in enumerate(els) if stream[i][1] == target_id]
    if not pos:
        return None
    n = min(len(els), rng.randint(10, 90))
    start = rng.randint(max(0, pos[0] - n + 1), min(pos[0], len(els) - n))
    window = els[start : start + n]
    lo, hi = window[0], window[-1]
    lines, options, target_ref, seen_text = [], {}, None, set()
    ref = 0
    keep = set(window)
    for i in range(lo, hi + 1):
        e = stream[i]
        if e[0] == "el" and i in keep:
            ref += 1
            _, node_id, role, name = e
            label = f'{role} "{name}"' if name else role
            lines.append(("el", f"[{ref}] {label}"))
            options[str(ref)] = label
            if node_id == target_id:
                target_ref = str(ref)
            seen_text.add(name)
        elif e[0] == "text" and e[1] not in seen_text:
            lines.append(("text", e[1]))
            seen_text.add(e[1])
    text_budget = MAX_PAGE_CHARS - sum(len(t) + 1 for kind, t in lines if kind == "el")
    page = []
    for kind, t in lines:
        if kind == "text":
            if text_budget < len(t):
                continue
            text_budget -= len(t) + 1
        page.append(t)
    return "\n".join(page), options, target_ref


def step_records(files: list[Path], out: Path, seed: int = 0, max_elements: int = 40) -> dict:
    """Lightweight per-step records (goal, history, elements, gold action) for building agent-step requests elsewhere."""
    rng = random.Random(seed)
    n = 0
    with out.open("w") as f:
        for path in files:
            for task in json.loads(path.read_text()):
                for step, action in enumerate(task["actions"]):
                    if not action["pos_candidates"]:
                        continue
                    target_id = action["pos_candidates"][0]["backend_node_id"]
                    ids = {c["backend_node_id"] for c in action["pos_candidates"] + action["neg_candidates"]}
                    detail: dict = {}
                    stream = page_elements(action["cleaned_html"], ids, detail)
                    els = [i for i, e in enumerate(stream) if e[0] == "el"]
                    pos = [k for k, i in enumerate(els) if stream[i][1] == target_id]
                    if not pos or not stream[els[pos[0]]][3]:
                        continue
                    k = min(len(els), rng.randint(8, max_elements))
                    start = rng.randint(max(0, pos[0] - k + 1), min(pos[0], len(els) - k))
                    lo, hi = els[start], els[start + k - 1]
                    elements, text, seen = [], [], set()
                    for i in range(lo, hi + 1):
                        e = stream[i]
                        if e[0] == "el" and e[1] in detail:
                            elements.append({"node": e[1], "role": e[2], "label": e[3], **detail[e[1]]})
                            seen.add(e[3])
                        elif e[0] == "text" and e[1] not in seen:
                            text.append(e[1])
                            seen.add(e[1])
                    op = action["operation"]
                    f.write(json.dumps({
                        "id": f"{task['annotation_id']}/{step}", "website": task["website"], "goal": task["confirmed_task"],
                        "history": task["action_reprs"][:step], "elements": elements, "text": "\n".join(text)[:MAX_PAGE_CHARS],
                        "target": target_id, "op": op["op"], "value": op.get("value", ""),
                    }, ensure_ascii=False) + "\n")
                    n += 1
            print(f"{path.name}: {n} step records", flush=True)
    return {"records": n, "out": str(out)}


def convert(files: list[Path], out: Path, seed: int = 0, history: int = 5) -> dict:
    rng = random.Random(seed)
    items, skipped = [], {"no_target": 0, "unnamed_target": 0, "few_options": 0}
    for path in files:
        for task in json.loads(path.read_text()):
            for step, action in enumerate(task["actions"]):
                if not action["pos_candidates"]:
                    skipped["no_target"] += 1
                    continue
                target_id = action["pos_candidates"][0]["backend_node_id"]
                ids = {c["backend_node_id"] for c in action["pos_candidates"] + action["neg_candidates"]}
                stream = page_elements(action["cleaned_html"], ids)
                target = next((e for e in stream if e[0] == "el" and e[1] == target_id), None)
                if target is None:
                    skipped["no_target"] += 1
                    continue
                if not target[3]:
                    skipped["unnamed_target"] += 1
                    continue
                rendered = render_window(stream, target_id, rng)
                if not rendered or len(rendered[1]) < 5 or rendered[2] is None:
                    skipped["few_options"] += 1
                    continue
                page, options, target_ref = rendered
                goal = task["confirmed_task"]
                done = task["action_reprs"][:step][-history:]
                if done:
                    goal += " Steps already taken: " + "; ".join(_clip(a, 80) for a in done) + "."
                question = Question(
                    type="choice",
                    instructions=f"The user's goal: {goal}\nWhich element should the agent interact with next?",
                    criteria=options,
                )
                items.append(EvalItem(
                    id=f"{SUITE}/{task['annotation_id']}/{step}",
                    suite=SUITE,
                    group="element_target",
                    state={"url": task["website"], "title": "", "page": page, "goal": goal},
                    questions={"target": question},
                    references={"target": Reference(probs={o: float(o == target_ref) for o in options}, soft=False)},
                    meta={"source": "mind2web", "website": task["website"], "domain": task["domain"],
                          "op": action["operation"]["op"], "k": len(options)},
                ))
        print(f"{path.name}: {len(items)} items so far", flush=True)
    write_jsonl(out, (it.to_json() for it in items))
    websites = {it.meta["website"] for it in items}
    return {"items": len(items), "websites": len(websites), "skipped": skipped, "out": str(out)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/mind2web/data/train"))
    p.add_argument("--out", type=Path, default=Path("data/train/mind2web/items.jsonl"))
    p.add_argument("--limit-files", type=int)
    p.add_argument("--records", action="store_true", help="write per-step records for agent-step requests")
    args = p.parse_args()
    files = sorted(args.data.glob("train_*.json"), key=lambda f: int(f.stem.split("_")[1]))[: args.limit_files]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(step_records(files, args.out) if args.records else convert(files, args.out), indent=2))
