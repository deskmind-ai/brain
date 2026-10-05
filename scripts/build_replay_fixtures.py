"""Build the replay fixture set for deskmind-ai/deskmind#16 (see deskmind_brain/eval/replay.py and docs/replay.md).

Sources are DeskMind's own labelled held-out sets (independent gold: the gym oracle, or the generator's rule), never a
model's answers:

- gym:   G19 held-out tasks in hands' synthetic gym apps: expense (the variant with the receipt text in the state),
         mail, mail+music, music, settings. Contiguous runs of steps per sampled task, so a warm replay sees
         consecutive steps of one goal. Public.
- pairs: done/not-done variants of one gym music step (the same screen with and without the finishing change; one of
         the three is DONE). Public.
- large: gym mail steps (mail, mail+music) whose gold is a click on the list, with synthetic filler rows appended
         until the list has 100-250 options (the app caps a question at 255). The gold row is untouched and the filler
         names never occur in the goal or on screen, so the label still holds. Public.
- ambiguity, text: generated from real harness request skeletons (structure only, every task string generated), so
         they stay private: written under --private-out, listed in the manifest without contents.

    uv run python scripts/build_replay_fixtures.py --openjev ~/projj/github.com/gxcsoccer/openjev \\
        --private-out ~/projj/github.com/gxcsoccer/openjev/private/replay-fixtures/v1

The output is frozen: the manifest records each file's count and sha256, and the replay refuses changed files.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path

from deskmind_brain.eval.data import EvalItem, read_jsonl
from deskmind_brain.eval.replay import categories_of, session_of, sha256_file, split_of, step_key

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_DIR = ROOT / "fixtures" / "replay" / "v1"
GYM = {  # set -> decisions to take
    "g19_gym_expense_vis_heldout": 40, "g19_gym_mail_heldout": 30, "g19_gym_mailmusic_heldout": 35,
    "g19_gym_music_heldout": 25, "g19_gym_settings_heldout": 20,
}
RUN = 8  # consecutive steps per sampled task
FILLER_NAMES = ["Quill Morrow", "Tessa Brandt", "Orin Vale", "Juno Pike", "Remy Castell", "Sable Okafor", "Wren Halvor",
                "Ines Darrow", "Pavel Lund", "Cora Whitlock", "Dario Fenn", "Mila Ostrow"]
FILLER_SUBJECTS = ["Weekly digest", "Build report", "Library hold ready", "Parking notice", "Webinar recording",
                   "Quarterly newsletter", "Survey reminder", "Room booking confirmed", "Shipping update", "Gym class schedule"]
FILLER_ROWS = (12, 24, 40)  # x5 elements per row (row + 4 cells, as the gym mail list renders them)


def items(path: Path) -> list[EvalItem]:
    return [EvalItem.from_json(d) for d in read_jsonl(path)]


def tag(it: EvalItem, set_name: str, session: str | None = None, **meta) -> EvalItem:
    it.meta = {**it.meta, **meta}
    session = session or session_of(it.id)
    it.meta["replay"] = {"set": set_name, "session": session, "split": split_of(session), "categories": categories_of(it)}
    return it


def runs(pool: list[EvalItem], n: int, rng: random.Random) -> list[EvalItem]:
    """n decisions as contiguous step runs from randomly ordered sessions."""
    by: dict[str, list[EvalItem]] = defaultdict(list)
    for it in pool:
        by[session_of(it.id)].append(it)
    sessions = sorted(by)
    rng.shuffle(sessions)
    out: list[EvalItem] = []
    for s in sessions:
        steps = sorted(by[s], key=lambda it: step_key(it.id))
        start = rng.randrange(max(1, len(steps) - RUN + 1))
        out += steps[start : start + min(RUN, n - len(out))]
        if len(out) >= n:
            break
    return out


def add_filler(it: EvalItem, rows: int, rng: random.Random) -> EvalItem:
    it = copy.deepcopy(it)
    state = it.state
    seen = json.dumps(state, ensure_ascii=False) + json.dumps(it.questions["operation"].instructions, ensure_ascii=False)
    names = [n for n in FILLER_NAMES if n not in seen and n.split()[0] not in seen]
    next_index = max(int(e["index"]) for e in state["elements"]) + 1
    added = []
    for r in range(rows):
        name, subject = rng.choice(names), rng.choice(FILLER_SUBJECTS)
        date = f"{rng.randint(1, 12)}/{rng.randint(1, 28)}"
        for role, label in (("row", f"{name} · {subject} · {date}"), ("cell", name), ("cell", subject), ("cell", "cell"),
                            ("cell", date)):
            added.append({"index": str(next_index), "id": f"filler_{r}_{role}_{next_index}", "role": role, "label": label,
                          "operations": ["CLICK", "OPEN"]})
            next_index += 1
    state["elements"] += added
    if isinstance(state.get("page"), dict):
        state["page"]["text"] = state["page"].get("text", "") + "\n" + "\n".join(e["label"] for e in added if e["role"] == "row")
    for qid, q in it.questions.items():
        if not qid.endswith("_target") or not isinstance(q.criteria, dict) or not q.criteria:
            continue
        op = qid[: -len("_target")].upper()
        shape = next(iter(q.criteria.values()))
        if not isinstance(shape, dict):
            continue
        for e in added:
            if op in e["operations"]:
                q.criteria[e["index"]] = {k: (f"[{e['index']}] {e['label']}" if k == "element" else e["role"] if k == "role"
                                              else "") for k in shape}
        if qid in it.references:
            it.references[qid].probs.update({e["index"]: 0.0 for e in added if op in e["operations"]})
    return it


def write(path: Path, rows: list[EvalItem]) -> None:
    """gzip with a fixed mtime, so the same fixtures always hash the same."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(it.to_json(), ensure_ascii=False) + "\n" for it in rows).encode()
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:
        f.write(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--openjev", type=Path, required=True, help="checkout holding data/eval and the private sources")
    ap.add_argument("--private-out", type=Path, required=True)
    ap.add_argument("--ambiguity", type=Path, help="generated ambiguity items (default: <private-out>/../ambiguity_s1016)")
    ap.add_argument("--seed", type=int, default=16)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    ev = args.openjev / "data" / "eval"

    gym = []
    for name, n in GYM.items():
        gym += [tag(it, "gym") for it in runs(items(ev / name / "items_labeled.jsonl"), n, rng)]

    pairs_pool = items(ev / "live_pairs_heldout" / "items_labeled.jsonl")
    by_step: dict[str, list[EvalItem]] = defaultdict(list)
    for it in pairs_pool:
        by_step[it.id.rsplit("/", 1)[0]].append(it)
    groups = [sorted(v, key=lambda it: it.id) for _, v in sorted(by_step.items()) if len(v) >= 2]
    pairs = [tag(it, "pairs", terminal_pair=True) for v in rng.sample(groups, 14) for it in v]

    clicks: dict[str, list[EvalItem]] = defaultdict(list)
    for name in ("g19_gym_mail_heldout", "g19_gym_mailmusic_heldout"):
        for it in items(ev / name / "items_labeled.jsonl"):
            if gold_is_row_click(it):
                clicks[session_of(it.id)].append(it)
    picked = []
    queues = [rng.sample(v, len(v)) for _, v in sorted(clicks.items())]
    while len(picked) < 30 and any(queues):  # round robin over tasks, so both splits get some
        for q in queues:
            if q and len(picked) < 30:
                picked.append(q.pop())
    large = []
    for i, it in enumerate(picked):
        big = add_filler(it, FILLER_ROWS[i % len(FILLER_ROWS)], rng)
        big.id = f"{it.id}~filler{FILLER_ROWS[i % len(FILLER_ROWS)]}"
        large.append(tag(big, "large", session=session_of(it.id), derived_from=it.id))

    amb_src = args.ambiguity or args.private_out.parent / "ambiguity_s1016" / "items_labeled.jsonl"
    amb_pool = items(amb_src)
    by_scn: dict[str, list[EvalItem]] = defaultdict(list)
    for it in amb_pool:
        by_scn[session_of(it.id)].append(it)
    kinds = {"ask_first", "ask_at_target", "ask_after_write", "done_full_row"}
    amb_scn = [s for s, v in sorted(by_scn.items()) if any(it.meta.get("ce_kind") == "ask_first" for it in v)]
    one_scn = [s for s, v in sorted(by_scn.items()) if any(it.meta.get("ce_kind") == "unambiguous_no_ask" for it in v)]
    ambiguity = [tag(it, "ambiguity", ambiguous=True) for s in rng.sample(amb_scn, 15) for it in by_scn[s]
                 if it.meta.get("ce_kind") in kinds]
    ambiguity += [tag(it, "ambiguity", ambiguous=True) for s in rng.sample(one_scn, 10) for it in by_scn[s]
                  if it.meta.get("ce_kind") == "unambiguous_no_ask"]

    text = [tag(it, "text") for it in runs(items(ev / "multirow_copy_heldout" / "items_labeled.jsonl"), 30, rng)]

    sets = []
    for name, rows, vis, source, lic, note in (
        ("gym", gym, "public", "openjev data/eval/g19_gym_*_heldout (hands gym oracle labels)", "Apache-2.0", ""),
        ("pairs", pairs, "public", "openjev data/eval/live_pairs_heldout (gym music done/not-done twins)", "Apache-2.0", ""),
        ("large", large, "public", "g19_gym_mail_heldout row clicks + synthetic filler rows (this script)", "Apache-2.0",
         "derived: labels inherited, filler rows are never the target"),
        ("ambiguity", ambiguity, "private", "openjev scripts/ambiguity_ask_data.py --seed 1016 (real skeleton, generated strings)",
         "internal", "no exact state overlap with the g14-g18 ambiguity training sets"),
        ("text", text, "private", "openjev data/eval/multirow_copy_heldout (real skeleton, generated strings)", "internal", ""),
    ):
        out = (PUBLIC_DIR if vis == "public" else args.private_out) / f"{name}.jsonl.gz"
        write(out, rows)
        sets.append({"name": name, "path": out.name, "visibility": vis, "source": source, "license": lic,
                     "count": len(rows), "sha256": sha256_file(out), "note": note})
        print(f"{name}: {len(rows)} -> {out}")
    manifest = {"name": "deskmind-brain-replay", "version": "1", "gates_version": "1", "seed": args.seed,
                "split": "by session, sha256(replay-v1:<session>) < 0.3 -> holdout",
                "do_not_train": True, "sets": sets}
    (PUBLIC_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(f"manifest -> {PUBLIC_DIR / 'manifest.json'}")


def gold_is_row_click(it: EvalItem) -> bool:
    ref = it.references.get("operation")
    if ref is None or max(ref.probs, key=ref.probs.get) != "CLICK" or "click_target" not in it.references:
        return False
    target = max(it.references["click_target"].probs, key=it.references["click_target"].probs.get)
    crit = it.questions["click_target"].criteria
    return isinstance(crit, dict) and isinstance(crit.get(target), dict) and crit[target].get("role") in ("row", "cell")


if __name__ == "__main__":
    main()
