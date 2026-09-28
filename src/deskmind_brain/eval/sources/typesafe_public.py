"""TypeSafe's public workflow evals (https://evals.typesafe.ai).

Each workflow page ships `<workflow>-cases.js` = `__VIEWER_DATA__({...})` with a handful of example cases.
A case runs a policy graph; every graph node that ran is one System One request (one document as state,
several questions). For each node we get:
  - reference answers: per-question label sets from GPT-6 Astra and Claude Fable 5.1 (high thinking),
    either full probabilities or just a hard value; TypeSafe scores models against their mean.
  - published answers from Opus 5, GPT-5.6 Sol and TypeSafe's own model (Jev).

One EvalItem = one (case, node). Published answers become Prediction files so they can be scored with
the same code as our own models.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from deskmind_brain.eval.data import EvalItem, Prediction, Reference, write_jsonl
from deskmind_brain.types import Question, normalize

SUITE = "typesafe_public"
BASE_URL = "https://evals.typesafe.ai"
WORKFLOWS = ("security_incidents", "agent_trace_observability", "invoice_processing", "customer_service")

_VIEWER_RE = re.compile(r"__VIEWER_DATA__\((.*)\)\s*;?\s*$", re.S)


def fetch_raw(raw_dir: Path, refresh: bool = False) -> dict[str, str]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for wf in WORKFLOWS:
        path = raw_dir / f"{wf}-cases.js"
        if refresh or not path.exists():
            req = urllib.request.Request(f"{BASE_URL}/{wf}-cases.js", headers={"User-Agent": "deskmind-brain-eval"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                path.write_bytes(resp.read())
        out[wf] = path.read_text(encoding="utf-8")
    return out


def parse_viewer(raw: str) -> dict[str, Any]:
    m = _VIEWER_RE.search(raw)
    if not m:
        raise ValueError("no __VIEWER_DATA__(...) payload found")
    return json.loads(m.group(1))["eval"]


def _canonical_value(value: Any, qtype: str) -> str | None:
    if value is None:
        return None
    if qtype == "noul":
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return "true" if float(value) >= 0.5 else "false"
        return "true" if str(value).strip().lower() in ("true", "yes", "1") else "false"
    if qtype == "score":
        try:
            return str(round(float(value)))
        except (TypeError, ValueError):
            return None
    return str(value)


def consensus(entry: dict[str, Any], question: Question) -> Reference | None:
    """Mean of the labelers' distributions; a set with only a hard value counts as one-hot."""
    options = question.options()
    dists, soft = [], True
    for s in entry.get("sets", []):
        probs = s.get("probabilities")
        if probs:
            dists.append(normalize({str(k): v for k, v in probs.items()}, options))
            continue
        value = _canonical_value(s.get("value"), question.type)
        if value not in options:
            continue
        soft = False
        dists.append({o: float(o == value) for o in options})
    if not dists:
        return None
    mean = {o: sum(d[o] for d in dists) / len(dists) for o in options}
    return Reference(probs=mean, n_sets=len(dists), soft=soft)


def build_workflow(ev: dict[str, Any]) -> tuple[list[EvalItem], dict[str, list[Prediction]]]:
    wf = ev["id"]
    catalog = ev["questions"]
    documents = ev["documents"]
    items: list[EvalItem] = []
    published: dict[str, list[Prediction]] = defaultdict(list)

    for example in ev["examples"]:
        case_id = example["case_id"]
        case = ev["cases"][case_id]

        # node name -> (doc index, {qid: catalog index}); taken from any model that ran the node.
        node_spec: dict[str, tuple[int, dict[str, int]]] = {}
        for model_key, run in case["models"].items():
            for node in run.get("nodes", []):
                if not node.get("ran") or node.get("doc") is None:
                    continue
                spec = (int(node["doc"]), {q: int(i) for q, i in (node.get("questions") or {}).items()})
                prev = node_spec.setdefault(node["node"], spec)
                if prev[0] != spec[0]:
                    raise ValueError(f"{wf}/{case_id}/{node['node']}: models disagree on the node's document")
                prev[1].update(spec[1])

        case_items: list[EvalItem] = []
        for node_name, ref_answers in case.get("reference_answers", {}).items():
            if node_name not in node_spec:
                continue
            doc_idx, qmap = node_spec[node_name]
            questions = {qid: Question.model_validate(catalog[idx]) for qid, idx in qmap.items()}
            references = {}
            for qid, entry in ref_answers.items():
                if qid in questions and (ref := consensus(entry, questions[qid])) is not None:
                    references[qid] = ref
            if not references:
                continue
            case_items.append(
                EvalItem(
                    id=f"{wf}/{case_id}/{node_name}",
                    suite=SUITE,
                    group=wf,
                    state=documents[doc_idx],
                    questions=questions,
                    references=references,
                    meta={
                        "case_id": case_id,
                        "case_name": example.get("name"),
                        "node": node_name,
                        "policy": example.get("policy"),
                        "reference_labelers": [r.get("title") for r in case.get("references", [])],
                    },
                )
            )
        items.extend(case_items)

        # Published answers. Case-level seconds/cost are spread over nodes by answer count.
        wanted = {it.meta["node"]: it for it in case_items}
        for model_key, run in case["models"].items():
            by_node = {n["node"]: n.get("answers") or {} for n in run.get("nodes", []) if n.get("ran")}
            total = sum(len(a) for a in by_node.values()) or 1
            for node_name, item in wanted.items():
                answers = by_node.get(node_name)
                if not answers:
                    continue
                share = len(answers) / total
                published[model_key].append(
                    Prediction(
                        item_id=item.id,
                        answers={qid: a for qid, a in answers.items() if qid in item.questions},
                        latency_s=run.get("seconds", 0.0) * share if run.get("seconds") is not None else None,
                        cost_usd=(run.get("cost") or {}).get("usd", 0.0) * share if run.get("cost") else None,
                        usage=None,
                        error=None,
                    )
                )
    return items, published


def build_suite(out_dir: Path, raw_dir: Path, refresh: bool = False) -> dict[str, Any]:
    raw = fetch_raw(raw_dir, refresh=refresh)
    all_items: list[EvalItem] = []
    all_published: dict[str, list[Prediction]] = defaultdict(list)
    model_ids: dict[str, str] = {}
    for wf in WORKFLOWS:
        ev = parse_viewer(raw[wf])
        items, published = build_workflow(ev)
        all_items.extend(items)
        for key, preds in published.items():
            all_published[key].extend(preds)
        for case in ev["cases"].values():
            for key, run in case["models"].items():
                model_ids.setdefault(key, run.get("model", key))

    write_jsonl(out_dir / "items.jsonl", (it.to_json() for it in all_items))
    for key, preds in all_published.items():
        write_jsonl(out_dir / "published" / f"{key}.jsonl", (p.to_json() for p in preds))

    summary = {
        "suite": SUITE,
        "items": len(all_items),
        "reference_pairs": sum(len(it.references) for it in all_items),
        "by_workflow": {
            wf: {
                "items": sum(1 for it in all_items if it.group == wf),
                "pairs": sum(len(it.references) for it in all_items if it.group == wf),
            }
            for wf in WORKFLOWS
        },
        "published_models": model_ids,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary
