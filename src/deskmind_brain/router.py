"""Routing between a fast and a strong decision tier: when the fast tier's answer is used, and when a step goes to the
strong tier. Shared by `deskmind-brain-serve --escalate-to` (both tiers in one process) and scripts/router_serve.py (tiers
behind HTTP).

Agent requests (they carry an `operation` question): escalate if the operation is DONE / BLOCKED, a chord outside
SAFE_KEYS, or a CLICK on an undo button, or if the weakest of p(operation) and the heads that operation needs is below
the threshold (0.94 by default). Plain judgements escalate below --judge-threshold. With keep_done_over_undo, a fast DONE
is kept when the strong tier's alternative is an undo click (on the real desktop the strong tier otherwise undid
finished work and looped).
"""

from __future__ import annotations

import json

RISKY_OPS = {"DONE", "BLOCKED"}
TERMINAL = {"DONE", "BLOCKED", "ASK"}
# Chords that cannot lose work or act outwardly may run on the fast tier's word; every other chord (undo, parent
# folder, delete, send, paste-move ...) escalates. In the first real-desktop router run 70 of 72 escalated chords were
# cmd+s, and the 4B chose the same chord every time.
SAFE_KEYS = {"cmd+s", "cmd+f", "cmd+c", "tab", "escape"}
TEXT_OPS = {"TYPE_TEXT", "REPLACE_TEXT", "APPEND_TEXT", "RENAME"}


def top_prob(answer: dict) -> float:
    if "probabilities" in answer and answer["probabilities"]:
        return max(answer["probabilities"].values())
    if "noul" in answer:
        return max(answer["noul"], 1 - answer["noul"])
    return 0.0


def heads_for(op: str, questions: dict) -> list[str]:
    prefix = op.lower() + "_"
    heads = [q for q in questions if q.startswith(prefix)]
    if op in TEXT_OPS and "type_text_value" in questions:
        heads.append("type_text_value")
    if op == "REPLACE_TEXT" and "replace_from" in questions:
        heads.append("replace_from")
    return heads


def is_undo_click(body: dict, answers: dict) -> bool:
    if (answers.get("operation") or {}).get("choice") != "CLICK":
        return False
    target = (answers.get("click_target") or {}).get("choice")
    option = (((body.get("questions") or {}).get("click_target") or {}).get("criteria") or {}).get(target, "")
    text = json.dumps(option, ensure_ascii=False)
    return "撤销" in text or "undo" in text.lower()


def decide(body: dict, answers: dict, threshold: float, judge_threshold: float) -> tuple[bool, str, float]:
    """(escalate?, reason, confidence) for the fast tier's answers."""
    questions = body.get("questions") or {}
    if "operation" not in answers:
        conf = min((top_prob(a) for a in answers.values()), default=0.0)
        return conf < judge_threshold, "judge_low_conf" if conf < judge_threshold else "judge_ok", conf
    op = answers["operation"].get("choice")
    conf = top_prob(answers["operation"])
    for h in heads_for(op or "", questions):
        conf = min(conf, top_prob(answers.get(h) or {}))
    if op in RISKY_OPS:
        return True, f"risky_{op}", conf
    if op == "KEY":
        # hands answers a one-option question itself and leaves it out of the request, so a missing key_target means
        # exactly one chord was on offer and the router cannot see which. In practice that chord was cmd+s; fall through to the confidence rule rather than escalate every such step (ask hands to send it).
        if "key_target" in questions:
            chord = (answers.get("key_target") or {}).get("choice") or ""
            if chord.lower() not in SAFE_KEYS:
                return True, "risky_KEY", conf
    if op == "CLICK":
        target = (answers.get("click_target") or {}).get("choice")
        option = ((questions.get("click_target") or {}).get("criteria") or {}).get(target, "")
        if "撤销" in json.dumps(option, ensure_ascii=False) or "undo" in json.dumps(option).lower():
            return True, "risky_undo", conf
    return conf < threshold, "low_conf" if conf < threshold else "fast_ok", conf


def route(body: dict, fast_answers: dict, strong, threshold: float = 0.94, judge_threshold: float = 0.8,
          keep_done_over_undo: bool = True) -> tuple[dict, dict]:
    """Final answers and a routing record. `strong(body) -> answers` is called only when the step escalates."""
    escalate, reason, conf = decide(body, fast_answers, threshold, judge_threshold)
    answers = fast_answers
    confirmed = False
    if escalate:
        answers = strong(body)
        op = (answers.get("operation") or {}).get("choice")
        if op in TERMINAL and (fast_answers.get("operation") or {}).get("choice") in TERMINAL:
            # the strong tier scored a DONE-class operation plus only that op's own heads (serve sets terminal_heads
            # off on it); every other head comes from the fast tier, which scored them all for its own DONE-class
            # answer, so hands' low-confidence override still has real heads
            own = {"operation", *heads_for(op, body.get("questions") or {})}
            answers = {**fast_answers, **{k: v for k, v in answers.items() if k in own}}
            confirmed = op == (fast_answers.get("operation") or {}).get("choice")
        if keep_done_over_undo and reason == "risky_DONE" and is_undo_click(body, answers):
            answers, reason, escalate = fast_answers, "done_kept_over_undo", False
    record = {"by": "strong" if escalate else "fast", "reason": reason, "fast_conf": round(conf, 4)}
    if confirmed:
        # both tiers chose the same DONE-class operation. The heads on this reply are the fast tier's, so a client
        # that overrides a low-confidence DONE with "best other op x its head" would act on heads the strong tier
        # never scored (real desktop: DONE@0.47 from the strong tier became an undo click). Clients should take
        # a confirmed DONE as final.
        record["confirmed"] = True
    return answers, record
