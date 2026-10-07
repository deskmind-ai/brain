"""Routing between a fast and a strong decision tier: when the fast tier's answer is used, and when a step goes to the
strong tier. Shared by `deskmind-brain-serve --escalate-to` (both tiers in one process) and scripts/router_serve.py (tiers
behind HTTP).

Agent requests (they carry an `operation` question): escalate if the operation is DONE / BLOCKED, a chord outside
SAFE_KEYS, a CLICK on an undo button or on a commit control (save, submit, send, delete, pay, publish, share), or if the
weakest of p(operation) and the heads that operation needs is below the threshold (0.94 by default). Plain judgements escalate below --judge-threshold. With keep_done_over_undo, a fast DONE
is kept when the strong tier's alternative is an undo click (on the real desktop the strong tier otherwise undid
finished work and looped).
"""

from __future__ import annotations

import json
import re

RISKY_OPS = {"DONE", "BLOCKED"}
#: hands' action effects that leave it open whether anything happened (state.last_effect).
UNCONFIRMED_EFFECTS = {"unverifiable", "suspected_noop"}
TERMINAL = {"DONE", "BLOCKED", "ASK"}
# Chords that cannot lose work or act outwardly may run on the fast tier's word; every other chord (undo, parent
# folder, delete, send, paste-move ...) escalates. In the first real-desktop router run 70 of 72 escalated chords were
# cmd+s, and the 4B chose the same chord every time.
SAFE_KEYS = {"cmd+s", "cmd+f", "cmd+c", "tab", "escape"}
# Deliberately asymmetric with COMMIT_WORDS below: cmd+s stays on the fast tier (70 of 72 escalated chords were cmd+s
# and the 4B always agreed), while a click on a Save button escalates -- the 0.8B's confident consequential mistakes
# were clicks on save/submit buttons (E1), a chord has no such look-alike targets.
from deskmind_brain.heads import TEXT_OPS, heads_for  # noqa: E402,F401 -- the scorer's definition, not a copy

#: Controls whose click commits something: hands' risky verbs (deskmind_hands/runtime/risk.py RISKY_WORDS, kept in
#: step by hand) plus save and submit. E1 (10-07): all three of the 0.8B's confident consequential mistakes on the
#: fixtures were clicks on save/submit buttons at 0.960-0.962, just over the threshold -- a hands floor cannot reach
#: them, so they go to the strong tier like a DONE (deskmind#63 part 2).
COMMIT_WORDS = (
    "发送", "发 送", "删除", "彻底删除", "永久删除", "确认删除", "移到废纸篓", "移至废纸篓", "清空废纸篓", "付款", "支付",
    "立即支付", "确认支付", "购买", "立即购买", "下单", "提交订单", "确认下单", "转账", "发布", "发表", "分享", "公开",
    "保存", "提交",
    "send", "delete", "delete message", "move to trash", "empty trash", "pay", "pay now", "buy", "buy now",
    "purchase", "place order", "checkout", "check out", "transfer", "publish", "post", "share", "make public",
    "save", "submit",
)
_COMMIT = re.compile(r"^\s*(?:" + "|".join(re.escape(w) for w in sorted(COMMIT_WORDS, key=len, reverse=True)) +
                     r")(?![A-Za-z])", re.I)
#: Names that start with a commit word and are not one (hands' _NOT_VERBS, plus save/submit look-alikes).
_NOT_COMMIT = re.compile(r"^\s*(?:删除线|发布会|发布日期|发布时间|发送时间|公开课|分享会|支付方式|付款方式|购买记录|保存位置|"
                         r"提交记录|delete(?:d)? items|shared with me|share sheet|posts?\b(?= by)|payment method|"
                         r"saved\b|save as\b|submissions?\b)", re.I)
#: Roles whose name describes a state or some text, not something that acts (hands' _NOT_ACTIONS).
_NOT_ACTIONS = {"checkbox", "switch", "axcheckbox", "textfield", "textarea", "statictext", "heading", "row", "cell",
                "table", "group", "radiobutton", "slider", "image"}

#: A strong-tier answer that overrules a fast DONE needs at least this much (min of p(op) and its heads); below it the
#: fast DONE stands, as long as the fast tier was at least DONE_KEEP sure. E1 (10-07): the 0.8B's DONE is right 98% of
#: the time (AUROC 0.985, no false DONE at >=0.96), and the one closed-loop failure was the 4B overruling a correct DONE
#: at 0.82 and then deleting the wrong mail (deskmind#63 part 2).
DONE_OVERRIDE = 0.96
DONE_KEEP = 0.8


def top_prob(answer: dict) -> float:
    if "probabilities" in answer and answer["probabilities"]:
        return max(answer["probabilities"].values())
    if "noul" in answer:
        return max(answer["noul"], 1 - answer["noul"])
    return 0.0


def is_undo_click(body: dict, answers: dict) -> bool:
    if (answers.get("operation") or {}).get("choice") != "CLICK":
        return False
    target = (answers.get("click_target") or {}).get("choice")
    option = _option(body.get("questions") or {}, "click_target", target)
    text = json.dumps(option, ensure_ascii=False)
    return "撤销" in text or "undo" in text.lower()


def _option(questions: dict, head: str, key):
    crit = (questions.get(head) or {}).get("criteria") or {}
    if isinstance(crit, list):   # v1 list form: [{"key": ..., ...}]
        return next((c for c in crit if isinstance(c, dict) and str(c.get("key")) == str(key)), {})
    return crit.get(key, "")


def is_commit_click(body: dict, answers: dict) -> bool:
    """A CLICK whose target is named by a commit word, on a control that acts."""
    if (answers.get("operation") or {}).get("choice") != "CLICK":
        return False
    option = _option(body.get("questions") or {}, "click_target", (answers.get("click_target") or {}).get("choice"))
    if isinstance(option, dict):
        role = str(option.get("role") or "").lower()
        label = str(option.get("element") or option.get("description") or "")
    else:
        role, label = "", str(option)
    label = re.sub(r"^\[\w+\]\s*", "", label).strip()
    if role in _NOT_ACTIONS or not label or len(label) > 30 or _NOT_COMMIT.match(label):
        return False
    return bool(_COMMIT.match(label))


def answer_conf(answers: dict, questions: dict) -> float:
    op = (answers.get("operation") or {}).get("choice") or ""
    conf = top_prob(answers.get("operation") or {})
    for h in heads_for(op, questions):
        conf = min(conf, top_prob(answers.get(h) or {}))
    return conf


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
    if op == "DONE" and ((body.get("state") or {}).get("last_effect") in UNCONFIRMED_EFFECTS):
        # Finishing right after an action that could not be confirmed (hands' last_effect): checked by the strong
        # tier like any DONE, and said so -- a pixel check missed a song pausing, and the run said DONE over it.
        return True, "unverified_last", conf
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
        option = _option(questions, "click_target", target)
        if "撤销" in json.dumps(option, ensure_ascii=False) or "undo" in json.dumps(option).lower():
            return True, "risky_undo", conf
        if is_commit_click(body, answers):
            return True, "risky_commit", conf
    return conf < threshold, "low_conf" if conf < threshold else "fast_ok", conf


def route(body: dict, fast_answers: dict, strong, threshold: float = 0.94, judge_threshold: float = 0.8,
          keep_done_over_undo: bool = True, done_override: float = DONE_OVERRIDE,
          done_keep: float = DONE_KEEP) -> tuple[dict, dict]:
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
        if keep_done_over_undo and reason in ("risky_DONE", "unverified_last") and is_undo_click(body, answers):
            answers, reason, escalate = fast_answers, "done_kept_over_undo", False
        elif (reason == "risky_DONE" and op != "DONE"   # not unverified_last: after an unconfirmed effect the 4B decides
              and top_prob(fast_answers.get("operation") or {}) >= done_keep
              and answer_conf(answers, body.get("questions") or {}) < done_override):
            # The strong tier overrules a fairly sure fast DONE, but not surely enough: the DONE stands.
            answers, reason, escalate = fast_answers, "done_kept_low_override", False
    record = {"by": "strong" if escalate else "fast", "reason": reason, "fast_conf": round(conf, 4)}
    if confirmed:
        # both tiers chose the same DONE-class operation. The heads on this reply are the fast tier's, so a client
        # that overrides a low-confidence DONE with "best other op x its head" would act on heads the strong tier
        # never scored (real desktop: DONE@0.47 from the strong tier became an undo click). Clients should take
        # a confirmed DONE as final.
        record["confirmed"] = True
    return answers, record
