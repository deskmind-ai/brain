"""Backend-independent part of the zero-training logit baseline.

Each question becomes a chat prompt (state first, then the question and its labelled options). The model is
never sampled: the backend returns logits at the answer position, restricted to the option label tokens
(A..Z for Choice, 0..9 for Score, Yes/No for Noul), and we softmax over those.

All prompts for one state share their leading tokens (system prompt + state), so backends prefill that prefix
once and score the question suffixes as branches off it.

Choice questions with more than 26 options run a tournament: options are split into chunks of 26, the top
candidates of each chunk advance to a final round; eliminated options are priced relative to their chunk's finalists
(see merge_tournament), so no option gets probability 0.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from deskmind_brain.eval.data import EvalItem, Prediction
from deskmind_brain.types import Question, normalize, to_answer

SYSTEM_PROMPT = (
    "You are a decision function inside a software system. Read the state, then answer the question "
    "with exactly one of the allowed labels and nothing else."
)
LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
# Options per tournament round. 26 (A..Z) is the default every model was first evaluated with; a model trained
# with more labels says so in `deskmind.json` next to its weights ({"round_size": 52} uses A..Z then a..z).
ROUND_SIZE = 26
FORMAT_FILE = "deskmind.json"  # the name this repo writes
FORMAT_FILES = (FORMAT_FILE,)
# Target heads repeat every element that the state already lists. A model trained with {"compact_targets": true}
# reads those options as bare indices instead, which is ~30% fewer tokens on a browser step.
COMPACT_TARGETS = False
ELEMENT_FIELDS = {"element", "current_value", "role", "checked", "selected", "expanded", "options"}
FINALISTS_PER_CHUNK = 3
LABELS = LETTERS + [str(i) for i in range(10)] + ["Yes", "No"]


# Prompt format, recorded as {"prompt_format": N} in deskmind.json next to the weights (absent = 1).
#   1: state as indent=2 JSON; instruction fields hoisted to the shared prefix only when identical in every question.
#   2: state as compact JSON (about half the tokens on desktop states), and list-valued instruction fields (the
#      agent `rules`) hoist their common leading items too, so rules shared by the operation and every target head
#      are sent once instead of once per head (~700 tokens x ~10 heads on hands requests).
#   3: format 2 with the shared instructions (goal + agent rules) *before* the state, so consecutive steps of one task
#      share a prompt prefix the server can keep prefilled; and decorative desktop elements (scroll bars, groups,
#      images, scroll areas) are left out of the state (~260 tokens a step). Target heads list every clickable
#      element, these included, so "not an option" would keep them all; none was ever the right target in the
#      desktop data (0 of ~2k click labels), and they stay selectable as options, just undescribed.
PROMPT_FORMAT = 1
DECORATIVE_ROLES = {"scrollBar", "scrollArea", "group", "image"}
STATE_TAG = "<state>\n"


def find_format_file(model_dir: str | Path) -> Path | None:
    """The prompt-format file next to a model's weights (see FORMAT_FILES), or None."""
    return next((Path(model_dir) / n for n in FORMAT_FILES if (Path(model_dir) / n).exists()), None)


def render_state(state: Any, fmt: int = 1) -> str:
    if isinstance(state, str):
        return state
    if fmt >= 2:
        return json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(state, indent=2, ensure_ascii=False)


def _common_prefix(lists: list[list]) -> list:
    out = []
    for items in zip(*lists):
        if any(x != items[0] for x in items[1:]):
            break
        out.append(items[0])
    return out


def hoist_shared(questions: dict[str, Question], fmt: int = 1) -> tuple[dict[str, Any], dict[str, Question]]:
    """Instruction fields identical across every question (e.g. a long `rules` text) move to the shared prefix.

    Only applies when at least two questions have object-valued instructions; plain string instructions and
    single-question requests are returned unchanged, so ordinary prompts are byte-identical to before.
    Format 2 also hoists the common leading items of list-valued fields present in every question.
    """
    objs = [q.instructions for q in questions.values()]
    if len(objs) < 2 or not all(isinstance(o, dict) for o in objs):
        return {}, questions
    shared = {k: v for k, v in objs[0].items() if all(k in o and o[k] == v for o in objs[1:])}
    partial: dict[str, int] = {}  # list field -> number of leading items hoisted
    if fmt >= 2:
        # Lists shared by most questions (hands: the agent rules on the operation and every target head, while the
        # value head carries its own string rules) hoist their common leading items; questions whose field is not
        # such a list keep theirs whole.
        keys = {k for o in objs for k, v in o.items() if isinstance(v, list) and k not in shared}
        for k in sorted(keys):
            lists = [o[k] for o in objs if isinstance(o.get(k), list)]
            prefix = _common_prefix(lists) if len(lists) >= 2 else []
            if prefix:
                shared[k] = prefix
                partial[k] = len(prefix)
    if not shared:
        return {}, questions

    def rest(ins: dict) -> dict:
        out = {}
        for k, v in ins.items():
            if k in partial and isinstance(v, list):
                if v[partial[k]:]:
                    out[k] = v[partial[k]:]
            elif k in partial:
                out[k] = v
            elif k not in shared:
                out[k] = v
        return out

    stripped = {qid: q.model_copy(update={"instructions": rest(q.instructions)}) for qid, q in questions.items()}
    return shared, stripped


def option_elements(questions: dict[str, Question]) -> set[str]:
    """Element indices any question offers as an option ("7", "7:2" -> "7")."""
    out = set()
    for q in questions.values():
        if isinstance(q.criteria, dict):
            out |= {str(k).split(":")[0] for k in q.criteria}
    return out


def prune_state(state: Any, keep: set[str] = frozenset()) -> Any:
    """Drop decorative elements (DECORATIVE_ROLES) from the state, except indices in `keep`."""
    if not isinstance(state, dict) or not isinstance(state.get("elements"), list):
        return state
    elements = [e for e in state["elements"]
                if not (isinstance(e, dict) and e.get("role") in DECORATIVE_ROLES and str(e.get("index")) not in keep)]
    return {**state, "elements": elements}


def render_context(state: Any, shared: dict[str, Any] | None = None, fmt: int = 1,
                   questions: dict[str, Question] | None = None) -> str:
    """Everything before the question block: the state and the instructions shared by every question.

    Formats 1-2 put the state first; format 3 puts the shared instructions first (see PROMPT_FORMAT) and needs the
    request's questions to know which elements must stay."""
    if fmt >= 3:
        state = prune_state(state)
    state_text = f"{STATE_TAG}{render_state(state, fmt)}\n</state>\n\n"
    shared_text = ""
    if shared:
        body = json.dumps(shared, ensure_ascii=False, separators=(",", ":")) if fmt >= 2 else json.dumps(shared, indent=1, ensure_ascii=False)
        shared_text = f"<shared_instructions>\n{body}\n</shared_instructions>\n\n"
    return shared_text + state_text if fmt >= 3 else state_text + shared_text


def render_value(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


@dataclass
class Prompt:
    qid: str
    options: list[str]  # option keys scored by this prompt
    labels: list[str]  # answer token text for each option
    text: str  # question block appended after the state


def duplicates_state(question: Question, options: list[str]) -> bool:
    """True when every option is an element already described in the state (agent-step target heads)."""
    if not isinstance(question.criteria, dict):
        return False
    values = [question.criteria.get(o) for o in options]
    return bool(values) and all(
        isinstance(v, dict) and v and set(v) <= ELEMENT_FIELDS and ":" not in o for v, o in zip(values, options))


def question_block(question: Question, options: list[str], compact: bool = False) -> tuple[str, list[str]]:
    """Question text for a subset of options, and the label token for each option."""
    instructions = render_value(question.instructions)
    if question.type == "noul":
        lines = [f"Question: {instructions}"]
        if isinstance(question.criteria, dict):
            if question.criteria.get("true"):
                lines.append(f"Yes means: {render_value(question.criteria['true'])}")
            if question.criteria.get("false"):
                lines.append(f"No means: {render_value(question.criteria['false'])}")
        elif question.criteria:
            lines.append(f"Criteria: {render_value(question.criteria)}")
        lines.append("Answer Yes or No.")
        return "\n".join(lines), ["No", "Yes"]
    if question.type == "score":
        levels = "\n".join(f"{i}. {render_value(level)}" for i, level in enumerate(question.criteria))
        text = f"Question: {instructions}\nLevels:\n{levels}\nAnswer with the level number only."
        return text, [str(i) for i in range(len(question.criteria))]
    labels = LETTERS[: len(options)]
    if compact and duplicates_state(question, options):
        lines = [f"{label}. element {key}" for label, key in zip(labels, options)]
    else:
        lines = [f"{label}. {key}: {render_value(question.criteria[key])}" for label, key in zip(labels, options)]
    text = f"Question: {instructions}\nOptions:\n" + "\n".join(lines) + "\nAnswer with the option letter only."
    return text, labels


def shared_prefix_len(sequences: list[list[int]]) -> int:
    """Longest common prefix, leaving every sequence at least one token of its own."""
    limit = min(len(s) for s in sequences) - 1
    first = sequences[0]
    for s in sequences[1:]:
        i = 0
        while i < limit and s[i] == first[i]:
            i += 1
        limit = i
    return max(limit, 0)


TEXT_OPS = {"TYPE_TEXT", "REPLACE_TEXT", "APPEND_TEXT", "RENAME"}
TERMINAL_OPS = {"DONE", "BLOCKED", "ASK"}


def heads_for(op: str, questions: dict) -> list[str]:
    """The questions an agent step with operation `op` actually uses (agent-step naming, as hands sends it: `<op>_target`,
    plus the value heads of text operations)."""
    prefix = op.lower() + "_"
    heads = [q for q in questions if q.startswith(prefix)]
    if op in TEXT_OPS and "type_text_value" in questions:
        heads.append("type_text_value")
    if op == "REPLACE_TEXT" and "replace_from" in questions:
        heads.append("replace_from")
    return heads


class LogitsPredictorBase:
    """Prompting and tournament logic. Backends set `tokenizer` and implement `_score`."""

    name = "logits"
    round_size = ROUND_SIZE
    compact_targets = COMPACT_TARGETS
    prompt_format = PROMPT_FORMAT
    tokenizer: Any
    # Agent requests ask ~10 questions (the operation plus one target head per operation) but a step uses two or three.
    # With two_stage the operation is scored first and then only the heads it needs, over the same cached prefix.
    two_stage = False
    terminal_heads = True  # two-stage: score every head when the operation is DONE/BLOCKED/ASK (see _predict_two_stage)
    _prefix_hint: int | None = None
    # Where the last predict() spent its time (see new_stats); read by scripts/replay_requests.py --manifest.
    stats: dict[str, Any] | None = None

    def _init_labels(self, model_id: str, model_dir: str | None = None) -> None:
        self.round_size = ROUND_SIZE
        self.compact_targets = COMPACT_TARGETS
        fmt = find_format_file(model_dir or model_id)
        if fmt is not None:
            cfg = json.loads(fmt.read_text())
            self.round_size = int(cfg.get("round_size", ROUND_SIZE))
            self.compact_targets = bool(cfg.get("compact_targets", COMPACT_TARGETS))
            self.prompt_format = int(cfg.get("prompt_format", PROMPT_FORMAT))
        self._label_ids: dict[str, int] = {}
        for label in LABELS:
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"label {label!r} is not a single token for {model_id}")
            self._label_ids[label] = ids[0]

    @staticmethod
    def new_stats() -> dict[str, Any]:
        """Counters for one predict(). Times are seconds; a backend that does not fill one leaves it 0.

        tokenize_s: chat template + tokenizer; prefill_s: the shared prefix (state + shared instructions), including
        restoring a checkpoint; score_s: the question branches. checkpoint: the format-3 head checkpoint of the first
        prefill ("hit", "miss", or None when the prompt has no stable head)."""
        return {"tokenize_s": 0.0, "prefill_s": 0.0, "score_s": 0.0, "total_s": 0.0, "prompt_tokens": 0,
                "prefix_tokens": 0, "head_tokens": 0, "prefilled_tokens": 0, "prefills": 0, "prefix_reused": 0,
                "checkpoint": None, "branches": 0, "score_calls": 0}

    def _encode(self, text: str) -> list[int]:
        start = time.perf_counter()
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if self.stats is not None:
            self.stats["tokenize_s"] += time.perf_counter() - start
        return ids

    def _prompt_text(self, context: str, block: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context + block},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def _prompt_ids(self, context: str, block: str) -> list[int]:
        start = time.perf_counter()
        text = self._prompt_text(context, block)
        if self.stats is not None:
            self.stats["tokenize_s"] += time.perf_counter() - start
        return self._encode(text)

    def _stable_head_len(self, context: str, block: str, ids: list[int]) -> int:
        """Format 3: token length of the prompt up to the state (system prompt + shared goal and rules), which stays
        the same across the steps of one task; 0 when not applicable or the tokenizer merges across the boundary."""
        if self.prompt_format < 3:
            return 0
        text = self._prompt_text(context, block)
        cut = text.find(STATE_TAG)
        if cut <= 0:
            return 0
        head = self._encode(text[:cut])
        return len(head) if ids[: len(head)] == head else 0

    def _score(self, context: str, prompts: list[Prompt]) -> list[dict[str, float]]:
        raise NotImplementedError

    def _release_memory(self) -> None:
        pass

    def _prompts(self, qid: str, question: Question, options: list[str]) -> list[Prompt]:
        if question.type != "choice" or len(options) <= self.round_size:
            text, labels = question_block(question, options, self.compact_targets)
            return [Prompt(qid, options, labels, text)]
        prompts = []
        for start in range(0, len(options), self.round_size):
            chunk = options[start : start + self.round_size]
            text, labels = question_block(question, chunk, self.compact_targets)
            prompts.append(Prompt(qid, chunk, labels, text))
        return prompts

    def predict(self, item: EvalItem) -> Prediction:
        start = time.perf_counter()
        self.stats = self.new_stats()
        self._unscored: set[str] = set()
        try:
            if self.two_stage and "operation" in item.questions:
                dists = self._predict_two_stage(item)
            else:
                dists = self._predict_distributions(item)
        finally:
            self._release_memory()
        answers = {qid: to_answer(q, normalize(dists[qid], q.options())) for qid, q in item.questions.items()}
        for qid in self._unscored:  # protocol v1 (deskmind#36 item 2): uniform here is a placeholder, not an answer
            answers[qid]["scored"] = False
        self.stats["total_s"] = time.perf_counter() - start
        return Prediction(item_id=item.id, answers=answers, latency_s=self.stats["total_s"], cost_usd=0.0)

    def _predict_two_stage(self, item: EvalItem) -> dict[str, dict[str, float]]:
        # The context (state + instructions shared by *all* questions) is identical in both stages, and the prefix is
        # pinned to its common length, so the second stage reuses the first stage's prefill instead of re-encoding.
        shared, questions = hoist_shared(item.questions, self.prompt_format)
        context = render_context(item.state, shared, self.prompt_format, item.questions)
        seqs = [self._prompt_ids(context, p.text) for qid, q in questions.items() if len(q.options()) > 1
                for p in self._prompts(qid, q, q.options())]
        self._prefix_hint = shared_prefix_len(seqs) if len(seqs) > 1 else None
        try:
            dists = self._run_rounds(context, {"operation": questions["operation"]})
            op = max(dists["operation"], key=dists["operation"].get)
            if op in TERMINAL_OPS and not self.terminal_heads:
                # only this op's own heads (ASK's question choice); the others come from elsewhere (router: the fast
                # tier already scored them for its own DONE/BLOCKED)
                needed = {h: questions[h] for h in heads_for(op, questions)}
            elif op in TERMINAL_OPS:
                # hands may override a low-confidence DONE/BLOCKED with the best other operation, scored as
                # p(op) x its target head -- so these (rare) steps need every head, as in one-pass scoring.
                needed = {q: v for q, v in questions.items() if q != "operation"}
            else:
                needed = {h: questions[h] for h in heads_for(op, questions)}
            dists.update(self._run_rounds(context, needed))
        finally:
            self._prefix_hint = None
        for qid, q in questions.items():  # not asked this step: uniform, and marked so (predict sets scored: false)
            if qid not in dists:
                opts = q.options()
                dists[qid] = {o: 1.0 / len(opts) for o in opts}
                self._unscored.add(qid)
        return dists

    def _predict_distributions(self, item: EvalItem) -> dict[str, dict[str, float]]:
        shared, questions = hoist_shared(item.questions, self.prompt_format)
        context = render_context(item.state, shared, self.prompt_format, item.questions)
        return self._run_rounds(context, questions)

    def _run_rounds(self, context: str, questions: dict[str, Question]) -> dict[str, dict[str, float]]:
        dists: dict[str, dict[str, float]] = {}
        pending = {}
        for qid, q in questions.items():
            options = q.options()
            if len(options) == 1:  # nothing to decide
                dists[qid] = {options[0]: 1.0}
            else:
                pending[qid] = options
        rounds: dict[str, list[list[dict[str, float]]]] = {}  # per question: chunk distributions of each round
        # Each round scores every question that still has candidates; questions with <= 26 finish in one round.
        while pending:
            prompts = [p for qid, opts in pending.items() for p in self._prompts(qid, questions[qid], opts)]
            scored = self._score(context, prompts)
            by_qid: dict[str, list[tuple[Prompt, dict[str, float]]]] = {}
            for p, d in zip(prompts, scored):
                by_qid.setdefault(p.qid, []).append((p, d))
            pending = {}
            for qid, parts in by_qid.items():
                if len(parts) == 1:
                    dist = parts[0][1]
                    for chunk_dists in reversed(rounds.get(qid, [])):
                        dist = merge_tournament(chunk_dists, dist)
                    dists[qid] = dist
                    continue
                rounds.setdefault(qid, []).append([d for _, d in parts])
                pending[qid] = [o for _, d in parts for o in sorted(d, key=d.get, reverse=True)[:FINALISTS_PER_CHUNK]]
        return dists


def merge_tournament(chunk_dists: list[dict[str, float]], final: dict[str, float]) -> dict[str, float]:
    """Full distribution from chunk rounds and the final round, with no option forced to zero.

    Finalists keep their final-round probability. Every other option is priced relative to its chunk's top finalist:
    P(o) = P_final(top) * P_chunk(o) / P_chunk(top). The result is renormalized.
    """
    out: dict[str, float] = {}
    for chunk in chunk_dists:
        in_final = [o for o in chunk if o in final]
        top = max(in_final, key=lambda o: final[o])
        for o, p in chunk.items():
            out[o] = final[o] if o in final else final[top] * p / max(chunk[top], 1e-12)
    total = sum(out.values())
    return {o: p / total for o, p in out.items()}
