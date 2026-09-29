"""LoRA distillation of a teacher's answer distributions into a small model.

The training objective mirrors inference exactly: one forward pass per (state, question) prompt, logits read at
the answer position and restricted to that question's label tokens. The loss mixes the teacher's distribution
(KL) with the dataset's own label (cross-entropy):

    L = alpha * KL(teacher || student) + (1 - alpha) * CE(gold)

Choice questions with more than 26 options are trained on one chunk of <= 26 options that contains the gold
label, which is exactly what the tournament scores at inference time.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

from deskmind_brain.eval.data import EvalItem, read_jsonl
from deskmind_brain.eval.logits_base import FORMAT_FILE, LABELS, ROUND_SIZE, hoist_shared, question_block, render_context
from deskmind_brain.train.data import OUT_DIR


@dataclass
class Example:
    input_ids: list[int]
    label_ids: list[int]  # candidate answer tokens, in option order
    teacher: list[float]  # teacher distribution over those options
    gold: int  # index of the dataset label


def build_examples(items: list[EvalItem], tokenizer, label_ids: dict[str, int], rng: random.Random,
                   round_size: int = ROUND_SIZE, compact_targets: bool = False, prompt_format: int = 1,
                   shuffle_options: bool = False) -> list[Example]:
    from deskmind_brain.eval.logits_base import SYSTEM_PROMPT

    def prompt_ids(context: str, block: str) -> list[int]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context + block},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return tokenizer.encode(text, add_special_tokens=False)

    examples = []
    for item in items:
        shared, questions = hoist_shared(item.questions, prompt_format)
        context = render_context(item.state, shared, prompt_format, item.questions)
        teacher_all = item.meta.get("teacher_probs", {})
        for qid, question in questions.items():
            if qid not in item.references or qid not in teacher_all:
                continue
            options = question.options()
            if len(options) < 2:
                continue
            ref = item.references[qid].probs
            top = max(ref.values())
            # several right answers (soft labels, e.g. the gym's "CLICK the play button or OPEN the row"): the CE
            # term takes one of them at random per example, so it does not always push toward the first
            gold = rng.choice([o for o, p in ref.items() if p >= top - 1e-9])
            teacher = teacher_all[qid]
            if question.type == "choice" and len(options) > round_size:
                others = [o for o in options if o != gold]
                options = sorted(rng.sample(others, round_size - 1) + [gold], key=options.index)
            if shuffle_options and question.type == "choice":
                # authored requests list the right answer first far more often than chance (measured on the eval
                # suites, not the training data); a random order per example keeps letter position from carrying it
                options = rng.sample(options, len(options))
            block, labels = question_block(question, options, compact_targets)
            weights = [max(teacher.get(o, 0.0), 1e-6) for o in options]
            total = sum(weights)
            examples.append(
                Example(
                    input_ids=prompt_ids(context, block),
                    label_ids=[label_ids[l] for l in labels],
                    teacher=[w / total for w in weights],
                    gold=options.index(gold),
                )
            )
    rng.shuffle(examples)
    return examples


def collate(batch: list[Example], pad_id: int, device, torch):
    width = max(len(e.input_ids) for e in batch)
    ids = torch.tensor([e.input_ids + [pad_id] * (width - len(e.input_ids)) for e in batch], device=device)
    mask = torch.tensor([[1] * len(e.input_ids) + [0] * (width - len(e.input_ids)) for e in batch], device=device)
    last = torch.tensor([len(e.input_ids) - 1 for e in batch], device=device)
    k = max(len(e.label_ids) for e in batch)
    # pad the candidate sets with the first label and mask those slots out of the softmax
    cand = torch.tensor([e.label_ids + [e.label_ids[0]] * (k - len(e.label_ids)) for e in batch], device=device)
    valid = torch.tensor([[1.0] * len(e.label_ids) + [0.0] * (k - len(e.label_ids)) for e in batch], device=device)
    teacher = torch.tensor([e.teacher + [0.0] * (k - len(e.teacher)) for e in batch], device=device)
    gold = torch.tensor([e.gold for e in batch], device=device)
    return ids, mask, last, cand, valid, teacher, gold


def token_batches(examples: list[Example], max_batch: int, token_budget: int) -> list[list[Example]]:
    """Greedy packing in the given order: a batch grows while (longest sequence x size) stays within the budget."""
    out, cur, width = [], [], 0
    for e in examples:
        w = max(width, len(e.input_ids))
        if cur and (len(cur) >= max_batch or w * (len(cur) + 1) > token_budget):
            out.append(cur)
            cur, w = [], len(e.input_ids)
        cur.append(e)
        width = w
    return out + ([cur] if cur else [])


def train(
    model_id: str = "Qwen/Qwen3.5-2B",
    data_dirs: tuple[Path, ...] = (OUT_DIR,),
    out_dir: Path = Path("runs/base"),
    alpha: float = 0.7,
    lr: float = 1e-4,
    epochs: int = 1,
    batch_size: int = 2,
    lora_rank: int = 16,
    max_tokens: int = 2560,
    token_budget: int = 3072,
    seed: int = 0,
    limit: int | None = None,
    round_size: int = ROUND_SIZE,
    compact_targets: bool = False,
    prompt_format: int = 1,
    shuffle_options: bool = False,
    save_every: int = 0,
) -> dict:
    import torch
    from modelscope import snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = snapshot_download(model_id)
    tokenizer = AutoTokenizer.from_pretrained(path)
    ids = {label: tokenizer.encode(label, add_special_tokens=False)[0] for label in LABELS}
    rng = random.Random(seed)
    items = []
    for data_dir in data_dirs:
        # "dir:N" samples N items from dir (repeating the whole set when N exceeds it)
        data_dir, _, n = str(data_dir).partition(":")
        labeled = Path(data_dir) / "items_labeled.jsonl"
        if not labeled.exists():
            raise SystemExit(f"{labeled} not found — run deskmind_brain.train.label_teacher first")
        rows = [EvalItem.from_json(row) for row in read_jsonl(labeled)]
        if n:
            n = int(n)
            rows = rows * (n // len(rows)) + rng.sample(rows, n % len(rows))
        items += rows
    items = [it for it in items if it.meta.get("teacher_probs")]
    if not items:
        raise SystemExit("no teacher-labeled items")
    examples = [e for e in build_examples(items, tokenizer, ids, rng, round_size, compact_targets, prompt_format, shuffle_options) if len(e.input_ids) <= max_tokens]
    if limit:
        examples = examples[:limit]

    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                            "in_proj_qkv", "in_proj_z", "out_proj"],
        ),
    )
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    steps = len(token_batches(examples, batch_size, token_budget)) * epochs  # packing varies a little per shuffle
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps, pct_start=0.1)
    pad_id = tokenizer.pad_token_id or 0
    history = []
    step = 0
    t0 = time.perf_counter()
    for epoch in range(epochs):
        rng.shuffle(examples)
        for batch in token_batches(examples, batch_size, token_budget):
            if step >= steps:
                break
            ids_t, mask, last, cand, valid, teacher, gold = collate(batch, pad_id, "cuda", torch)
            # use_cache would keep a KV cache we never read during training
            hidden = model.base_model.model.model(input_ids=ids_t, attention_mask=mask, use_cache=False).last_hidden_state
            picked = hidden[torch.arange(len(batch), device="cuda"), last]
            logits = model.base_model.model.lm_head(picked).float().gather(1, cand)
            logits = logits.masked_fill(valid == 0, float("-inf"))
            log_probs = torch.log_softmax(logits, dim=-1)
            # padded slots hold -inf; 0 * -inf is NaN, so mask them out of both sums
            keep = valid > 0
            cross = torch.where(keep, teacher * log_probs, torch.zeros_like(teacher)).sum(-1)
            entropy = torch.where(keep, teacher * teacher.clamp_min(1e-9).log(), torch.zeros_like(teacher)).sum(-1)
            kl = -cross + entropy
            ce = -log_probs.gather(1, gold[:, None]).squeeze(1)
            loss = (alpha * kl + (1 - alpha) * ce).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if save_every and step % save_every == 0:
                ckpt = out_dir / f"{model_id.split('/')[-1]}-lora-step{step}"
                model.save_pretrained(str(ckpt))
                (ckpt / FORMAT_FILE).write_text(json.dumps({"round_size": round_size, "compact_targets": compact_targets, "prompt_format": prompt_format}) + "\n")
            if step % 20 == 0 or step == 1:
                row = {"step": step, "of": steps, "epoch": epoch, "loss": round(loss.item(), 4),
                       "kl": round(kl.mean().item(), 4), "ce": round(ce.mean().item(), 4),
                       "s/step": round((time.perf_counter() - t0) / step, 2),
                       "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
                history.append(row)
                print(json.dumps(row), flush=True)

    out = out_dir / f"{model_id.split('/')[-1]}-lora"
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    (out / FORMAT_FILE).write_text(json.dumps({"round_size": round_size, "compact_targets": compact_targets, "prompt_format": prompt_format}) + "\n")
    (out / "train_history.json").write_text(json.dumps({"examples": len(examples), "steps": step, "history": history}, indent=2))
    return {"examples": len(examples), "steps": step, "out": str(out)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--data", type=Path, action="append", help="training dir with items_labeled.jsonl, optionally dir:N to sample N items (repeatable)")
    p.add_argument("--out", type=Path, default=Path("runs/base"))
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=2560)
    p.add_argument("--token-budget", type=int, default=3072)
    p.add_argument("--save-every", type=int, default=0, help="also save the adapter every N steps")
    p.add_argument("--round-size", type=int, default=ROUND_SIZE, help="choice options per prompt (26 = A..Z, 52 = A..Z a..z)")
    p.add_argument("--compact-targets", action="store_true", help="render element target options as bare indices (~25%% fewer tokens)")
    p.add_argument("--limit", type=int)
    p.add_argument("--shuffle-options", action="store_true", help="random option order per choice example (order robustness)")
    p.add_argument("--prompt-format", type=int, default=1, help="1 = original; 2 = compact state JSON + rules hoisted once; 3 = 2 with shared instructions before the state and decorative elements pruned (see logits_base)")
    args = p.parse_args()
    print(json.dumps(train(model_id=args.model, data_dirs=tuple(args.data or [OUT_DIR]), out_dir=args.out, alpha=args.alpha, lr=args.lr, epochs=args.epochs,
                           batch_size=args.batch_size, max_tokens=args.max_tokens, token_budget=args.token_budget, limit=args.limit, round_size=args.round_size, compact_targets=args.compact_targets, prompt_format=args.prompt_format, shuffle_options=args.shuffle_options,
                           save_every=args.save_every), indent=2))
