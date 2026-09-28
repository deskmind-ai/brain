"""Training data for the base distillation run (public hf task train splits).

Only the *train* splits of the datasets marked `intask` in the eval suite are used; every `heldout` dataset stays
untouched so the held-out numbers keep their meaning. Items are the same EvalItem format as the eval suites:
the dataset's own label goes into `references` (hard), and the teacher's distribution is added later by
`label_teacher.py` under meta["teacher"].
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deskmind_brain.eval.data import EvalItem, write_jsonl
from deskmind_brain.eval.sources import hf_tasks as hf

OUT_DIR = Path("data/train/base")

# Train-split file for each intask task; heldout tasks are deliberately absent.
TRAIN_PATHS: dict[str, str] = {
    "banking77": "data/train-00000-of-00001.parquet",
    "emotion": "data/train-00000-of-00001.parquet",
    "mmlu": "all/auxiliary_train-00000-of-00001.parquet",
    "boolq": "data/train-00000-of-00001.parquet",
    "tweet_hate": "hate/train-00000-of-00001.parquet",
    "sst2": "data/train-00000-of-00001.parquet",
    "toxic_chat": "data/0124/toxic-chat_annotation_train.csv",
    "helpsteer2": "train.jsonl.gz",
    "amazon_reviews": "en/train.jsonl",
}


def train_tasks() -> list[hf.Task]:
    tasks = []
    for task in hf.TASKS:
        if task.role != "intask" or task.name not in TRAIN_PATHS:
            continue
        tasks.append(hf.Task(**{**task.__dict__, "path": TRAIN_PATHS[task.name]}))
    return tasks


def build(n_per_task: int = 800, seed: int = 100, raw_dir: Path = Path("data/raw/hf_tasks")) -> dict[str, Any]:
    """Sample training requests. The seed differs from the eval suite's, and eval rows come from other splits."""
    items: list[EvalItem] = []
    summary = {}
    for task in train_tasks():
        built = hf.build_task(task, raw_dir, n=n_per_task, seed=seed)
        for it in built:
            it.meta["split"] = "train"
        items.extend(built)
        summary[task.name] = {"items": len(built), "questions": sum(len(i.references) for i in built)}
    write_jsonl(OUT_DIR / "items.jsonl", (it.to_json() for it in items))
    result = {"items": len(items), "questions": sum(len(i.references) for i in items), "by_task": summary, "out": str(OUT_DIR)}
    (OUT_DIR / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
