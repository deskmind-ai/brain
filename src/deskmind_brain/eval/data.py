"""Eval suite format.

A suite is a directory:
  items.jsonl              one EvalItem per line (one System One request + reference distributions)
  published/<model>.jsonl  optional Prediction files shipped with the source (third-party answers, scored for comparison only;
                           never used as training labels)

Predictions are also JSONL, one Prediction per item, so runs can be resumed and re-scored.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from deskmind_brain.types import Distribution, Question


@dataclass
class Reference:
    probs: Distribution
    # How many labelers were averaged, and whether all of them gave probabilities (vs hard values).
    n_sets: int = 1
    soft: bool = True


@dataclass
class EvalItem:
    id: str
    suite: str
    group: str
    state: Any
    questions: dict[str, Question]
    references: dict[str, Reference]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["questions"] = {qid: q.model_dump(exclude_none=True) for qid, q in self.questions.items()}
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> EvalItem:
        return cls(
            id=d["id"],
            suite=d["suite"],
            group=d["group"],
            state=d["state"],
            questions={qid: Question.model_validate(q) for qid, q in d["questions"].items()},
            references={qid: Reference(**r) for qid, r in d["references"].items()},
            meta=d.get("meta", {}),
        )


@dataclass
class Prediction:
    item_id: str
    answers: dict[str, dict[str, Any]]  # qid -> /v1/systemone answer object
    latency_s: float | None = None
    cost_usd: float | None = None
    usage: dict[str, int] | None = None
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Prediction:
        return cls(**d)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_items(suite_dir: Path) -> list[EvalItem]:
    return [EvalItem.from_json(d) for d in read_jsonl(suite_dir / "items.jsonl")]


def load_predictions(path: Path) -> dict[str, Prediction]:
    return {p.item_id: p for p in (Prediction.from_json(d) for d in read_jsonl(path))}
