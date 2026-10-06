"""Wire types for `POST /v1/systemone` (wire-compatible with /v1/systemone).

Internally every answer is a probability distribution over a question's option keys:
  choice -> criteria keys, score -> "0".."K-1", noul -> "false", "true".
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, StrictInt, model_validator

QuestionType = Literal["choice", "score", "noul"]
Distribution = dict[str, float]

NOUL_OPTIONS = ("false", "true")
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS, MAX_SCORE_LEVELS = 2, 10


class Question(BaseModel):
    type: QuestionType
    instructions: Any
    criteria: Any = None

    @model_validator(mode="before")
    @classmethod
    def _criteria_list_form(cls, data: Any) -> Any:
        """Protocol v1 also sends choice criteria as `[{"key", "description"}]`, which keeps the option order explicit
        instead of leaving it to JSON object order. A description is any JSON value, as in the map form (agent heads
        mostly send objects). Both forms become the same ordered map, so `options()` and the prompt are identical.
        Brain renders options in the order sent; order rules are checked by the harness."""
        if not isinstance(data, dict) or data.get("type") != "choice" or not isinstance(data.get("criteria"), list):
            return data
        criteria: dict[str, Any] = {}
        for entry in data["criteria"]:
            if not isinstance(entry, dict) or set(entry) != {"key", "description"}:
                raise ValueError('choice criteria entries must be {"key", "description"}')
            key, description = entry["key"], entry["description"]
            if not isinstance(key, str) or not key:
                raise ValueError("choice criteria key must be a non-empty string")
            if key in criteria:
                raise ValueError(f"duplicate choice criteria key: {key!r}")
            criteria[key] = description
        return {**data, "criteria": criteria}

    @model_validator(mode="after")
    def _check_criteria(self) -> Question:
        if self.type == "choice":
            if not isinstance(self.criteria, dict) or not (1 <= len(self.criteria) <= MAX_CHOICE_OPTIONS):
                raise ValueError(f"choice criteria must be a map or a list with 1..{MAX_CHOICE_OPTIONS} options")
        elif self.type == "score":
            if not isinstance(self.criteria, list) or not (MIN_SCORE_LEVELS <= len(self.criteria) <= MAX_SCORE_LEVELS):
                raise ValueError(f"score criteria must be a list of {MIN_SCORE_LEVELS}..{MAX_SCORE_LEVELS} levels")
        return self

    def options(self) -> list[str]:
        if self.type == "choice":
            return list(self.criteria)
        if self.type == "score":
            return [str(i) for i in range(len(self.criteria))]
        return list(NOUL_OPTIONS)


class SystemOneRequest(BaseModel):
    state: Any
    model: str = "deskmind-brain-local"
    questions: dict[str, Question]
    # Request identity (protocol v1, deskmind#36 item 4): which step of which run this is for. Optional; validated as
    # the schema says; never part of the prompt or the answer cache key.
    request_id: str | None = Field(default=None, min_length=1)
    session_id: str | None = Field(default=None, min_length=1)
    step: StrictInt | None = Field(default=None, ge=1)   # strict: "3", True and 3.0 are not a step
    observation_id: str | None = Field(default=None, min_length=1)
    state_digest: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")

    def identity(self) -> dict[str, Any]:
        """What a reply echoes and the routing log records: request_id, session_id, step, as sent."""
        return {k: getattr(self, k) for k in ("request_id", "session_id", "step") if getattr(self, k) is not None}


def normalize(dist: dict[str, float], options: list[str]) -> Distribution:
    """Restrict to `options` (missing -> 0) and renormalize; uniform if all mass is zero."""
    vals = [max(float(dist.get(o, 0.0)), 0.0) for o in options]
    total = sum(vals)
    if total <= 0:
        return {o: 1.0 / len(options) for o in options}
    return {o: v / total for o, v in zip(options, vals)}


def choice_confidence(dist: Distribution) -> float:
    """(K * p_max - 1) / (K - 1). Matches every published Opus/Sol answer in the public eval suite exactly."""
    k = len(dist)
    if k <= 1:
        return 1.0
    return (k * max(dist.values()) - 1.0) / (k - 1)


def score_value(dist: Distribution) -> float:
    return sum(int(level) * p for level, p in dist.items())


def score_confidence(dist: Distribution) -> float:
    """1 - 2 * E|level - score| / (K - 1). Approximates the (unpublished) /v1/systemone score confidence."""
    k = len(dist)
    if k <= 1:
        return 1.0
    mean = score_value(dist)
    mad = sum(p * abs(int(level) - mean) for level, p in dist.items())
    return max(0.0, 1.0 - 2.0 * mad / (k - 1))


def to_answer(question: Question, dist: Distribution) -> dict[str, Any]:
    """Distribution -> /v1/systemone answer object."""
    dist = normalize(dist, question.options())
    if question.type == "noul":
        return {"type": "noul", "noul": dist["true"]}
    if question.type == "choice":
        return {
            "type": "choice",
            "choice": max(dist, key=dist.__getitem__),
            "probabilities": dist,
            "confidence": choice_confidence(dist),
        }
    return {
        "type": "score",
        "score": score_value(dist),
        "legend": {str(i): level for i, level in enumerate(question.criteria)},
        "probabilities": dist,
        "confidence": score_confidence(dist),
    }


def from_answer(question: Question, answer: dict[str, Any]) -> Distribution:
    """/v1/systemone answer object -> distribution over the question's options."""
    options = question.options()
    if question.type == "noul":
        p = float(answer["noul"])
        if math.isnan(p):
            raise ValueError("noul is NaN")
        p = min(max(p, 0.0), 1.0)
        return {"false": 1.0 - p, "true": p}
    probs = answer.get("probabilities")
    if probs:
        return normalize({str(k): v for k, v in probs.items()}, options)
    # Answer carries only a hard value: treat it as one-hot.
    value = str(answer.get("choice") if question.type == "choice" else round(float(answer["score"])))
    return normalize({value: 1.0}, options)
