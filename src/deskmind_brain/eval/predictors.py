"""Predictors turn an EvalItem into a Prediction. Model-backed predictors plug in here."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Protocol

from deskmind_brain.eval.data import EvalItem, Prediction
from deskmind_brain.types import to_answer


class Predictor(Protocol):
    name: str

    def predict(self, item: EvalItem) -> Prediction: ...


class UniformPredictor:
    """Floor: every option equally likely."""

    name = "uniform"

    def predict(self, item: EvalItem) -> Prediction:
        answers = {}
        for qid, q in item.questions.items():
            opts = q.options()
            answers[qid] = to_answer(q, {o: 1.0 / len(opts) for o in opts})
        return Prediction(item_id=item.id, answers=answers, latency_s=0.0, cost_usd=0.0)


class ReferencePredictor:
    """Ceiling / harness sanity check: answers with the reference distribution itself."""

    name = "reference"

    def predict(self, item: EvalItem) -> Prediction:
        answers = {qid: to_answer(item.questions[qid], ref.probs) for qid, ref in item.references.items()}
        return Prediction(item_id=item.id, answers=answers, latency_s=0.0, cost_usd=0.0)


class SystemOneAPIPredictor:
    """Any server speaking TypeSafe's `POST /v1/systemone` (TypeSafe itself, or our own server later)."""

    def __init__(
        self,
        base_url: str = "https://api.typesafe.ai",
        model: str = "jev-latest",
        api_key: str | None = None,
        usd_per_mtok_input: float = 0.042,
        timeout_s: float = 120.0,
    ):
        self.name = f"systemone:{model}"
        self.url = base_url.rstrip("/") + "/v1/systemone"
        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get("SYSTEMONE_API_KEY")
        self.usd_per_mtok_input = usd_per_mtok_input
        self.timeout_s = timeout_s

    def predict(self, item: EvalItem) -> Prediction:
        body = {
            "state": item.state,
            "model": self.model,
            "questions": {qid: q.model_dump(exclude_none=True) for qid, q in item.questions.items()},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers, method="POST")
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            payload = json.loads(resp.read())
        latency = time.perf_counter() - start
        usage = payload.get("usage") or {}
        cost = usage.get("input_tokens", 0) * self.usd_per_mtok_input / 1e6 if usage else None
        return Prediction(
            item_id=item.id, answers=payload["answers"], latency_s=latency, cost_usd=cost, usage=usage or None
        )


class CuaS1Predictor:
    """cua-ai/cua-s1-forms behind the same interface (research comparison only).

    Each question's criteria, in order, are the option strings; the state is the element context string.
    """

    def __init__(self, checkpoint: str, source: str | None = None):
        import sys as _sys

        import torch

        # `source` (or $CUA_S1_SRC) is a checkout's libs/cua-s1/python/src, e.g.
        # `export CUA_S1_SRC=~/src/cua/libs/cua-s1/python/src` in the shell profile.
        source = source or os.environ.get("CUA_S1_SRC")
        if not source:
            raise RuntimeError("cua-s1 needs its source: set CUA_S1_SRC to <cua checkout>/libs/cua-s1/python/src")
        _sys.path.insert(0, os.path.expanduser(source))
        from cua_s1.model import ChoiceExample, make_system

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model, self.collator = make_system(ckpt["config"], torch.device("cpu"))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.torch, self.example = torch, ChoiceExample
        self.name = "cua-s1"

    def predict(self, item: EvalItem) -> Prediction:
        start = time.perf_counter()
        answers = {}
        for qid, q in item.questions.items():
            keys = list(q.criteria)
            options = [str(q.criteria[k]) for k in keys]
            state = item.state if isinstance(item.state, str) else json.dumps(item.state)
            batch = self.collator([self.example(context=state, options=tuple(options), label=0)])
            with self.torch.no_grad():
                probs = self.torch.softmax(self.model(batch)[0][: len(options)].float(), -1).tolist()
            answers[qid] = to_answer(q, dict(zip(keys, probs)))
        return Prediction(item_id=item.id, answers=answers, latency_s=time.perf_counter() - start, cost_usd=0.0)


def make_predictor(spec: str) -> Predictor:
    """Any spec below, optionally suffixed `~<temps.json>` to apply fitted temperatures (see eval/calibrate.py).

    `uniform`, `reference`, `systemone[:model[@base_url]]`, `local[-noshare]:<model>[+<lora dir>]`, or `mlx:<model>[@4bit|@8bit]`."""
    if "~" in spec:
        from deskmind_brain.eval.calibrate import CalibratedPredictor

        inner, _, temps = spec.rpartition("~")
        return CalibratedPredictor(make_predictor(inner), temps)
    if spec.startswith("cua-s1:"):
        return CuaS1Predictor(spec[len("cua-s1:") :])
    if spec == "uniform":
        return UniformPredictor()
    if spec == "reference":
        return ReferencePredictor()
    if spec.startswith("systemone"):
        rest = spec.partition(":")[2]
        model, _, base_url = rest.partition("@")
        return SystemOneAPIPredictor(base_url=base_url or "https://api.typesafe.ai", model=model or "jev-latest")
    if spec.startswith(("local:", "local-noshare:")):
        from deskmind_brain.eval.local_logits import LocalLogitsPredictor

        kind, _, rest = spec.partition(":")
        model_id, _, adapter = rest.partition("+")  # local:<model>+<lora dir>
        return LocalLogitsPredictor(model_id, adapter=adapter or None, share_prefix=kind == "local")
    if spec.startswith("mlx:"):
        from deskmind_brain.eval.mlx_logits import MLXLogitsPredictor

        model_id, _, quant = spec[len("mlx:") :].partition("@")
        return MLXLogitsPredictor(model_id, bits=int(quant.rstrip("bit")) if quant else None)
    raise ValueError(f"unknown predictor: {spec}")
