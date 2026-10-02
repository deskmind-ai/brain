"""A local `POST /v1/systemone` server (wire-compatible with /v1/systemone), backed by any deskmind-brain predictor.

    uv run --extra mlx deskmind-brain-serve --predictor mlx:models/brain-4b --port 8793
    uv run --extra mlx deskmind-brain-serve --predictor mlx:models/brain-0.8b --escalate-to mlx:models/brain-4b --two-stage

With --escalate-to, one process serves both tiers: the fast model answers every request, and steps the routing rules
send up (see deskmind_brain.router) are answered by the strong model instead. The reply carries a `routing` record.

Requests are served one at a time (MLX is not thread-safe, and the Mac is compute-bound anyway). Existing
/v1/systemone clients only need their base URL pointed here. When DESKMIND_BRAIN_TOKEN is set and non-empty, every request
must carry `Authorization: Bearer <token>` (otherwise 401); when it is unset, the Authorization header is ignored.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import hmac
import json
import os
import threading
import time
import uuid
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from pydantic import ValidationError

from deskmind_brain.eval.data import EvalItem
from deskmind_brain.eval.predictors import make_predictor
from deskmind_brain.router import route
from deskmind_brain.types import SystemOneRequest


class Server:
    def __init__(self, spec: str, model_name: str, cache_size: int = 64, escalate_to: str | None = None,
                 threshold: float = 0.94, keep_done_over_undo: bool = True, routing_log: str | None = None):
        self.predictor = make_predictor(spec)
        self.strong = make_predictor(escalate_to) if escalate_to else None
        self.threshold = threshold
        self.keep_done_over_undo = keep_done_over_undo
        self.routed = collections.Counter()
        self.routing_log = routing_log
        self.model_name = model_name
        self.lock = threading.Lock()
        # An agent loop re-asks about a page it has already seen (a click that changed nothing, a WAIT, a
        # re-observation): same state, same questions, same answer. Caching those cuts a typical browser run by ~20%.
        self.cache: collections.OrderedDict[str, dict] = collections.OrderedDict()
        self.cache_size = cache_size
        self.hits = self.misses = 0

    def input_tokens(self, item: EvalItem) -> int | None:
        tokenizer = getattr(getattr(self.predictor, "inner", self.predictor), "tokenizer", None)
        if tokenizer is None:
            return None
        text = json.dumps(item.state, ensure_ascii=False) + json.dumps(
            {k: q.model_dump(exclude_none=True) for k, q in item.questions.items()}, ensure_ascii=False
        )
        return len(tokenizer.encode(text))

    def answer(self, body: dict) -> dict:
        request = SystemOneRequest(**body)
        item = EvalItem(id=uuid.uuid4().hex, suite="serve", group="serve", state=request.state,
                        questions=request.questions, references={})
        key = hashlib.sha256(json.dumps({"state": body.get("state"), "questions": body.get("questions")},
                                        sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with self.lock:
            cached = self.cache.get(key) if self.cache_size else None
            if cached is not None:
                self.cache.move_to_end(key)
                self.hits += 1
                return {**cached, "id": item.id, "cached": True}
            self.misses += 1
            started = time.perf_counter()
            prediction = self.predictor.predict(item)
            if prediction.error:
                raise RuntimeError(prediction.error)
            answers, routing = prediction.answers, None
            if self.strong is not None:
                answers, routing = route(body, self._plain(prediction.answers), self._ask_strong(item),
                                         self.threshold, keep_done_over_undo=self.keep_done_over_undo)
                self.routed[routing["reason"]] += 1
                if self.routing_log:  # metadata only, never the state or question text
                    fast_op = ((self._plain(prediction.answers).get("operation") or {}).get("choice"))
                    final_op = ((answers.get("operation") or {}) if isinstance(answers.get("operation"), dict) else {}).get("choice")
                    with open(self.routing_log, "a") as f:
                        f.write(json.dumps({"t": round(time.time(), 3), "escalated": routing["by"] == "strong",
                                            "reason": routing["reason"], "fast_op": fast_op, "final_op": final_op,
                                            "conf": routing["fast_conf"],
                                            "total_s": round(time.perf_counter() - started, 3)}) + "\n")
            elapsed = time.perf_counter() - started
        reply = {
            "id": item.id,
            "model": self.model_name,
            "answers": answers,
            "usage": {"input_tokens": self.input_tokens(item), "output_tokens": 0},
            "latency_ms": round(elapsed * 1000, 1),
        }
        if routing is not None:
            reply["routing"] = routing
        if self.cache_size:
            with self.lock:
                self.cache[key] = reply
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
        return reply


    @staticmethod
    def _plain(answers: dict) -> dict:
        return {k: (v.model_dump(exclude_none=True) if hasattr(v, "model_dump") else v) for k, v in answers.items()}

    def _ask_strong(self, item: EvalItem):
        def ask(_body: dict) -> dict:
            prediction = self.strong.predict(item)
            if prediction.error:
                raise RuntimeError(prediction.error)
            return self._plain(prediction.answers)
        return ask


def make_handler(server: Server, token: str | None = None):
    """With a non-empty `token`, every request must send `Authorization: Bearer <token>`; others get 401. The Mac app
    sets one per install, so a process that merely listens on the expected port cannot pose as its model server."""
    expected = f"Bearer {token}".encode() if token else None

    class Handler(BaseHTTPRequestHandler):
        def _authorized(self) -> bool:
            if expected is None:
                return True
            if hmac.compare_digest(self.headers.get("Authorization", "").encode(), expected):
                return True
            self._send(401, {"error": "unauthorized"})
            return False

        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 — http.server naming
            if not self._authorized():
                return
            if self.path.rstrip("/") == "/v1/models":
                entry = {"id": server.model_name, "object": "model"}
                if server.strong is not None:
                    entry["routing"] = dict(server.routed)
                self._send(200, {"data": [entry]})
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorized():
                return
            if self.path.rstrip("/") != "/v1/systemone":
                self._send(404, {"error": {"message": "not found"}})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                self._send(200, server.answer(body))
            except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as e:
                self._send(400, {"error": {"message": str(e)[:500]}})
            except Exception as e:  # noqa: BLE001 — report instead of dropping the connection
                self._send(500, {"error": {"message": f"{type(e).__name__}: {e}"[:500]}})

        def do_PUT(self) -> None:  # noqa: N802
            if self._authorized():
                self._send(404, {"error": {"message": "not found"}})

        do_DELETE = do_PATCH = do_PUT

        def log_message(self, fmt: str, *args) -> None:
            print(f"{self.address_string()} {fmt % args}", flush=True)

    return Handler


def default_threshold(spec: str, fallback: float = 0.94) -> float:
    """The routing threshold a release ships with: `router_threshold` in the fast tier's deskmind.json.

    Each round's fast model has its own confidence spread (G18b's 0.8B sits at 0.94-0.97), so the threshold travels with
    the weights instead of being a constant of the server.
    """
    path = Path(spec.split(":", 1)[-1]) / "deskmind.json"
    try:
        return float(json.loads(path.read_text()).get("router_threshold", fallback))
    except (OSError, ValueError):
        return fallback


def main() -> None:
    p = argparse.ArgumentParser(prog="deskmind-brain-serve")
    p.add_argument("--predictor", required=True, help="any deskmind-brain-eval predictor spec, e.g. mlx:models/brain-4b")
    p.add_argument("--model-name", default="deskmind-brain-local")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--cache-size", type=int, default=64, help="repeat-request cache entries (0 disables)")
    p.add_argument("--two-stage", action="store_true",
                   help="agent requests: score the operation first, then only the heads that operation needs")
    p.add_argument("--escalate-to", help="predictor spec of a stronger tier served in the same process (routing rules: "
                   "deskmind_brain.router); --predictor is then the fast tier")
    p.add_argument("--threshold", type=float, help="with --escalate-to: escalate below this confidence (default: the "
                   "fast tier's deskmind.json router_threshold, else 0.94)")
    p.add_argument("--no-keep-done-over-undo", action="store_true",
                   help="with --escalate-to: let the strong tier overrule a fast DONE with an undo click")
    p.add_argument("--routing-log", help="with --escalate-to: append one metadata line per request (who answered, why)")
    args = p.parse_args()
    if args.threshold is None:
        args.threshold = default_threshold(args.predictor)
    server = Server(args.predictor, args.model_name, args.cache_size, args.escalate_to, args.threshold,
                    not args.no_keep_done_over_undo, args.routing_log)
    if args.two_stage:
        for pred in (server.predictor, server.strong):
            if pred is not None:
                getattr(pred, "inner", pred).two_stage = True
        if server.strong is not None:
            # a DONE the strong tier confirms reuses the fast tier's heads (deskmind_brain.router.route): no head scoring
            getattr(server.strong, "inner", server.strong).terminal_heads = False
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(server, os.environ.get("DESKMIND_BRAIN_TOKEN") or None))
    print(f"serving {args.predictor} on http://{args.host}:{args.port}/v1/systemone", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
