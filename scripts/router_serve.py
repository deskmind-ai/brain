"""Two-tier `POST /v1/systemone`: a fast model answers, a stronger one takes over when the fast one should not act.

The fast tier (deskmind-brain 0.8B, ~1–2 s per desktop decision on a Mac) answers every request. Its answer is used only
when it is confident and the step is not one that is costly to get wrong; otherwise the same request goes to the
escalation tier and that answer is returned. The escalation tier is any server speaking the same API: a local 4B
(slower), or a remote API such as TypeSafe's Jev (fast, but the request text leaves the machine) via
--strong/--strong-key-env.

Agent requests (they carry an `operation` question):
  escalate if the operation is DONE / BLOCKED, a chord outside SAFE_KEYS, or a CLICK on the undo button -- ending a task, giving up and
  undoing are where earlier 0.8B checkpoints failed on the real desktop while the 4B did not;
  escalate if min(p(operation), p(each head that operation needs)) < --threshold (0.94: on the held-out desktop tasks
  the 0.8B's decisions above it were wrong <= 5% of the time; that set proved optimistic, so this is a starting point).
Other requests (plain judgements): escalate if any question's top probability is below --judge-threshold.

Every decision is logged (who answered, operation, confidence, reason; no state text) to --log.

    python scripts/router_serve.py --fast http://127.0.0.1:8794 --strong http://127.0.0.1:8793 --port 8796
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from deskmind_brain.router import SAFE_KEYS, decide, heads_for, is_undo_click, top_prob  # noqa: F401


def call(url: str, body: dict, key: str | None, model: str | None, timeout: float = 300) -> dict:
    payload = dict(body)
    if model:
        payload["model"] = model
    req = urllib.request.Request(url.rstrip("/") + "/v1/systemone", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key or 'local'}"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", required=True, help="base URL of the fast tier (deskmind-brain 0.8B)")
    ap.add_argument("--strong", required=True, help="base URL of the escalation tier (local 4B, or https://... for Jev)")
    ap.add_argument("--strong-key-env", help="env var holding the escalation tier's API key (e.g. SYSTEMONE_API_KEY)")
    ap.add_argument("--strong-model", help="model name to send to the escalation tier (e.g. jev-latest)")
    ap.add_argument("--threshold", type=float, default=0.94)
    ap.add_argument("--judge-threshold", type=float, default=0.8)
    ap.add_argument("--port", type=int, default=8796)
    ap.add_argument("--model-name", default="router")
    ap.add_argument("--log", default="runs/router_decisions.jsonl")
    ap.add_argument("--keep-done-over-undo", action="store_true",
                    help="when the fast tier says DONE and the escalation tier answers an undo click, keep DONE")
    args = ap.parse_args()
    strong_key = os.environ.get(args.strong_key_env) if args.strong_key_env else None
    lock = threading.Lock()
    stats = {"requests": 0, "escalated": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.startswith("/v1/models"):
                return self._send(200, {"data": [{"id": args.model_name, "object": "model"}], "stats": stats})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.startswith("/v1/systemone"):
                return self._send(404, {"error": "not found"})
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            t0 = time.time()
            try:
                fast = call(args.fast, body, None, None)
                escalate, reason, conf = decide(body, fast.get("answers") or {}, args.threshold, args.judge_threshold)
                t_fast = time.time() - t0
                out = call(args.strong, body, strong_key, args.strong_model) if escalate else fast
                if escalate and args.keep_done_over_undo and reason == "risky_DONE" and is_undo_click(body, out.get("answers") or {}):
                    # seen on the real-desktop diagnostic suite: the 4B overruled correct DONEs from the fast tier by
                    # undoing the finished moves, then looped move/undo until the budget ran out. Undoing is never how a
                    # goal gets finished.
                    out, reason, escalate = fast, "done_kept_over_undo", False
            except Exception as e:  # noqa: BLE001 — surface backend failures to the caller
                return self._send(502, {"error": f"{type(e).__name__}: {e}"})
            op = ((fast.get("answers") or {}).get("operation") or {}).get("choice")
            chord = ((fast.get("answers") or {}).get("key_target") or {}).get("choice") if op == "KEY" else None
            final_op = ((out.get("answers") or {}).get("operation") or {}).get("choice")
            rec = {"t": round(t0, 3), "escalated": escalate, "reason": reason, "fast_op": op, "chord": chord, "final_op": final_op,
                   "conf": round(conf, 4), "fast_s": round(t_fast, 3), "total_s": round(time.time() - t0, 3)}
            with lock:
                stats["requests"] += 1
                stats["escalated"] += escalate
                with open(args.log, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            out = dict(out)
            out["model"] = args.model_name
            out["routing"] = {"by": "strong" if escalate else "fast", "reason": reason, "fast_conf": rec["conf"]}
            self._send(200, out)

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
