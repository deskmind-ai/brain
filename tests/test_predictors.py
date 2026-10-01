import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from deskmind_brain.eval.predictors import SystemOneAPIPredictor, make_predictor
from deskmind_brain.eval.runner import run_predictions
from deskmind_brain.eval.data import load_predictions
from tests.test_metrics import make_item


class _Handler(BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.seen.append((self.path, self.headers.get("Authorization"), body))
        answers = {qid: {"type": "noul", "noul": 0.7} for qid, q in body["questions"].items() if q["type"] == "noul"}
        payload = json.dumps({"model": body["model"], "answers": answers, "usage": {"input_tokens": 1000, "output_tokens": 5}})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload.encode())

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_systemone_predictor_wire_format(server):
    _Handler.seen.clear()
    item = make_item()
    pred = SystemOneAPIPredictor(base_url=server, model="brain-4b", api_key="k").predict(item)
    path, auth, body = _Handler.seen[0]
    assert path == "/v1/systemone" and auth == "Bearer k"
    assert body["state"] == item.state and body["model"] == "brain-4b"
    assert body["questions"]["team"]["criteria"] == {"billing": "", "technical": "", "sales": ""}
    assert "criteria" not in body["questions"]["urgent"]
    assert pred.answers == {"urgent": {"type": "noul", "noul": 0.7}}
    assert pred.cost_usd == pytest.approx(1000 * 0.042 / 1e6)


def test_runner_resumes_and_retries_errors(tmp_path):
    item = make_item()
    out = tmp_path / "run.jsonl"

    class Flaky:
        name = "flaky"
        calls = 0

        def predict(self, it):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise RuntimeError("boom")
            return make_predictor("uniform").predict(it)

    assert run_predictions(Flaky(), [item], out)[item.id].error == "RuntimeError: boom"
    assert run_predictions(Flaky(), [item], out)[item.id].error is None
    run_predictions(Flaky(), [item], out)  # already done: no new call
    assert Flaky.calls == 2
    assert load_predictions(out)[item.id].error is None
