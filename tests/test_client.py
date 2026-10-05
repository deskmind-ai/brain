import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from deskmind_brain.serve import make_handler


ROOT = Path(__file__).resolve().parents[1]
ANSWERS = {
    "operation": {"type": "choice", "choice": "CLICK", "probabilities": {"CLICK": 0.8, "TYPE": 0.2}},
    "ready": {"type": "noul", "noul": 0.9},
    "quality": {"type": "score", "score": 1.5},
}


class FakeServer:
    model_name = "test-model"
    strong = None

    def __init__(self):
        self.bodies = []

    def answer(self, body):
        self.bodies.append(body)
        return {"answers": ANSWERS}


@pytest.fixture
def serve():
    servers = []

    def start(token=None, handler=None):
        fake = FakeServer()
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler or make_handler(fake, token))
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        servers.append((httpd, thread))
        return f"http://127.0.0.1:{httpd.server_address[1]}", fake

    yield start
    for httpd, thread in servers:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def run_client(*args, token=None):
    env = dict(os.environ)
    env.pop("DESKMIND_BRAIN_TOKEN", None)
    if token is not None:
        env["DESKMIND_BRAIN_TOKEN"] = token
    return subprocess.run([sys.executable, "examples/client.py", *args], cwd=ROOT,
                          env=env, capture_output=True, text=True, timeout=10)


@pytest.mark.parametrize("token", [None, "test-token"])
def test_client_posts_default_request_and_prints_all_answer_types(serve, token):
    url, fake = serve(token)
    result = run_client("--url", url, token=token)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "operation: CLICK (P=0.800)", "ready: P(yes)=0.900", "quality: score=1.5",
    ]
    assert fake.bodies == [json.loads((ROOT / "examples/request.json").read_text())]


def test_client_accepts_an_explicit_request_file(serve, tmp_path):
    url, fake = serve()
    path = tmp_path / "request.json"
    path.write_text('{"state": {}, "questions": {}}')
    assert run_client(str(path), "--url", url + "/").returncode == 0
    assert fake.bodies == [{"state": {}, "questions": {}}]


def test_client_reports_auth_failure_without_a_traceback(serve):
    url, fake = serve("test-token")
    result = run_client("--url", url)
    assert result.returncode != 0
    assert "401" in result.stderr
    assert "Traceback" not in result.stderr
    assert fake.bodies == []


@pytest.mark.parametrize("url", ["http://example.com", "https://127.0.0.1.example.com",
                                "http://192.168.1.1"])
def test_remote_urls_are_refused_before_reading_or_sending(url):
    result = run_client("missing.json", "--url", url)
    assert result.returncode != 0
    assert "non-local URL refused" in result.stderr
    assert "--allow-remote" in result.stderr


def test_allow_remote_is_an_explicit_opt_in():
    # A missing file proves URL validation passed, without making a remote call.
    result = run_client("missing.json", "--url", "https://example.com", "--allow-remote")
    assert result.returncode != 0
    assert "missing.json" in result.stderr
    assert "non-local URL refused" not in result.stderr


@pytest.mark.parametrize("url", ["file:///tmp/request.json", "http://localhost:bad",
                                "http://user:password@localhost", "http://localhost/path"])
def test_invalid_base_urls_are_rejected(url):
    result = run_client("--url", url)
    assert result.returncode != 0
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("url", ["http://localhost", "http://[::1]", "http://127.0.0.2"])
def test_loopback_forms_pass_url_validation(url):
    result = run_client("missing.json", "--url", url)
    assert "missing.json" in result.stderr
    assert "non-local URL refused" not in result.stderr


def test_redirects_do_not_forward_request_or_token(serve):
    destination, fake = serve()

    class Redirect(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(307)
            self.send_header("Location", destination + "/v1/systemone")
            self.end_headers()

        def log_message(self, *args):
            pass

    url, _ = serve(handler=Redirect)
    result = run_client("--url", url, token="test-token")
    assert result.returncode != 0
    assert "307" in result.stderr
    assert fake.bodies == []
