"""One error shape for every status (deskmind#36 item 5): {"error": {"code", "message", "field"?, "retryable"}}."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from deskmind_brain.serve import Server, make_handler

GOOD = {"state": {"page": {"title": "w"}},
        "questions": {"operation": {"type": "choice", "instructions": "go?", "criteria": {"CLICK": "c", "DONE": "d"}}}}


class Broken(Server):
    def answer(self, body):
        super().answer(body)            # validates the request first, as the real one does
        raise ValueError("a bug while answering")


@pytest.fixture
def serve():
    servers = []

    def start(server):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(server))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append(httpd)
        return f"http://127.0.0.1:{httpd.server_address[1]}"

    yield start
    for h in servers:
        h.shutdown()


def post(url, raw: bytes):
    req = urllib.request.Request(url, data=raw, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def shape(err):
    assert set(err) >= {"code", "message", "retryable"} and set(err) <= {"code", "message", "retryable", "field"}
    assert isinstance(err["retryable"], bool)
    return err


def test_bad_json_and_non_objects(serve):
    base = serve(Server("uniform", "t", cache_size=0)) + "/v1/systemone"
    status, body = post(base, b"{not json")
    assert status == 400 and shape(body["error"])["code"] == "invalid_json"
    status, body = post(base, b"[1, 2]")
    assert status == 400 and shape(body["error"])["code"] == "invalid_request"


def test_an_invalid_request_names_the_field(serve):
    base = serve(Server("uniform", "t", cache_size=0)) + "/v1/systemone"
    bad = json.loads(json.dumps(GOOD))
    bad["questions"]["operation"]["criteria"] = [{"key": "A", "description": ""}, {"key": "A", "description": ""}]
    status, body = post(base, json.dumps(bad).encode())
    err = shape(body["error"])
    assert status == 400 and err["code"] == "invalid_request" and err["field"].startswith("questions.operation")
    assert err["retryable"] is False


def test_a_failure_while_answering_is_the_servers_500_not_a_400(serve):
    base = serve(Broken("uniform", "t", cache_size=0)) + "/v1/systemone"
    status, body = post(base, json.dumps(GOOD).encode())
    assert status == 500 and shape(body["error"])["code"] == "internal_error"
    assert "a bug while answering" in body["error"]["message"]


def test_not_found(serve):
    base = serve(Server("uniform", "t", cache_size=0))
    status, body = post(base + "/v1/other", json.dumps(GOOD).encode())
    assert status == 404 and shape(body["error"])["code"] == "not_found"
