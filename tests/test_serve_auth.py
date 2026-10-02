import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from deskmind_brain.serve import make_handler


class FakeServer:
    model_name = "test-model"
    strong = None

    def answer(self, body):
        return {"ok": True}


@pytest.fixture
def serve():
    servers = []

    def start(token):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(FakeServer(), token))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append(httpd)
        return f"http://127.0.0.1:{httpd.server_address[1]}"

    yield start
    for h in servers:
        h.shutdown()


def call(url, auth=None, body=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": auth} if auth else {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def test_no_token_ignores_header(serve):
    base = serve(None)
    assert call(base + "/v1/models")[0] == 200
    assert call(base + "/v1/models", auth="Bearer anything")[0] == 200
    assert call(base + "/v1/systemone", body={})[0] == 200


def test_token_required_on_every_route(serve):
    base = serve("s3cret")
    for path, body in [("/v1/models", None), ("/v1/systemone", {}), ("/nope", None)]:
        assert call(base + path, body=body) == (401, {"error": "unauthorized"})
        assert call(base + path, auth="Bearer wrong", body=body) == (401, {"error": "unauthorized"})
    assert call(base + "/v1/models", auth="Bearer s3cret")[0] == 200
    assert call(base + "/v1/systemone", auth="Bearer s3cret", body={}) == (200, {"ok": True})
    assert call(base + "/nope", auth="Bearer s3cret")[0] == 404
