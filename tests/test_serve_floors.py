"""Recommended act-or-not floors travel with the weights and are advertised in /v1/models (deskmind#63 part 2)."""
import json
import tempfile
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from deskmind_brain.serve import default_floors, make_handler, parse_floors


class FakeServer:
    model_name = "test-model"
    strong = None

    def __init__(self, floors=None):
        self.floors = floors or {}


def models(server):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(server, None))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_address[1]}/v1/models", timeout=5) as r:
            return json.load(r)["data"][0]
    finally:
        httpd.shutdown()


def test_floors_are_advertised_when_set():
    entry = models(FakeServer({"consequential": 0.9, "value": 0.5}))
    assert entry["floors"] == {"consequential": 0.9, "value": 0.5}


def test_no_floors_no_field():
    assert "floors" not in models(FakeServer())


def test_floors_come_from_the_release_folder_or_the_flag():
    d = Path(tempfile.mkdtemp())
    (d / "deskmind.json").write_text(json.dumps({"prompt_format": 3, "floors": {"consequential": 0.9, "value": 0.5}}))
    assert default_floors(f"mlx:{d}") == {"consequential": 0.9, "value": 0.5}
    assert default_floors(f"mlx:{Path(tempfile.mkdtemp())}") == {}
    assert parse_floors("consequential=0.95, value=0.4") == {"consequential": 0.95, "value": 0.4}
