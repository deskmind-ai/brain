"""Send one request to Brain and print typed answers; no third-party dependencies."""

import argparse
import ipaddress
import json
import os
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Do not forward private state or credentials to a different endpoint.
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", nargs="?", type=Path, default=Path("examples/request.json"))
    parser.add_argument("--url", default="http://127.0.0.1:8793", help="Brain server base URL")
    parser.add_argument("--allow-remote", action="store_true",
                        help="Allow sending the request and token to a non-loopback server")
    args = parser.parse_args()
    try:
        url = urllib.parse.urlsplit(args.url)
        # Reject malformed ports as well as paths that would change the endpoint.
        url.port
        if (url.scheme not in {"http", "https"} or not url.hostname or url.username is not None
                or url.password is not None or url.path not in {"", "/"} or url.query or url.fragment):
            raise ValueError("use an http(s) base URL without credentials, a path, query or fragment")
        try:
            local = ipaddress.ip_address(url.hostname).is_loopback
        except ValueError:
            local = url.hostname.lower() == "localhost"
        if not local and not args.allow_remote:
            parser.error("non-local URL refused; use --allow-remote only if you trust the server")
        body = json.loads(args.request.read_text(encoding="utf-8"))
        headers = {"Content-Type": "application/json"}
        if token := os.environ.get("DESKMIND_BRAIN_TOKEN"):
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(args.url.rstrip("/") + "/v1/systemone",
                                         data=json.dumps(body).encode("utf-8"), headers=headers)
        # A proxy must not turn a local-only request into a remote disclosure.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        with opener.open(request, timeout=30) as response:
            answers = json.load(response)["answers"]
        for name, answer in answers.items():
            if answer["type"] == "choice":
                chosen = answer["choice"]
                print(f"{name}: {chosen} (P={answer['probabilities'][chosen]:.3f})")
            elif answer["type"] == "noul":
                print(f"{name}: P(yes)={answer['noul']:.3f}")
            elif answer["type"] == "score":
                print(f"{name}: score={answer['score']}")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"client: {exc}\n")


if __name__ == "__main__":
    main()
