import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from privyscope_local_llm.providers import OpenAICompatProvider, ProviderError


def _server(handler_body):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.server.seen.append((self.path, req))
            code, payload = handler_body(req)
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    srv.seen = []
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1"


def test_single_token_logprob_request_and_parse():
    srv, url = _server(lambda r: (200, {"choices": [{"logprobs": {"top_logprobs": [{" A": -0.1, " C": -2.3}]}}]}))
    try:
        out = OpenAICompatProvider(url, "m", concurrency=2).score(["p1", "p2", "p3"])
    finally:
        srv.shutdown()
    assert out == [{" A": -0.1, " C": -2.3}] * 3
    path, req = srv.seen[0]
    assert path == "/v1/completions"
    assert req["max_tokens"] == 1 and req["temperature"] == 0 and req["logprobs"] == 20


def test_server_without_logprobs_raises():
    srv, url = _server(lambda r: (200, {"choices": [{"text": "A", "logprobs": None}]}))
    try:
        with pytest.raises(ProviderError):
            OpenAICompatProvider(url, "m").score(["p"])
    finally:
        srv.shutdown()


def test_unreachable_server_raises():
    with pytest.raises(ProviderError):
        OpenAICompatProvider("http://127.0.0.1:1/v1", "m", timeout=1).score(["p"])
