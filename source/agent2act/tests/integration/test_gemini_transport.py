# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent2act.adapters.planners.gemini import GeminiPlanner


@pytest.mark.integration
def test_native_auth_and_signed_tool_continuation_retry(observation, motion_decision, monkeypatch):
    requests = []
    model_content = {
        "role": "model",
        "parts": [
            {
                "thoughtSignature": "opaque-offline-signature",
                "functionCall": {
                    "id": "depth1",
                    "name": "read_depth_pixels",
                    "args": {
                        "camera": "head",
                        "pixels": [{"label": "test", "u": 0, "v": 0}],
                    },
                },
            }
        ],
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append(
                (self.path, self.headers.get("x-goog-api-key"), self.headers.get("Authorization"), raw)
            )
            if len(requests) == 2:
                self.send_response(503)
                self.end_headers()
                return
            content = (
                model_content
                if len(requests) == 1
                else {
                    "role": "model",
                    "parts": [{"text": json.dumps(motion_decision)}],
                }
            )
            response = {
                "modelVersion": "offline",
                "candidates": [{"finishReason": "STOP", "content": content}],
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        def log_message(self, *args):
            pass

    monkeypatch.setenv("OFFLINE_GEMINI_KEY", "offline-key")
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        planner = GeminiPlanner(
            model="offline",
            endpoint=f"http://127.0.0.1:{server.server_port}/v1beta/models/offline:generateContent",
            api_key_env="OFFLINE_GEMINI_KEY",
            reasoning_effort="high",
            use_depth=True,
            enable_raw_depth_query=True,
            max_retries=1,
            retry_backoff_s=0.001,
            retry_max_delay_s=0.001,
        )
        try:
            assert planner.decide(observation, {}).mode == "motion"
            assert len(planner._geometry_memory) == 1
        finally:
            planner.close()
            server.shutdown()
            thread.join(timeout=5)
    assert len(requests) == 3 and requests[1] == requests[2]
    path, key, authorization, raw = requests[-1]
    assert (path, key, authorization) == ("/v1beta/models/offline:generateContent", "offline-key", None)
    body = json.loads(raw)
    assert body["contents"][-2] == model_content
    assert body["contents"][-1]["parts"][0]["functionResponse"]["id"] == "depth1"
    assert body["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "HIGH"


@pytest.mark.integration
def test_redirect_to_another_origin_receives_no_credential(monkeypatch):
    """urllib copies ordinary headers onto a redirect target, including another origin."""
    received = []

    def capture(where, headers):
        # email.message.Message is case insensitive; dict() would not be.
        received.append((where, {name.lower(): value for name, value in headers.items()}))

    class Sink(BaseHTTPRequestHandler):
        def do_GET(self):
            capture("target", self.headers)
            body = b'{"candidates": []}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    class Redirect(BaseHTTPRequestHandler):
        def do_POST(self):
            capture("origin", self.headers)
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{sink.server_port}/elsewhere")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    with (
        ThreadingHTTPServer(("127.0.0.1", 0), Sink) as sink,
        ThreadingHTTPServer(("127.0.0.1", 0), Redirect) as server,
    ):
        threads = [threading.Thread(target=each.serve_forever, daemon=True) for each in (sink, server)]
        for thread in threads:
            thread.start()
        monkeypatch.setenv("OFFLINE_GEMINI_KEY", "offline-key")
        planner = GeminiPlanner(
            model="offline",
            endpoint=f"http://127.0.0.1:{server.server_port}/v1beta/models/offline:generateContent",
            api_key_env="OFFLINE_GEMINI_KEY",
            max_retries=0,
        )
        try:
            assert planner._post({"model": "offline"}) == {"candidates": []}
        finally:
            planner.close()
            for each in (server, sink):
                each.shutdown()
            for thread in threads:
                thread.join(timeout=5)

    origin = dict(received)["origin"]
    assert origin["x-goog-api-key"] == "offline-key"
    assert "authorization" not in origin
    target = dict(received)["target"]
    assert "x-goog-api-key" not in target
    assert "authorization" not in target
