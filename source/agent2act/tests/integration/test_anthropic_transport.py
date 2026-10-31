# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent2act.adapters.planners.anthropic import AnthropicPlanner
from agent2act.bootstrap import build_session


@pytest.mark.integration
@pytest.mark.parametrize("status", [503, 400])
def test_messages_auth_retry_and_session(config, request_payload, monkeypatch, status):
    decision = config["planner"]["decisions"][0]
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(
                (self.path, self.headers.get("x-api-key"), self.headers.get("anthropic-version"), body)
            )
            if len(requests) == 1:
                self.send_response(status)
                self.end_headers()
                return
            response = {
                "type": "message",
                "model": "offline",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": json.dumps(decision)}],
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        def log_message(self, *args):
            pass

    monkeypatch.setenv("OFFLINE_MESSAGES_KEY", "offline-test-key")
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            config["planner"] = {
                "type": "anthropic",
                "model": "offline",
                "endpoint": f"http://127.0.0.1:{server.server_port}/v1/messages",
                "api_key_env": "OFFLINE_MESSAGES_KEY",
                "reasoning_effort": "high",
                "retry_backoff_s": 0.001,
                "retry_max_delay_s": 0.001,
            }
            with build_session(config) as session:
                session.handle(request_payload)
                assert session.runtime._last_issued.steps == config["execution"]["execute_steps"]
                session.runtime.reset()
        finally:
            server.shutdown()
            thread.join(timeout=5)
    assert len(requests) == 2 and requests[0] == requests[1]
    path, key, version, body = requests[0]
    assert (path, key, version) == ("/v1/messages", "offline-test-key", "2023-06-01")
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "high"}
    assert "messages" in body and "input" not in body


@pytest.mark.integration
def test_retry_signature_http_400_sends_identical_signed_continuation(observation, motion_decision):
    requests = []
    first = {
        "type": "message",
        "stop_reason": "tool_use",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "offline-signed-block"},
            {
                "type": "tool_use",
                "id": "depth1",
                "name": "read_depth_pixels",
                "input": {"camera": "head", "pixels": [{"label": "test", "u": 0, "v": 0}]},
            },
        ],
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append(raw)
            if len(requests) == 1:
                response, status = first, 200
            elif len(requests) == 2:
                response, status = {"error": {"message": "Invalid `signature` in `thinking` block"}}, 400
            else:
                response, status = (
                    {
                        "type": "message",
                        "stop_reason": "end_turn",
                        "content": [{"type": "text", "text": json.dumps(motion_decision)}],
                    },
                    200,
                )
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        planner = AnthropicPlanner(
            model="offline",
            endpoint=f"http://127.0.0.1:{server.server_port}/v1/messages",
            api_key_env=None,
            reasoning_effort="high",
            max_retries=2,
            retry_backoff_s=0.001,
            retry_max_delay_s=0.001,
            use_depth=True,
            enable_raw_depth_query=True,
        )
        try:
            assert planner.decide(observation, {}).mode == "motion"
            assert len(planner._geometry_memory) == 1
        finally:
            planner.close()
            server.shutdown()
            thread.join(timeout=5)
    assert len(requests) == 3 and requests[1] == requests[2]
    body = json.loads(requests[2])
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "high"}
    assert body["messages"][-2]["content"] == first["content"]
    assert body["messages"][-1]["content"][0]["tool_use_id"] == "depth1"
