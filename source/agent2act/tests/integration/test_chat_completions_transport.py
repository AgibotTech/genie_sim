# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent2act.bootstrap import build_session


@pytest.mark.integration
def test_chat_http_auth_retry_and_session(config, request_payload, monkeypatch):
    decision = config["planner"]["decisions"][0]
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(
                (self.path, self.headers.get("Authorization"), self.headers.get("User-Agent"), body)
            )
            if len(requests) == 1:
                self.send_response(503)
                self.end_headers()
                return
            response = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(decision)}}]}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        def log_message(self, *args):
            pass

    monkeypatch.setenv("TEST_CHAT_KEY", "offline-test-key")
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            config["planner"] = {
                "type": "chat_completions",
                "model": "qwen3-vl-plus",
                "endpoint": f"http://127.0.0.1:{server.server_port}/compatible-mode/v1/chat/completions",
                "api_key_env": "TEST_CHAT_KEY",
                "retry_backoff_s": 0.001,
                "retry_max_delay_s": 0.001,
                "enable_thinking": False,
            }
            with build_session(config) as session:
                session.handle(request_payload)
                assert session.runtime._last_issued.steps == config["execution"]["execute_steps"]
                session.runtime.reset()
        finally:
            server.shutdown()
            thread.join(timeout=5)
    assert len(requests) == 2 and requests[0] == requests[1]
    path, auth, user_agent, body = requests[0]
    assert path == "/compatible-mode/v1/chat/completions"
    assert auth == "Bearer offline-test-key"
    assert user_agent == "Agent2Act/0.1"
    assert body["model"] == "qwen3-vl-plus"
    assert body["messages"][1]["role"] == "user"
    assert "input" not in body and "reasoning" not in body
