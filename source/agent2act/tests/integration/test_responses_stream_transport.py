# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent2act.adapters.planners.responses import ResponsesPlanner
from agent2act.bootstrap import build_session


def _event(kind, **fields):
    return f"event: {kind}\ndata: {json.dumps({'type': kind, **fields})}\n\n".encode()


@pytest.mark.integration
def test_streamed_responses_reassemble_the_final_response_and_retry_a_cut_stream(
    config, request_payload, monkeypatch
):
    decision = config["planner"]["decisions"][0]
    final = {
        "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(decision)}]}],
    }
    requests = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(_event("response.created", response={"status": "in_progress"}))
            self.wfile.write(_event("response.reasoning_summary_text.delta", delta="Checking the target"))
            if len(requests) == 1:
                # A gateway dropping the connection before completion.
                self.wfile.flush()
                self.close_connection = True
                return
            self.wfile.write(_event("response.output_text.delta", delta="{"))
            self.wfile.write(_event("response.completed", response=final))
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *args):
            pass

    monkeypatch.setenv("TEST_RESPONSES_KEY", "offline-test-key")
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            config["planner"] = {
                "type": "responses",
                "model": "test",
                "endpoint": f"http://127.0.0.1:{server.server_port}/v1/responses",
                "api_key_env": "TEST_RESPONSES_KEY",
                "stream": True,
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
    assert requests[0]["stream"] is True


@pytest.mark.integration
def test_stream_option_is_validated_and_off_by_default(config):
    config["planner"] = {"type": "responses", "model": "test", "stream": "yes"}
    with pytest.raises(Exception, match="stream must be boolean"):
        build_session(config)
    assert "stream" not in ResponsesPlanner(model="test")._request_body([])
    assert ResponsesPlanner(model="test", stream=True)._request_body([])["stream"] is True


@pytest.mark.integration
@pytest.mark.parametrize(
    "error, retried",
    [
        ({"type": "server_error", "code": "server_error", "message": "retry"}, True),
        ({"type": "invalid_request_error", "code": "bad"}, False),
    ],
)
def test_stream_server_errors_are_retried_like_5xx(monkeypatch, error, retried):
    final = {
        "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "{}"}]}],
    }
    requests = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            requests.append(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(_event("response.created", response={"status": "in_progress"}))
            if len(requests) == 1:
                self.wfile.write(_event("response.failed", response={"status": "failed", "error": error}))
            else:
                self.wfile.write(_event("response.completed", response=final))
            self.close_connection = True

        def log_message(self, *args):
            pass

    monkeypatch.setenv("TEST_RESPONSES_KEY", "offline-test-key")
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            planner = ResponsesPlanner(
                model="test",
                endpoint=f"http://127.0.0.1:{server.server_port}/v1/responses",
                api_key_env="TEST_RESPONSES_KEY",
                stream=True,
                retry_backoff_s=0.001,
                retry_max_delay_s=0.001,
            )
            result = planner._post({"model": "test", "input": "x", "stream": True})
        finally:
            server.shutdown()
            thread.join(timeout=5)
    if retried:
        assert len(requests) == 2 and result["status"] == "completed"
    else:
        assert len(requests) == 1 and result["status"] == "failed"
