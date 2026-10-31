# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json
import threading
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent2act.bootstrap import build_session


@pytest.mark.integration
def test_session_recovers_encrypted_continuation_without_duplicate_actions(config, request_payload, tmp_path):
    decision = deepcopy(config["planner"]["decisions"][0])
    requests, events = [], []
    config["logging"]["trace_dir"] = str(tmp_path / "trace")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            number = len(requests)
            status = 400 if number in (3, 4) else 200
            if status == 400:
                body = {"error": {"type": "invalid_request_error", "code": "invalid_encrypted_content"}}
            elif number in (2, 5):
                body = {
                    "output": [
                        {
                            "type": "reasoning",
                            "id": str(number),
                            "summary": [],
                            "encrypted_content": "opaque",
                        },
                        {
                            "type": "function_call",
                            "name": "compose_grasp_pose",
                            "call_id": str(number),
                            "arguments": json.dumps(
                                {"arm": "left", "center_m": [0.5, 0.1, 0.8], "orientation_xyzw": [1, 0, 0, 0]}
                            ),
                        },
                    ]
                }
            else:
                body = {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": json.dumps(decision)}],
                        }
                    ]
                }
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    original = deepcopy(request_payload)
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            config["planner"] = {
                "type": "embodied",
                "model": "offline",
                "api_key_env": None,
                "endpoint": f"http://127.0.0.1:{server.server_port}/v1/responses",
                "wrist_to_center_m": {"left": [0, 0, 0.15], "right": [0, 0, 0.15]},
                "invalid_encrypted_content_retries": 1,
                "invalid_encrypted_content_replans": 1,
                "retry_backoff_s": 0.001,
                "retry_max_delay_s": 0.001,
                "retry_budget_s": 10,
            }
            with build_session(config, event_sink=events.append) as session:
                session.handle(request_payload)
                response = session.handle(request_payload)
                assert "error" not in response
                assert session.runtime._last_issued.steps == config["execution"]["execute_steps"]
        finally:
            server.shutdown()
            thread.join(timeout=5)
    assert len(requests) == 6
    assert requests[2] == requests[3]  # Retry the exact rejected continuation.
    assert requests[1] == requests[4]  # Rebuild this planning round with prior action history intact.
    assert requests[5]["input"][-1]["call_id"] == "5"
    assert request_payload == original
    assert len([event for event in events if event.get("status") == "issued"]) == 2
    recovery = [json.loads(path.read_text()) for path in tmp_path.rglob("*-llm-planning-recovery.json")]
    assert len(recovery) == 1 and recovery[0]["action_issued"] is False
