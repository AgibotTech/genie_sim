# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json
import logging

from agent2act.service.action_trace import action_log_path
from agent2act.service.trace import RequestTrace


def test_live_action_trace_flushes_public_evidence_and_preserves_observation_link(tmp_path):
    root = tmp_path / "agent.trace"
    output = action_log_path(root)
    first = RequestTrace(root / "session" / "first", action_log=output)
    first("simulator-request", {"params": {"timestamps": {"head": 123}, "prompt": "pick"}})
    assert len(output.read_text().splitlines()) == 1  # Available before inference completes.
    first("runtime-event", {"status": "phase_started", "sequence": 1, "attempt": 0})
    first("llm-request", {"model": "requested", "input": "not copied"})
    first(
        "llm-response",
        {
            "model": "reported",
            "output": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Public summary"}],
                    "encrypted_content": "PRIVATE_CONTENT",
                }
            ],
        },
    )
    first("runtime-event", {"status": "issued", "sequence": 1, "reason": "Approach", "issued_steps": 10})
    second = RequestTrace(root / "session" / "second", action_log=output)
    second("runtime-event", {"status": "observed_tracking", "sequence": 2, "previous_sequence": 1})
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 5
    assert rows[0]["data"]["observation_timestamps"] == {"head": 123}
    assert rows[0]["data"]["video_frame_index"] is None
    assert rows[2]["data"]["model_identity_status"] == "mismatch"
    assert rows[2]["data"]["reasoning_summary"] == ["Public summary"]
    assert rows[2]["sequence"] == 1 and rows[2]["attempt"] == 0
    assert rows[3]["data"]["reason"] == "Approach"
    assert rows[-1]["inference_id"] == "second"
    assert rows[-1]["data"]["previous_sequence"] == 1
    assert all(row["timestamp"] and row["monotonic_ns"] and row["artifact_path"] for row in rows)
    assert "PRIVATE_CONTENT" not in output.read_text()


def test_action_trace_write_failure_does_not_fail_request(tmp_path, caplog):
    output = tmp_path / "directory"
    output.mkdir()
    trace = RequestTrace(tmp_path / "trace" / "session" / "request", action_log=output)
    with caplog.at_level(logging.WARNING):
        trace("runtime-event", {"status": "issued", "sequence": 1})
    assert (trace.directory / "001-runtime-event.json").exists()
    assert "action_trace_failed" in caplog.text


def test_diagnostic_events_are_immediately_available_with_request_identity(tmp_path):
    output = tmp_path / "events.jsonl"
    trace = RequestTrace(tmp_path / "trace" / "session" / "request", action_log=output)
    trace("runtime-event", {"status": "phase_started", "sequence": 9, "attempt": 1})
    for kind in ("diagnostic-input", "decision-diagnostics", "reasoning-summary"):
        trace(kind, {"status": "available", "source": "fixture"})
        row = json.loads(output.read_text().splitlines()[-1])
        assert row["kind"] == kind
        assert row["sequence"] == 9 and row["attempt"] == 1
        assert row["inference_id"] == "request"
