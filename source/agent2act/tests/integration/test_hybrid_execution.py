# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Real loopback OpenPI, offline Chat Completions, and the public CoRobot session."""

import json
import threading
from copy import deepcopy
from io import BytesIO

import numpy as np
import pytest
from openpi_client import msgpack_numpy
from PIL import Image
from websockets.sync.server import serve

from agent2act.adapters.planners.chat_completions import ChatCompletionsPlanner
from agent2act.bootstrap import ApplicationFactory, default_registry
from agent2act.configuration import parse_settings

pytestmark = pytest.mark.integration


def test_supervisor_policy_loop_and_review(config, request_payload, pi_config, tmp_path):
    buffer = BytesIO()
    Image.new("RGB", (32, 32), "red").save(buffer, "JPEG")
    request_payload["params"]["images"] = {"head": {"encoding": "JPEG", "image_data": buffer.getvalue()}}
    policy_inputs, model_calls = [], []
    decision = {
        "mode": "policy",
        "backend": "local",
        "instruction": "pick the test block",
        "reason": "use declared test skill",
        "policy_chunks": 2,
        "success_condition": "block visibly lifted",
        "motion": None,
    }
    complete = {
        "mode": "motion",
        "instruction": "review the scene",
        "reason": "offline test completion claim",
        "status": "complete",
        "motion": None,
    }

    def supervisor(body):
        model_calls.append(body)
        value = decision if len(model_calls) == 1 else complete
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}

    def handle(connection):
        connection.send(msgpack_numpy.Packer().pack({"checkpoint": "offline-test"}))
        for raw in connection:
            obs = msgpack_numpy.unpackb(raw)
            policy_inputs.append(obs)
            state = obs["state"].astype(float)
            actions = np.stack([state + [0.005 * i, 0, 0, 0] for i in range(1, 4)])
            connection.send(msgpack_numpy.Packer().pack({"actions": actions}))

    with serve(handle, "127.0.0.1", 0) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            config["execution"].update(
                mode="auto", allowed_modes=["motion", "policy"], policy_execute_steps=2, max_policy_chunks=2
            )
            config["policy"] = {
                "default": "local",
                "backends": {
                    "local": {
                        "type": "pi05",
                        **pi_config,
                        "port": server.socket.getsockname()[1],
                        "timeout_s": 2,
                    }
                },
            }
            config["planner"] = {"type": "offline"}
            config["logging"] = {"trace_dir": str(tmp_path)}
            registry = default_registry()
            registry.planners.register(
                "offline", lambda _: ChatCompletionsPlanner(model="test", post=supervisor)
            )
            with ApplicationFactory(parse_settings(config), registry).create_session() as session:
                first = session.handle(request_payload)["result"]["agent2act"]
                assert first["policy_run"]["issued_chunks"] == 1
                second_request = deepcopy(request_payload)
                second_request["params"]["states"]["arm_joint_states"][0] = 0.01
                second = session.handle(second_request)["result"]["agent2act"]
                assert second["policy_run"]["issued_chunks"] == 2
                assert len(model_calls) == 1
                third = session.handle(second_request)["result"]["agent2act"]
                assert third["status"] == "planner_complete_unverified"
                assert session.runtime.episode_key is not None
                assert session.runtime._policy_run is None
            assert len(policy_inputs) == len(model_calls) == 2
            assert [o["prompt"] for o in policy_inputs] == ["pick the test block"] * 2
            assert policy_inputs[1]["state"][0] == pytest.approx(0.01)
            assert policy_inputs[1]["image"].shape == (16, 16, 3)
            context = json.loads(model_calls[-1]["messages"][1]["content"][0]["text"])["context"]
            assert context["policy_review"]["issued_steps"] == 4
            assert context["policy_review"]["reason"] == "budget_exhausted"
            assert context["policy_review"]["task_success"] == "unverified"
            assert len(list(tmp_path.rglob("*-llm-request.json"))) == 2
            assert len(list(tmp_path.rglob("*-simulator-response.json"))) == 3
        finally:
            server.shutdown()
            worker.join(timeout=3)
