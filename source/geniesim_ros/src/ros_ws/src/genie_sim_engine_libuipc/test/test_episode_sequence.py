# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""CPU-only episode ownership checks; native recovery has a separate GPU probe."""

from types import SimpleNamespace
import pytest
from genie_sim_engine_libuipc.episode_sequence import EpisodeRequests


class Engine:
    _source_id = "owner"
    _epoch = 0
    _sequence_index = 0

    def __init__(self):
        self.calls = []

    def sequence_operation(self, operation, index):
        self.calls.append((operation, index))
        if operation == "reset":
            self._epoch += 1
            self._sequence_index = index
        return {"source_id": self._source_id, "epoch": self._epoch}


def request(operation="finish", index=0, epoch=0, id="one"):
    return dict(version=1, source_id="owner", epoch=epoch, request_id=id, operation=operation, index=index)


def test_callbacks_only_enqueue_and_duplicate_never_reexecutes():
    engine = Engine()
    queue = EpisodeRequests(engine)
    assert queue.request(request()) is None
    assert engine.calls == []
    response = queue.drain()[0]
    assert response["status"] == "completed"
    assert queue.request(request()) == response
    assert queue.drain() == []
    assert engine.calls == [("finish", 0)]


def test_epoch_change_requires_ack_and_stale_request_cannot_reset_twice():
    engine = Engine()
    queue = EpisodeRequests(engine)
    packet = request("reset", 1)
    queue.request(packet)
    reply = queue.drain()[0]
    assert reply["epoch"] == 0 and reply["identity"]["epoch"] == 1
    assert queue.request(packet) == reply
    assert queue.request(request("reset", 2, id="stale"))["status"] == "rejected"
    assert engine.calls == [("reset", 1)]


def test_changed_retry_payload_and_foreign_identity_rejected():
    queue = EpisodeRequests(Engine())
    queue.request(request())
    queue.drain()
    assert queue.request(request("begin"))["status"] == "rejected"
    packet = request(id="foreign")
    packet["source_id"] = "other"
    assert queue.request(packet)["status"] == "rejected"


def test_owner_failure_is_terminal_and_never_acknowledged_as_completed():
    engine = Engine()

    def fail(*args):
        raise RuntimeError("native recover failed")

    engine.sequence_operation = fail
    queue = EpisodeRequests(engine)
    queue.request(request("reset", 1))
    with pytest.raises(RuntimeError, match="native recover failed"):
        queue.drain()
    assert queue.request(request("reset", 1))["status"] == "failed"


def base_scene():
    first = dict(
        state_recording=dict(enabled=True, output_dir="archive-0"),
        material=dict(episode_plan={}, receipt_path="material-0"),
        robot_base_pose=dict(x=0.03, y=-0.02, z=0.0, theta=0.0),
    )
    second = dict(
        state_recording=dict(enabled=True, output_dir="archive-1"),
        material=dict(episode_plan={}, receipt_path="material-1"),
        robot_base_pose=dict(x=0.045, y=0.02, z=0.0, theta=0.0),
    )
    return dict(
        robot=dict(init_base_pose=first["robot_base_pose"].copy()),
        libuipc=dict(state_recording=first["state_recording"], episode_sequence=[first, second]),
        generalization=dict(material=first["material"]),
    )


def test_preloaded_xy_plans_keep_authored_values_and_return_owned_copy():
    from genie_sim_engine_libuipc.episode_sequence import sequence_entries

    scene = base_scene()
    result = sequence_entries(scene)
    assert result[1]["robot_base_pose"]["x"] == 0.045
    result[1]["robot_base_pose"]["x"] = 10
    assert scene["libuipc"]["episode_sequence"][1]["robot_base_pose"]["x"] == 0.045


@pytest.mark.parametrize("key,value", [("z", 0.1), ("theta", 0.1), ("x", float("nan")), ("y", True)])
def test_base_plan_rejects_unimplemented_axes_and_nonfinite_values(key, value):
    from genie_sim_engine_libuipc.episode_sequence import sequence_entries

    scene = base_scene()
    scene["libuipc"]["episode_sequence"][1]["robot_base_pose"][key] = value
    with pytest.raises(ValueError):
        sequence_entries(scene)


def test_base_plan_requires_complete_sequence_and_original_first_pose():
    from genie_sim_engine_libuipc.episode_sequence import sequence_entries

    scene = base_scene()
    del scene["libuipc"]["episode_sequence"][1]["robot_base_pose"]
    with pytest.raises(ValueError, match="every sequence entry"):
        sequence_entries(scene)
    scene = base_scene()
    scene["libuipc"]["episode_sequence"][0]["robot_base_pose"]["x"] += 0.01
    with pytest.raises(ValueError, match="initial robot base pose"):
        sequence_entries(scene)
