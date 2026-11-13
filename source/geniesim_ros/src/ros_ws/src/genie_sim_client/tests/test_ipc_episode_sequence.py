# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from collections import deque
from pathlib import Path
import pytest
from genie_sim_client.ipc import IPCClient, IPCClientConfig


def identity(epoch=0, tick=0):
    return dict(
        version=1,
        source_id="same-world",
        epoch=epoch,
        tick=tick,
        sim_time=tick / 60,
        joint_names=["arm"],
        reported_joint_names=["arm"],
        physics_hz=60,
        status="ready",
        ready=True,
    )


class Transport:
    def __init__(self):
        self.messages = deque([("identity", identity())])
        self.sent = []
        self.fail = False

    def start(self):
        pass

    def close(self):
        pass

    def next(self, timeout):
        if not self.messages:
            raise TimeoutError("test has no reply")
        return self.messages.popleft()

    def publish(self, kind, packet):
        self.sent.append((kind, packet))
        if self.fail:
            return
        new_identity = identity(packet["epoch"] + (packet["operation"] == "reset"))
        if packet["operation"] == "reset":
            self.messages.append(("identity", new_identity))
        self.messages.append(("episode", dict(packet, status="completed", identity=new_identity)))


def client():
    transport = Transport()
    config = IPCClientConfig(Path("scene"), Path("launcher"), "/action", "/obs", 60, 1, 2)
    return IPCClient(config, transport=transport).start(), transport


def test_begin_finish_reset_preserves_source_and_requires_committed_epoch():
    c, t = client()
    c.begin_episode()
    c.finish_episode()
    reply = c.reset(1)
    assert reply["identity"]["source_id"] == "same-world"
    assert c.identity["epoch"] == 1
    c.begin_episode()
    c.finish_episode()
    assert [p[1]["operation"] for p in t.sent] == ["begin", "finish", "reset", "begin", "finish"]
    with pytest.raises(ValueError):
        c.reset(2)
    assert len(t.sent) == 5


def test_unsolicited_epoch_change_fails_and_uncertain_reset_never_replays():
    c, t = client()
    t.messages.append(("identity", identity(1)))
    with pytest.raises(RuntimeError, match="epoch changed"):
        c.begin_episode()
    assert not t.sent
    c, t = client()
    c.begin_episode()
    c.finish_episode()
    t.fail = True
    with pytest.raises(TimeoutError):
        c.reset(1)
    with pytest.raises(RuntimeError, match="session failed"):
        c.reset(1)
    assert len(t.sent) == 3
