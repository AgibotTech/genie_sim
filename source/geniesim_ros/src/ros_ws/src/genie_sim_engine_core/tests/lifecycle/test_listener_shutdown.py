# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Context shutdown ends the listener quietly; unexpected failures stay visible."""

import sys
import threading
from types import SimpleNamespace

import pytest

from genie_sim_engine_core.common.lifecycle_ros import RosResetCoordinator


@pytest.mark.parametrize("expected_shutdown", [True, False])
def test_listener_distinguishes_context_shutdown_from_executor_failure(monkeypatch, expected_shutdown):
    class ContextShutdown(Exception):
        pass

    failure = ContextShutdown() if expected_shutdown else RuntimeError("executor failed")
    errors = []
    monkeypatch.setattr(threading, "excepthook", lambda args: errors.append(args.exc_value))

    class Executor:
        def add_node(self, node):
            pass

        def spin(self):
            raise failure

    fake_node = SimpleNamespace(create_subscription=lambda *args: None)
    modules = {
        "rclpy": SimpleNamespace(ok=lambda: True),
        "rclpy.executors": SimpleNamespace(SingleThreadedExecutor=Executor, ExternalShutdownException=ContextShutdown),
        "rclpy.node": SimpleNamespace(Node=lambda name: fake_node),
        "rclpy.qos": SimpleNamespace(
            HistoryPolicy=SimpleNamespace(KEEP_LAST=1),
            ReliabilityPolicy=SimpleNamespace(RELIABLE=1, BEST_EFFORT=2),
            QoSProfile=lambda **kwargs: kwargs,
        ),
        "std_msgs.msg": SimpleNamespace(Int32MultiArray=object, Int64MultiArray=object),
        "sensor_msgs.msg": SimpleNamespace(JointState=object),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    warnings = []
    coordinator = RosResetCoordinator(None, SimpleNamespace(info=lambda *_: None, warn=warnings.append))
    coordinator.start_listener()
    assert coordinator._spin_thread is not None, warnings
    coordinator._spin_thread.join(timeout=2)
    assert not coordinator._spin_thread.is_alive()
    assert warnings == []
    assert errors == ([] if expected_shutdown else [failure])
