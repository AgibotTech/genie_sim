# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Opt-in real DDS transport smoke, using a fake engine without Newton/GPU.

Run after sourcing ROS:
  python3 ros_transport_smoke.py --domain-id 173
The caller must supply an unused domain. This is not a Newton physics test.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from genie_sim_client import ClientConfig, SimulationClient


def run(domain_id: int):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from tf2_msgs.msg import TFMessage
    from geometry_msgs.msg import TransformStamped
    from std_msgs.msg import Int32MultiArray, Int64MultiArray

    rclpy.init(domain_id=domain_id)
    node = rclpy.create_node("genie_sim_client_smoke_engine")
    joints = node.create_publisher(JointState, "joint_states", qos_profile_sensor_data)
    bodies = node.create_publisher(TFMessage, "tf_render", 10)
    targets = [0.0, 0.0]
    epoch = 0
    engaged = False
    seeded_requests = []

    def publish():
        stamp = node.get_clock().now().to_msg()
        js = JointState()
        js.header.stamp = stamp
        js.header.frame_id = f"reset_epoch:{epoch}"
        js.name = ["joint_a", "joint_b"]
        js.position = list(targets)
        js.velocity = [0.0, 0.0]
        joints.publish(js)
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = "world"
        transform.child_frame_id = "/World/body"
        transform.transform.rotation.w = 1.0
        message = TFMessage()
        message.transforms = [transform]
        bodies.publish(message)

    def command(message):
        nonlocal targets, engaged
        engaged = True
        targets = list(message.position)
        publish()

    def reset(message):
        nonlocal targets, epoch
        targets = [0.0, 0.0]
        epoch += 1
        if isinstance(message, Int64MultiArray):
            seeded_requests.append(list(message.data))
        publish()

    node.create_subscription(JointState, "joint_command", command, 1)
    node.create_subscription(Int32MultiArray, "/rlinf/reset_envs", reset, 4)
    node.create_subscription(Int64MultiArray, "/rlinf/reset_envs_seeded", reset, 10)
    node.create_timer(0.05, lambda: None if engaged else publish())
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    try:
        with tempfile.TemporaryDirectory(prefix="genie_client_ros_smoke_") as directory:
            temporary = Path(directory)
            (temporary / "scene.yaml").write_text("robot: {}\n")
            manifest = temporary / "manifest.json"
            manifest.write_text(json.dumps({"scene_yaml": str(temporary / "scene.yaml")}))
            config = ClientConfig.from_files(
                Path(__file__).resolve().parents[1] / "config/launcher_client_newton.yaml", manifest
            )
            with SimulationClient(config, spawn=False, ready_timeout_s=10, step_timeout_s=5) as client:
                result = client.step(np.array([[0.25, -0.5]], dtype=np.float32))
                np.testing.assert_array_equal(result.snapshot.joint_positions, [[0.25, -0.5]])
                assert result.snapshot.body_stamp >= result.snapshot.joint_stamp
                snapshot = client.reset([0], seeds=[42], instances=[3])
                assert snapshot.reset_epoch == 1
                assert seeded_requests == [[0, 42, 3]]
                np.testing.assert_array_equal(snapshot.joint_positions, [[0, 0]])
            assert rclpy.ok(), "client closed the shared ROS context"
        print("real ROS transport smoke passed: action, body freshness, seeded reset, shared context")
    finally:
        executor.shutdown()
        spinner.join(timeout=2)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-id", type=int, required=True)
    run(parser.parse_args().domain_id)
