#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import argparse
import time

import rclpy
from geniesim_msg.msg import GeniesimJointCommand, GeniesimJointState
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

from geniesim_teleop.utils.chassis_command import ChassisCommandAdapter
from geniesim_teleop.utils.joint_command_mux import JointCommandMux

QOS_PROFILE_LATEST = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)

QOS_PROFILE_VOLATILE = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
)

# BEST_EFFORT accepts the engine's SensorDataQoS joint-state publisher.
QOS_PROFILE_SENSOR = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=5,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
)


class BridgeSender(Node):
    def __init__(self, args, node_name="bridge_sender"):
        super().__init__(node_name)

        self.command_mux = JointCommandMux()
        self.chassis_adapter = ChassisCommandAdapter()
        self.pub_chassis_state = self.create_publisher(JointState, "/teleop/chassis_state", QOS_PROFILE_SENSOR)

        self.sub_arm_r_hal = self.create_subscription(
            JointState,
            "/joint_states",
            self._callback_hal,
            QOS_PROFILE_SENSOR,
        )
        self.sub_joint_cmd = self.create_subscription(
            GeniesimJointCommand,
            "/hal/joint_cmd",
            self._callback_joint_cmd,
            QOS_PROFILE_VOLATILE,
        )
        self.sub_playback = self.create_subscription(
            Bool,
            "/sim/playback_flag",
            self._callback_playback,
            QOS_PROFILE_VOLATILE,
        )

        self.playback_flag = False
        self.pub_js_hal = self.create_publisher(GeniesimJointState, "/hal/joint_state", QOS_PROFILE_VOLATILE)
        self.pub_joint_command = self.create_publisher(JointState, "/joint_command", QOS_PROFILE_VOLATILE)
        self.sub_gripper_cmd = self.create_subscription(
            JointState, "/teleop/gripper_command", lambda msg: self._cache_command("gripper", msg), QOS_PROFILE_VOLATILE
        )
        self.sub_chassis_cmd = self.create_subscription(
            JointState, "/cmd_4ws", self._callback_chassis, QOS_PROFILE_VOLATILE
        )
        self.sub_chassis_twist = self.create_subscription(
            Twist, "/cmd_twist", self._callback_chassis_twist, QOS_PROFILE_VOLATILE
        )
        self.command_timer = self.create_timer(0.01, self._publish_command)

    def _callback_hal(self, msg):
        try:
            self.chassis_adapter.update_feedback(msg.name, msg.position, msg.velocity, time.monotonic())
        except ValueError:
            pass  # Partial upper-body feedback must not refresh the chassis watchdog.
        else:
            self.pub_chassis_state.publish(msg)
        self._publish_hal(msg)

    def _callback_chassis(self, msg):
        try:
            self.chassis_adapter.update_command(msg.name, msg.position, msg.velocity, time.monotonic())
        except ValueError as exc:
            self.get_logger().error(f"Rejected chassis command: {exc}")

    def _callback_chassis_twist(self, msg):
        self.chassis_adapter.update_twist(msg.linear.x, msg.linear.y, msg.angular.z, time.monotonic())

    def _callback_joint_cmd(self, msg: GeniesimJointCommand):
        if self.playback_flag:
            return
        js = JointState()
        js.header = msg.header
        js.name = list(msg.name)
        js.position = list(msg.position)
        js.velocity = list(msg.velocity)
        js.effort = list(msg.effort)
        self._cache_command("mc", js)

    def _cache_command(self, source, msg):
        try:
            self.command_mux.update(source, msg.name, msg.position, msg.velocity, msg.effort, time.monotonic())
        except ValueError as exc:
            self.get_logger().error(f"Rejected {source} joint command: {exc}")

    def _publish_command(self):
        now = time.monotonic()
        chassis = self.chassis_adapter.snapshot(now, self.playback_flag)
        if chassis is not None:
            self.command_mux.update("chassis", *chassis, now)
        msg = JointState()
        msg.name, msg.position, msg.velocity, msg.effort = self.command_mux.snapshot(now, self.playback_flag)
        if msg.name:
            msg.header.stamp = self.get_clock().now().to_msg()
            self.pub_joint_command.publish(msg)

    def _callback_playback(self, msg):
        self.playback_flag = msg.data

    def _publish_hal(self, msg):
        armmsgl = GeniesimJointState()
        armmsgl.header = msg.header

        jsmsg = GeniesimJointState()
        jsmsg.header = msg.header
        velocity_thresh = 0.1
        body_tuples = []
        for i in range(len(msg.name)):
            if "body" in msg.name[i]:
                vel = 0.0 if abs(msg.velocity[i]) < velocity_thresh else msg.velocity[i]
                body_tuples.append(
                    (
                        msg.name[i],
                        msg.position[i],
                        vel,
                        msg.effort[i],
                    )
                )
        for name, pos, vel, eff in sorted(body_tuples, key=lambda t: t[0]):
            jsmsg.name.append(name)
            jsmsg.motor_position.append(pos)
            jsmsg.motor_velocity.append(vel)
            jsmsg.effort.append(eff)
            jsmsg.error_code.append(0)
        for i in range(len(msg.name)):
            if "head" in msg.name[i]:
                vel = 0.0 if abs(msg.velocity[i]) < velocity_thresh else msg.velocity[i]
                jsmsg.name.append(msg.name[i])
                jsmsg.motor_position.append(msg.position[i])
                jsmsg.motor_velocity.append(vel)
                jsmsg.effort.append(msg.effort[i])
                jsmsg.error_code.append(0)
        for i in range(len(msg.name)):
            if "arm_l" in msg.name[i]:
                vel = 0.0 if abs(msg.velocity[i]) < velocity_thresh else msg.velocity[i]
                jsmsg.name.append(msg.name[i])
                jsmsg.motor_position.append(msg.position[i])
                jsmsg.motor_velocity.append(vel)
                jsmsg.effort.append(msg.effort[i])
                jsmsg.error_code.append(0)
        for i in range(len(msg.name)):
            if "arm_r" in msg.name[i]:
                vel = 0.0 if abs(msg.velocity[i]) < velocity_thresh else msg.velocity[i]
                jsmsg.name.append(msg.name[i])
                jsmsg.motor_position.append(msg.position[i])
                jsmsg.motor_velocity.append(vel)
                jsmsg.effort.append(msg.effort[i])
                jsmsg.error_code.append(0)
        self.pub_js_hal.publish(jsmsg)


def run_bridge(args):
    rclpy.init()
    sender = None
    try:
        sender = BridgeSender(args)
        rclpy.spin(sender)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if sender is not None:
                sender.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Merge teleop commands and forward robot joint feedback")
    parser.add_argument(
        "--mode",
        choices=["inprocess"],
        help="Run mode: inprocess for in-process mode",
        default="inprocess",
    )
    args = parser.parse_args()

    print("Starting in-process pub/sub mode...")
    run_bridge(args)


if __name__ == "__main__":
    main()
