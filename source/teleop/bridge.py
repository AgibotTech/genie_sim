#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
import time

import threading
import cv2
import os, sys
import argparse

from pprint import pprint

# pprint(sys.path)


from builtin_interfaces.msg import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import (
    JointState,
    Imu,
    Image,
    PointCloud,
    PointCloud2,
    ChannelFloat32,
)
from geometry_msgs.msg import Point32, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from std_msgs.msg import String

# Import geniesim_msg ROS2 message types
from geniesim_msg.msg import (
    GeniesimJointState,
    GeniesimJointCommand,
    GeniesimHeadState,
    GeniesimWaistState,
    GeniesimMotorState,
)

import rclpy
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)
from rclpy.node import Node
from cv_bridge import CvBridge
from std_msgs.msg import Bool


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


class BridgeBase(Node):
    def __init__(self, node_name):
        super().__init__(node_name)

        self.node_name = node_name

    def run(self):
        pass


class BridgeSender(BridgeBase):
    def __init__(self, args, node_name="bridge_sender"):
        super().__init__(node_name=node_name)

        self.bridge = CvBridge()

        self.sub_arm_r_hal = self.create_subscription(
            JointState,
            "/joint_states",
            self._callback_hal,
            QOS_PROFILE_VOLATILE,
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

    def _callback_hal(self, msg):
        # Convert JointState to geniesim_msg and publish
        self._publish_hal(msg)

    def _callback_joint_cmd(self, msg: GeniesimJointCommand):
        if self.playback_flag:
            return
        js = JointState()
        js.header = msg.header
        js.name = list(msg.name)
        js.position = list(msg.position)
        js.velocity = list(msg.velocity)
        js.effort = list(msg.effort)
        self.pub_joint_command.publish(js)

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


class BridgeReceiver(BridgeBase):
    def __init__(self, node_name="bridge_receiver"):
        super().__init__(node_name=node_name)


def run_demo(args):
    global running

    rclpy.init()

    sender, receiver = BridgeSender(args), BridgeReceiver()

    def receiver_thread():
        receiver.run()

    try:
        # start subscriber thread
        sub_thread = threading.Thread(target=receiver_thread)
        sub_thread.daemon = True

        # init receiver
        sub_thread.start()
        time.sleep(1)

        # init sender
        rclpy.spin(sender)

    except KeyboardInterrupt:
        print("\nStop signal received, shutting down...")
        running = False
        # sub_thread.join()

    try:
        rclpy.shutdown()
    except Exception as e:
        if "already called" not in str(e):
            print(f"Shutdown warning: {e}")


# exit()

running = True


def main():
    parser = argparse.ArgumentParser(description="Image pub/sub test program")
    parser.add_argument(
        "--mode",
        choices=["inprocess"],
        help="Run mode: inprocess for in-process mode",
        default="inprocess",
    )
    args = parser.parse_args()

    print("Starting in-process pub/sub mode...")
    run_demo(args)


if __name__ == "__main__":
    main()
