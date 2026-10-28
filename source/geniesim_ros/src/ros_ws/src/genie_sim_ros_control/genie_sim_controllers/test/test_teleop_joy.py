# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Exercise trigger commands through the actual ROS joystick node."""

import importlib.util
from pathlib import Path

import pytest
import rclpy
from sensor_msgs.msg import Joy

_spec = importlib.util.spec_from_file_location(
    'teleop_joy', Path(__file__).resolve().parents[1] / 'scripts' / 'teleop_joy.py'
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


@pytest.fixture
def node():
    rclpy.init()
    instance = _module.TeleopJoyNode()
    yield instance
    instance.destroy_node()
    rclpy.try_shutdown()


@pytest.mark.parametrize('mode', ['OptimalServo', 'GeneralServo'])
@pytest.mark.parametrize('idle', [0.0, 1.0, -0.01])
def test_released_triggers_do_not_move_chassis(node, mode, idle):
    node.current_servo_mode = mode
    node.subscribe_joy(Joy(axes=[0.0, 0.0, 0.0, 0.0, idle, idle], buttons=[0] * 15))
    node.timer_callback()
    assert node.twist_cmd.linear.x == 0.0
    assert node.twist_cmd.linear.y == 0.0
    assert node.twist_cmd.angular.z == 0.0


@pytest.mark.parametrize('mode', ['OptimalServo', 'GeneralServo'])
@pytest.mark.parametrize('axis,expected', [(5, 0.5), (4, -0.5)])
def test_trigger_direction_and_magnitude(node, mode, axis, expected):
    node.current_servo_mode = mode
    axes = [0.0] * 6
    axes[axis] = -0.5
    node.subscribe_joy(Joy(axes=axes, buttons=[0] * 15))
    node.timer_callback()
    assert node.twist_cmd.linear.x == pytest.approx(expected)


def test_trigger_noise_on_one_side_does_not_creep(node):
    node.subscribe_joy(Joy(axes=[0.0, 0.0, 0.0, 0.0, 0.0, -0.01], buttons=[0] * 15))
    node.timer_callback()
    assert node.twist_cmd.linear.x == 0.0
