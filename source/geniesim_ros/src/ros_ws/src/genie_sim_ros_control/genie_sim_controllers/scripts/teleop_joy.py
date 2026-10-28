#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import rclpy
from rclpy.parameter import Parameter
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_msgs.msg import String


class TeleopJoyNode(Node):
    def __init__(self):
        super().__init__(
            "teleop_joy",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )

        self.max_linear_x = 1.0
        self.max_linear_y = 1.0
        self.max_angular_z = 0.5
        # Axes smaller than this are treated as zero after trigger
        # normalization so small controller noise at startup doesn't creep.
        self.axis_deadband = 0.05

        self.twist_cmd = Twist()

        self.pub_cmd_twist = self.create_publisher(Twist, "/cmd_twist", 1)
        self.pub_set_servo_mode = self.create_publisher(String, "/set_servo_mode", 1)
        self.sub_joy = self.create_subscription(Joy, "/joy", self.subscribe_joy, 1)
        self.sub_servo_mode = self.create_subscription(String, "/set_servo_mode", self.subscribe_servo_mode, 1)

        self.current_servo_mode = "OptimalServo"

        self.last_button_7 = False
        self.last_button_8 = False
        self.last_button_11 = False
        self.last_button_12 = False
        self.last_button_13 = False
        self.last_button_14 = False

        self.timer_c = self.create_timer(
            timer_period_sec=1 / 100.0,
            callback=self.timer_callback,
        )

        self.joy_msg = None

    def subscribe_joy(self, msg: Joy):
        self.joy_msg = msg

    def subscribe_servo_mode(self, msg: String):
        self.current_servo_mode = msg.data

    def _trigger_value(self, idx: int) -> float:
        # Negative-going trigger convention: idle sits at 0 (or +1 on some
        # controllers whose released state is +1), full press drives the axis
        # to -1. So the actionable magnitude is ``-raw`` clipped to [0, 1] --
        # released=0 -> 0, released=+1 -> 0, pressed=-1 -> 1, half-press -> 0.5.
        # Deadband filters low-magnitude noise at idle.
        if self.joy_msg is None or idx >= len(self.joy_msg.axes):
            return 0.0
        raw = float(self.joy_msg.axes[idx])
        value = max(0.0, min(1.0, -raw))
        return 0.0 if value < self.axis_deadband else value

    def timer_callback(self):
        if not self.joy_msg:
            return

        forward = self._trigger_value(5)
        backward = self._trigger_value(4)
        trigger_x = forward - backward
        button_yaw = self.joy_msg.buttons[9] - self.joy_msg.buttons[10]

        if self.current_servo_mode == "OptimalServo":
            self.twist_cmd.linear.x = self.max_linear_x * (self.joy_msg.axes[1] + trigger_x)
            self.twist_cmd.linear.y = self.max_linear_y * self.joy_msg.axes[0]
            self.twist_cmd.angular.z = self.max_angular_z * (self.joy_msg.axes[2] + button_yaw)
        else:
            self.twist_cmd.linear.x = self.max_linear_x * (
                trigger_x + abs(self.joy_msg.buttons[0]) - abs(self.joy_msg.buttons[1])
            )
            self.twist_cmd.linear.y = 0.0
            self.twist_cmd.angular.z = self.max_angular_z * (self.joy_msg.axes[2] + self.joy_msg.axes[0])

        self.pub_cmd_twist.publish(self.twist_cmd)

        mode_map = {
            7: "OptimalServo",
            8: "ParkingServo",
            11: "GeneralServo",
            12: "SpinServo",
            13: "SelftestServo",
        }
        last_buttons = {
            7: self.last_button_7,
            8: self.last_button_8,
            11: self.last_button_11,
            12: self.last_button_12,
            13: self.last_button_13,
            14: self.last_button_14,
        }
        for btn_idx, servo_mode in mode_map.items():
            current = bool(self.joy_msg.buttons[btn_idx])
            last = last_buttons[btn_idx]
            if current and not last:
                msg = String()
                msg.data = servo_mode
                self.pub_set_servo_mode.publish(msg)
                self.get_logger().info(f"Button {btn_idx} pressed, switching to servo mode: {servo_mode}")
        self.last_button_7 = bool(self.joy_msg.buttons[7])
        self.last_button_8 = bool(self.joy_msg.buttons[8])
        self.last_button_11 = bool(self.joy_msg.buttons[11])
        self.last_button_12 = bool(self.joy_msg.buttons[12])
        self.last_button_13 = bool(self.joy_msg.buttons[13])
        self.last_button_14 = bool(self.joy_msg.buttons[14])


def main():
    rclpy.init()

    node = TeleopJoyNode()

    try:
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # Guard against the ros2 launch SIGINT race: when ``ros2 launch`` forwards
        # Ctrl-C, rclpy's default signal handler may have already invoked
        # ``rcl_shutdown`` on this context. Calling ``rclpy.shutdown()`` a second
        # time raises ``RCLError: rcl_shutdown already called``. ``rclpy.ok()``
        # returns False once the context has been shut down, so we only call
        # shutdown when it's still needed.
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass


if __name__ == '__main__':
    main()
