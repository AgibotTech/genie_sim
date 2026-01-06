#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""
Read observation.state from parquet file and send ROS JointState messages frame by frame to /joint_command
"""
import sys
import os
import time
import argparse
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError:
    print("Error: pyarrow library is required")
    print("Please run: pip install pyarrow")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    np = None

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
except ImportError:
    print("Error: ROS2 related libraries are required")
    print("Please ensure rclpy and sensor_msgs are installed")
    sys.exit(1)


# Joint name list (refer to cmd_msg.name in genie_sim_ros.py)
JOINT_NAMES = [
    "idx21_arm_l_joint1",
    "idx22_arm_l_joint2",
    "idx23_arm_l_joint3",
    "idx24_arm_l_joint4",
    "idx25_arm_l_joint5",
    "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
    "idx41_gripper_l_outer_joint1",
    "idx81_gripper_r_outer_joint1",
]

# Gripper joint indices in the positions list
GRIPPER_L_INDEX = 14  # idx41_gripper_l_outer_joint1
GRIPPER_R_INDEX = 15  # idx81_gripper_r_outer_joint1


class JointCommandPublisher(Node):
    """ROS2 node for publishing joint commands"""

    def __init__(self):
        super().__init__("joint_command_publisher")

        # Create publisher for /joint_command topic
        self.publisher = self.create_publisher(JointState, "/joint_command", 10)

        self.get_logger().info(f"Publisher created: /joint_command")

    def publish_joint_state(self, positions, frame_id="base_link"):
        """
        Publish joint state

        Args:
            positions: Joint position list (16 values)
            frame_id: Frame ID
        """
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.name = JOINT_NAMES
        msg.position = list(positions)

        self.publisher.publish(msg)
        self.get_logger().debug(f"Published joint state: {len(positions)} joints")


def load_parquet_data(file_path):
    """
    Load data from parquet file

    Returns:
        observation_states: observation.state data list
        timestamps: Timestamp list
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")

    print(f"Reading file: {file_path}")

    # Read parquet file
    table = pq.read_table(file_path)

    # Convert to pandas DataFrame (if available) or use pyarrow
    try:
        import pandas as pd

        df = table.to_pandas()
        # Get observation.state column, which may be list or numpy.ndarray
        obs_col = df["observation.state"]
        observation_states = []
        for val in obs_col:
            # If already a list, use directly; if numpy.ndarray, convert to list
            if isinstance(val, list):
                observation_states.append(val)
            elif np is not None and isinstance(val, np.ndarray):
                observation_states.append(val)
            else:
                # Try to convert to list
                observation_states.append(list(val) if hasattr(val, "__iter__") else [val])

        timestamps = df["timestamp"].tolist() if "timestamp" in df.columns else None
    except ImportError:
        # If pandas is not available, read directly from pyarrow table
        obs_col = table["observation.state"]
        observation_states = []
        for i in range(len(table)):
            chunk = obs_col.chunk(0) if obs_col.num_chunks > 0 else None
            if chunk is not None:
                values = chunk.to_pylist()
                if i < len(values):
                    observation_states.append(values[i])

        timestamp_col = table["timestamp"] if "timestamp" in table.column_names else None
        timestamps = None
        if timestamp_col is not None:
            timestamps = []
            for i in range(len(table)):
                chunk = timestamp_col.chunk(0) if timestamp_col.num_chunks > 0 else None
                if chunk is not None:
                    values = chunk.to_pylist()
                    if i < len(values):
                        timestamps.append(values[i])

    print(f"Successfully loaded {len(observation_states)} frames of data")

    return observation_states, timestamps


def main():
    parser = argparse.ArgumentParser(
        description="Read observation.state from parquet file and send ROS JointState messages"
    )
    parser.add_argument(
        "parquet_file",
        type=str,
        nargs="?",
        default="episode_000000.parquet",
        help="Parquet file path (default: episode_000000.parquet)",
    )
    parser.add_argument("--rate", type=float, default=30.0, help="Publish frequency (Hz, default: 30.0)")
    parser.add_argument(
        "--use-timestamps", action="store_true", help="Use timestamps from parquet file to control publish rate"
    )
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index (default: 0)")
    parser.add_argument("--end-frame", type=int, default=None, help="End frame index (default: to end of file)")

    args = parser.parse_args()

    # Initialize ROS2
    rclpy.init()

    try:
        # Create node
        node = JointCommandPublisher()

        # Load data
        observation_states, timestamps = load_parquet_data(args.parquet_file)

        # Check data dimensions
        if len(observation_states) == 0:
            node.get_logger().error("No data to play")
            return

        # Check observation.state dimensions
        first_state = observation_states[0]

        # Support list and numpy.ndarray types
        if isinstance(first_state, list):
            state_dim = len(first_state)
        elif np is not None and isinstance(first_state, np.ndarray):
            state_dim = first_state.shape[0] if len(first_state.shape) > 0 else 0
        else:
            node.get_logger().error(f"observation.state format error: {type(first_state)}")
            node.get_logger().error("Supported types: list or numpy.ndarray")
            return

        node.get_logger().info(f"observation.state dimension: {state_dim}")

        if state_dim < len(JOINT_NAMES):
            node.get_logger().warn(
                f"Warning: observation.state dimension ({state_dim}) is less than joint count ({len(JOINT_NAMES)})"
            )
            node.get_logger().warn(f"Will use first {len(JOINT_NAMES)} dimensions")
        elif state_dim > len(JOINT_NAMES):
            node.get_logger().info(
                f"observation.state dimension ({state_dim}) is greater than joint count ({len(JOINT_NAMES)}), will use first {len(JOINT_NAMES)} dimensions"
            )

        # Determine frame range
        start_frame = args.start_frame
        end_frame = args.end_frame if args.end_frame is not None else len(observation_states)
        end_frame = min(end_frame, len(observation_states))

        if start_frame >= end_frame:
            node.get_logger().error(f"Invalid frame range: {start_frame} to {end_frame}")
            return

        node.get_logger().info(
            f"Will play frames {start_frame} to {end_frame-1} (total {end_frame-start_frame} frames)"
        )

        # Calculate publish interval
        if args.use_timestamps and timestamps:
            node.get_logger().info("Using timestamps from file to control publish rate")
        else:
            period = 1.0 / args.rate
            node.get_logger().info(f"Using fixed frequency: {args.rate} Hz (period: {period:.4f} seconds)")

        # Publish loop
        loop_count = 0
        while True:
            loop_count += 1
            if loop_count > 1:
                node.get_logger().info(f"Starting loop {loop_count}")

            for frame_idx in range(start_frame, end_frame):
                # Get current frame's observation.state
                state = observation_states[frame_idx]

                # Extract joint positions (use first len(JOINT_NAMES) dimensions)
                # Support list and numpy.ndarray types
                if isinstance(state, list):
                    positions = state[: len(JOINT_NAMES)].copy()
                elif np is not None and isinstance(state, np.ndarray):
                    positions = state[: len(JOINT_NAMES)].tolist()
                else:
                    node.get_logger().error(f"Frame {frame_idx} data format error: {type(state)}")
                    continue

                # Special handling for gripper joints
                # Left gripper: normalize to [0, 1], then transform to 1-v
                if GRIPPER_L_INDEX < len(positions):
                    # Use numpy clip for normalization
                    normalized = np.clip(1 - (positions[GRIPPER_L_INDEX] - 30) / 90, 0.0, 1.0)
                    positions[GRIPPER_L_INDEX] = float(normalized)

                # Right gripper: normalize to [0, 1], then transform to 1-v
                if GRIPPER_R_INDEX < len(positions):
                    # Use numpy clip for normalization
                    normalized = np.clip(1 - (positions[GRIPPER_R_INDEX] - 30) / 90, 0.0, 1.0)
                    print("bef normalized", positions[GRIPPER_R_INDEX])
                    print("normalized", normalized)
                    positions[GRIPPER_R_INDEX] = float(normalized)

                # Publish joint state
                node.publish_joint_state(positions)

                # Control publish frequency
                if args.use_timestamps and timestamps:
                    if frame_idx < len(timestamps) - 1:
                        # Calculate time difference
                        dt = timestamps[frame_idx + 1] - timestamps[frame_idx]
                        time.sleep(dt)
                else:
                    time.sleep(period)

                # Print progress every 30 frames
                if (frame_idx - start_frame) % 30 == 0:
                    node.get_logger().info(f"Published frame {frame_idx}/{end_frame-1}")

            node.get_logger().info(f"Playback completed (total {end_frame-start_frame} frames)")

            if not args.loop:
                break

            # Brief pause when looping
            time.sleep(0.5)

    except KeyboardInterrupt:
        node.get_logger().info("Received interrupt signal, exiting...")
    except Exception as e:
        node.get_logger().error(f"Error occurred: {str(e)}")
        import traceback

        traceback.print_exc()
    finally:
        # Cleanup
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
