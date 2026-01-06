# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import omni.graph.core as og
from isaacsim.core.nodes.scripts.utils import set_target_prims

from common.base_utils.logger import logger
from common.base_utils.ros_nodes.sim_ros_node import JointStatePubRosNode


class USDBase:
    def __init__(self):
        pass

    def _init_sensor(self, ros_domain_id):
        self.ros_domain_id = (int)(ros_domain_id)
        logger.info(f"ROS domain ID: {self.ros_domain_id}")

    def step(self):
        pass

    def init_all_sensors(self):
        if "sensors" in self.config:
            for sensor in self.config["sensors"]:
                if sensor["type"] == "Camera":
                    self._init_camera(sensor)
                if sensor["type"] == "Lidar":
                    self._init_lidar(sensor)
                if sensor["type"] == "IMU":
                    self._init_imu(sensor)

    def _init_lidar(self, param):
        from omni.isaac.sensor import LidarRtx

        from .lidar import publish_lidar_pointcloud, publish_lidar_scan

        lidar = LidarRtx(prim_path=param["path"])
        lidar.initialize()
        approx_freq = param["frequency"]
        for publish in param["publish"]:
            if publish is None:
                continue
            split = publish.split(":", 2)
            topic = ""
            if len(split) > 1:
                publish = split[0]
                topic = split[1]
            if publish == "pointcloud":
                publish_lidar_pointcloud(lidar, approx_freq, topic)
            elif publish == "scan":
                publish_lidar_scan(lidar, approx_freq, topic)

    def _init_camera(self, param):
        from isaacsim.sensors.camera import Camera

        from .camera import (
            publish_boundingbox2d_loose,
            publish_boundingbox2d_tight,
            publish_boundingbox3d,
            publish_camera_info,
            publish_depth,
            publish_noised_rgb,
            publish_pointcloud_from_depth,
            publish_rgb,
            publish_semantic_segment,
        )

        camera = Camera(
            prim_path=param["path"],
            frequency=param["frequency"],
            resolution=(param["resolution"]["width"], param["resolution"]["height"]),
        )
        camera.initialize()

        step_size = param["frequency"]
        camera_graph = []
        ros_nodes = []
        for publish in param["publish"]:
            if publish is None:
                continue
            split = publish.split(":", 2)
            topic = ""
            if len(split) > 1:
                publish = split[0]
                topic = split[1]
            if publish == "rgb":
                if not param.get("noised", False):
                    camera_graph.append(publish_rgb(camera, step_size, ""))
                else:
                    ros_nodes.append(
                        publish_noised_rgb(
                            camera=camera,
                            step_size=step_size,
                            topic="",
                            **param["noise_parameters"],
                        )
                    )
            elif publish == "info":
                camera_graph.append(publish_camera_info(camera, step_size, topic))
            elif publish == "pointcloud":
                publish_pointcloud_from_depth(camera, step_size, topic)
            elif publish == "depth":
                camera_graph.append(publish_depth(camera, step_size, ""))
            elif publish == "bbox2_loose":
                publish_boundingbox2d_loose(camera, step_size, topic)
            elif publish == "bbox2_tight":
                publish_boundingbox2d_tight(camera, step_size, topic)
            elif publish == "bbox3":
                publish_boundingbox3d(camera, step_size, topic)
            elif publish == "semantic":
                publish_semantic_segment(camera, step_size, topic)

        # Note: camera.initialize() already called above, no need to call again
        return camera_graph, ros_nodes

    def _init_imu(self, param):
        from omni.isaac.sensor import IMUSensor

        from .imu import publish_imu

        imu = IMUSensor(prim_path=param["path"])
        # imu.initialize()
        approx_freq = param["frequency"]
        for publish in param["publish"]:
            if publish is None:
                continue
            split = publish.split(":", 2)
            topic = ""
            if len(split) > 1:
                publish = split[0]
                topic = split[1]
            else:
                topic = param["path"] + "_imu"
            if publish == "imu":
                publish_imu(imu, approx_freq, topic)

    def reset_graph(self):
        ros_tf_graph_path = "/World/RobotTFActionGraph"

        set_target_prims(
            primPath=ros_tf_graph_path + "/RosPublishTransformTree",
            inputName="inputs:targetPrims",
            targetPrimPaths=[],
        )

    def publish_tf(self, robot_prim, targets, approx_freq, delta_time):
        ros_tf_graph_path = "/World/RobotTFActionGraph"
        (int)(approx_freq)
        og.Controller.edit(
            {
                "graph_path": ros_tf_graph_path,
                "evaluator_name": "execution",
                # "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            },
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("IsaacSimulationGate", "isaacsim.core.nodes.IsaacSimulationGate"),
                    (
                        "RosPublishTransformTree",
                        "isaacsim.ros2.bridge.ROS2PublishTransformTree",
                    ),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                    ("RosContext", "isaacsim.ros2.bridge.ROS2Context"),
                ],
                og.Controller.Keys.CONNECT: [
                    (
                        "OnPlaybackTick.outputs:tick",
                        "IsaacSimulationGate.inputs:execIn",
                    ),
                    (
                        "IsaacSimulationGate.outputs:execOut",
                        "RosPublishTransformTree.inputs:execIn",
                    ),
                    (
                        "RosContext.outputs:context",
                        "RosPublishTransformTree.inputs:context",
                    ),
                    (
                        "ReadSimTime.outputs:simulationTime",
                        "RosPublishTransformTree.inputs:timeStamp",
                    ),
                    (
                        "IsaacSimulationGate.outputs:execOut",
                        "PublishJointState.inputs:execIn",
                    ),
                    (
                        "ReadSimTime.outputs:simulationTime",
                        "PublishJointState.inputs:timeStamp",
                    ),
                    ("RosContext.outputs:context", "PublishJointState.inputs:context"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("OnPlaybackTick.outputs:deltaSeconds", delta_time),
                    ("PublishJointState.inputs:targetPrim", robot_prim),
                    # ("RosPublishTransformTree.inputs:parentPrim", robot_prim),
                    # ("IsaacSimulationGate.inputs:step", step),
                ],
            },
        )
        set_target_prims(
            primPath=ros_tf_graph_path + "/RosPublishTransformTree",
            inputName="inputs:targetPrims",
            targetPrimPaths=targets,
        )

    def publish_joint(self, robot_prim, approx_freq, delta_time, topic_name="/joint_state"):
        step = (int)(approx_freq)
        og.Controller.edit(
            {
                "graph_path": "/World/RobotJointActionGraph",
                "evaluator_name": "execution",
            },
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("IsaacSimulationGate", "isaacsim.core.nodes.IsaacSimulationGate"),
                    ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("RosContext", "isaacsim.ros2.bridge.ROS2Context"),
                ],
                og.Controller.Keys.CONNECT: [
                    (
                        "OnPlaybackTick.outputs:tick",
                        "IsaacSimulationGate.inputs:execIn",
                    ),
                    (
                        "IsaacSimulationGate.outputs:execOut",
                        "PublishJointState.inputs:execIn",
                    ),
                    (
                        "ReadSimTime.outputs:simulationTime",
                        "PublishJointState.inputs:timeStamp",
                    ),
                    ("RosContext.outputs:context", "PublishJointState.inputs:context"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("OnPlaybackTick.outputs:deltaSeconds", delta_time),
                    ("PublishJointState.inputs:targetPrim", robot_prim),
                    ("IsaacSimulationGate.inputs:step", step),
                    ("PublishJointState.inputs:topicName", topic_name),
                ],
            },
        )

    def publish_articulation_action(self, robot, step_size, topic_name="/articulation_action"):
        frame_id = topic_name

        def get_articulation_action():
            ac = robot.get_articulation_controller()
            articulation_action = ac.get_applied_action()
            action_dict = {}
            if articulation_action is not None:
                action_dict["position"] = articulation_action.joint_positions
                action_dict["velocity"] = articulation_action.joint_velocities
                action_dict["effort"] = articulation_action.joint_efforts
                action_dict["joint_names"] = robot.dof_names
                return action_dict
            else:
                return None

        node = JointStatePubRosNode(
            topic_name,
            get_articulation_action,
            frame_id,
            node_name="articulation_action_node",
            step_size=step_size,
        )
        return node

    def publish_clock(self, clock_graph_path="/ClockActionGraph"):
        ros_clock_graph_path = clock_graph_path
        og.Controller.edit(
            {
                "graph_path": ros_clock_graph_path,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            },
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("RosContext", "isaacsim.ros2.bridge.ROS2Context"),
                    ("RosPublisher", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "RosPublisher.inputs:execIn"),
                    ("OnPlaybackTick.outputs:time", "RosPublisher.inputs:timeStamp"),
                    ("RosContext.outputs:context", "RosPublisher.inputs:context"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("RosContext.inputs:domain_id", self.ros_domain_id),
                ],
            },
        )
