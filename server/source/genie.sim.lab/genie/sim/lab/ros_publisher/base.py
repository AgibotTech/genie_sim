# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os
import omni
import omni.graph.core as og

from base_utils.logger import Logger

logger = Logger()  # Create singleton instance

from isaacsim.core.nodes.scripts.utils import set_target_prims

import usdrt.Sdf


class USDBase:
    def __init__(self):
        # self.init_all_sensors()
        # tf_targets = []
        # for sensor in self.config["tf"]:
        #     tf_targets.append(sensor)
        # self.publish_tf(tf_targets, self.config)
        pass
        # og.Controller.create_node("/Render/PostProcess/SDGPipeline/"+"RosContext", "isaacsim.ros2.bridge.ROS2Context")

    def _init_sensor(self):
        # self.ros_domain_id = (int)(ros_domain_id)
        # og.Controller.delete_node("/Render/PostProcess/SDGPipeline/"+"RosContext")
        # og.Controller.create_node("/Render/PostProcess/SDGPipeline/"+"RosContext_{}".format(ros_domain_id), "isaacsim.ros2.bridge.ROS2Context")
        # og.Controller.attribute("/Render/PostProcess/SDGPipeline/"+"RosContext_{}.inputs:domain_id".format(ros_domain_id)).set(self.ros_domain_id)
        # logger.info(self.ros_domain_id)
        self.publish_clock()
        self.publish_rtf()

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

    def _init_camera(self, physics_dt, param):
        from .camera import (
            Camera,
            publish_camera_info,
            publish_rgb,
            publish_pointcloud_from_depth,
            publish_depth,
            publish_boundingbox2d_loose,
            publish_boundingbox2d_tight,
            publish_boundingbox3d,
            publish_semantic_segmant,
        )

        camera = Camera(
            prim_path=param["path"],
            frequency=param["frequency"],
            resolution=(
                param["resolution"]["width"],
                param["resolution"]["height"],
            ),
        )
        camera.initialize()

        step_size = int((1.0 / physics_dt) / param["frequency"])
        camera_graph = []
        for publish in param["publish"]:
            if publish is None:
                continue
            split = publish.split(":", 2)
            topic = ""
            if len(split) > 1:
                publish = split[0]
                topic = split[1]
            if publish == "rgb":
                camera_graph.append(publish_rgb(camera, step_size, ""))
            elif publish == "depth":
                camera_graph.append(publish_depth(camera, step_size, ""))
            elif publish == "info":
                camera_graph.append(publish_camera_info(camera, step_size, topic))
            elif publish == "pointcloud":
                publish_pointcloud_from_depth(camera, step_size, topic)
            elif publish == "bbox2_loose":
                publish_boundingbox2d_loose(camera, step_size, topic)
            elif publish == "bbox2_tight":
                publish_boundingbox2d_tight(camera, step_size, topic)
            elif publish == "bbox3":
                publish_boundingbox3d(camera, step_size, topic)
            elif publish == "semantic":
                publish_semantic_segmant(camera, step_size, topic)
        camera.initialize()
        return camera_graph

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
        ros_tf_graph_path = "/RobotTFActionGraph"

        set_target_prims(
            primPath=ros_tf_graph_path + "/RosPublishTransformTree",
            inputName="inputs:targetPrims",
            targetPrimPaths=[],
        )

    def publish_tf(
        self, robot_prim, targets, approx_freq, delta_time, topic_name="/obj_tf"
    ):
        ros_tf_graph_path = "/RobotTFActionGraph"
        step = (int)(approx_freq)
        og.Controller.edit(
            {
                "graph_path": ros_tf_graph_path,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            },
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    (
                        "IsaacSimulationGate",
                        "isaacsim.core.nodes.IsaacSimulationGate",
                    ),
                    (
                        "RosPublishTransformTree",
                        "isaacsim.ros2.bridge.ROS2PublishTransformTree",
                    ),
                    (
                        "ReadSimTime",
                        "isaacsim.core.nodes.IsaacReadSimulationTime",
                    ),
                    (
                        "PublishJointState",
                        "isaacsim.ros2.bridge.ROS2PublishJointState",
                    ),
                    (
                        "RosContext",
                        "isaacsim.ros2.bridge.ROS2Context",
                    ),
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
                    ("IsaacSimulationGate.inputs:step", step),
                    ("RosPublishTransformTree.inputs:topicName", topic_name),
                ],
            },
        )
        set_target_prims(
            primPath=ros_tf_graph_path + "/RosPublishTransformTree",
            inputName="inputs:targetPrims",
            targetPrimPaths=targets,
        )

    def publish_joint(self, robot_prim):
        og.Controller.edit(
            {
                "graph_path": "/RobotJointStateActionGraph",
                "evaluator_name": "execution",
            },
            {
                og.Controller.Keys.CREATE_NODES: [
                    (
                        "OnPlaybackTick",
                        "omni.graph.action.OnPlaybackTick",
                    ),
                    (
                        "RosContext",
                        "isaacsim.ros2.bridge.ROS2Context",
                    ),
                    (
                        "IsaacSimulationGate",
                        "isaacsim.core.nodes.IsaacSimulationGate",
                    ),
                    (
                        "ReadSimTime",
                        "isaacsim.core.nodes.IsaacReadSimulationTime",
                    ),
                    (
                        "PublisherJointState",
                        "isaacsim.ros2.bridge.ROS2PublishJointState",
                    ),
                    (
                        "SubscriberJointState",
                        "isaacsim.ros2.bridge.ROS2SubscribeJointState",
                    ),
                    (
                        "ArticulationController",
                        "isaacsim.core.nodes.IsaacArticulationController",
                    ),
                ],
                og.Controller.Keys.CONNECT: [
                    (
                        "OnPlaybackTick.outputs:tick",
                        "PublisherJointState.inputs:execIn",
                    ),
                    (
                        "RosContext.outputs:context",
                        "PublisherJointState.inputs:context",
                    ),
                    (
                        "ReadSimTime.outputs:simulationTime",
                        "PublisherJointState.inputs:timeStamp",
                    ),
                    (
                        "OnPlaybackTick.outputs:tick",
                        "SubscriberJointState.inputs:execIn",
                    ),
                    (
                        "RosContext.outputs:context",
                        "SubscriberJointState.inputs:context",
                    ),
                    (
                        "OnPlaybackTick.outputs:tick",
                        "ArticulationController.inputs:execIn",
                    ),
                    (
                        "SubscriberJointState.outputs:positionCommand",
                        "ArticulationController.inputs:positionCommand",
                    ),
                    (
                        "SubscriberJointState.outputs:velocityCommand",
                        "ArticulationController.inputs:velocityCommand",
                    ),
                    (
                        "SubscriberJointState.outputs:effortCommand",
                        "ArticulationController.inputs:effortCommand",
                    ),
                    (
                        "SubscriberJointState.outputs:jointNames",
                        "ArticulationController.inputs:jointNames",
                    ),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("ReadSimTime.inputs:resetOnStop", False),
                    ("PublisherJointState.inputs:topicName", "/joint_state"),
                    ("PublisherJointState.inputs:targetPrim", robot_prim),
                    ("SubscriberJointState.inputs:topicName", "/joint_command"),
                    ("ArticulationController.inputs:targetPrim", robot_prim),
                    ("ArticulationController.inputs:robotPath", robot_prim),
                ],
            },
        )

    # def publish_state(self, step, topic_name, data):
    #     og.Controller.edit(
    #         {"graph_path": "/StringGraph", "evaluator_name": "execution"},
    #         {
    #             og.Controller.Keys.CREATE_NODES: [
    #                 ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
    #                 ("Publisher", "isaacsim.ros2.bridge.ROS2Publisher"),
    #                 ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime")
    #             ],
    #             og.Controller.Keys.CONNECT:[
    #                 ("OnPlaybackTick.outputs:tick", "Publisher.inputs:execIn"),
    #             ],
    #             og.Controller.Keys.SET_VALUES: [
    #                 ("Publisher.inputs:messageName", "String"),
    #                 ("Publisher.inputs:messagePackage", "std_msgs"),
    #                 ("Publisher.inputs:message", "msg"),
    #                 ("Publisher.inputs:topicName", topic_name)
    #                 ("Publisher.inputs:data", data)
    #             ]

    #         }
    #     )

    def publish_clock(self, graph_path="/RosClockActionGraph"):
        og.Controller.edit(
            {
                "graph_path": graph_path,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            },
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    (
                        "RosContext",
                        "isaacsim.ros2.bridge.ROS2Context",
                    ),
                    (
                        "RosPublisherClock",
                        "isaacsim.ros2.bridge.ROS2PublishClock",
                    ),
                ],
                og.Controller.Keys.CONNECT: [
                    (
                        "OnPlaybackTick.outputs:tick",
                        "RosPublisherClock.inputs:execIn",
                    ),
                    (
                        "OnPlaybackTick.outputs:time",
                        "RosPublisherClock.inputs:timeStamp",
                    ),
                    (
                        "RosContext.outputs:context",
                        "RosPublisherClock.inputs:context",
                    ),
                ],
            },
        )

    def publish_rtf(self, graph_path="/RosRTFActionGraph"):
        og.Controller.edit(
            {
                "graph_path": graph_path,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
                "fc_backing_type": og.GraphBackingType.GRAPH_BACKING_TYPE_FLATCACHE_SHARED,
                "evaluation_mode": og.GraphEvaluationMode.GRAPH_EVALUATION_MODE_AUTOMATIC,
            },
            {
                og.Controller.Keys.CREATE_NODES: [
                    (
                        "OnPlaybackTick",
                        ("omni.graph.action.OnPlaybackTick"),
                    ),
                    (
                        "RosContext",
                        ("isaacsim.ros2.bridge.ROS2Context"),
                    ),
                    (
                        "RTF",
                        ("isaacsim.core.nodes.IsaacRealTimeFactor"),
                    ),
                    (
                        "RosPublisherRTF",
                        ("isaacsim.ros2.bridge.ROS2Publisher"),
                    ),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("RosPublisherRTF.inputs:messageName", "Float32"),
                    ("RosPublisherRTF.inputs:messagePackage", "std_msgs"),
                    ("RosPublisherRTF.inputs:messageSubfolder", "msg"),
                    ("RosPublisherRTF.inputs:topicName", "rtf_factor"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "RosPublisherRTF.inputs:execIn"),
                    ("RosContext.outputs:context", "RosPublisherRTF.inputs:context"),
                    # ("RTF.outputs:rtf", "RosPublisherRTF.inputs:data"),
                ],
            },
        )
        # og.Controller.attribute(
        #     graph_path + "/RosPublisherRTF.inputs:messageName"
        # ).set("Float32")
        # og.Controller.attribute(
        #     graph_path + "/RosPublisherRTF.inputs:messagePackage"
        # ).set("std_msgs")
        # og.Controller.attribute(
        #     graph_path + "/RosPublisherRTF.inputs:messageSubfolder"
        # ).set("msg")
        og.Controller.connect(
            og.Controller.attribute(graph_path + "/RTF.outputs:rtf"),
            og.Controller.attribute(graph_path + "/RosPublisherRTF.inputs:data"),
        )
