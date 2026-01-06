# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import math

import omni.graph.core as og
from isaacsim.core.nodes.scripts.utils import set_target_prims
from omni.isaac.sensor import IMUSensor


def publish_imu(imu: IMUSensor, approx_freq, topic):
    prim_path = imu.prim_path
    ros_publish_imu_graph_path = "/World/ImuActionGraph"
    step = math.floor(6.0 / approx_freq)
    og.Controller.edit(
        {
            "graph_path": ros_publish_imu_graph_path,
            "evaluator_name": "execution",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
        },
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("IsaacSimulationGate", "isaacsim.core.nodes.IsaacSimulationGate"),
                ("IsaacReadIMUNode", "omni.isaac.sensor.IsaacReadIMU"),
                ("RosPublishImu", "isaacsim.ros2.bridge.ROS2PublishImu"),
                ("RosContext", "isaacsim.ros2.bridge.ROS2Context"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("RosPublishImu.inputs:topicName", topic),
                ("IsaacSimulationGate.inputs:step", step),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "IsaacSimulationGate.inputs:execIn"),
                (
                    "IsaacSimulationGate.outputs:execOut",
                    "IsaacReadIMUNode.inputs:execIn",
                ),
                ("RosContext.outputs:context", "RosPublishImu.inputs:context"),
                (
                    "IsaacReadIMUNode.outputs:angVel",
                    "RosPublishImu.inputs:angularVelocity",
                ),
                (
                    "IsaacReadIMUNode.outputs:linAcc",
                    "RosPublishImu.inputs:linearAcceleration",
                ),
                (
                    "IsaacReadIMUNode.outputs:orientation",
                    "RosPublishImu.inputs:orientation",
                ),
                ("IsaacReadIMUNode.outputs:execOut", "RosPublishImu.inputs:execIn"),
                ("OnPlaybackTick.outputs:time", "RosPublishImu.inputs:timeStamp"),
            ],
        },
    )
    set_target_prims(
        primPath=ros_publish_imu_graph_path + "/IsaacReadIMUNode",
        inputName="inputs:imuPrim",
        targetPrimPaths=[prim_path],
    )
