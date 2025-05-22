# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import omni
import omni.graph.core as og
import omni.replicator.core as rep
from omni.isaac.sensor import Camera


def publish_lidar_pointcloud(camera: Camera, freq: int, topic=""):
    render_product = camera._render_product_path
    step_size = int(60 / freq)
    topic_name = camera.prim_path + "_pointcloud" if topic == "" else topic
    queue_size = 10
    node_namespace = ""
    frame_id = camera.prim_path.split("/")[-1]

    writer = rep.writers.get("RtxLidarROS2PublishPointCloudBuffer")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name,
    )
    writer.attach([render_product])
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        "PostProcessDispatchIsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)


def publish_lidar_scan(camera: Camera, freq: int, topic=""):
    render_product = camera._render_product_path
    step_size = int(60 / freq)
    topic_name = camera.prim_path + "_scan" if topic == "" else topic
    queue_size = 10
    node_namespace = ""
    frame_id = camera.prim_path.split("/")[-1]

    writer = rep.writers.get("RtxLidarROS2PublishLaserScan")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name,
    )
    writer.attach([render_product])
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        "PostProcessDispatchIsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)
