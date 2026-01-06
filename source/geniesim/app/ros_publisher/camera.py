# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os

import omni
import omni.graph.core as og
import omni.replicator.core as rep
import omni.syntheticdata
import omni.syntheticdata._syntheticdata as sd

from isaacsim.sensors.camera import Camera

NODE_NAMESPACE = "genie_sim"


def publish_boundingbox2d_loose(camera: Camera, step_size: int, topic=""):
    render_product = camera._render_product_path
    topic_name = camera.prim_path + "_bbox2_loose" if topic == "" else topic
    queue_size = 1
    node_namespace = NODE_NAMESPACE
    frame_id = camera.prim_path.split("/")[-1]

    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar("BoundingBox2DLoose")
    writer = rep.writers.get("ROS2PublishBoundingBox2DLoose")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name,
    )
    writer.attach([render_product])
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(rv + "IsaacSimulationGate", render_product)
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)


def publish_semantic_segment(camera: Camera, step_size: int, topic=""):
    render_product = camera._render_product_path
    topic_name = camera.prim_path + "_semantic" if topic == "" else topic
    queue_size = 1
    node_namespace = NODE_NAMESPACE
    frame_id = camera.prim_path.split("/")[-1]

    og.Controller.edit(
        {
            "graph_path": "/World/" + frame_id + "_semantic",
            "evaluator_name": "execution",
        },
        {
            og.Controller.Keys.CREATE_NODES: [
                (
                    "publish_semantic",
                    "isaacsim.ros2.bridge.ROS2CameraHelper",
                ),
                (
                    "OnPlaybackTick",
                    "omni.graph.action.OnPlaybackTick",
                ),
                (
                    "RosContext",
                    "isaacsim.ros2.bridge.ROS2Context",
                ),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("publish_semantic" + ".inputs:topicName", topic_name),
                ("publish_semantic" + ".inputs:type", "semantic_segmentation"),
                ("publish_semantic" + ".inputs:resetSimulationTimeOnStop", True),
                ("publish_semantic" + ".inputs:frameId", frame_id),
                ("publish_semantic" + ".inputs:nodeNamespace", node_namespace),
                ("publish_semantic" + ".inputs:enableSemanticLabels", True),
                ("publish_semantic" + ".inputs:renderProductPath", render_product),
                ("publish_semantic" + ".inputs:frameSkipCount", step_size),
            ],
            og.Controller.Keys.CONNECT: [
                (
                    "OnPlaybackTick" + ".outputs:tick",
                    "publish_semantic" + ".inputs:execIn",
                ),
            ],
        },
    )


def publish_boundingbox2d_tight(camera: Camera, step_size: int, topic=""):
    render_product = camera._render_product_path
    topic_name = camera.prim_path + "_bbox2_tight" if topic == "" else topic
    queue_size = 1
    node_namespace = NODE_NAMESPACE
    frame_id = camera.prim_path.split("/")[-1]

    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar("BoundingBox2DTight")
    writer = rep.writers.get("ROS2PublishBoundingBox2DTight")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name,
    )
    writer.attach([render_product])
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(rv + "IsaacSimulationGate", render_product)
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)


def publish_boundingbox3d(camera: Camera, step_size: int, topic=""):
    render_product = camera._render_product_path
    topic_name = camera.prim_path + "_bbox3" if topic == "" else topic
    queue_size = 1
    node_namespace = NODE_NAMESPACE
    frame_id = camera.prim_path.split("/")[-1]

    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar("BoundingBox3D")
    writer = rep.writers.get("ROS2PublishBoundingBox3D")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name,
    )
    writer.attach([render_product])
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(rv + "IsaacSimulationGate", render_product)
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)


def publish_rgb(camera: Camera, step_size: int, topic=""):
    render_product = camera._render_product_path
    topic_name = camera.prim_path.split("/")[-1] + "_rgb" if topic == "" else topic
    queue_size = 1
    node_namespace = NODE_NAMESPACE
    frame_id = camera.prim_path.split("/")[-1]

    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
    writer = rep.writers.get(rv + "ROS2PublishImage")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name,
    )
    writer.attach([render_product])

    gate_path = omni.syntheticdata.SyntheticData._get_node_path(rv + "IsaacSimulationGate", render_product)
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    return gate_path


def publish_camera_info(camera: Camera, step_size: int, topic=""):
    from .camera_info import (
        read_camera_info,
    )  # isaacsim.ros2.bridge -> read_camera_info has bug in computing cx, cy

    render_product = camera._render_product_path
    topic_name = camera.prim_path.split("/")[-1] + "_camera_info" if topic == "" else topic
    queue_size = 1
    node_namespace = NODE_NAMESPACE
    frame_id = camera.prim_path.split("/")[-1]

    writer = rep.writers.get("ROS2PublishCameraInfo")
    camera_info = read_camera_info(render_product_path=render_product)
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name,
        width=camera_info["width"],
        height=camera_info["height"],
        projectionType=camera_info["projectionType"],
        physicalDistortionModel=camera_info["physicalDistortionModel"],
        physicalDistortionCoefficients=camera_info["physicalDistortionCoefficients"],
    )
    writer.attach([render_product])

    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        "PostProcessDispatch" + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)
    return gate_path


def publish_pointcloud_from_depth(camera: Camera, step_size: int, topic=""):
    render_product = camera._render_product_path
    topic_name = camera.prim_path if topic == "" else topic
    queue_size = 1
    node_namespace = NODE_NAMESPACE
    frame_id = camera.prim_path.split("/")[-1]

    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.DistanceToImagePlane.name)

    writer = rep.writers.get(rv + "ROS2PublishPointCloud")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name,
    )
    writer.attach([render_product])

    gate_path = omni.syntheticdata.SyntheticData._get_node_path(rv + "IsaacSimulationGate", render_product)
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)


def publish_depth(camera: Camera, step_size: int, topic=""):
    render_product = camera._render_product_path
    topic_name = camera.prim_path.split("/")[-1] + "_depth" if topic == "" else topic
    queue_size = 1
    node_namespace = NODE_NAMESPACE
    frame_id = camera.prim_path.split("/")[-1]

    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.DistanceToImagePlane.name)
    writer = rep.writers.get(rv + "ROS2PublishImage")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name,
    )
    writer.attach([render_product])

    gate_path = omni.syntheticdata.SyntheticData._get_node_path(rv + "IsaacSimulationGate", render_product)
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)
    return gate_path
