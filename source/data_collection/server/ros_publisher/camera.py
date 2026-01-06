# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
import omni
import omni.graph.core as og
import omni.replicator.core as rep
import omni.syntheticdata
import omni.syntheticdata._syntheticdata as sd
from isaacsim.sensors.camera import Camera

from common.base_utils.logger import logger
from common.base_utils.ros_nodes.sim_ros_node import ImagePubRosNode
from server.ros_publisher.camera_noiser import apply_noise_to_image, get_random_parameters


def publish_boundingbox2d_loose(camera: Camera, freq: int, topic=""):
    render_product = camera._render_product_path
    step_size = int(60 / freq)
    topic_name = camera.prim_path + "_bbox2_loose" if topic == "" else topic
    queue_size = 10
    node_namespace = ""
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
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)


def publish_semantic_segment(camera: Camera, freq: int, topic=""):
    render_product = camera._render_product_path
    step_size = freq - 1
    topic_name = camera.prim_path + "_semantic" if topic == "" else topic
    node_namespace = camera.prim_path
    frame_id = camera.prim_path.split("/")[-1]
    og.Controller.edit(
        {
            "graph_path": "/World/" + frame_id + "_semantic",
            "evaluator_name": "execution",
        },
        {
            og.Controller.Keys.CREATE_NODES: [
                ("publish_semantic", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("RosContext", "isaacsim.ros2.bridge.ROS2Context"),
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


def publish_boundingbox2d_tight(camera: Camera, freq: int, topic=""):
    render_product = camera._render_product_path
    step_size = int(60 / freq)
    topic_name = camera.prim_path + "_bbox2_tight" if topic == "" else topic
    queue_size = 10
    node_namespace = ""
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
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)


def publish_boundingbox3d(camera: Camera, freq: int, topic=""):
    render_product = camera._render_product_path
    step_size = int(60 / freq)
    topic_name = camera.prim_path + "_bbox3" if topic == "" else topic
    queue_size = 10
    node_namespace = ""
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
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)


def publish_rgb(camera: Camera, freq: int, topic=""):
    render_product = camera._render_product_path
    step_size = freq
    topic_name = "/" + camera.prim_path.split("/")[-1] + "_rgb" if topic == "" else topic
    queue_size = 50
    node_namespace = ""
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

    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    return gate_path


def publish_noised_rgb(camera: Camera, step_size: int, topic="", **kwargs):
    resolution = camera.get_resolution()
    rp = rep.create.render_product(camera.prim_path, (resolution[0], resolution[1]))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach(rp)
    topic_name = "/" + camera.prim_path.split("/")[-1] + "_rgb" if topic == "" else topic
    frame_id = camera.prim_path.split("/")[-1]
    seed = np.random.randint(0, 99999)
    noise_type = (
        kwargs.pop("noise_type")
        if "noise_type" in kwargs
        else np.random.choice(["gaussian", "poisson", "salt_pepper", "speckle", "quantization"])
    )
    logger.info(f"Noise type: {noise_type}, topic: {topic_name}")
    random_parameters = get_random_parameters(noise_type)
    for key, value in random_parameters.items():
        if key not in kwargs:
            kwargs[key] = value
            logger.info(f"set noise parameters: {key} = {value}")

    def get_msg_callback():
        rgb = annotator.get_data(device="cuda")
        if rgb:
            rgb = apply_noise_to_image(rgb[:, :, :3], seed=seed, noise_type=noise_type, **kwargs)
            return rgb
        return None

    ros_node = ImagePubRosNode(
        topic_name, get_msg_callback, frame_id, node_name=f"{frame_id}_node", step_size=step_size
    )
    return ros_node


def publish_camera_info(camera: Camera, freq: int, topic=""):
    from .camera_info import (  # isaacsim.ros2.bridge's built-in read_camera_info cannot correctly calculate cx and cy
        read_camera_info,
    )

    render_product = camera._render_product_path
    step_size = int(60 / freq)
    topic_name = camera.prim_path.split("/")[-1] + "_camera_info" if topic == "" else topic
    queue_size = 30
    node_namespace = ""
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


def publish_pointcloud_from_depth(camera: Camera, freq: int, topic=""):
    render_product = camera._render_product_path
    step_size = int(60 / freq)
    topic_name = camera.prim_path if topic == "" else topic
    queue_size = 10
    node_namespace = ""
    frame_id = camera.prim_path.split("/")[-1]

    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
        sd.SensorType.DistanceToImagePlane.name
    )

    writer = rep.writers.get(rv + "ROS2PublishPointCloud")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name,
    )
    writer.attach([render_product])

    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)


def publish_depth(camera: Camera, freq: int, topic=""):
    render_product = camera._render_product_path
    step_size = freq
    topic_name = "/" + camera.prim_path.split("/")[-1] + "_depth" if topic == "" else topic
    queue_size = 50
    node_namespace = ""
    frame_id = camera.prim_path.split("/")[-1]

    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
        sd.SensorType.DistanceToImagePlane.name
    )
    writer = rep.writers.get(rv + "ROS2PublishImage")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name,
    )
    writer.attach([render_product])

    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)
    return gate_path
