# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from geniesim.plugins.logger import Logger

logger = Logger()  # Create singleton instance

import omni
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, UsdPhysics, PhysxSchema
from isaacsim.core.utils.prims import get_prim_at_path, get_prim_object_type
from isaacsim.core.prims import SingleXFormPrim, SingleArticulation
from isaacsim.core.utils.xforms import get_world_pose

from pprint import pprint
import re


def get_articulated_object_prims(articulate_object_name):
    # Get the current USD stage
    stage = omni.usd.get_context().get_stage()
    # List to store prims with both RigidBody and Collider
    articulated_prims = []
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh" and str(prim.GetPrimPath()).startswith(articulate_object_name):
            articulated_prims.append(str(prim.GetPrimPath()))
    return articulated_prims


def is_valid_path(path):
    pattern = r"^/World/objects/[^/]+$"
    return re.fullmatch(pattern, path) is not None


def get_rigidbody_collider_prims(robot_name, extra_prim_paths=None):
    stage = omni.usd.get_context().get_stage()

    objects_path = "/World/objects"
    objects_prim = stage.GetPrimAtPath(objects_path)

    if not objects_prim.IsValid():
        print(f"warning: {objects_path} not found")
        return []

    direct_children = []

    if extra_prim_paths is not None:
        for prim_path in extra_prim_paths:
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid() and prim not in direct_children:
                direct_children.append(prim)

    for child_prim in objects_prim.GetAllChildren():
        if child_prim not in direct_children:
            entity_prim = child_prim.GetChild("entity")
            if entity_prim:
                direct_children.append(entity_prim)

    return direct_children


def get_camera_prims(robot_path):
    # Get the current USD stage
    stage = omni.usd.get_context().get_stage()

    # List to store camera prims
    camera_prims = []
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Camera":
            if not robot_path:
                camera_prims.append(prim)
            else:
                robot_id = str(robot_path).split("/")[-1]
                if robot_id in str(prim.GetPrimPath()):
                    camera_prims.append(prim)

    return camera_prims


def collect_physics(physics_info):
    logger.info("========Collect physics==========")
    if not physics_info:
        physics_info["articulation"] = []
        physics_info["rigidbody"] = []

    ignore_keys = ["background", "G1", "G2", "genie"]
    preserved_keys = ["genie_"]
    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        prim_path = str(prim.GetPrimPath())
        prim_type = get_prim_object_type(prim_path)

        if any(key in prim_path for key in ignore_keys) and not any(key in prim_path for key in preserved_keys):
            continue
        if prim_type == "articulation":
            logger.info(f"{prim_path}")
            logger.info("  -articulation")
            if prim.IsActive():
                physics_info["articulation"].append(prim_path)
        else:
            if prim.GetAttribute("physics:rigidBodyEnabled"):
                logger.info(f"{prim_path}")
                logger.info("  -rigidbody")
                physics_info["rigidbody"].append(prim_path)


def disable_physics(physics_info):
    logger.info("========Disable physics==========")
    ignore_keys = []
    set_gravity(False)


def restore_physics(physics_info):
    logger.info("========Restore physics==========")
    set_gravity(True)


def store_history_physics(robot_articulation, physics_info, history_info, timestamp):
    single_frame = {"articulation": {}}
    for prim_path in physics_info.get("rigidbody", []):
        try:
            pos, quat = get_world_pose(prim_path)
            single_frame[prim_path] = list(pos) + list(quat)
        except:
            continue
        pos, quat = get_world_pose(prim_path)
        single_frame[prim_path] = list(pos) + list(quat)

    for prim_path in physics_info.get("articulation", []):
        object = SingleArticulation(prim_path=prim_path)
        single_frame["articulation"][prim_path] = {
            "joint_state": object.get_joint_positions(),
            "world_pose": object.get_world_pose(),
        }

    single_frame["robot"] = robot_articulation.get_joint_positions()
    single_frame["timestamp"] = timestamp
    history_info.append(single_frame)


def playback_once(robot_articulation, history_info):
    if not history_info:
        logger.warning("hist info empty")
        return -1
    latest_frame = history_info.pop()
    logger.info(f"=========playback once========")
    for prim_path in latest_frame:
        if prim_path == "robot":
            robot_articulation.set_joint_positions(latest_frame[prim_path])
        elif prim_path == "timestamp":
            continue
        elif prim_path == "articulation":
            arti_infos = latest_frame[prim_path]
            for arti_path, val in arti_infos.items():
                joint_state = val["joint_state"]
                pos, quat = val["world_pose"]
                arti_object = SingleArticulation(arti_path)
                arti_object.set_joint_positions(joint_state)
                arti_object.set_world_pose(pos, quat)
        else:
            SingleXFormPrim(prim_path=prim_path).set_world_pose(
                latest_frame[prim_path][:3], latest_frame[prim_path][3:]
            )
    return latest_frame["timestamp"]


def reset_one_frame(robot_articulation, one_frame):
    logger.info(f"=========reset one frame========")
    # logger.info(one_frame)
    for prim_path in one_frame:
        if prim_path == "robot":
            continue
            robot_articulation.set_joint_positions(one_frame[prim_path])
        elif prim_path == "timestamp":
            continue
        elif prim_path == "articulation":
            arti_infos = one_frame[prim_path]
            for arti_path, val in arti_infos.items():
                joint_state = val["joint_state"]
                pos, quat = val["world_pose"]
                arti_object = SingleArticulation(arti_path)
                arti_object.set_joint_positions(joint_state)
                arti_object.set_world_pose(pos, quat)
        else:
            SingleXFormPrim(prim_path=prim_path).set_world_pose(one_frame[prim_path][:3], one_frame[prim_path][3:])


def store_init_physics(robot_articulation, physics_info):
    single_frame = {}
    for prim_path in physics_info.get("rigidbody", []):
        object = SingleXFormPrim(prim_path)
        pos, quat = object.get_world_pose()
        single_frame[prim_path] = list(pos) + list(quat)
    if physics_info.get("articulation"):
        single_frame["articulation"] = {}
        for prim_path in physics_info.get("articulation", []):
            object = SingleArticulation(prim_path=prim_path)
            single_frame["articulation"][prim_path] = {
                "joint_state": object.get_joint_positions(),
                "world_pose": object.get_world_pose(),
            }
    if single_frame:
        single_frame["robot"] = robot_articulation.get_joint_positions()
    return single_frame


def set_gravity(enabled):
    stage = omni.usd.get_context().get_stage()
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
    if enabled:
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
    else:
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        scene.CreateGravityMagnitudeAttr().Set(0.0)

    return scene
