# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from pxr import Usd, UsdGeom, Gf, Sdf, UsdUtils, UsdPhysics
from typing import NamedTuple, Any, Callable, Literal
import os
from pprint import pprint

current_path = os.path.abspath(__file__)
GENIESIM_PATH = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
ASSETS_PATH = os.path.join(GENIESIM_PATH, "assets")


def load_scene(scene_path: str) -> Usd.Stage:
    abs_scene_path = os.path.join(ASSETS_PATH, scene_path)
    if scene_path and os.path.exists(abs_scene_path):
        print(f"loading scene: {abs_scene_path}")
        # usd_context = Usd.Stage.CreateNew("s.usda")
        usd_context = Usd.Stage.Open(abs_scene_path)  # , load=Usd.Stage.LoadNone
        print("loaded")
    else:
        print(f"file not exist {abs_scene_path}, creating new")
        # usd_context = Usd.Stage.CreateInMemory()
        usd_context = Usd.Stage.CreateNew(scene_path)

    return usd_context


def add_objects_to_stage(stage: Usd.Stage, object_info_list: list[dict]) -> Usd.Stage:
    """
    Add objects to the stage and return the context.
    """
    if stage is None:
        print.log_error("stage load failed")
        return None

    for object_info in object_info_list:
        object_name = object_info["id"]
        object_path = os.path.join(ASSETS_PATH, object_info["url"])
        object_type = object_info.get("type", "unknown")
        object_scale = object_info["scale"]
        object_quat = object_info["rotation"]
        object_trans = object_info["translation"]

        print(f"Adding object: {object_name} at {object_trans}")

        prim_path = Sdf.Path(f"/World/Objects/{object_name}")

        xform = stage.DefinePrim(prim_path, object_name)

        ok = xform.GetPrim().GetPayloads().AddPayload(os.path.join(".", "assets", object_info["url"]))
        print("xform.GetPrim().GetPayloads().AddPayload(object_path)", object_path, ok)
        print(f"Added ref: {object_path}  ->  {prim_path}")

    return stage


def dump_scene(stage: Usd.Stage, output_path: str) -> str:
    abs_output_path = os.path.join(ASSETS_PATH, output_path)

    output_dir = os.path.dirname(abs_output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Creating output directory: {output_dir}")

    stage.Export(output_path, args={"exportReferences": "false"})
    return output_path


import numpy as np
from scipy.spatial.transform import Rotation as R


def quaternion_to_euler(q):
    # Scipy expects the quaternion in the form [x, y, z, w]
    q_array = np.array([q[1], q[2], q[3], q[0]])  # Reorder to [x, y, z, w]

    # Create a Rotation object using the quaternion
    r = R.from_quat(q_array)

    # Convert to Euler angles in radians (XYZ order)
    euler_angles_rad = r.as_euler("xyz", degrees=False)

    # Convert to degrees and return as Gf.Vec3f
    return Gf.Vec3f(
        np.degrees(euler_angles_rad[0]),
        np.degrees(euler_angles_rad[1]),
        np.degrees(euler_angles_rad[2]),
    )


def gen_scene_usda(scene_path: str, object_info_list: list[dict]):
    print(f"\nstep1: load scene...")
    stage = load_scene(scene_path)
    print("✅ scene loaded")

    axis: UsdGeom.Tokens = UsdGeom.Tokens.z
    UsdGeom.SetStageUpAxis(stage, axis)
    stage.SetMetadata(UsdGeom.Tokens.metersPerUnit, 1.0)

    prim_sdf_path_list = []

    print(f"\nstep2: add objects to scene...")
    for object_info in object_info_list:
        object_name = object_info["id"]
        object_path = os.path.join(ASSETS_PATH, object_info["url"])
        object_type = object_info.get("type", "unknown")
        object_scale = object_info["scale"]
        object_quat = object_info["rotation"]
        object_trans = object_info["translation"]

        prim_path = Sdf.Path(f"/World/Objects/{object_name}")
        prim_sdf_path_list.append(f"{prim_path}")

        relative_path = os.path.relpath(object_path, os.path.dirname(scene_path))

        xform = UsdGeom.Xform.Define(stage, prim_path)
        xform.ClearXformOpOrder()
        prim = xform.GetPrim()
        prim.GetPayloads().AddPayload(relative_path)
        print(f"Added payload: {object_path}  ->  {prim_path}")

        translate_op = UsdGeom.Xformable(prim).AddTranslateOp()
        orient_op = UsdGeom.Xformable(prim).AddOrientOp()
        scale_op = UsdGeom.Xformable(prim).AddScaleOp()

        # translate
        translate_op.Set(Gf.Vec3d(object_trans[0], object_trans[1], object_trans[2]))

        # quaternion (real, i, j, k) -> (w, x, y, z)
        orient_op.Set(
            Gf.Quatf(
                object_quat[3],  # w (real part)
                object_quat[0],  # x (i)
                object_quat[1],  # y (j)
                object_quat[2],  # z (k)
            )
        )

        # scale
        scale_op.Set(Gf.Vec3f(object_scale[0], object_scale[1], object_scale[2]))

        print(f"  -> trans: {object_trans}")
        print(f"  -> quat : {object_quat}")
        # print(f"  -> scale: {object_scale}")

        # Remove rigidbody for benchmark_hanger objects
        if "benchmark_hanger" in object_path:
            print(f"  -> Detected benchmark_hanger, removing rigidbody...")
            for child_prim in Usd.PrimRange(prim):
                if child_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    child_prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                    print(f"     Removed RigidBodyAPI from: {child_prim.GetPath()}")

    chosen_one_particle_system = None
    prim_path_list = []
    for prim_sdf_path in prim_sdf_path_list:
        for prim in stage.Traverse():
            prim_name = str(prim.GetName())
            prim_path = str(prim.GetPath())
            if prim_path.startswith(str(prim_sdf_path)) and prim_path.endswith("visual"):
                prim_path_list.append(prim)

                props = prim.GetProperties()
                tartget = None
                for prop in props:
                    prop_name_str = prop.GetName()
                    if "physxParticle:particleSystem" == prop_name_str:
                        rel = prim.GetRelationship("physxParticle:particleSystem")
                        if rel:
                            print(prim_path)
                            tartget = rel.GetTargets()
                            if not chosen_one_particle_system:
                                print(f"  <- {tartget[0]} set to Default")
                                chosen_one_particle_system = tartget[0]
                            else:
                                print(f"  -> {tartget[0]} use Default")
                                rel.SetTargets([chosen_one_particle_system])

                # print(prim.GetRelationship())

    print("✅ objects added")

    print(f"\nstep3: save scene to {scene_path}...")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    stage.GetRootLayer().Save()
    print(f"✅ scene saved")
