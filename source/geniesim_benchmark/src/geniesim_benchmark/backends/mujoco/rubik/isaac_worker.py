# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Standalone Isaac Sim renderer; no native physics imports or stepping."""

import argparse
import json
import os
from pathlib import Path
import select
import socket
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
from render_protocol import receive_message, send_message
from robot_render import RobotPoseTree, joint_frame
from cameras import projection_parameters


def _require_complete_scene(stage):
    errors = [error for prim in stage.TraverseAll() for error in prim.GetPrimIndex().localErrors]
    if errors:
        raise RuntimeError(
            "Supplied USD scene has unresolved assets; check scene asset mounts before inference:\n"
            + "\n".join(str(error) for error in errors)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(args.socket)
    state = receive_message(connection)
    app = None
    try:
        import numpy as np

        from isaacsim import SimulationApp

        app = SimulationApp(
            {
                "headless": args.headless,
                "renderer": "RaytracedLighting",
                "multi_gpu": False,
                "width": 1280,
                "height": 720,
                "window_width": 1600,
                "window_height": 1000,
                "enable_crashreporter": False,
            }
        )
        import carb
        import omni.usd
        import omni.timeline
        import omni.replicator.core as rep
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, UsdPhysics, Vt
        from isaacsim.core.utils.viewports import set_camera_view

        settings = carb.settings.get_settings()
        settings.set("/rtx/dataWindow/fitOutputToDataWindow", True)
        for i, value in enumerate((0.0, 0.0, 1.0, 1.0)):
            settings.set(f"/rtx/dataWindowNDC/{i}", value)
        settings.set("/rtx/post/motionblur/enabled", False)
        omni.timeline.get_timeline_interface().stop()
        context = omni.usd.get_context()
        scene = json.loads((args.directory / "scene.json").read_text())
        if not context.open_stage(scene["scene_usda"]):
            raise RuntimeError("Could not open supplied USD scene")
        stage = context.get_stage()
        _require_complete_scene(stage)
        stage.SetEditTarget(stage.GetSessionLayer())
        robot = UsdGeom.Xform.Define(stage, "/genie")
        robot.GetPrim().GetReferences().AddReference(scene["robot_usd"])
        background = UsdGeom.Xform.Define(stage, "/World/Background")
        background.GetPrim().GetReferences().AddReference(scene["environment_usd"])
        robot_joints = []
        for prim in stage.Traverse():
            if not str(prim.GetPath()).startswith("/genie/") or not prim.IsA(UsdPhysics.Joint):
                continue
            parent = prim.GetRelationship("physics:body0").GetTargets()
            child = prim.GetRelationship("physics:body1").GetTargets()
            if len(child) != 1:
                raise RuntimeError("USD robot joint must have exactly one child")
            frames = []
            for index in (0, 1):
                position = prim.GetAttribute(f"physics:localPos{index}").Get()
                quaternion = prim.GetAttribute(f"physics:localRot{index}").Get()
                frames.append(joint_frame(position, [quaternion.GetReal(), *quaternion.GetImaginary()]).tolist())
            robot_joints.append(
                {
                    "name": prim.GetName(),
                    "kind": prim.GetTypeName(),
                    "parent": str(parent[0]) if parent else "",
                    "child": str(child[0]),
                    "frame0": frames[0],
                    "frame1": frames[1],
                    "axis": prim.GetAttribute("physics:axis").Get(),
                }
            )
        missing = set(scene["required_robot_joints"]) - {joint["name"] for joint in robot_joints}
        if missing:
            raise RuntimeError(f"Standard robot is missing controlled joints: {sorted(missing)}")
        robot_tree = RobotPoseTree(robot_joints)
        robot_ops = {}
        for joint in robot_joints:
            xform = UsdGeom.Xformable(stage.GetPrimAtPath(joint["child"]))
            xform.ClearXformOpOrder()
            robot_ops[joint["child"]] = xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
        for prim in list(stage.Traverse()):
            if prim.IsA(UsdPhysics.Joint) or prim.GetTypeName() == "PhysicsScene":
                prim.SetActive(False)
                continue
            physics = [
                str(schema) for schema in prim.GetAppliedSchemas() if str(schema).startswith(("Physics", "Physx"))
            ]
            if physics:
                prim.SetMetadata("apiSchemas", Sdf.TokenListOp.Create(deletedItems=physics))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        transforms = []
        materials = {}
        for geom in scene["geoms"]:
            path = f"/World/Geometries/g{geom['id']:04d}"
            xform = UsdGeom.Xform.Define(stage, path)
            transforms.append(xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble))
            xform.GetPrim().SetCustomDataByKey("nativeGeomName", geom["name"] or "")
            kind, size = geom["type"], geom["size"]
            shape_path = path + "/Shape"
            if kind != 0:
                raise RuntimeError("Only the native floor is added to the supplied scene")
            shape = UsdGeom.Mesh.Define(stage, shape_path)
            sx, sy = max(size[0], 3.0), max(size[1], 3.0)
            shape.CreatePointsAttr([(-sx, -sy, 0), (sx, -sy, 0), (sx, sy, 0), (-sx, sy, 0)])
            shape.CreateFaceVertexCountsAttr([4])
            shape.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            shape.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            rgba = tuple(geom["rgba"])
            if rgba not in materials:
                mp = f"/World/Materials/m{len(materials)}"
                material = UsdShade.Material.Define(stage, mp)
                shader = UsdShade.Shader.Define(stage, mp + "/Surface")
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgba[:3]))
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
                shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(rgba[3])
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                materials[rgba] = material
            UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(materials[rgba])
        light = UsdLux.DomeLight.Define(stage, "/World/Lighting/Dome")
        light.CreateIntensityAttr(1000.0)
        light.CreateTextureFileAttr(scene["hdr_texture"])
        light.CreateTextureFormatAttr("latlong")
        cube_ops = []
        cube_paths = []
        for binding in scene["cube_bindings"]:
            path = "/World/objects/rubik_cube"
            if binding["name"] != "cube_middle":
                path += "/" + binding["name"]
            prim = stage.GetPrimAtPath(path)
            if not prim:
                raise RuntimeError(f"Missing supplied USD cubie: {path}")
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            cube_ops.append(xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble))
            cube_paths.append(path)
        camera_ops, annotators, camera_paths = {}, {}, {}
        for name, camera in {**scene["cameras"], **scene["inspection_cameras"]}.items():
            source = camera.get("usd_prim")
            path = source or f"/World/Cameras/{name}"
            if source and not stage.GetPrimAtPath(source).IsA(UsdGeom.Camera):
                raise RuntimeError(f"Standard robot camera is missing: {source}")
            usd_camera = UsdGeom.Camera(stage.GetPrimAtPath(path)) if source else UsdGeom.Camera.Define(stage, path)
            if source:
                # The robot asset retains legacy fisheye settings. Collection
                # frames use OpenCV pinhole calibration, including distortion.
                prim = usd_camera.GetPrim()
                prim.ApplyAPI("OmniLensDistortionOpenCvPinholeAPI")
                prim.GetAttribute("omni:lensdistortion:model").Set("opencvPinhole")
                prim.CreateAttribute("cameraProjectionType", Sdf.ValueTypeNames.Token).Set("pinhole")
                prefix = "omni:lensdistortion:opencvPinhole:"
                prim.GetAttribute(prefix + "imageSize").Set(Gf.Vec2i(camera["width"], camera["height"]))
                for key, number in camera["intrinsics"].items():
                    prim.GetAttribute(prefix + key).Set(number)
                for key in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4"):
                    prim.GetAttribute(prefix + key).Set(camera.get("distortion", {}).get(key, 0.0))
            projection = projection_parameters(camera)
            usd_camera.CreateFocalLengthAttr(projection["focal"])
            usd_camera.CreateVerticalApertureAttr(projection["vertical"])
            usd_camera.CreateHorizontalApertureAttr(projection["horizontal"])
            usd_camera.CreateHorizontalApertureOffsetAttr(projection["horizontal_offset"])
            usd_camera.CreateVerticalApertureOffsetAttr(projection["vertical_offset"])
            if not source:
                usd_camera.CreateClippingRangeAttr(Gf.Vec2f(*scene["clip_range"]))
                camera_ops[name] = usd_camera.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
            camera_paths[name] = path
            if name in scene["cameras"]:
                product = rep.create.render_product(path, (camera["width"], camera["height"]))
                annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                annotator.attach(product)
                annotators[name] = annotator

        def apply_state(value):
            positions = np.frombuffer(value["geom_positions"], dtype=np.float64).reshape(-1, 3)
            rotations = np.frombuffer(value["geom_rotations"], dtype=np.float64).reshape(-1, 3, 3)
            poses = robot_tree.evaluate(value["robot_joint_positions"], value["robot_anchors"]["/genie/base_link"])
            robot_error = 0.0
            for path, pose in poses.items():
                robot_ops[path].Set(Gf.Matrix4d(pose.T.tolist()))
            for path, expected in value["robot_anchors"].items():
                if path not in poses:
                    raise RuntimeError("Native robot link missing from USD: " + path)
                robot_error = max(robot_error, float(np.max(np.abs(poses[path] - np.asarray(expected)))))
            if robot_error > 2e-5:
                raise RuntimeError(f"Standard USD robot differs from native kinematics: {robot_error}")
            maximum_error = 0.0
            for op, position, rotation in zip(transforms, positions, rotations):
                matrix = np.eye(4)
                matrix[:3, :3] = rotation.T
                matrix[3, :3] = position
                op.Set(Gf.Matrix4d(matrix.tolist()))
                actual = np.asarray(op.Get(), dtype=float)
                maximum_error = max(maximum_error, float(np.max(np.abs(actual - matrix))))
            positions = np.frombuffer(value["cube_positions"], dtype=np.float64).reshape(-1, 3)
            rotations = np.frombuffer(value["cube_rotations"], dtype=np.float64).reshape(-1, 3, 3)
            root_position, root_rotation = positions[0], rotations[0]
            for index, (op, position, rotation) in enumerate(zip(cube_ops, positions, rotations)):
                matrix = np.eye(4)
                matrix[:3, :3] = rotation.T if index == 0 else (root_rotation.T @ rotation).T
                matrix[3, :3] = position if index == 0 else root_rotation.T @ (position - root_position)
                op.Set(Gf.Matrix4d(matrix.tolist()))
            cache = UsdGeom.XformCache()
            for path, position, rotation in zip(cube_paths, positions, rotations):
                world_matrix = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)), dtype=float)
                maximum_error = max(
                    maximum_error,
                    float(np.max(np.abs(world_matrix[3, :3] - position))),
                    float(np.max(np.abs(world_matrix[:3, :3] - rotation.T))),
                )
            camera_error = 0.0
            for name, pose in {**value["cameras"], **value["inspection_cameras"]}.items():
                forward = np.asarray(pose["forward"])
                up = np.asarray(pose["up"])
                matrix = np.eye(4)
                matrix[:3, :3] = np.stack((np.cross(forward, up), up, -forward))
                matrix[3, :3] = pose["eye"]
                if name in camera_ops:
                    camera_ops[name].Set(Gf.Matrix4d(matrix.tolist()))
                else:
                    actual = np.asarray(
                        UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(camera_paths[name]))
                    )
                    camera_error = max(camera_error, float(np.max(np.abs(actual - matrix))))
            if camera_error > 2e-5:
                raise RuntimeError(f"USD mounted camera differs from collection calibration: {camera_error}")
            return maximum_error, robot_error, camera_error

        initial_cube_error, initial_robot_error, initial_camera_error = apply_state(state)
        stage.GetSessionLayer().Export(str(args.directory / "render-overrides.usda"))
        render_layer = Sdf.Layer.CreateNew(str(args.directory / "render-scene.usda"))
        render_layer.subLayerPaths = ["render-overrides.usda", scene["scene_usda"]]
        render_layer.Save()
        if not args.headless:
            set_camera_view(eye=np.array([1.5, 1.3, 1.55]), target=np.array([0.45, 0, 0.98]))
        for _ in range(20):
            app.update()
        rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=True)
        forbidden = [
            str(p.GetPath())
            for p in stage.Traverse()
            if p.GetTypeName() == "PhysicsScene"
            or any("RigidBody" in str(a) or "CollisionAPI" in str(a) for a in p.GetAppliedSchemas())
        ]
        if forbidden:
            raise RuntimeError(f"Render stage contains physics schemas: {forbidden}")
        metadata = {
            "initial_cube_pose_error": initial_cube_error,
            "initial_robot_pose_error": initial_robot_error,
            "initial_camera_pose_error": initial_camera_error,
            "policy_camera_paths": {name: camera_paths[name] for name in annotators},
            "inspection_camera_paths": {name: camera_paths[name] for name in scene["inspection_cameras"]},
            "camera_intrinsics": {
                name: projection_parameters(camera)["intrinsics"] for name, camera in scene["cameras"].items()
            },
            "camera_distortion_models": {
                name: str(stage.GetPrimAtPath(camera_paths[name]).GetAttribute("omni:lensdistortion:model").Get())
                for name in annotators
            },
            "camera_distortion": {name: camera.get("distortion", {}) for name, camera in scene["cameras"].items()},
            "robot_usd": scene["robot_usd"],
            "robot_usd_sha256": scene["robot_usd_sha256"],
            "environment_usd": scene["environment_usd"],
            "environment_usd_sha256": scene["environment_usd_sha256"],
            "hdr_texture_sha256": scene["hdr_texture_sha256"],
            "environment_alignment": "industrial HDR asset visually matched to sampled collection frames; exposure not pixel-certified",
            "renderer": "isaac",
            "isaac_version": Path(os.environ["ISAAC_PATH"]).joinpath("VERSION").read_text().strip(),
            "render_mode": "RaytracedLighting",
            "additional_geometry_count": len(transforms),
            "synchronized_robot_links": len(robot_ops),
            "controlled_robot_joints": len(scene["required_robot_joints"]),
            "physics_schemas": forbidden,
            "scene_usda": scene["scene_usda"],
            "source_scene_id": scene["source_scene_id"],
            "source_scene_sha256": scene["source_scene_sha256"],
            "render_scene_usda": str(args.directory / "render-scene.usda"),
            "synchronized_cube_bodies": len(cube_ops),
            "camera_names": list(annotators),
        }
        send_message(connection, {"metadata": metadata})
        print(json.dumps(metadata), flush=True)
        while app.is_running():
            if select.select([connection], [], [], 0)[0]:
                value = receive_message(connection)
                if value["method"] == "close":
                    break
                error, robot_error, camera_error = apply_state(value)
                response = {
                    "simulation_time": value["simulation_time"],
                    "maximum_pose_error": error,
                    "maximum_robot_pose_error": robot_error,
                    "maximum_camera_pose_error": camera_error,
                }
                if value["capture"]:
                    rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=True)
                    images = {}
                    for name, annotator in annotators.items():
                        image = np.asarray(annotator.get_data())
                        camera = scene["cameras"][name]
                        if image.shape[:2] != (camera["height"], camera["width"]) or image.ndim != 3:
                            raise RuntimeError(f"Isaac camera {name} returned {image.shape}")
                        rgb = np.ascontiguousarray(image[:, :, :3], dtype=np.uint8)
                        images[name] = {"rgb": rgb.tobytes(), "shape": rgb.shape}
                    response["images"] = images
                else:
                    app.update()
                send_message(connection, response)
            else:
                app.update()
                time.sleep(0.01)
    except (EOFError, BrokenPipeError):
        pass
    except Exception as error:
        import traceback

        traceback.print_exc()
        try:
            send_message(connection, {"error": f"{type(error).__name__}: {error}"})
        except OSError:
            pass
    finally:
        connection.close()
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
