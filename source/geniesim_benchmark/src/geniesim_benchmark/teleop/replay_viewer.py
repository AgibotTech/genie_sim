# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License 2.0

"""Render recorded states in a private Isaac Sim process without a control server."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from geniesim_benchmark.teleop.replay_trace import (
    ReplayTraceError,
    _matrix,
    _vector,
    _verified_runtime,
    frame_delay,
)


def validate_inputs(state, source_root, assets_root, assets_sha):
    """Check evidence and immutable scene inputs before importing Isaac Sim."""
    source_root, assets_root = Path(source_root).resolve(), Path(assets_root).resolve()
    replay = state.get("replay", {})
    if (
        replay.get("purpose") != "replay"
        or replay.get("source_schema") != "genie-master.geniesim-source-samples.v1"
    ):
        raise ReplayTraceError("viewer requires an adapted compact source trace")
    if replay.get("record_source", {}).get("assets_sha") != assets_sha:
        raise ReplayTraceError("player assets differ from the recorded assets")
    runtime = dict(replay.get("runtime_scenario", {}))
    runtime["robot_config"] = "source/geniesim_benchmark/src/geniesim_benchmark/app/robot_cfg/G2_90d.json"
    task_file = source_root / runtime.get("task_config", "")
    scene, _ = _verified_runtime(runtime, task_file, source_root)
    task = json.loads(task_file.read_text())
    config = json.loads((source_root / runtime["robot_config"]).read_text())
    inputs = {"scene": scene}
    for name, relative in (
        ("robot", config["robot"]["robot_usd"]),
        ("background", task["scene"]["scene_usd"].lstrip("/")),
    ):
        path = (assets_root / relative).resolve()
        if not path.is_relative_to(assets_root) or not path.is_file():
            raise ReplayTraceError(f"{name} asset is absent or outside the asset root")
        inputs[name] = path
    camera_path = "/genie/head_link3/head_front_Camera"
    resolution = config.get("camera", {}).get(camera_path, [])[:2]
    if len(resolution) != 2 or any(
        type(value) is not int or value <= 0 for value in resolution
    ):
        raise ReplayTraceError("canonical head RGB camera resolution is unavailable")
    frames = state.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ReplayTraceError("replay has no frames")
    previous = None
    joint_names = None
    object_paths = None
    for frame in frames:
        frame_delay(previous, frame)
        root = frame.get("robot", {}).get("root_pose", {})
        if root.get("prim_path") != "/genie/base_link":
            raise ReplayTraceError("recorded robot root must identify /genie/base_link")
        _vector(root.get("position"), 3, "robot root position")
        quaternion = _vector(root.get("orientation_xyzw"), 4, "robot root quaternion")
        if abs(sum(value * value for value in quaternion) - 1) > 1e-4:
            raise ReplayTraceError("robot root quaternion is not normalized")
        names = frame["robot"]["joints"]["joint_name"]
        if joint_names is None:
            joint_names = names
        named_joint_values(frame, joint_names)
        paths = {}
        for name, value in frame.get("objects", {}).items():
            path = value.get("pose_prim_path")
            if (
                value.get("pose_frame") != "world"
                or path != f"/Workspace/Objects/{name}/entity"
            ):
                raise ReplayTraceError(
                    "object state must identify its recorded rigid body pose path"
                )
            _matrix(value.get("pose"), "object pose")
            paths[name] = path
        if object_paths is None:
            object_paths = paths
        elif paths != object_paths:
            raise ReplayTraceError("object body membership changed inside replay")
        previous = frame
    return {
        **inputs,
        "robot_prim": config["robot"]["base_prim_path"],
        "object_paths": object_paths,
        "camera_path": camera_path,
        "camera_resolution": resolution,
    }


def named_joint_values(frame, live_names, mimic_joints=None):
    """Use recorded q, and only explicitly selected asset mimic constraints."""
    joints = frame["robot"]["joints"]
    names = joints.get("joint_name", [])
    values = _vector(joints.get("joint_position"), len(names), "joint positions")
    if not names or len(set(names)) != len(names) or not set(names) <= set(live_names):
        raise ReplayTraceError("recorded and loaded articulation DOFs differ")
    by_name = dict(zip(names, values))
    missing = set(live_names) - set(names)
    rules = mimic_joints or {}
    if missing - set(rules):
        raise ReplayTraceError(
            "recorded and loaded articulation DOFs differ; unrecorded joints have no selected mimic rule"
        )
    while missing:
        resolved = set()
        for name in missing:
            rule = rules[name]
            if rule["reference"] in by_name:
                # PhysX constraint: q + gearing * reference_q + offset = 0.
                by_name[name] = (
                    -rule["gearing"] * by_name[rule["reference"]] - rule["offset"]
                )
                resolved.add(name)
        if not resolved:
            raise ReplayTraceError(
                "mimic reconstruction has an unresolved reference or cycle"
            )
        missing -= resolved
    return [by_name[name] for name in live_names]


def read_mimic_rules(stage, missing_names):
    """Read only the asset-authored scalar mimic constraints for absent DOFs."""
    result = {}
    for prim in stage.Traverse():
        name = prim.GetName()
        if name not in missing_names:
            continue
        references = [
            relationship
            for relationship in prim.GetRelationships()
            if relationship.GetName().startswith("physxMimicJoint:")
            and relationship.GetName().endswith(":referenceJoint")
        ]
        if len(references) != 1 or len(references[0].GetTargets()) != 1:
            raise ReplayTraceError(
                f"unrecorded joint has no unique mimic constraint: {name}"
            )
        reference = stage.GetPrimAtPath(references[0].GetTargets()[0])
        namespace = references[0].GetName().removesuffix(":referenceJoint")
        gearing = prim.GetAttribute(namespace + ":gearing").Get()
        offset = prim.GetAttribute(namespace + ":offset").Get()
        result[name] = {
            "reference": reference.GetName(),
            "gearing": 1.0 if gearing is None else float(gearing),
            "offset": 0.0 if offset is None else float(offset),
        }
    if set(result) != set(missing_names):
        raise ReplayTraceError("unrecorded articulation joint is absent from the asset")
    return result


def apply_frame(frame, backend):
    """Restore exactly one state, then render without a dynamics step."""
    backend.set_robot(
        frame["robot"]["root_pose"],
        named_joint_values(
            frame, backend.joint_names, getattr(backend, "mimic_joints", {})
        ),
    )
    for name, value in frame["objects"].items():
        backend.set_object(name, value["pose"])
    return backend.render_and_measure(frame)


class ReplayRestoreError(ReplayTraceError):
    def __init__(self, details):
        self.details = details
        super().__init__("rendered state differs from recording; see failure_details")


def prepare_replay_root(stage, robot_path, recorded_root_path):
    """Pin the free articulation to the recorded rigid link, not an auto root."""
    from pxr import Usd, UsdPhysics

    robot = stage.GetPrimAtPath(robot_path)
    root = stage.GetPrimAtPath(recorded_root_path)
    if (
        not robot.IsValid()
        or not root.IsValid()
        or not root.HasAPI(UsdPhysics.RigidBodyAPI)
    ):
        raise ReplayTraceError("recorded articulation root is not a loaded rigid body")
    if not root.GetPath().HasPrefix(robot.GetPath()):
        raise ReplayTraceError("recorded root is outside the loaded robot")
    prims = list(Usd.PrimRange(robot))
    previous_roots = [
        prim for prim in prims if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if any(
        prim.GetPath() not in {robot.GetPath(), root.GetPath()}
        for prim in previous_roots
    ):
        raise ReplayTraceError("robot contains an unexpected articulation root")
    anchors = []
    for prim in prims:
        if prim.IsA(UsdPhysics.FixedJoint):
            joint = UsdPhysics.FixedJoint(prim)
            if (
                not joint.GetBody0Rel().GetTargets()
                or not joint.GetBody1Rel().GetTargets()
            ):
                anchors.append(str(prim.GetPath()))
                # Inactive prims cannot be parsed as world anchors at all.
                prim.SetActive(False)
    for prim in previous_roots:
        if prim != root:
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    schemas = robot.GetMetadata("apiSchemas")
    if (
        robot != root
        and schemas
        and "PhysxArticulationAPI" in schemas.GetAppliedItems()
    ):
        root.AddAppliedSchema("PhysxArticulationAPI")
        for attr in robot.GetAuthoredAttributes():
            if attr.GetName().startswith("physxArticulation:"):
                root.CreateAttribute(
                    attr.GetName(), attr.GetTypeName(), custom=attr.IsCustom()
                ).Set(attr.Get())
        robot.RemoveAppliedSchema("PhysxArticulationAPI")
    return {
        "recorded_root_path": recorded_root_path,
        "articulation_view_path": recorded_root_path,
        "previous_root_paths": [str(prim.GetPath()) for prim in previous_roots],
        "inactive_world_anchors": anchors,
    }


def pose_measurement(
    expected_position, expected_orientation, actual_position, actual_orientation
):
    position = [float(value) for value in actual_position]
    orientation = [float(value) for value in actual_orientation]
    if (
        len(position) != 3
        or len(orientation) != 4
        or not all(math.isfinite(value) for value in position + orientation)
    ):
        raise ReplayTraceError("pose readback is not finite")
    return {
        "position": position,
        "orientation_wxyz": orientation,
        "position_error_m": max(
            abs(a - b) for a, b in zip(position, expected_position)
        ),
        "quaternion_dot_error": abs(
            1 - abs(sum(a * b for a, b in zip(orientation, expected_orientation)))
        ),
    }


def select_replay_camera(stage, viewport, camera_path, resolution):
    """Use the same authored head Camera and render size as canonical head RGB."""
    from pxr import UsdGeom

    camera = stage.GetPrimAtPath(camera_path)
    if not camera.IsValid() or not camera.IsA(UsdGeom.Camera):
        raise ReplayTraceError(
            "canonical head RGB camera is absent from the replay stage"
        )
    viewport.set_active_camera(camera_path)
    viewport.resolution = tuple(resolution)
    if str(viewport.camera_path) != camera_path or tuple(viewport.resolution) != tuple(
        resolution
    ):
        raise ReplayTraceError("viewport did not select the canonical head RGB view")
    return {
        "mode": "canonical_head_rgb",
        "prim_path": camera_path,
        "resolution": list(resolution),
    }


class IsaacReplay:
    def __init__(
        self, app, inputs, first_frame, allow_mimic_reconstruction=False, headless=False
    ):
        import numpy as np
        import omni.physx
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
        from omni.kit.viewport.utility import get_active_viewport_and_window

        self.np = np
        self.app = app
        self.physx = omni.physx.get_physx_interface()
        self.world = World(
            stage_units_in_meters=1,
            physics_dt=first_frame["physics_dt_s"],
            device="cpu",
        )
        add_reference_to_stage(str(inputs["background"]), "/World")
        add_reference_to_stage(str(inputs["scene"]), "/Workspace")
        add_reference_to_stage(str(inputs["robot"]), inputs["robot_prim"])
        stage = get_current_stage()
        self.root_setup = prepare_replay_root(
            stage, inputs["robot_prim"], first_frame["robot"]["root_pose"]["prim_path"]
        )
        self.robot = self.world.scene.add(
            SingleArticulation(
                self.root_setup["articulation_view_path"],
                name="replay_robot",
                reset_xform_properties=False,
            )
        )
        self.objects = {}
        for name, path in inputs["object_paths"].items():
            if not stage.GetPrimAtPath(path).IsValid():
                raise ReplayTraceError(
                    f"recorded body is absent from the scene: {path}"
                )
            self.objects[name] = self.world.scene.add(
                SingleRigidPrim(path, name=name, reset_xform_properties=False)
            )
        self.world.reset()
        if any(
            stage.GetPrimAtPath(path).IsActive()
            for path in self.root_setup["inactive_world_anchors"]
        ):
            raise ReplayTraceError(
                "world anchor became active during replay initialization"
            )
        self.joint_names = list(self.robot.dof_names)
        recorded_names = first_frame["robot"]["joints"]["joint_name"]
        missing = set(self.joint_names) - set(recorded_names)
        self.mimic_joints = (
            read_mimic_rules(stage, missing) if allow_mimic_reconstruction else {}
        )
        named_joint_values(first_frame, self.joint_names, self.mimic_joints)
        self.fixed_simulation_time = self.world.current_time
        viewport, _ = get_active_viewport_and_window()
        if viewport is None:
            if not headless:
                raise ReplayTraceError("GUI replay has no active viewport")
            self.camera_setup = {"mode": "headless_without_viewport"}
        else:
            self.camera_setup = select_replay_camera(
                stage, viewport, inputs["camera_path"], inputs["camera_resolution"]
            )

    def set_robot(self, root, joints):
        np = self.np
        xyzw = root["orientation_xyzw"]
        self.robot.set_world_pose(
            np.asarray(root["position"]), np.asarray([xyzw[3], *xyzw[:3]])
        )
        self.robot.set_joint_positions(np.asarray(joints))
        self.robot.set_joint_velocities(np.zeros(len(joints)))
        self.robot.set_linear_velocity(np.zeros(3))
        self.robot.set_angular_velocity(np.zeros(3))

    def set_object(self, name, pose):
        from isaacsim.core.utils.rotations import rot_matrix_to_quat

        matrix = self.np.asarray(pose)
        self.objects[name].set_world_pose(
            matrix[:3, 3], rot_matrix_to_quat(matrix[:3, :3])
        )
        self.objects[name].set_linear_velocity(self.np.zeros(3))
        self.objects[name].set_angular_velocity(self.np.zeros(3))

    def render_and_measure(self, frame):
        from isaacsim.core.utils.rotations import rot_matrix_to_quat
        from isaacsim.core.utils.xforms import get_world_pose

        self.world.physics_sim_view.update_articulations_kinematic()
        self.physx.update_transformations(False, True, True, False)
        self.world.render()
        if self.world.current_time != self.fixed_simulation_time:
            raise ReplayTraceError("physics advanced during state-only replay")
        np = self.np
        root = frame["robot"]["root_pose"]
        xyzw = root["orientation_xyzw"]
        expected_root_orientation = [xyzw[3], *xyzw[:3]]
        bodies = {}

        def capture(path, expected_position, expected_orientation, physical_pose):
            bodies[path] = {
                "expected_position": list(map(float, expected_position)),
                "expected_orientation_wxyz": list(map(float, expected_orientation)),
                "physics": pose_measurement(
                    expected_position, expected_orientation, *physical_pose
                ),
                "usd": pose_measurement(
                    expected_position,
                    expected_orientation,
                    *get_world_pose(path, fabric=False),
                ),
            }

        capture(
            root["prim_path"],
            root["position"],
            expected_root_orientation,
            self.robot.get_world_pose(),
        )
        for name, value in frame["objects"].items():
            expected = np.asarray(value["pose"])
            capture(
                value["pose_prim_path"],
                expected[:3, 3],
                rot_matrix_to_quat(expected[:3, :3]),
                self.objects[name].get_world_pose(),
            )
        joint_error = float(
            np.max(
                np.abs(
                    self.robot.get_joint_positions()
                    - named_joint_values(frame, self.joint_names, self.mimic_joints)
                )
            )
        )
        result = {
            key: max(
                body[channel][key]
                for body in bodies.values()
                for channel in ("physics", "usd")
            )
            for key in ("position_error_m", "quaternion_dot_error")
        }
        result["joint_error"] = joint_error
        if (
            result["position_error_m"] > 1e-4
            or joint_error > 1e-4
            or result["quaternion_dot_error"] > 1e-5
        ):
            raise ReplayRestoreError(
                {
                    "sequence": frame["sequence"],
                    "simulation_time_s": frame["simulation_time_s"],
                    "root_setup": self.root_setup,
                    "errors": result,
                    "bodies": bodies,
                }
            )
        return result

    def refresh(self):
        self.world.render()
        if self.world.current_time != self.fixed_simulation_time:
            raise ReplayTraceError("physics advanced during replay refresh")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--assets-sha", required=True)
    parser.add_argument("--player-source-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--allow-mimic-reconstruction",
        action="store_true",
        help="derive unrecorded passive joint q from pinned asset mimic constraints",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if (
        not math.isfinite(args.speed)
        or args.speed <= 0
        or not math.isfinite(args.hold_seconds)
        or args.hold_seconds < 0
    ):
        parser.error("speed must be positive and hold-seconds must be nonnegative")
    state = json.loads(args.state_file.read_text())
    inputs = validate_inputs(state, args.source_root, args.assets_root, args.assets_sha)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / (
        "validation.json" if args.validate_only else "replay-receipt.json"
    )
    if output.exists():
        raise ReplayTraceError(f"refusing to overwrite replay evidence: {output}")
    receipt = {
        "purpose": "replay",
        "player_source_sha": args.player_source_sha,
        "state_sha256": hashlib.sha256(args.state_file.read_bytes()).hexdigest(),
        "record_source": state["replay"]["record_source"],
        "clip": state["replay"]["clip"],
        "gaps": state["replay"]["gaps"],
        "frames": len(state["frames"]),
        "applied_frames": 0,
        "headless": args.headless,
        "speed": args.speed,
        "physics_steps_during_replay": 0,
        "status": "validated" if args.validate_only else "starting",
        "media_capture": False,
    }
    if args.validate_only:
        output.write_text(json.dumps(receipt, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "status": "validated",
                    "frames": receipt["frames"],
                    "gaps": len(receipt["gaps"]),
                }
            )
        )
        return
    app = None
    try:
        # All Omni/Isaac imports follow app creation. --help/--validate-only are CPU-only.
        from isaacsim import SimulationApp

        app = SimulationApp(
            {
                "headless": args.headless,
                "renderer": "RaytracedLighting",
                "limit_cpu_threads": 16,
            }
        )
        backend = IsaacReplay(
            app,
            inputs,
            state["frames"][0],
            args.allow_mimic_reconstruction,
            headless=args.headless,
        )
        receipt["root_setup"] = backend.root_setup
        receipt["camera_setup"] = backend.camera_setup
        receipt["joint_reconstruction"] = {
            "recorded_names": state["frames"][0]["robot"]["joints"]["joint_name"],
            "derived_mimic_joints": backend.mimic_joints,
        }
        began = time.monotonic()
        first_time = state["frames"][0]["simulation_time_s"]
        maxima = {
            "position_error_m": 0.0,
            "quaternion_dot_error": 0.0,
            "joint_error": 0.0,
        }
        for frame in state["frames"]:
            deadline = began + (frame["simulation_time_s"] - first_time) / args.speed
            while app.is_running() and time.monotonic() < deadline:
                backend.refresh()
                time.sleep(min(0.005, max(0, deadline - time.monotonic())))
            if not app.is_running():
                raise ReplayTraceError("viewer was closed before the clip finished")
            errors = apply_frame(frame, backend)
            maxima = {key: max(value, errors[key]) for key, value in maxima.items()}
            receipt.update(
                applied_frames=receipt["applied_frames"] + 1,
                last_sequence=frame["sequence"],
            )
        receipt.update(
            status="completed",
            maximum_restore_error=maxima,
            elapsed_s=time.monotonic() - began,
        )
        deadline = time.monotonic() + args.hold_seconds
        while app.is_running() and time.monotonic() < deadline:
            backend.refresh()
            time.sleep(0.005)
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        if isinstance(error, ReplayRestoreError):
            receipt["failure_details"] = error.details
        raise
    finally:
        # Isaac may terminate Python in close(); persist restoration facts first.
        output.write_text(json.dumps(receipt, indent=2) + "\n")
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
