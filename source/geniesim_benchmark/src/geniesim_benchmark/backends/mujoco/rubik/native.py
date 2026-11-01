# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Adapter for the hash-verified native cube physics assets."""

import hashlib
import importlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .cameras import camera_state, projection_parameters
from .actuation import NativePositionController
from .robot_render import native_to_usd_joint
from .cube_solved_checker import CubeSolvedChecker


class PhysicsError(RuntimeError):
    pass


class TaskFailure(RuntimeError):
    pass


def load_package(root):
    """Verify the source manifest before importing its executable physics code."""
    root = Path(root).expanduser().resolve()
    if mujoco.__version__ != "3.12.0":
        raise RuntimeError(f"Native cube requires MuJoCo 3.12.0; installed {mujoco.__version__}")
    manifest_path = root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("Rubik assets require schema 2; update assets feature/benchmark_baseline")
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing/unsafe native package file: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Native package checksum mismatch: {relative}")
    modules = {}
    for name in ("cube", "passive", "g2_contact_friction", "g2_hand_profile", "g2_closed_loop", "interarm_clearance"):
        modules[name] = importlib.import_module(f"{__package__}.physics.{name}")
    return root, modules, hashlib.sha256(manifest_path.read_bytes()).hexdigest()


class NativeEnvironment:
    def __init__(self, config):
        self.config = config
        self.root, self.modules, self.manifest_hash = load_package(config.package_root)
        mujoco.set_mjcb_passive(None)
        original = mujoco.MjModel.from_xml_path(str(self.root / "scene.xml"))
        scene = ET.parse(self.root / "scene.xml").getroot()
        scene.find("compiler").set("meshdir", str(self.root / "assets"))
        model = self.modules["g2_contact_friction"].with_internal_cube_friction(
            original, config.internal_friction, source_xml=ET.tostring(scene, encoding="unicode")
        )
        self.friction_changes = self.modules["g2_contact_friction"].apply_surface_friction(
            model, config.hand_friction_scale, config.cube_friction
        )
        self.model = model
        self.cube = self.modules["cube"].CubeState(model)
        if model.nu != 54 or model.neq != 0 or len(self.cube.items) != 26:
            raise ValueError("Native cube topology is not 54 robot motors / 26 cubies / zero equalities")
        joints = model.actuator_trnid[:, 0]
        if np.any(model.jnt_bodyid[joints] >= self.cube.root) or model.opt.timestep != 0.001:
            raise ValueError("Cube must have no motors and use the original 1 ms physics timestep")
        urdf = self.root / "assets/g2sim/urdf/G2_t2v2_crsB/G2_t2v2_crsB_wuji.urdf"
        self.native_velocity_limits = {
            joint.get("name"): float(joint.find("limit").get("velocity"))
            for joint in ET.parse(urdf).getroot().findall("joint")
            if joint.find("limit") is not None
        }
        self.motor_velocity_limits = np.array(
            [self.native_velocity_limits[native_to_usd_joint(model.joint(j).name)] for j in joints]
        )
        self.motor_joints = joints
        self.motor_q = model.jnt_qposadr[joints]
        initial = json.loads((self.root / "initial_state.json").read_text())
        self.initial_qpos = np.asarray(initial["qpos"], dtype=float)
        if self.initial_qpos.shape != (model.nq,) or not np.isfinite(self.initial_qpos).all():
            raise ValueError("Rubik initial state must contain one finite pose matching the compiled model")
        if initial["joint_names"] != [model.joint(i).name for i in range(model.njnt)]:
            raise ValueError("Rubik initial-state joint ordering does not match the compiled model")
        self.initial_joints = json.loads((self.root / "initial_joints.json").read_text())
        self.tips = self.modules["g2_closed_loop"].native_tips(model)
        self.interarm = self.modules["interarm_clearance"].InterArmClearance(model)
        self.renderers = {}
        self.viewer = None
        self.graphics = None
        self.task_name = "rubik_cube_solve_1turn"
        self.scene_catalog = None
        if config.scene_archive:
            from .scenes import SceneCatalog

            self.scene_catalog = SceneCatalog(config.scene_archive)
        self.maximum_integration_error = 0.0
        self.steps = 0
        self.data = None
        self.solved_checker = None

    def reset(self, scramble, scene=None):
        """Only reset/initialize before stepping a fresh MjData."""
        self.close_viewer()
        mujoco.set_mjcb_passive(None)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = self.initial_qpos
        if scene is None:
            self.cube.initialize(self.data, scramble)
        else:
            from .scenes import initialize_scene

            initialize_scene(self.model, self.data, self.cube, scene)
        self.current_scene = scene
        self.solved_checker = CubeSolvedChecker(
            self.model,
            angle_tolerance_deg=self.config.alignment_degrees,
            hold_seconds=self.config.hold_seconds,
            cube_module=self.modules["cube"],
        )
        self.modules["passive"].install_detents(
            self.model, self.config.detent_stiffness, self.config.center_detent_stiffness, self.config.guide_stiffness
        )
        mujoco.mj_forward(self.model, self.data)
        if self.config.hand_controller == "natural":
            self.controller = self.modules["g2_hand_profile"].NaturalG2Controller(self.model, self.data)
        else:
            groups = self.modules["g2_closed_loop"].G2Controller(self.model, self.data)
            self.controller = NativePositionController(
                self.model,
                self.data,
                groups,
                self.modules["g2_hand_profile"].NativeHandProfile(self.model),
                self.motor_velocity_limits,
                self.config.action_time_scale,
            )
        self.target = self.data.qpos.copy()
        self.data.ctrl[:] = self.target[self.motor_q]
        self.steps = 0
        self.maximum_integration_error = 0.0
        self.initial_height = float(self.data.xpos[self.cube.root, 2])
        self.arm_q = np.r_[self.controller.armq["l"], self.controller.armq["r"]]
        self.hand_q = np.r_[self.controller.handq["l"], self.controller.handq["r"]]
        self.action_q = np.r_[self.arm_q, self.hand_q]
        self.action_joints = np.array([np.flatnonzero(self.model.jnt_qposadr == q)[0] for q in self.action_q])
        self.action_bounds = self.model.jnt_range[self.action_joints].copy()
        self.arm_velocity_limits = np.array(
            [
                self.native_velocity_limits[native_to_usd_joint(self.model.joint(j).name)]
                for j in self.action_joints[:14]
            ]
        )
        self.maximum_arm_speeds = np.zeros(14)
        self.arm_overspeed_ticks = np.zeros(14, dtype=int)
        self.maximum_robot_penetration = 0.0
        self.maximum_cross_arm_penetration = 0.0
        self.robot_collision_ticks = 0
        self.cross_arm_collision_ticks = 0
        for offset, side in ((14, "l"), (34, "r")):
            lower, upper = self.controller.hand_profile.bounds(side)
            self.action_bounds[offset : offset + 20] = np.c_[lower, upper]
        expected = (
            scene.facelets
            if scene is not None
            else self.modules["cube"].facelets(self.modules["cube"].logical_cubies(scramble))[0]
        )
        if self.cube.decode(self.data)[0] != expected:
            raise PhysicsError("Initialization did not produce the requested scramble")
        return self.sample()

    def action_contract(self):
        return {
            "version": 1,
            "kind": "JOINT_ABS",
            "units": "radians",
            "dimension": 54,
            "order": ["left_arm[7]", "right_arm[7]", "left_effector[20]", "right_effector[20]"],
            "joint_names": [self.model.joint(j).name for j in self.action_joints],
            "bounds": self.action_bounds.tolist(),
            "control_hz": self.config.control_hz,
            "action_time_scale": self.config.action_time_scale,
            "interpolation": "quintic smoothstep" if self.config.hand_controller == "native" else "expert filter",
            "hand_target_tolerance": self.config.hand_target_tolerance,
            "physics_hz": 1000,
            "waypoint_timing": "nominal control interval stretched to satisfy motor reference speed limits",
            "motor_velocity_limits_radians_per_second": self.motor_velocity_limits.tolist(),
            "motor_joint_names": [self.model.joint(j).name for j in self.motor_joints],
            "motor_command_speed_limits_radians_per_second": (
                self.controller.velocity_limits.tolist()
                if isinstance(self.controller, NativePositionController)
                else None
            ),
            "fingertip_force_feedforward": False,
            "arm_bias_compensation": self.config.hand_controller == "natural",
            "arm_velocity_limits_radians_per_second": self.arm_velocity_limits.tolist(),
            "arm_speed_limit_enforcement": "substep_interpolation_with_actual_speed_check",
            "motor_target_semantics": (
                "speed-bounded substep interpolation of position actuator waypoints"
                if self.config.hand_controller == "native"
                else "expert filtered joint poses"
            ),
            "hand_controller": type(self.controller).__name__,
            "hand_joint_limits": self.config.hand_controller,
        }

    def set_action(self, action, minimum_steps=None):
        action = np.asarray(action, dtype=float)
        if action.shape != (54,) or not np.isfinite(action).all():
            raise ValueError("Expected 54 finite absolute joint targets")
        if np.any(action < self.action_bounds[:, 0] - 1e-6) or np.any(action > self.action_bounds[:, 1] + 1e-6):
            raise ValueError("Policy joint target is outside the declared bounds")
        self.target[self.action_q] = action
        if minimum_steps is None:
            minimum_steps = max(1, round(1 / (self.config.control_hz * self.model.opt.timestep)))
        if isinstance(self.controller, NativePositionController):
            return self.controller.set_target(self.target[self.motor_q], minimum_steps)
        return max(1, int(np.ceil(minimum_steps * self.config.action_time_scale)))

    def configure_renderer(self, headless, output):
        if not self.config.render:
            return
        if self.config.renderer == "isaac":
            from .isaac_render import IsaacRenderer

            self.graphics = IsaacRenderer(self, headless, output)

    def open_viewer(self):
        from .viewer import NativeViewer

        self.close_viewer()
        self.viewer = self.graphics if self.graphics is not None else NativeViewer(self.model, self.data)

    def close_viewer(self):
        if self.viewer is not None:
            if self.viewer is not self.graphics:
                self.viewer.close()
            self.viewer = None

    def tick(self):
        previous = self.data.qpos.copy()
        self.controller.step(self.target)
        self.steps += 1
        integrated = previous.copy()
        mujoco.mj_integratePos(self.model, integrated, self.data.qvel, self.model.opt.timestep)
        error = float(np.max(np.abs(integrated - self.data.qpos)))
        self.maximum_integration_error = max(self.maximum_integration_error, error)
        if error > 1e-9 or abs(self.data.time - self.steps * 0.001) > 1e-7:
            raise PhysicsError("State integration or simulation clock discontinuity")
        if (
            not np.isfinite(self.data.qpos).all()
            or not np.isfinite(self.data.qvel).all()
            or any(w.number for w in self.data.warning)
        ):
            raise PhysicsError("Nonfinite dynamics state or MuJoCo warning")
        if np.any(self.data.qfrc_applied) or np.any(self.data.xfrc_applied):
            raise PhysicsError("External generalized/body force is forbidden")
        mujoco.mj_forward(self.model, self.data)
        limited = self.model.actuator_forcelimited.astype(bool)
        forces, bounds = self.data.actuator_force[limited], self.model.actuator_forcerange[limited]
        if np.any(forces < bounds[:, 0] - 1e-8) or np.any(forces > bounds[:, 1] + 1e-8):
            raise PhysicsError("Actuator effort limit violated")
        q, limits = self.data.qpos[self.motor_q], self.model.jnt_range[self.motor_joints]
        if np.any(q < limits[:, 0] - 0.01) or np.any(q > limits[:, 1] + 0.01):
            raise TaskFailure("joint_limit")
        # Interpolating actuator references bounds commanded speed; physical
        # contact/inertia can still violate it, so keep a separate actual check.
        speeds = np.abs(self.data.qvel[self.model.jnt_dofadr[self.action_joints[:14]]])
        self.maximum_arm_speeds = np.maximum(self.maximum_arm_speeds, speeds)
        self.arm_overspeed_ticks += speeds > self.arm_velocity_limits
        if np.any(speeds > 1.01 * self.arm_velocity_limits):
            raise TaskFailure("arm_speed_limit")
        robot_pairs = self.model.geom_bodyid[self.data.contact.geom]
        enabled = np.all((robot_pairs > 0) & (robot_pairs < self.cube.root), axis=1)
        penetration = max(0.0, -float(np.min(self.data.contact.dist[enabled]))) if np.any(enabled) else 0.0
        gap, _ = self.interarm.minimum(self.data, 0.0)
        cross_penetration = max(0.0, -float(gap))
        self.maximum_robot_penetration = max(self.maximum_robot_penetration, penetration)
        self.maximum_cross_arm_penetration = max(self.maximum_cross_arm_penetration, cross_penetration)
        self.robot_collision_ticks += penetration > 0.002
        self.cross_arm_collision_ticks += cross_penetration > 0.0005
        # This switches failure classification only. Keep all native collision
        # masks, exclusions and contact forces intact during diagnosis.
        if self.config.terminate_on_robot_collision:
            if penetration > 0.002:
                raise TaskFailure("self_collision")
            if cross_penetration > 0.0005:
                raise TaskFailure("cross_arm_collision")
        sample = self.sample()
        if sample["height"] < self.initial_height - self.config.drop_distance:
            raise TaskFailure("drop")
        if sample["legal"] is False:
            raise PhysicsError("Aligned cube state is illegal")
        return sample

    def robot_collision_diagnostics(self):
        return {
            "termination_enabled": self.config.terminate_on_robot_collision,
            "physical_contacts_unchanged": True,
            "self_collision_threshold_meters": 0.002,
            "cross_arm_collision_threshold_meters": 0.0005,
            "maximum_robot_contact_penetration_meters": self.maximum_robot_penetration,
            "maximum_cross_arm_penetration_meters": self.maximum_cross_arm_penetration,
            "self_collision_threshold_ticks": self.robot_collision_ticks,
            "cross_arm_collision_threshold_ticks": self.cross_arm_collision_ticks,
        }

    def arm_speed_diagnostics(self):
        return {
            "enforcement": "substep_interpolation_with_actual_speed_check",
            "joint_names": [self.model.joint(j).name for j in self.action_joints[:14]],
            "asset_limits_radians_per_second": self.arm_velocity_limits.tolist(),
            "maximum_radians_per_second": self.maximum_arm_speeds.tolist(),
            "ticks_above_asset_limit": self.arm_overspeed_ticks.tolist(),
            "physics_timestep_seconds": float(self.model.opt.timestep),
        }

    def checker_contract(self):
        provenance = json.loads(Path(__file__).with_name("cube_checker_provenance.json").read_text())
        return {
            **provenance,
            "implementation_sha256": hashlib.sha256(
                Path(__file__).with_name("cube_solved_checker.py").read_bytes()
            ).hexdigest(),
            "angle_tolerance_degrees": self.config.alignment_degrees,
            "position_tolerance_meters": (
                self.solved_checker.position_tolerance_m if self.solved_checker is not None else None
            ),
            "hold_seconds": self.config.hold_seconds,
            "requires_pickup": False,
            "requires_support_contacts": False,
            "evaluation_frequency_hz": 1000,
        }

    def sample(self):
        checker = self.solved_checker.check(self.data)
        state, error = checker["facelets"], checker["max_angle_error_deg"]
        fingers = self.modules["g2_closed_loop"].supporting_fingers(self.model, self.data, self.cube.root, self.tips)
        return {
            "time": float(self.data.time),
            "facelets": state,
            "angle_error": error,
            "cube_checker": checker,
            "legal": self.modules["cube"].legal_state(state),
            "support_fingers": len(fingers),
            "supporting_pads": sorted(fingers),
            "height": float(self.data.xpos[self.cube.root, 2]),
            "cube_position": self.data.xpos[self.cube.root].tolist(),
        }

    def states(self):
        fixed = self.initial_joints["init_joint_pos"]
        head = np.deg2rad([fixed[f"idx1{i}_head_joint{i}"] for i in range(1, 4)]).tolist()
        waist = np.deg2rad([fixed[f"idx0{i}_body_joint{i}"] for i in range(1, 6)]).tolist()
        from scipy.spatial.transform import Rotation

        end_pose = {}
        for frame in ("base_link", "arm_base_link"):
            base = self.model.body(frame).id
            R = self.data.xmat[base].reshape(3, 3)
            end_pose[frame] = {}
            for side, name in (("l", "left_arm"), ("r", "right_arm")):
                body = self.model.body("arm_" + side + "_end_link").id
                end_pose[frame][name] = {
                    "position": (R.T @ (self.data.xpos[body] - self.data.xpos[base])).tolist(),
                    "orientation": Rotation.from_matrix(R.T @ self.data.xmat[body].reshape(3, 3)).as_quat().tolist(),
                }
        return {
            "arm_joint_states": self.data.qpos[self.arm_q].tolist(),
            "gripper_states": self.data.qpos[self.hand_q].tolist(),
            "head_joint_states": head,
            "waist_joint_states": waist,
            "end_pose": end_pose,
        }

    def images(self):
        """Render configured mounted cameras without editing physical model arrays."""
        if self.graphics is not None:
            return self.graphics.images()
        if set(self.config.cameras) != {"head", "hand_left", "hand_right"}:
            raise ValueError("Configure head, hand_left and hand_right cameras")
        images = {}
        for name, camera in self.config.cameras.items():
            width, height = camera["width"], camera["height"]
            if type(width) is not int or type(height) is not int or not 1 <= width <= 2048 or not 1 <= height <= 2048:
                raise ValueError("Camera dimensions must be positive integers <=2048")
            pose = camera_state(self.model, self.data, camera)
            eye, forward, up = (np.asarray(pose[key]) for key in ("eye", "forward", "up"))
            intrinsics = projection_parameters(camera)["intrinsics"]
            size = (width, height)
            if size not in self.renderers:
                self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, width)
                self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, height)
                self.renderers[size] = mujoco.Renderer(self.model, height=height, width=width)
            renderer = self.renderers[size]
            option = mujoco.MjvOption()
            option.geomgroup[2] = 0
            renderer.update_scene(self.data, scene_option=option)
            for gl_camera in renderer.scene.camera:
                gl_camera.pos[:] = eye
                gl_camera.forward[:] = forward
                gl_camera.up[:] = up
                near = gl_camera.frustum_near
                fx, fy, cx, cy = (intrinsics[key] for key in ("fx", "fy", "cx", "cy"))
                gl_camera.frustum_top = near * cy / fy
                gl_camera.frustum_bottom = -near * (height - cy) / fy
                gl_camera.frustum_width = near * width / fx
                gl_camera.frustum_center = near * (width / 2 - cx) / fx
                gl_camera.orthographic = 0
            images[name] = renderer.render().copy()
        return images

    def close(self):
        self.close_viewer()
        if self.graphics is not None:
            self.graphics.close()
            self.graphics = None
        for renderer in self.renderers.values():
            renderer.close()
        mujoco.set_mjcb_passive(None)
