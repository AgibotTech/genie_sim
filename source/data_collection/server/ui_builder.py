# Copyright (c) 2023-2026, AgiBot Inc.
# SPDX-License-Identifier: MPL-2.0

import asyncio
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import yaml
import omni.kit.app
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.prims import (
    delete_prim,
    get_prim_at_path,
    get_prim_children,
    get_prim_object_type,
)
from isaacsim.core.utils.types import ArticulationAction

# Perception sensors

# ros2bridge
from pxr import Usd, UsdUtils

from common.deformable_compat import create_deformable_body_view, select_deformable_bodies
from common.base_utils.logger import logger
from common.base_utils.transform_utils import mat2quat_wxyz, make_pose, quat2mat_wxyz
from curobo.types import Pose as CuroboPose
from sim_backend import physics, render
from sim_backend import backend as _bk
from server.controllers.kinematics_solver import KinematicsSolver
from server.controllers.ruckig_move import RuckigController
from server.motion_generator.motion_gen_reacher import CuroboMotion


class UIBuilder:
    def __init__(self, world: World, debug=False, enable_deformable: bool = False):
        self.debug = debug
        self.enable_deformable = enable_deformable
        # FEM deformable tensor-view containers:
        #   fem_sim_view: create_simulation_view("torch") handle, None for an empty scene.
        #   cloth_views[prim_path]: surface or volume deformable body views.
        # Initialized after my_world.reset() triggers FEM cooking.
        self.fem_sim_view = None
        self.cloth_views: Dict[str, Any] = {}
        self.frames = []
        self.wrapped_ui_elements = []
        self.collision_paths = []
        self._Joint_Info_Sliders = []
        self._Sensor_parent = None
        self._Camera_parent = None
        self.camera: Any = None
        self.depth_camera: Any = None
        self.articulation = None
        self.articulation_rmpflow = None
        self.right_articulation_rmpflow = None
        self._taskspace_trajectory_generator = None
        self._target = None
        self._right_target = None
        self._currentCamera = ""
        self._followingPos = np.array([0, 0, 0])
        self._followingOrientation = np.array([1, 0, 0, 0])
        self.my_world: World = world
        self.currentImg = None
        self.currentCamInfo = None
        self.curoboMotion: Dict[str, CuroboMotion] = {}
        self.current_curobo_motion: Optional[CuroboMotion] = None
        self.camera_prim_list = []
        self.camera_list = []
        self.rmp_move = False
        self.cmd_list = None
        self.reached = False
        self.cameras = []
        self.art_controllers = []
        # Per-arm dual trajectory state (used by DualLinearMove)
        self.dual_cmd_lists = {}   # {arm_key: list | None}
        self.dual_idx_lists = {}   # {arm_key: list[int]}
        self.dual_cmd_idxs = {}    # {arm_key: int}
        self.dual_reached = {}     # {arm_key: bool}

    def init_cloth_views(self) -> Dict[str, Any]:
        """Create one Isaac 6 tensor view per deformable body hierarchy.

        View factories bind the body root. Both the root and simulation-prim
        paths are registered as aliases. A zero-count view or unreadable nodal
        tensor means cooking is incomplete and is retried on the next call.
        """
        if not self.enable_deformable:
            return self.cloth_views
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return self.cloth_views
        if self.fem_sim_view is None:
            try:
                from omni.physics.tensors import create_simulation_view  # noqa: PLC0415
                stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
                self.fem_sim_view = create_simulation_view("torch", stage_id=stage_id)
                self.fem_sim_view.set_subspace_roots("/")
            except Exception as exc:
                logger.warning(f"[cloth] create_simulation_view('torch') failed: {exc}")
                self.fem_sim_view = None
                return self.cloth_views
        records = []
        for prim in stage.Traverse():
            try:
                records.append((str(prim.GetPath()), tuple(prim.GetAppliedSchemas())))
            except Exception as exc:
                logger.warning(f"[cloth] read applied schemas failed for {prim.GetPath()}: {exc}")
        specs = select_deformable_bodies(records)
        for spec in specs:
            root_path, simulation_path, kind = spec
            if root_path in self.cloth_views:
                continue
            try:
                view = create_deformable_body_view(self.fem_sim_view, spec)
                if view is None or int(getattr(view, "count", 0)) == 0:
                    logger.warning(
                        f"[cloth] {kind} view root={root_path} count=0; "
                        "deformable cook may still be incomplete"
                    )
                    continue
                positions = view.get_simulation_nodal_positions()
                if positions is None or positions.shape[0] == 0 or positions.shape[1] == 0:
                    logger.warning(
                        f"[cloth] {kind} view root={root_path} has no cooked simulation nodes"
                    )
                    continue
                import torch  # noqa: PLC0415
                view._fem_body_idx_cache = torch.arange(
                    int(view.count), dtype=torch.int32, device=positions.device
                )
                view._fem_simulation_path = simulation_path
                self.cloth_views[root_path] = view
                self.cloth_views[simulation_path] = view
                logger.info(
                    f"[cloth] {kind} deformable view ready root={root_path} "
                    f"simulation={simulation_path} count={view.count} "
                    f"nodes/body={positions.shape[1]}"
                )
            except Exception as exc:
                logger.warning(
                    f"[cloth] create {kind} view root={root_path} failed: {exc}"
                )
        if not self.cloth_views:
            logger.warning(
                f"[cloth] no cooked Isaac 6 deformable body view found; specs={specs}"
            )
        return self.cloth_views

    def _init_solver(self, robot, enable_curobo, batch_num):
        self.enable_curobo = enable_curobo
        self.robot_name = robot.robot_name
        self.robot_prim_path = robot.robot_prim_path
        self.dof_nums = robot.dof_nums
        self.lock_joints = robot.lock_joints
        self.joint_delta_time = robot.joint_delta_time
        self.curobo_config_file = robot.curobo_config_file
        self.cameras = robot.cameras
        self.init_joint_position = robot.init_joint_position
        self.end_effector_prim_path = robot.end_effector_prim_path
        # Set arm_type before initializing the articulation so dual-arm robots can
        # reuse the planner and avoid allocating a second CUDA planner.
        self.arm_type = robot.arm_type
        self.end_effector_name = robot.end_effector_name
        self.initialize_articulation(batch_num)
        self.active_arm_joints = robot.active_arm_joints

    def _init_kinematic_solver(self, robot):
        self._robot_cfg_for_solver = robot
        if robot.arm_type == "dual":
            self.kinematics_solver = {
                "left": KinematicsSolver(
                    robot_description_path=robot.robot_description_path["left"],
                    urdf_path=robot.urdf_name,
                    end_effector_name=robot.end_effector_name["left"],
                    articulation=self.articulation,
                ),
                "right": KinematicsSolver(
                    robot_description_path=robot.robot_description_path["right"],
                    urdf_path=robot.urdf_name,
                    end_effector_name=robot.end_effector_name["right"],
                    articulation=self.articulation,
                ),
            }
        else:
            self.kinematics_solver = KinematicsSolver(
                robot_description_path=robot.robot_description_path,
                urdf_path=robot.urdf_name,
                end_effector_name=robot.end_effector_name,
                articulation=self.articulation,
            )
        self._lula_rotated = False

    def _check_and_update_kinematics_solver(self):
        """Compare articulation fixed-joint values with the kinematics YAML.
        Update the YAML and rebuild the solver when they differ."""
        if not hasattr(self, 'kinematics_solver') or self.articulation is None:
            return

        tolerance = 1e-2

        def _get_fixed_joints_from_yaml(robot_description_path):
            with open(robot_description_path, "r") as f:
                desc = yaml.safe_load(f)
            fixed_joints = {}
            for rule in desc.get("cspace_to_urdf_rules", []):
                if rule.get("rule") == "fixed":
                    fixed_joints[rule["name"]] = rule["value"]
            return fixed_joints

        def _needs_update(solver):
            fixed_joints = _get_fixed_joints_from_yaml(solver.robot_description_path)
            for joint_name, yaml_value in fixed_joints.items():
                dof_idx = self.articulation.get_dof_index(joint_name)
                current_positions = physics.get_joint_positions(self.articulation, as_torch=False)
                actual_value = current_positions[dof_idx]
                if abs(actual_value - yaml_value) > tolerance:
                    return True
            return False

        def _rebuild_solver(solver):
            robot_description_path = solver.robot_description_path
            with open(robot_description_path, "r") as f:
                desc = yaml.safe_load(f)
            current_positions = physics.get_joint_positions(self.articulation, as_torch=False)
            for rule in desc.get("cspace_to_urdf_rules", []):
                if rule.get("rule") == "fixed":
                    dof_idx = self.articulation.get_dof_index(rule["name"])
                    rule["value"] = float(current_positions[dof_idx])
            rotated_path = robot_description_path.replace("_tmp.yaml", "_tmp_rotated.yaml")
            with open(rotated_path, "w") as f:
                yaml.dump(desc, f, default_flow_style=False)
            logger.info(f"Updated lula robot_description: {rotated_path}")
            return rotated_path

        if isinstance(self.kinematics_solver, dict):
            for key in self.kinematics_solver:
                if _needs_update(self.kinematics_solver[key]):
                    rotated_path = _rebuild_solver(self.kinematics_solver[key])
                    robot = self._robot_cfg_for_solver
                    self.kinematics_solver[key] = KinematicsSolver(
                        robot_description_path=rotated_path.replace(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/robot_cfg", ""
                        ),
                        urdf_path=robot.urdf_name,
                        end_effector_name=robot.end_effector_name[key],
                        articulation=self.articulation,
                    )
                    self._lula_rotated = True
        else:
            if _needs_update(self.kinematics_solver):
                rotated_path = _rebuild_solver(self.kinematics_solver)
                robot = self._robot_cfg_for_solver
                self.kinematics_solver = KinematicsSolver(
                    robot_description_path=rotated_path.replace(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/robot_cfg", ""
                    ),
                    urdf_path=robot.urdf_name,
                    end_effector_name=robot.end_effector_name,
                    articulation=self.articulation,
                )
                self._lula_rotated = True

    def on_physics_step(self, step):
        for curoboMotion in self.curoboMotion.values():
            curoboMotion.on_physics_step()

    def _on_capture_cam(self, isRGB, isDepth, isSemantic):
        if self._currentCamera:
            resolution = [640, 480]
            if self._currentCamera in self.cameras:
                resolution = self.cameras[self._currentCamera]
            _Camera = render.make_camera(prim_path=self._currentCamera, resolution=resolution)
            # This path returns metadata and empty image fields. Initializing
            # a 6.2 CameraSensor creates a distinct render product per call;
            # repeated resets would retain those unused GPU resources. Read
            # authoring properties only; ROS image publishers own their sensors.
            focal_length = _Camera.get_focal_length()
            horizontal_aperture = _Camera.get_horizontal_aperture()
            vertical_aperture = _Camera.get_vertical_aperture()
            width, height = _Camera.get_resolution()
            fx = width * focal_length / horizontal_aperture
            fy = height * focal_length / vertical_aperture
            ppx = width * 0.5
            ppy = height * 0.5
            self.currentCamInfo = {
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
                "ppx": ppx,
                "ppy": ppy,
            }
            self.currentImg = {}
            self.currentImg["camera_info"] = self.currentCamInfo
            self.currentImg["rgb"] = []
            self.currentImg["depth"] = []
            self.currentImg["semantic"] = ()

    def _Generate_following_position(self, isRight=True):
        curoboMotion = self.get_curobo_motion(isRight)
        if curoboMotion:
            curoboMotion.target = XFormPrim(
                "/World/target",
                position=self._followingPos,
                orientation=self._followingOrientation,
            )
            # On GPU, XFormPrim construction writes the initial pose to USD by default.
            # Fabric can still hold the old pose read by cuRobo target queries.
            # A stale IK target can make the planned path collide with the table.
            # Explicitly call the patched wrapper setter to synchronize the target pose.
            # The torch path writes Fabric; the CPU path repeats the USD write.
            curoboMotion.target.set_world_pose(
                position=self._followingPos,
                orientation=self._followingOrientation,
            )
            target_world = XFormPrim(
                f"{self.robot_prim_path}/target",
            )
            target_world.set_local_pose(
                translation=self._followingPos, orientation=self._followingOrientation
            )
            if self.arm_type == "dual":
                key = self.end_effector_name["left"]
                if isRight:
                    key = self.end_effector_name["right"]
                    curoboMotion.target_links[key] = XFormPrim(
                        "/World/target_" + key,
                        position=np.array(self._followingPos),
                        orientation=np.array(self._followingOrientation),
                    )
                    # Likewise, synchronize target_link through the wrapper setter.
                    curoboMotion.target_links[key].set_world_pose(
                        position=np.array(self._followingPos),
                        orientation=np.array(self._followingOrientation),
                    )
            self._target = None

    def _follow_target(
        self,
        isRight=True,
        goal_offset=[0, 0, 0, 1, 0, 0, 0],
        path_constraint=None,
        offset_and_constraint_in_goal_frame=True,
        disable_collision_links=[],
        from_current_pose=False,
    ):
        self._Generate_following_position(isRight)
        self.set_locked_joint_positions(isRight)
        curoboMotion = self.get_curobo_motion(isRight)
        curoboMotion.set_obstacles()
        # Pass following position/orientation directly to calculate_ik_goal.
        # This bypasses reading the target marker prim, which can return stale
        # poses if the wrapper patch is unavailable or Fabric has not updated.
        # A common symptom is the second target equaling the current EE pose,
        # producing a stationary plan with an apparent pre-grasp offset.
        # CPU USD reads and writes are synchronized and do not show this issue.
        curoboMotion.calculate_ik_goal(
            goal_offset=goal_offset,
            path_constraint=path_constraint,
            offset_and_constraint_in_goal_frame=offset_and_constraint_in_goal_frame,
            disable_collision_links=disable_collision_links,
            from_current_pose=from_current_pose,
            target_position=self._followingPos,
            target_orientation=self._followingOrientation,
        )
        if self.arm_type == "dual":
            js_names = self.active_arm_joints["left"]
            if isRight:
                js_names = self.active_arm_joints["right"]
            curoboMotion.exclude_js(js_names)

    def _follow_trajectory(
        self,
        trajectory_positions,
        trajectory_orientations,
        is_right,
        active_ee_name,
        solvable_threshold=0.5,
        max_consecutive_failures=3,
    ):
        curoboMotion = self.get_curobo_motion(is_right)
        self.set_locked_joint_positions(is_right)
        curoboMotion.set_obstacles()
        curoboMotion.follow_trajectory(
            trajectory_positions,
            trajectory_orientations,
            solvable_threshold=solvable_threshold,
            max_consecutive_failures=max_consecutive_failures,
            active_ee_name=active_ee_name,
        )
        if self.arm_type == "dual":
            js_names = self.active_arm_joints["left"]
            if is_right:
                js_names = self.active_arm_joints["right"]
            curoboMotion.exclude_js(js_names)

    def _follow_bimanual_rigid(
        self,
        target_position,
        target_rotation,
        leader_arm,
        leader_ee_name,
        goal_offset=None,
        path_constraint=None,
        offset_and_constraint_in_goal_frame=True,
        disable_collision_links=None,
        max_consecutive_failures=1,
        from_current_pose=False,
    ):
        is_leader_right = leader_arm == "right"
        follower_arm = "left" if is_leader_right else "right"
        leader_motion = self.get_curobo_motion(is_leader_right)
        leader_tool = leader_ee_name or leader_motion.ee_link_name
        follower_tool = None
        if isinstance(self.end_effector_name, dict):
            leader_tool = self.end_effector_name.get(leader_arm, leader_tool)
            follower_tool = self.end_effector_name.get(follower_arm)
        follower_motion = self.get_curobo_motion(not is_leader_right)
        if follower_tool is None:
            follower_tool = next(
                (ln for ln in leader_motion.link_names if ln != leader_tool), None
            )
        if leader_tool not in leader_motion.link_names or follower_tool not in leader_motion.link_names:
            logger.error(
                "_follow_bimanual_rigid: shared planner must contain leader=%s and follower=%s tools; got %s",
                leader_tool,
                follower_tool,
                leader_motion.link_names,
            )
            leader_motion.reached = True
            leader_motion.success = False
            follower_motion.reached = True
            follower_motion.success = False
            return

        self.set_locked_joint_positions(is_leader_right)
        if follower_motion is not leader_motion:
            self.set_locked_joint_positions(not is_leader_right)
        leader_motion.ee_link_name = leader_tool
        leader_motion._active_is_right = is_leader_right
        leader_motion.set_obstacles()
        if follower_motion is not leader_motion:
            follower_motion.set_obstacles()

        leader_cu_js, _ = leader_motion._get_current_joint_state()

        leader_kin = leader_motion.planner.compute_kinematics(leader_cu_js[0:1])
        pose_leader0 = leader_kin.tool_poses.get_link_pose(leader_tool)
        pose_leader0 = CuroboPose(
            position=pose_leader0.position.contiguous(),
            quaternion=pose_leader0.quaternion.contiguous(),
        )

        pose_follower0 = leader_kin.tool_poses.get_link_pose(follower_tool)
        pose_follower0 = CuroboPose(
            position=pose_follower0.position.contiguous(),
            quaternion=pose_follower0.quaternion.contiguous(),
        )
        pose_rel = pose_leader0.inverse().multiply(pose_follower0)

        if from_current_pose:
            target_goal = pose_leader0.clone()
        else:
            target_goal = CuroboPose(
                position=leader_motion.tensor_args.to_device(
                    np.asarray(target_position, dtype=np.float64).reshape(1, 3)
                ),
                quaternion=leader_motion.tensor_args.to_device(
                    np.asarray(target_rotation, dtype=np.float64).reshape(1, 4)
                ),
            )
        if goal_offset is not None:
            offset = CuroboPose.from_list(goal_offset)
            if offset_and_constraint_in_goal_frame:
                target_goal_with_offset = target_goal.clone().multiply(offset)
            else:
                target_goal_with_offset = offset.clone().multiply(target_goal.clone())
        else:
            target_goal_with_offset = target_goal
        follower_goal = target_goal_with_offset.multiply(pose_rel)

        leader_trajectories = leader_motion.plan_bimanual_leader(
            target_position=target_position,
            target_rotation=target_rotation,
            follower_ee_link=follower_tool,
            follower_ee_pose=follower_goal,
            goal_offset=goal_offset,
            path_constraint=path_constraint,
            offset_and_constraint_in_goal_frame=offset_and_constraint_in_goal_frame,
            disable_collision_links=disable_collision_links,
            from_current_pose=from_current_pose,
            active_tool=leader_tool,
        )

        if leader_trajectories is None or len(leader_trajectories) == 0:
            logger.info("_follow_bimanual_rigid: bimanual planning failed")
            leader_motion.reached = True
            leader_motion.success = False
            follower_motion.reached = True
            follower_motion.success = False
            return

        _, chosen_traj = leader_trajectories[0]
        sim_js_names = [
            n for n in physics.get_dof_names(leader_motion.robot)
            if n not in leader_motion.lock_js_names
        ]
        leader_motion.idx_list = []
        common_js_names = []
        for x in sim_js_names:
            if x in chosen_traj.joint_names:
                leader_motion.idx_list.append(leader_motion.robot.get_dof_index(x))
                common_js_names.append(x)
        leader_motion.cmd_plan = chosen_traj.reorder(common_js_names)
        leader_motion.cmd_idx = 0
        leader_motion.reached = False
        leader_motion.success = True
        if follower_motion is not leader_motion:
            follower_motion.cmd_plan = None
            follower_motion.cmd_idx = 0
            follower_motion.reached = False
            follower_motion.success = True
            leader_js_names = self.active_arm_joints["right" if is_leader_right else "left"]
            follower_js_names = self.active_arm_joints["left" if is_leader_right else "right"]
            leader_motion.exclude_js(leader_js_names)
            follower_motion.exclude_js(follower_js_names)

        logger.info(
            "_follow_bimanual_rigid: planning complete, %d steps for shared bimanual trajectory",
            len(leader_motion.cmd_plan.position),
        )

    def _get_ee_pose(self, is_right, is_local=False):
        # SingleArticulation world poses may be CUDA tensors with the torch backend.
        # Lula C++ bindings, SciPy Rotation, and numpy pose helpers require CPU arrays.
        # Convert explicitly before passing poses to those consumers.
        robot_base_translation, robot_base_orientation = self.articulation.get_world_pose()
        robot_base_translation = _bk.to_numpy(robot_base_translation)
        robot_base_orientation = _bk.to_numpy(robot_base_orientation)
        if is_local:
            robot_base_translation = np.array([0.0, 0.0, 0.0])
            robot_base_orientation = np.array([1.0, 0.0, 0.0, 0.0])

        if self.arm_type == "dual":
            key = "left"
            if is_right:
                key = "right"
            self.kinematics_solver[key]._kinematics_solver.set_robot_base_pose(
                robot_base_translation, robot_base_orientation
            )
            return self.kinematics_solver[
                key
            ]._articulation_kinematics_solver.compute_end_effector_pose()
        self.kinematics_solver._kinematics_solver.set_robot_base_pose(
            robot_base_translation, robot_base_orientation
        )
        return self.kinematics_solver._articulation_kinematics_solver.compute_end_effector_pose()

    def _get_ee_pose_curobo(self, is_right, is_local=False):
        # As in _get_ee_pose, numpy pose helpers require CPU arrays.
        robot_base_translation, robot_base_orientation = self.articulation.get_world_pose()
        robot_base_translation = _bk.to_numpy(robot_base_translation)
        robot_base_orientation = _bk.to_numpy(robot_base_orientation)
        if is_local:
            robot_base_translation = np.array([0.0, 0.0, 0.0])
            robot_base_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        # curoboMotion uses left/right string keys; indexing it with a boolean fails.
        # Use get_curobo_motion to select the arm consistently.
        position, rotation = self.get_curobo_motion(is_right).get_ee_pose()
        ee_rel_robot_pose = make_pose(position, quat2mat_wxyz(rotation))
        robot_pose = make_pose(robot_base_translation, quat2mat_wxyz(robot_base_orientation))
        ee_pose = robot_pose @ ee_rel_robot_pose
        position = ee_pose[:3, 3]
        rotation_matrix = ee_pose[:3, :3]
        return position, rotation_matrix

    def _get_ik_status(self, target_position, target_orientation, isRight):
        # As in _get_ee_pose, Lula base-pose updates require SciPy/C++ compatible arrays.
        robot_base_translation, robot_base_orientation = self.articulation.get_world_pose()
        robot_base_translation = _bk.to_numpy(robot_base_translation)
        robot_base_orientation = _bk.to_numpy(robot_base_orientation)
        if self.arm_type == "dual":
            key = "left"
            if isRight:
                key = "right"
            self.kinematics_solver[key]._kinematics_solver.set_robot_base_pose(
                robot_base_translation, robot_base_orientation
            )
            actions, success = self.kinematics_solver[
                key
            ]._articulation_kinematics_solver.compute_inverse_kinematics(
                target_position, target_orientation
            )
        else:
            self.kinematics_solver._kinematics_solver.set_robot_base_pose(
                robot_base_translation, robot_base_orientation
            )
            actions, success = (
                self.kinematics_solver._articulation_kinematics_solver.compute_inverse_kinematics(
                    target_position, target_orientation
                )
            )
        return success, actions

    def _limit_joint_positions(self, positions, joint_indices=None):
        if joint_indices is None:
            joint_indices = np.arange(len(positions))
        # Avoid dof_properties, which inserts CUDA tensors into a numpy structured array
        # and triggers an unsupported implicit numpy conversion.
        # Use sim_backend.physics.get_dof_limits for numpy lower/upper limits.
        lower_all, upper_all = physics.get_dof_limits(self.articulation, as_torch=False)
        lowers = lower_all[joint_indices]
        uppers = upper_all[joint_indices]
        positions = np.clip(positions, lowers, uppers)
        return positions

    def _safe_set_joint_positions(self, positions, joint_indices=None):
        positions = self._limit_joint_positions(positions, joint_indices)
        physics.set_joint_positions(self.articulation, positions, joint_indices=joint_indices)

    def _move_to(self, target_positions, joint_indices=None, is_trajectory=False, is_action=False):
        if not self.articulation:
            return
        self._currentLeftTask = 10
        self._currentRightTask = 10
        actions = ArticulationAction(joint_positions=target_positions)
        if not is_trajectory:
            if joint_indices is None:
                joint_indices = []
                positions = []
                for idx, joint_position in enumerate(target_positions):
                    if joint_position is not None:
                        positions.append(joint_position)
                        joint_indices.append(idx)
                self._safe_set_joint_positions(positions, joint_indices=joint_indices)
            else:
                self._safe_set_joint_positions(target_positions, joint_indices=joint_indices)
        else:
            self.reached = True
            physics.apply_action(self.articulation, actions)

    def _trajectory_list_follow_target(
        self,
        target_position,
        target_orientation,
        is_right,
        ee_interpolation=False,
        distance_frame=0.01,
    ):
        current_position, rotation_matrix = self._get_ee_pose(is_right)
        XFormPrim("/ruckig", position=target_position, orientation=target_orientation)
        current_rotation = mat2quat_wxyz(rotation_matrix)
        if not self._get_ik_status(target_position, target_orientation, is_right)[0]:
            self.cmd_list = None
            self.reached = True
            logger.info("IK not success")
            return
        target_arm_positions = self._get_ik_status(target_position, target_orientation, is_right)[
            1
        ].joint_positions
        self.idx_list = self._get_ik_status(target_position, target_orientation, is_right)[
            1
        ].joint_indices
        current_arm_positions = physics.get_joint_positions(self.articulation, joint_indices=self.idx_list, as_torch=False)

        def lerp(start, end, t):
            return start + t * (end - start)

        def slerp(q0, q1, t):
            dot = np.dot(q0, q1)
            if dot < 0.0:
                q1 = -q1
                dot = -dot

            dot = min(max(dot, -1.0), 1.0)
            theta_0 = np.arccos(dot)
            theta = theta_0 * t
            sin_theta = np.sin(theta)
            sin_theta_0 = np.sin(theta_0)
            if sin_theta == 0:
                return q0
            s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
            s1 = sin_theta / sin_theta_0

            return s0 * q0 + s1 * q1

        self.cmd_list = []
        if ee_interpolation:
            distance = np.linalg.norm(target_position - current_position)
            joint_distance = 0
            step = (int)(distance / distance_frame)
            if step > 1:
                for i in range(step):
                    t = i / (step - 1)
                    position = lerp(current_position, target_position, t)
                    rotation = slerp(current_rotation, target_orientation, t)
                    issuccess, arm_position = self._get_ik_status(position, rotation, is_right)
                    joint_distance = np.linalg.norm(
                        arm_position.joint_positions - current_arm_positions
                    )
                    if joint_distance < 1:
                        self.cmd_list.append(arm_position.joint_positions)
                        current_arm_positions = arm_position.joint_positions
        else:
            cmd_list = self.ruckig_controller.caculate_trajectory(
                current_arm_positions, target_arm_positions
            )
            distance = np.linalg.norm(target_position - current_position)
            for position in cmd_list:
                joint_distance = np.linalg.norm(
                    np.array(current_arm_positions) - np.array(position)
                )
                current_arm_positions = position
                self.cmd_list.append(position)
        self.cmd_idx = 0
        self.reached = False
        self.time_index = 0
        if not self.cmd_list:
            self.reached = True

    def _on_every_frame_trajectory_list(self):
        if self.cmd_list and self.articulation:
            self.time_index += 1
            cmd_state = self.cmd_list[self.cmd_idx]
            art_action = ArticulationAction(joint_positions=cmd_state, joint_indices=self.idx_list)
            for robot in self.art_controllers:
                physics.apply_action(robot, art_action)
            physics.apply_action(self.articulation, art_action)
            self.cmd_idx += 1
            if self.cmd_idx >= len(self.cmd_list):
                self.cmd_idx = 0
                self.cmd_list = None
                self.reached = True

    def _trajectory_list_follow_target_arm(
        self,
        target_position,
        target_orientation,
        arm_key,
        ee_interpolation=False,
        distance_frame=0.01,
    ):
        """Per-arm version of _trajectory_list_follow_target for simultaneous dual-arm execution."""
        is_right = arm_key == "right"
        current_position, rotation_matrix = self._get_ee_pose(is_right)
        XFormPrim("/ruckig", position=target_position, orientation=target_orientation)
        current_rotation = mat2quat_wxyz(rotation_matrix)
        if not self._get_ik_status(target_position, target_orientation, is_right)[0]:
            self.dual_cmd_lists[arm_key] = None
            self.dual_reached[arm_key] = True
            logger.info(f"IK not success for arm {arm_key}")
            return
        target_arm_positions = self._get_ik_status(target_position, target_orientation, is_right)[
            1
        ].joint_positions
        idx_list = self._get_ik_status(target_position, target_orientation, is_right)[
            1
        ].joint_indices
        current_arm_positions = physics.get_joint_positions(self.articulation, joint_indices=idx_list, as_torch=False)

        def lerp(start, end, t):
            return start + t * (end - start)

        def slerp(q0, q1, t):
            dot = np.dot(q0, q1)
            if dot < 0.0:
                q1 = -q1
                dot = -dot
            dot = min(max(dot, -1.0), 1.0)
            theta_0 = np.arccos(dot)
            theta = theta_0 * t
            sin_theta = np.sin(theta)
            sin_theta_0 = np.sin(theta_0)
            if sin_theta == 0:
                return q0
            s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
            s1 = sin_theta / sin_theta_0
            return s0 * q0 + s1 * q1

        cmd_list = []
        if ee_interpolation:
            distance = np.linalg.norm(target_position - current_position)
            step = int(distance / distance_frame)
            if step > 1:
                for i in range(step):
                    t = i / (step - 1)
                    position = lerp(current_position, target_position, t)
                    rotation = slerp(current_rotation, target_orientation, t)
                    issuccess, arm_position = self._get_ik_status(position, rotation, is_right)
                    joint_distance = np.linalg.norm(
                        arm_position.joint_positions - current_arm_positions
                    )
                    if joint_distance < 1:
                        cmd_list.append(arm_position.joint_positions)
                        current_arm_positions = arm_position.joint_positions
        else:
            ruckig_cmds = self.ruckig_controller.caculate_trajectory(
                current_arm_positions, target_arm_positions
            )
            for position in ruckig_cmds:
                cmd_list.append(position)
                current_arm_positions = position

        self.dual_cmd_lists[arm_key] = cmd_list if cmd_list else None
        self.dual_idx_lists[arm_key] = idx_list
        self.dual_cmd_idxs[arm_key] = 0
        self.dual_reached[arm_key] = not bool(cmd_list)

    def _on_every_frame_dual_trajectory(self):
        """Apply one step of each arm's trajectory simultaneously (for DualLinearMove)."""
        if not self.dual_cmd_lists or not self.articulation:
            return
        combined_positions = []
        combined_indices = []
        for arm_key in sorted(self.dual_cmd_lists.keys()):
            if self.dual_reached.get(arm_key, True):
                continue
            cmd_list = self.dual_cmd_lists.get(arm_key)
            if not cmd_list:
                self.dual_reached[arm_key] = True
                continue
            idx = self.dual_cmd_idxs[arm_key]
            combined_positions.extend(cmd_list[idx])
            combined_indices.extend(self.dual_idx_lists[arm_key])
            self.dual_cmd_idxs[arm_key] += 1
            if self.dual_cmd_idxs[arm_key] >= len(cmd_list):
                self.dual_cmd_lists[arm_key] = None
                self.dual_reached[arm_key] = True
        if combined_positions:
            art_action = ArticulationAction(
                joint_positions=combined_positions, joint_indices=combined_indices
            )
            for robot in self.art_controllers:
                physics.apply_action(robot, art_action)
            physics.apply_action(self.articulation, art_action)
        self.articulation = None

    def initialize_articulation(self, batch_num=0):
        scene = self.my_world.scene
        if scene._scene_registry.name_exists(self.robot_name):
            self.articulation = scene.get_object(self.robot_name)
        else:
            # On GPU, register new articulations before PhysX warmup constructs the view.
            # Otherwise PxDirectGPUAPI can receive invalid articulation GPU buffers
            # and fail in _on_physics_ready with
            # "data and/or gpuIndices has to be valid pointer".
            # The caller may have already warmed up via my_world.play(). Stop first to
            # invalidate the old view, register the articulation, then reset to rebuild it.
            # Reset performs stop/play and initializes physics with the new articulation
            # before finalizing registered scene prims.
            # CPU views do not use the direct GPU buffers; preserve the original path
            # without adding another reset.
            need_rebuild = _bk.tensors_as() == "torch"
            if need_rebuild:
                need_rebuild = SimulationManager.get_physics_sim_view() is not None
            if need_rebuild:
                self.my_world.stop()
            self.articulation = Articulation(prim_path=self.robot_prim_path, name=self.robot_name)
            scene.add(self.articulation)
            if need_rebuild:
                self.my_world.reset()
        self.articulation.initialize()
        self.ruckig_controller = RuckigController(self.dof_nums, self.joint_delta_time)
        robot_list = []
        for idx in range(batch_num):
            articulation = Articulation(
                prim_path=self.robot_prim_path + "_{}".format(idx),
                name=self.robot_name + "_{}".format(idx),
            )
            articulation.initialize()
            robot_list.append(articulation)
            self.art_controllers = [r.get_articulation_controller() for r in robot_list]
        if self.enable_curobo:
            for key, cfg in self.curobo_config_file.items():
                if not self.curoboMotion.get(key):
                    # v2 dual-arm YAML includes both tool frames, so one BatchMotionPlanner is enough.
                    # Alias the second arm key to the existing CuroboMotion instance
                    # instead of building another large CUDA planner.
                    # Duplicate planners can exhaust GPU memory.
                    # Reuse only when the existing planner tool_frames includes the requested
                    # gripper link, read from the robot end_effector_name configuration.
                    # Dual-arm configs map left/right to links; single-arm configs use a string.
                    # Avoid hardcoding gripper_r_center_link because robot link names differ.
                    if self.arm_type == "dual" and self.curoboMotion:
                        want_tool = None
                        if isinstance(self.end_effector_name, dict):
                            want_tool = self.end_effector_name.get(key)
                        existing_key, existing_inst = next(iter(self.curoboMotion.items()))
                        existing_tools = getattr(existing_inst, "tool_frames", []) or []
                        if want_tool and want_tool in existing_tools:
                            logger.info(
                                f"[dual] aliasing curoboMotion['{key}'] → '{existing_key}' "
                                f"(shared planner, target tool '{want_tool}' present)"
                            )
                            self.curoboMotion[key] = existing_inst
                            continue
                    before = time.time()
                    curobo_motion = self._init_curobo(cfg, key)
                    after = time.time()
                    logger.info(f"init curobo motion for {key} takes {after - before}")
                    self.curoboMotion[key] = curobo_motion
                else:
                    if self.arm_type == "dual":
                        for effector_key, value in self.end_effector_name.items():
                            if effector_key == "left":
                                current_position, rotation_matrix = self._get_ee_pose(
                                    False, is_local=True
                                )
                            else:

                                current_position, rotation_matrix = self._get_ee_pose(
                                    True, is_local=True
                                )
                            self.curoboMotion.get(key).init_ee_pose[value] = {
                                "position": current_position,
                                "orientation": mat2quat_wxyz(rotation_matrix),
                            }
                        self.curoboMotion.get(key).reset_link()
                    else:
                        # Recreate IK target markers for single-arm robots after reset clears them.
                        # Otherwise solve_batch_ik cannot find /World/target_<ee>.
                        is_right = self.arm_type != "left"
                        current_position, rotation_matrix = self._get_ee_pose(
                            is_right, is_local=True
                        )
                        self.curoboMotion.get(key).init_ee_pose[self.end_effector_name] = {
                            "position": current_position,
                            "orientation": mat2quat_wxyz(rotation_matrix),
                        }
                        self.curoboMotion.get(key).reset_link()
                    self.curoboMotion.get(key).reset()

    def _init_curobo(self, curobo_config, name):
        if self.articulation:
            curoboMotion = CuroboMotion(
                name,
                self.articulation,
                self.my_world,
                curobo_config,
                self.robot_prim_path,
                self.art_controllers,
                step=32,
                debug=self.debug,
                simplify_mesh=True,
            )
            curoboMotion.set_obstacles()
            return curoboMotion
        return None

    def get_curobo_motion(self, is_right=True):
        # v2 dual-arm keys can alias the same CuroboMotion instance.
        # Refresh ee_link_name and _active_is_right for the selected arm
        # so default active_tool and CRS filtering use the correct side.
        # Read the link name from end_effector_name rather than hardcoding it.
        key = "right" if is_right else "left"
        inst = self.curoboMotion.get(key)
        if inst is None:
            raise RuntimeError(
                f"get_curobo_motion(is_right={is_right}): curoboMotion['{key}'] is None. "
                f"Init sequence broken — check initialize_articulation()."
            )
        target_tool = None
        if isinstance(self.end_effector_name, dict):
            target_tool = self.end_effector_name.get(key)
        elif isinstance(self.end_effector_name, str):
            target_tool = self.end_effector_name
        tool_frames = getattr(inst, "tool_frames", None) or []
        if target_tool and target_tool in tool_frames and inst.ee_link_name != target_tool:
            inst.ee_link_name = target_tool
        # The v2 CRS bound filter needs the arm side; store it explicitly
        # instead of inferring it from a hardcoded right-gripper link name.
        inst._active_is_right = is_right
        return inst

    def attach_objs(self, prim_path_list, is_right=True):
        result = False
        curoboMotion = self.get_curobo_motion(is_right)
        if curoboMotion:
            logger.info("Attach!!!!")

            link_name = "attached_object"
            position, rotation_matrix = self._get_ee_pose(is_right, is_local=True)
            rotation = mat2quat_wxyz(rotation_matrix)
            if self.arm_type == "dual" and not is_right:
                link_name = "left_attached_object"
            result = curoboMotion.attach_obj(prim_path_list, link_name, position, rotation)
            curoboMotion.view_debug_world()
        return result

    def detach_objs(self):
        if self.articulation:
            for curoboMotion in self.curoboMotion.values():
                curoboMotion.detach_obj()
                logger.info("Detach!!!!!")
                curoboMotion.view_debug_world()

    def remove_objects_from_world(self, prim_paths):
        if self.articulation:
            for curoboMotion in self.curoboMotion.values():
                curoboMotion.remove_objects_from_world(prim_paths)

    def set_locked_joint_positions(self, is_right=True):
        if not self.lock_joints:
            return
        articulation = self.articulation
        joint_positions = physics.get_joint_positions(articulation, as_torch=False)
        curoboMotion = self.get_curobo_motion(is_right)
        ids = {}
        dof_names = physics.get_dof_names(articulation)
        for idx in range(len(dof_names)):
            name = dof_names[idx]
            if name in curoboMotion.lock_js_names:
                ids[name] = float(joint_positions[idx])
        curoboMotion.update_lock_joints(ids)

    def _on_reset(self):
        async def _on_rest_async():
            await omni.kit.app.get_app().next_update_async()
            self.initialize_articulation()
            self.rmp_move = False

        asyncio.ensure_future(_on_rest_async())

        return

    def _find_all_objects_of_type(self, obj_type):
        items = []
        stage = omni.usd.get_context().get_stage()
        if stage:
            for prim in Usd.PrimRange(stage.GetPrimAtPath("/")):
                path = str(prim.GetPath())
                type = get_prim_object_type(path)
                if type == obj_type:
                    items.append(path)
        return items

    def _initialize_object_articulations(self):
        articulations = self._find_all_objects_of_type("articulation")
        for art in articulations:
            articulation = Articulation(art)
            if articulation:
                articulation.initialize()

    def set_articulation_state(self, state: bool):
        articulations = self._find_all_objects_of_type("articulation")
        for art in articulations:
            _prim = get_prim_at_path(art)
            if art not in self.robot_prim_path:
                _prim.GetAttribute("physxArticulation:articulationEnabled").Set(state)

    def remove_objects(self):
        parent_prim = get_prim_at_path("/World/Objects")
        if parent_prim.IsValid():
            prims = get_prim_children(parent_prim)
            for prim in prims:
                delete_prim(prim.GetPath())

    def remove_graph(self, prim_paths):
        for prim in prim_paths:
            replicator_prim = get_prim_at_path(prim)
            if replicator_prim.IsValid():
                delete_prim(prim)
