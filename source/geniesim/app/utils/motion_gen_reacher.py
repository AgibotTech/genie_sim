# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

# @misc{curobo_report23,
#       title={cuRobo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation},
#       author={Balakumar Sundaralingam and Siva Kumar Sastry Hari and Adam Fishman and Caelan Garrett
#               and Karl Van Wyk and Valts Blukis and Alexander Millane and Helen Oleynikova and Ankur Handa
#               and Fabio Ramos and Nathan Ratliff and Dieter Fox},
#       year={2023},
#       eprint={2310.17274},
#       archivePrefix={arXiv},
#       primaryClass={cs.RO}
# }

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig, Cuboid
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.types.camera import CameraObservation
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)
from curobo.geom.sphere_fit import SphereFitType
import os

from isaacsim.core.api.objects import cuboid, sphere
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.sensors.camera import Camera


import omni.replicator.core as rep
from scipy.spatial.transform import Rotation as R
import carb
import numpy as np
import time
import torch


class CuroboMotion:
    def __init__(
        self,
        robot: SingleArticulation,
        world: World,
        robot_cfg,
        robot_prim_path,
        robot_list,
        step=80,
    ):
        self.usd_help = UsdHelper()
        self.target_pose = None
        self.target_orientation = None
        self.past_pose = None
        self.past_orientation = None
        self.robot_list = robot_list
        tensor_args = TensorDeviceType()
        self.robot_prim_path = robot_prim_path
        n_obstacle_cuboids = 30
        self.init_ee_pose = {}
        n_obstacle_mesh = 100

        robot_cfg_path = get_robot_configs_path()
        self.robot_cfg = load_yaml(join_path(robot_cfg_path, robot_cfg))["robot_cfg"]
        self.robot_cfg["kinematics"]["extra_collision_spheres"] = {
            "attached_object": 30,
            "left_attached_object": 30,
        }
        self.lock_joints = self.robot_cfg["kinematics"]["lock_joints"]
        self.lock_js_names = []
        if self.lock_joints:
            for key in self.lock_joints:
                self.lock_js_names.append(key)
        j_names = self.robot_cfg["kinematics"]["cspace"]["joint_names"]
        default_config = self.robot_cfg["kinematics"]["cspace"]["retract_config"]

        self.world_cfg = WorldConfig()
        motion_gen_config = MotionGen.load_from_robot_config(
            robot_cfg=self.robot_cfg,
            world_model=self.world_cfg,
            tensor_args=tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=True,
            num_trajopt_seeds=4,
            num_graph_seeds=2,
            num_ik_seeds=32,
            num_batch_ik_seeds=96,
            interpolation_dt=0.01,
            interpolation_steps=5000,
            collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
            optimize_dt=True,
            trajopt_dt=None,
            trajopt_tsteps=step,
            num_trajopt_noisy_seeds=1,
            num_batch_trajopt_seeds=2,
            collision_activation_distance=0.0005,
        )

        self.tensor_args = tensor_args
        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup(parallel_finetune=True)
        self.world_model = self.motion_gen.world_collision
        self.plan_config = MotionGenPlanConfig(
            enable_graph=False,
            enable_graph_attempt=4,
            max_attempts=10,
            enable_finetune_trajopt=True,
            parallel_finetune=True,
            time_dilation_factor=0.6,
        )
        self.target = SingleXFormPrim(
            "/World/target",
            position=np.array([0.5, 0, 0.5]),
            orientation=np.array([0, 1, 0, 0]),
        )
        self.cmd_plan = None
        self.cmd_plans = []
        self.cmd_idx = 0
        self.num_targets = 0
        self.past_cmd = None
        self.pose_metic = None
        self.robot = robot
        self.robot._articulation_view.initialize()
        self.idx_list = [self.robot.get_dof_index(x) for x in j_names]
        self.robot.set_joint_positions(default_config, self.idx_list)
        self.robot._articulation_view.set_max_efforts(
            values=np.array([5000 for i in range(len(self.idx_list))]),
            joint_indices=self.idx_list,
        )
        self.reached = False
        self.success = False
        self.saved_poses = []
        self.my_world = World(stage_units_in_meters=1.0)
        stage = self.my_world.stage
        self.usd_help.load_stage(stage)
        self.time_index = 0
        self.spheres = None
        self.set_obstacles()
        self.link_names = self.motion_gen.kinematics.link_names
        self.ee_link_name = self.motion_gen.kinematics.ee_link
        kin_state = self.motion_gen.kinematics.get_state(self.motion_gen.get_retract_config().view(1, -1))
        link_retract_pose = kin_state.link_pose
        self.target_links = {}
        for i in self.link_names:
            k_pose = np.ravel(link_retract_pose[i].to_list())
            self.target_links[i] = SingleXFormPrim(
                "/World/target_" + i,
                position=np.array(k_pose[:3]),
                orientation=np.array(k_pose[3:]),
            )

    def reset(self):
        self.motion_gen.clear_world_cache()

    def reset_link(self):
        self.link_names = self.motion_gen.kinematics.link_names
        self.ee_link_name = self.motion_gen.kinematics.ee_link
        for i in self.link_names:
            self.target_links[i] = SingleXFormPrim(
                "/World/target_" + i,
                position=self.init_ee_pose[i]["position"],
                orientation=self.init_ee_pose[i]["orientation"],
            )

    def set_obstacles(self):
        obstacle = self.usd_help.get_obstacles_from_stage(
            reference_prim_path=self.robot_prim_path,
            ignore_substring=[
                "/World/background",
                "/World/huojia/Xform_01",
                "/G1",
                "/curobo",
            ],
        ).get_collision_check_world()
        self.motion_gen.update_world(obstacle)
        self.world_cfg = obstacle

    def solve_batch_ik(self, positions, rotations, end_effector_name):
        goal_pose = Pose(
            position=self.tensor_args.to_device(positions),
            quaternion=self.tensor_args.to_device(rotations),
        )
        link_poses = None
        if len(self.link_names) > 1:
            link_poses = {}
            for i in self.target_links.keys():
                c_p, c_rot = self.target_links[i].get_world_pose()
                if i == end_effector_name:
                    link_poses[i] = Pose(
                        position=self.tensor_args.to_device(positions),
                        quaternion=self.tensor_args.to_device(rotations),
                    )
                else:
                    link_poses[i] = Pose(
                        position=self.tensor_args.to_device(c_p),
                        quaternion=self.tensor_args.to_device(c_rot),
                    )
            c_p, c_rot = self.target_links[self.ee_link_name].get_world_pose()
            if end_effector_name == self.ee_link_name:
                goal_pose = Pose(
                    position=self.tensor_args.to_device(positions),
                    quaternion=self.tensor_args.to_device(rotations),
                )
            else:
                goal_pose = Pose(
                    position=self.tensor_args.to_device(c_p),
                    quaternion=self.tensor_args.to_device(c_rot),
                )
        result = self.motion_gen.ik_solver.solve_single(goal_pose, link_poses=link_poses)
        return result.success, result.js_solution

    # nvblox camera
    def add_camera(self, camera_prim, position, orientation):
        depth_camera = Camera(
            prim_path=camera_prim,
            resolution=[320, 240],
            position=np.array([0, 0, 0]),
            orientation=np.array([0.707, 0.707, 0, 0]),
        )
        depth_camera.initialize()
        depth_camera.set_world_pose(position=position, orientation=orientation, camera_axes="usd")
        rp = depth_camera._render_product_path
        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        rgb_annot.attach(rp)
        depth_annot.attach(rp)
        rgb_np = rgb_annot.get_data()
        depth_np = depth_annot.get_data()
        return depth_camera

    def update_camera(self, world_model, tensor_args, depth_camera, camera_prim):
        def matrix_to_quaternion_and_translation(matrix):
            translation = matrix[:3, 3]
            rotation_matrix = matrix[:3, :3]
            rotation = R.from_matrix(rotation_matrix)
            quaternion_xyzw = rotation.as_quat()  # quaternion [x, y, z, w]
            quaternion = np.array(
                [
                    quaternion_xyzw[3],
                    quaternion_xyzw[0],
                    quaternion_xyzw[1],
                    quaternion_xyzw[2],
                ]
            )
            return translation, quaternion

        def get_pose(xyz: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
            def get_rotation_matrix_from_quaternion(quat: np.ndarray) -> np.ndarray:
                w, x, y, z = quat
                rot = np.array(
                    [
                        [
                            2 * (w**2 + x**2) - 1,
                            2 * (x * y - w * z),
                            2 * (x * z + w * y),
                        ],
                        [
                            2 * (x * y + w * z),
                            2 * (w**2 + y**2) - 1,
                            2 * (y * z - w * x),
                        ],
                        [
                            2 * (x * z - w * y),
                            2 * (y * z + w * x),
                            2 * (w**2 + z**2) - 1,
                        ],
                    ]
                )
                return rot

            pose = np.eye(4)
            pose[:3, :3] = get_rotation_matrix_from_quaternion(quat_wxyz)
            pose[:3, 3] = xyz
            return pose

        rotation_x_180 = np.array([[1.0, 0.0, 0.0, 0], [0.0, -1.0, 0.0, 0], [0.0, 0.0, -1.0, 0], [0, 0, 0, 1]])
        depth_camera_prim = SingleXFormPrim(prim_path=camera_prim)
        rp = depth_camera._render_product_path
        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        semantic_annot = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")
        if rp:
            rgb_annot.attach(rp)
            depth_annot.attach(rp)
            semantic_annot.attach(rp)
        semantics = semantic_annot.get_data()

        depth_np = depth_annot.get_data()
        depth_np[depth_np > 2] = 0
        depth_np[semantics["data"] == 2] = 0
        intrinsics_np = depth_camera.get_intrinsics_matrix()
        pose_np = get_pose(*depth_camera_prim.get_world_pose()) @ rotation_x_180
        depth = torch.from_numpy(depth_np).float()
        pose = torch.from_numpy(pose_np).float()
        intrinsics = torch.from_numpy(intrinsics_np).float()
        data = {
            "depth": depth,
            "pose": pose,
            "intrinsics": intrinsics,
            "raw_depth": depth_np,
        }
        cube_position, cube_orientation = matrix_to_quaternion_and_translation(pose_np)
        camera_pose = Pose(
            position=tensor_args.to_device(cube_position),
            quaternion=tensor_args.to_device(cube_orientation),
        )
        data_camera = CameraObservation(  # rgb_image = data["rgba_nvblox"],
            depth_image=data["depth"], intrinsics=data["intrinsics"], pose=camera_pose
        )
        data_camera = data_camera.to(device=tensor_args.device)
        world_model.add_camera_frame(data_camera, "world")

    def update_lock_joints(self, locked_joints):
        self.motion_gen.update_locked_joints(locked_joints, self.robot_cfg)

    def caculate_ik_goal(self, plan_js=False, joint_state=None):
        self.reached = False
        cube_position, cube_orientation = self.target.get_world_pose()
        if self.past_pose is None:
            self.past_pose = cube_position
        if self.target_pose is None:
            self.target_pose = cube_position
        if self.target_orientation is None:
            self.target_orientation = cube_orientation
        if self.past_orientation is None:
            self.past_orientation = cube_orientation
        sim_js = self.robot.get_joints_state()
        js_names = self.robot.dof_names
        sim_js_names = []
        lock_idx = []
        sim_js_positions = []
        sim_js_velocities = []
        for idx, name in enumerate(js_names):
            if name not in self.lock_js_names:
                sim_js_names.append(name)
            else:
                lock_idx.append(idx)
        for idx, position in enumerate(sim_js.positions):
            if idx not in lock_idx:
                sim_js_positions.append(position)
        for idx, velocity in enumerate(sim_js.velocities):
            if idx not in lock_idx:
                sim_js_velocities.append(velocity)
        cu_js = JointState(
            position=self.tensor_args.to_device(sim_js_positions),
            velocity=self.tensor_args.to_device(sim_js_velocities) * 0.0,
            acceleration=self.tensor_args.to_device(sim_js_velocities) * 0.0,
            jerk=self.tensor_args.to_device(sim_js_velocities) * 0.0,
            joint_names=sim_js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)
        start_time = time.time()

        ee_translation_goal = cube_position
        ee_orientation_teleop_goal = cube_orientation
        ik_goal = Pose(
            position=self.tensor_args.to_device(ee_translation_goal),
            quaternion=self.tensor_args.to_device(ee_orientation_teleop_goal),
        )
        self.plan_config.pose_cost_metric = self.pose_metic
        link_poses = None
        if len(self.link_names) > 1:
            link_poses = {}
            for i in self.target_links.keys():
                c_p, c_rot = self.target_links[i].get_world_pose()
                link_poses[i] = Pose(
                    position=self.tensor_args.to_device(c_p),
                    quaternion=self.tensor_args.to_device(c_rot),
                )
                if i == self.ee_link_name:
                    ik_goal = Pose(
                        position=self.tensor_args.to_device(c_p),
                        quaternion=self.tensor_args.to_device(c_rot),
                    )
        if plan_js:
            target_js_positions = []
            for name, value in joint_state.items():
                if name in sim_js_names:
                    target_js_positions.append(value)
            target_js = JointState(
                position=self.tensor_args.to_device(target_js_positions),
                velocity=self.tensor_args.to_device(sim_js_velocities) * 0.0,
                acceleration=self.tensor_args.to_device(sim_js_velocities) * 0.0,
                jerk=self.tensor_args.to_device(sim_js_velocities) * 0.0,
                joint_names=sim_js_names,
            )
            target_js = target_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)
            result = self.motion_gen.plan_single_js(
                cu_js.unsqueeze(0), target_js.unsqueeze(0), self.plan_config.clone()
            )
        else:
            result = self.motion_gen.plan_single(
                cu_js.unsqueeze(0),
                ik_goal,
                self.plan_config.clone(),
                link_poses=link_poses,
            )
        succ = result.success.item()

        if succ:
            self.reached = False
            self.success = True
            carb.log_warn("end_time is{}".format(time.time() - start_time))
            self.num_targets += 1
            self.cmd_plan = result.get_interpolated_plan()
            self.cmd_plan = self.motion_gen.get_full_js(self.cmd_plan)

            self.idx_list = []
            common_js_names = []
            for x in sim_js_names:
                if x in self.cmd_plan.joint_names:
                    self.idx_list.append(self.robot.get_dof_index(x))
                    common_js_names.append(x)
            self.cmd_plan = self.cmd_plan.get_ordered_joint_state(common_js_names)
            self.cmd_idx = 0
            carb.log_warn("success")
        else:
            self.reached = True
            self.success = False
            carb.log_warn("plan did not converge to a solution")
        self.target_pose = cube_position
        self.target_orientation = cube_orientation
        self.past_pose = cube_position
        self.past_orientation = cube_orientation

    def exclude_js(self, joint_names):
        if self.cmd_plan:
            positions = []
            velocities = []
            self.idx_list = []
            for name in joint_names:
                self.idx_list.append(self.robot.get_dof_index(name))
            for index, pos in enumerate(self.cmd_plan.position):
                position = []
                velocity = []
                for name in joint_names:
                    idx = self.cmd_plan.joint_names.index(name)
                    position.append(pos.cpu().numpy()[idx])
                    velocity.append(self.cmd_plan.velocity.cpu().numpy()[index][idx])
                positions.append(position)
                velocities.append(velocity)
            self.cmd_plan = JointState(
                position=self.tensor_args.to_device(positions),
                velocity=self.tensor_args.to_device(velocities) * 0.0,
                acceleration=self.tensor_args.to_device(velocities) * 0.0,
                jerk=self.tensor_args.to_device(velocities) * 0.0,
                joint_names=joint_names,
            )

    def detach_obj(self):
        self.motion_gen.detach_object_from_robot()
        self.set_obstacles()

    def visualize_spheres(
        self,
        sph_list,
        prim_prefix="/curobo/robot_sphere_",
        color=np.array([0, 0.8, 0.2]),
    ):
        if self.spheres is None:
            self.spheres = []
            for si, s in enumerate(sph_list[0]):
                sp = sphere.VisualSphere(
                    prim_path=prim_prefix + str(si),
                    position=np.array([s.position[0], s.position[1], s.position[2]]),
                    radius=float(s.radius),
                    color=color,
                )
                self.spheres.append(sp)
        else:
            for si, s in enumerate(sph_list[0]):
                if not np.isnan(s.position[0]):
                    self.spheres[si].set_world_pose(
                        position=np.array([s.position[0] - 0.4, s.position[1], s.position[2] - 0.55])
                    )
                    self.spheres[si].set_radius(float(s.radius))

    def visualize_robot_spheres(self):
        sim_js = self.robot.get_joints_state()
        sim_js_names = self.robot.dof_names
        cu_js = JointState(
            position=self.tensor_args.to_device(sim_js.positions),
            velocity=self.tensor_args.to_device(sim_js.velocities),
            acceleration=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )
        cu_js.acceleration *= 0.0
        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)
        sph_list = self.motion_gen.kinematics.get_robot_as_spheres(cu_js.position)
        self.visualize_spheres(sph_list, prim_prefix="/curobo/robot_sphere_")

    def attach_obj(
        self,
        prim_paths,
        link_name="attached_object",
        ee_position=[0, 0, 0],
        ee_rotation=[1, 0, 0, 0],
    ):
        attach_result = False
        self.set_obstacles()

        ee_pose = Pose(
            position=self.tensor_args.to_device(ee_position),
            quaternion=self.tensor_args.to_device(ee_rotation),
        )
        attach_result = self.attach_objects_to_robot(
            object_names=prim_paths,
            link_name=link_name,
            sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
            surface_sphere_radius=0.005,
            world_objects_pose_offset=Pose.from_list([0, 0, 0.005, 1, 0, 0, 0], self.tensor_args),
            remove_obstacles_from_world_config=True,
            ee_pose=ee_pose,
        )

        return attach_result

    def attach_objects_to_robot(
        self,
        object_names,
        surface_sphere_radius: float = 0.001,
        link_name: str = "attached_object",
        sphere_fit_type: SphereFitType = SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
        voxelize_method: str = "ray",
        world_objects_pose_offset=None,
        remove_obstacles_from_world_config: bool = False,
        ee_pose=None,
    ) -> bool:
        if world_objects_pose_offset is not None:
            ee_pose = world_objects_pose_offset.inverse().multiply(ee_pose)
        ee_pose = ee_pose.inverse()  # ee_T_w to multiply all objects later
        max_spheres = self.motion_gen.robot_cfg.kinematics.kinematics_config.get_number_of_spheres(link_name)
        n_spheres = int(max_spheres / len(object_names))
        sphere_tensor = torch.zeros((max_spheres, 4))
        sphere_tensor[:, 3] = -10.0
        sph_list = []
        if n_spheres == 0:
            return False
        for i, x in enumerate(object_names):
            obs = self.motion_gen.world_model.get_obstacle(x)
            sph = obs.get_bounding_spheres(
                n_spheres,
                surface_sphere_radius,
                pre_transform_pose=ee_pose,
                tensor_args=self.tensor_args,
                fit_type=sphere_fit_type,
                voxelize_method=voxelize_method,
            )
            sph_list += [s.position + [s.radius] for s in sph]
            self.motion_gen.world_coll_checker.enable_obstacle(enable=False, name=x)
            if remove_obstacles_from_world_config:
                self.motion_gen.world_model.remove_obstacle(x)
        spheres = self.tensor_args.to_device(torch.as_tensor(sph_list))

        if spheres.shape[0] > max_spheres:
            spheres = spheres[: spheres.shape[0]]
        sphere_tensor[: spheres.shape[0], :] = spheres.contiguous()

        self.motion_gen.attach_spheres_to_robot(sphere_tensor=sphere_tensor, link_name=link_name)
        return True

    def on_physics_step(self):
        self.time_index += 1

        if self.cmd_plan is not None:
            cmd_state = self.cmd_plan[self.cmd_idx]
            self.past_cmd = cmd_state.clone()
            art_action = ArticulationAction(
                cmd_state.position.cpu().numpy(),
                cmd_state.velocity.cpu().numpy(),
                joint_indices=self.idx_list,
            )
            self.robot.apply_action(art_action)
            self.cmd_idx += 1
            if self.cmd_idx >= len(self.cmd_plan.position):
                self.cmd_idx = 0
                self.cmd_plan = None
                self.past_cmd = None
                self.reached = True
