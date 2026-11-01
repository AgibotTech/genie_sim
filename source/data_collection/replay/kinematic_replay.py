# Copyright (c) 2024-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""
KinematicReplay loads recorded state, the scene, robot, and clones, then
applies each frame without physics. Scene and recording helpers live in other
replay modules. The render rate is configurable.
"""

import json
import os
import sys
import time

import numpy as np

# Add the project root to the import path for package execution.
_replay_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(_replay_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
geniesim_path = os.path.join(project_root, "source", "geniesim")
if geniesim_path not in sys.path:
    sys.path.insert(0, geniesim_path)


def _robot_root_pose_from_recorded_link_pose(
    recorded_link_pose: np.ndarray,
    root_to_recorded_link: np.ndarray,
) -> np.ndarray:
    """Convert a recorded link world pose into the robot root world pose."""
    return np.asarray(recorded_link_pose, dtype=np.float64) @ np.linalg.inv(
        np.asarray(root_to_recorded_link, dtype=np.float64)
    )


def _apply_advance_render(world, apply_frame, frame_index: int) -> None:
    """Apply the recorded state, advance articulation FK, then render."""
    apply_frame(frame_index)
    world.step(render=True)


def _set_articulation_joint_state(articulation, joint_positions: np.ndarray) -> None:
    """Pin an articulation to a recorded pose for the following FK step."""
    articulation.set_joint_positions(joint_positions)
    articulation.set_joint_velocities(np.zeros_like(joint_positions))
    articulation._articulation_view.set_joint_position_targets(joint_positions)


from client import robot
from common.base_utils.logger import logger
from replay import constants
from replay import scene_utils
from replay import transform_utils
from replay.cloth_replay import ClothReplayer
from replay.generated_geometry_replay import GeneratedGeometryReplayer
from replay.recorder import Recorder


def disable_physics_on_stage(robot_prim_path: str, include_robot: bool = False):
    """Disable physics, articulation, and collision APIs on the stage.

    Preserve the robot in joint mode; include it in link mode so articulation
    control does not override recorded per-link world poses.
    """
    import omni
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, PhysxSchema
    stage = omni.usd.get_context().get_stage()
    if not stage:
        return
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if not prim.IsValid():
            continue
        is_robot = path == robot_prim_path or path.startswith(robot_prim_path + "/")
        if include_robot or not is_robot:
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
            if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            rigid_api = UsdPhysics.RigidBodyAPI.Get(stage, path)
            if rigid_api:
                rigid_api.GetRigidBodyEnabledAttr().Set(False)
            collision_api = UsdPhysics.CollisionAPI.Get(stage, path)
            if collision_api:
                collision_api.GetCollisionEnabledAttr().Set(False)
        
        
class KinematicReplay:
    """Replay recorded poses at the configured rate without physics."""

    SAVE_WORKERS = constants.SAVE_WORKERS
    MAX_INFLIGHT_SAVE_TASKS = constants.MAX_INFLIGHT_SAVE_TASKS

    def __init__(
        self,
        input_dir: str,
        robot_config: str = "G2_omnipicker_fixed_dual.json",
        num_clones: int = 1,
        clone_spacing: float = 3.0,
        backgrounds: list[dict] = [],
        enable_object_compensation: bool = False,
        filter_unmoved_objects: bool = True,
        enable_unmoved_object_disappear: bool = False,
        enable_recording: bool = True,
        launch_wall_start: float | None = None,
        recording_root: str | None = None,
        record_depth: str = "NONE",
        depth_realtime: bool = False,
        image_mode: bool = False,
        render_hz: int = 30,
        video_fps: int = 30,
        video_crf: int = 24,
        video_gop: int = 8,
        video_preset: str = "medium",
        rename_camera: bool = True,
        save_depth_in_uint16: bool = True,
        record_all_camera_depth: bool = False,
        add_default_ground_plane: bool = False,
        cloth_subdivision_levels: int = 2,
    ):
        self.input_dir = input_dir.rstrip("/")
        self.cloth_subdivision_levels = cloth_subdivision_levels
        self._recording_root = recording_root
        self.robot_config = robot_config
        self.num_clones = max(1, int(num_clones))
        self.clone_spacing = float(clone_spacing)
        self.RENDER_HZ = render_hz
        self.enable_object_compensation = enable_object_compensation
        self.enable_unmoved_object_disappear = enable_unmoved_object_disappear
        self._launch_wall_start = float(launch_wall_start) if launch_wall_start is not None else time.perf_counter()
        self._setup_wall_start = None
        self._setup_wall_end = None
        self.enable_recording = enable_recording
        self.filter_unmoved_objects = filter_unmoved_objects
        self.add_default_ground_plane = add_default_ground_plane
        self._recording_output_dir = None
        self._recording_output_dir_by_env = {}
        self._record_env_ids_for_round: list[int] | None = None  # None records every environment.
        self._recorder: Recorder | None = None
        self._robot_name = ""

        self._state_path = os.path.join(self.input_dir, "state.json")
        self._recording_info_path = os.path.join(self.input_dir, "recording_info.json")
        if not os.path.isfile(self._state_path):
            raise FileNotFoundError(f"state.json not found: {self._state_path}")
        with open(self._state_path, "r", encoding="utf-8") as f:
            self._state = json.load(f)
        if os.path.isfile(self._recording_info_path):
            with open(self._recording_info_path, "r", encoding="utf-8") as f:
                self._recording_info = json.load(f)
        else:
            self._recording_info = {}
        self._record_config = self._get_record_config_from_robot()
        if self._record_config:
            self._record_config["record_depth"] = record_depth
            self._record_config["depth_realtime"] = depth_realtime
            self._record_config["image_mode"] = image_mode
            self._record_config["video_crf"] = video_crf
            self._record_config["video_gop"] = video_gop
            self._record_config["video_fps"] = video_fps
            self._record_config["video_preset"] = video_preset
            self._record_config["rename_camera"] = rename_camera
            self._record_config["save_depth_in_uint16"] = save_depth_in_uint16
            self._record_config["record_all_camera_depth"] = record_all_camera_depth
        self._frames = self._state.get("frames", [])
        if not self._frames:
            raise ValueError("state.json has no frames")
        self._robot_mode, self._robot_link_prefix = self._detect_robot_mode()
        self._robot_link_prims_by_key: dict[str, str] = {}
        self._objects_dir_name = self._detect_objects_dir_name()
        scene_info = self._state.get("scene", {})
        scene_usd_raw = scene_info.get("scene_usd", "")
        self._backgrounds = backgrounds
        self._resolved_scene_usd = scene_utils.resolve_scene_usd(self.input_dir, scene_usd_raw)

    def _build_robot_link_prims(self, stage, robot_root_prim_path: str, init_objects: dict) -> dict:
        """Map recorded link keys to prim paths under the robot root.

        State keys combine a robot prefix and URDF link name; the USD tree
        normally contains prims with matching link names.
        """
        from pxr import Usd
        prefix = self._robot_link_prefix
        mapping: dict[str, str] = {}
        if not prefix:
            return mapping
        root = stage.GetPrimAtPath(robot_root_prim_path)
        if not root or not root.IsValid():
            logger.warning("Robot root prim not found: %s", robot_root_prim_path)
            return mapping
        name_to_path: dict[str, str] = {}
        for prim in Usd.PrimRange(root):
            n = prim.GetName()
            if n and n not in name_to_path:
                name_to_path[n] = prim.GetPath().pathString
        missing: list[str] = []
        for key in init_objects:
            if not key.startswith(prefix):
                continue
            link_name = key[len(prefix):]
            full = name_to_path.get(link_name)
            if full is None:
                missing.append(link_name)
                continue
            mapping[key] = full
        if missing:
            logger.warning(
                "Link replay: %d link key(s) without prim under %s: %s",
                len(missing), robot_root_prim_path, missing[:10],
            )
        return mapping

    def _detect_robot_mode(self) -> tuple[str, str]:
        """Detect whether state.json uses joint or per-link robot poses.

        Return the mode and link prefix. Joint mode reads frame.robot; link
        mode reads world poses from frame.objects keys such as base_link.
        """
        first_frame = self._frames[0] if self._frames else {}
        first_robot = first_frame.get("robot") or {}
        if first_robot.get("joints"):
            return "joint", ""
        for key in first_frame.get("objects", {}) or {}:
            if key.endswith("_base_link"):
                prefix = key[: -len("base_link")]
                return "link", prefix
        return "joint", ""

    def _detect_objects_dir_name(self) -> str:
        first_frame = self._frames[0] if self._frames else {}
        for key in first_frame.get("objects", {}):
            if "_Objects_" in key or key.startswith("Objects_"):
                return "Objects"
            if "_objects_" in key or key.startswith("objects_"):
                return "objects"
        return "Objects"

    def _get_task_name_from_data_dir(self) -> str:
        path = os.path.join(self.input_dir, "task_result.json")
        if not os.path.isfile(path):
            return "replay"
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            name = data.get("task_name")
            return str(name) if name else "replay"
        except Exception:
            return "replay"

    def _get_record_config_from_robot(self) -> dict:
        robot_cfg_path = os.path.join(project_root, "config", "robot_cfg", self.robot_config)
        if not self.enable_recording or not os.path.isfile(robot_cfg_path):
            return {}
        try:
            with open(robot_cfg_path, "r", encoding="utf-8") as f:
                robot_cfg = json.load(f)
        except Exception:
            return {}
        robot_name = (robot_cfg.get("robot") or {}).get("robot_name", "")
        recording_cfg = robot_cfg.get("recording_config")
        if recording_cfg:
            camera_list = recording_cfg.get("camera_list") or []
            fps = int(recording_cfg.get("fps", 30))
            noised_probability = float(recording_cfg.get("noised_probability", 0.0))
        else:
            raw = constants.G1_RECORDING_SETTING if "G1" in robot_name else constants.G2_RECORDING_SETTING
            if "ARX" in robot_name:
                raw = constants.ARX_RECORDING_SETTING
            camera_list = raw.get("camera_list") or []
            fps = int(raw.get("fps", 30))
            noised_probability = float(raw.get("noised_probability", 0.2))
        # Keep recording frequency independent of Isaac Sim stepping.
        # Source frame timestamps remain unchanged in copied state.json.
        source_fps = self._recording_info.get("fps")
        if source_fps is not None:
            fps = float(source_fps)
        if not camera_list:
            return {}
        task_name = self._get_task_name_from_data_dir()
        return {
            "task_name": task_name,
            "fps": fps,
            "camera_prim_list": list(camera_list),
            "additional_cam_parameters": {
                "noised_probability": noised_probability,
                "noised_camera_indices": list(range(len(camera_list))),
                "noise_parameters": {},
            },
        }

    def _get_recording_dirs_by_env(self, num_envs: int, root: str | None = None) -> dict:
        if not self._record_config or num_envs <= 0:
            return {}
        root = root or getattr(self, "_recording_root", None) or os.path.join(project_root, "recording_data")
        task_name = self._record_config["task_name"]
        id_base = 0
        while True:
            paths = []
            for i in range(num_envs):
                folder_id = id_base + i
                dir_name = task_name if folder_id == 0 else (task_name + str(folder_id))
                path = os.path.join(root, dir_name)
                paths.append(path)
            if all(not os.path.isdir(p) for p in paths):
                return {i: paths[i] for i in range(num_envs)}
            id_base += 1

    def _load_payload(self):
        from client.llm_gen.launcher import change_payload
        change_payload(scene_usda_path=self._resolved_scene_usd)

    def setup(self, simulation_app):
        """Load the scene, robot, clones, lights, and recorder after Isaac Sim starts."""
        self._setup_wall_start = time.perf_counter()
        logger.info("Setup started. wall_since_launch=%.3fs", self._setup_wall_start - self._launch_wall_start)

        import omni
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics
        from isaacsim.core.api import World
        from isaacsim.core.utils import extensions
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.prims import SingleArticulation as Articulation
        from isaacsim.core.prims import SingleXFormPrim as XFormPrim
        from common.base_utils.transform_utils import euler2quat_wxyz, mat2euler

        if self.num_clones > 1:
            extensions.enable_extension("isaacsim.core.cloner")

        rendering_dt = 1.0 / self.RENDER_HZ
        
        self._world = World(
            stage_units_in_meters=1,
            physics_dt=rendering_dt,
            rendering_dt=rendering_dt,
            device="cpu",
        )
        world = self._world
        # Replay needs PhysX only to update articulation forward kinematics. Gravity
        # makes the recorded joints sag before rendering and shifts end links by cm.
        world.get_physics_context().set_gravity(0.0)

        def set_prim_world_pose(prim_path: str, position, rotation_wxyz):
            stage = omni.usd.get_context().get_stage()
            if not stage or not stage.GetPrimAtPath(prim_path).IsValid():
                return
            pos = np.array(position, dtype=np.float64)
            quat = np.array(rotation_wxyz, dtype=np.float64)
            if pos.size != 3 or quat.size != 4:
                return
            xform = XFormPrim(prim_path=prim_path)
            xform.set_world_pose(pos, quat)

        sim_assets = os.environ.get("SIM_ASSETS", "")
        if not sim_assets:
            raise RuntimeError("SIM_ASSETS not set")

        with open(os.path.join(project_root, "config", "robot_cfg", self.robot_config), "r", encoding="utf-8") as f:
            robot_cfg = json.load(f)
        robot_prim_path = robot_cfg["robot"]["base_prim_path"]
        robot_usd_rel = robot_cfg["robot"]["robot_usd"]
        robot_usd_path = os.path.join(sim_assets, robot_usd_rel)
        robot_name = robot_cfg["robot"]["robot_name"]
        recorded_pose_prim_path = robot_cfg["robot"].get("recorded_pose_prim_path", "")
        self._robot_name = robot_name

        init_frame = self._frames[0]
        if self._robot_mode == "link":
            base_link_key = self._robot_link_prefix + "base_link"
            base_link_data = (init_frame.get("objects") or {}).get(base_link_key)
            if not base_link_data or "pose" not in base_link_data:
                raise ValueError(
                    f"link-mode replay requires '{base_link_key}' in frames[0].objects with 'pose' field"
                )
            init_robot_pose = base_link_data["pose"]
        else:
            init_robot_pose = init_frame["robot"]["pose"]
        init_translation_matrix = np.array(init_robot_pose)
        scene_object_paths = scene_utils.get_scene_objects(self._resolved_scene_usd, self._objects_dir_name)
        object_asset_dict = self._recording_info.get("object_asset_dict", {})
        unmoved_objects = (
            []
            if (not object_asset_dict or not self.filter_unmoved_objects)
            else scene_utils.filter_unmoved_objects(self._frames)
        )

        env_paths = []
        robot_prim_paths = []
        articulations_list = []
        clone_positions = np.zeros((1, 3))
        clone_orientations = np.array([[1.0, 0.0, 0.0, 0.0]])
        _base_path = "/World"
        scene_objects_mapping = {}
        task_objects = {}
        object_list = []
        
        def add_backgrounds(base_path):
            _base_path = base_path
            stage = omni.usd.get_context().get_stage()
            for background in self._backgrounds:
                background_usd = background.get("usd")
                background_position = background.get("position", [0, 0, 0])
                background_rotation = background.get("rotation", [1, 0, 0, 0])
                if not background_usd:
                    logger.warning("Background USD not found: %s", background_usd)
                    continue
                background_usd = os.path.join(sim_assets, background_usd)
                if not os.path.isfile(background_usd):
                    logger.warning("Background USD not found: %s", background_usd)
                    continue
                background_path = _base_path +"/background"
                add_reference_to_stage(background_usd, background_path)
                set_prim_world_pose(background_path, background_position, background_rotation)

        def add_task_objects(base_path):
            nonlocal _base_path, scene_objects_mapping, task_objects, object_list
            _base_path = base_path
            stage = omni.usd.get_context().get_stage()
            scene_objects_mapping = {}
            for obj_id in init_frame.get("objects", {}):
                for p in scene_object_paths:
                    if p.lstrip("/").replace("/", "_") == obj_id:
                        scene_objects_mapping[obj_id] = _base_path + (p[len("/World"):] if p.startswith("/World") else "/" + p)
                        break
            task_objects = {}
            for obj_id in init_frame.get("objects", {}):
                if obj_id in scene_objects_mapping or obj_id in unmoved_objects:
                    continue
                prim_path = _base_path + "/" + self._objects_dir_name + "/" + obj_id
                if prim_path not in object_asset_dict:
                    continue
                obj_usd = object_asset_dict[prim_path]
                rel_path = obj_usd.split("/assets/")[-1] if "/assets/" in obj_usd else obj_usd
                full_usd = obj_usd if os.path.isabs(obj_usd) and os.path.isfile(obj_usd) else os.path.join(sim_assets, rel_path)
                if not os.path.isfile(full_usd):
                    continue
                data_info_dir = os.path.dirname(obj_usd)
                scale = scene_utils.get_object_scale(data_info_dir, sim_assets)
                pose = np.array(init_frame["objects"][obj_id]["pose"])
                target_matrix = init_translation_matrix @ (np.linalg.inv(np.array(init_robot_pose)) @ pose)
                R, t = target_matrix[:3, :3], target_matrix[:3, 3]
                local_center = scene_utils.compute_local_bbox_center(obj_usd)
                if self.enable_object_compensation:
                    t = t - R @ (local_center * np.array(scale))
                rot = euler2quat_wxyz(mat2euler(R, order="zyx"), order="zyx")
                task_objects[obj_id] = {"scale": scale, "local_center": local_center}
                add_reference_to_stage(full_usd, prim_path)
                set_prim_world_pose(prim_path, t.tolist(), rot.tolist())
                prim = stage.GetPrimAtPath(prim_path)
                scale_attr = prim.GetAttribute("xformOp:scale")
                if scale_attr.IsValid():
                    scale_attr.Set(Gf.Vec3d(*(scale if isinstance(scale, list) else [scale] * 3)))
            for obj_id in unmoved_objects:
                if obj_id in scene_objects_mapping:
                    continue
                prim_path = _base_path + "/" + self._objects_dir_name + "/" + obj_id
                if prim_path not in object_asset_dict:
                    continue
                obj_usd = object_asset_dict[prim_path]
                rel_path = obj_usd.split("/assets/")[-1] if "/assets/" in obj_usd else obj_usd
                full_usd = obj_usd if os.path.isabs(obj_usd) and os.path.isfile(obj_usd) else os.path.join(sim_assets, rel_path)
                if not os.path.isfile(full_usd):
                    continue
                data_info_dir = os.path.dirname(obj_usd)
                scale = scene_utils.get_object_scale(data_info_dir, sim_assets)
                pose = np.array(init_frame["objects"][obj_id]["pose"])
                target_matrix = init_translation_matrix @ (np.linalg.inv(np.array(init_robot_pose)) @ pose)
                R, t = target_matrix[:3, :3], target_matrix[:3, 3]
                local_center = scene_utils.compute_local_bbox_center(obj_usd)
                if self.enable_object_compensation:
                    t = t - R @ (local_center * np.array(scale))
                rot = euler2quat_wxyz(mat2euler(R, order="zyx"), order="zyx")
                add_reference_to_stage(full_usd, prim_path)
                set_prim_world_pose(prim_path, t.tolist(), rot.tolist())
                prim = stage.GetPrimAtPath(prim_path)
                scale_attr = prim.GetAttribute("xformOp:scale")
                if scale_attr.IsValid():
                    scale_attr.Set(Gf.Vec3d(*(scale if isinstance(scale, list) else [scale] * 3)))
            if self.enable_unmoved_object_disappear and unmoved_objects:
                for n in unmoved_objects:
                    path = scene_objects_mapping.get(n, _base_path + "/" + self._objects_dir_name + "/" + n)
                    set_prim_world_pose(path, [0, 0, -1000], [1, 0, 0, 0])
            object_list = [
                k for k in init_frame.get("objects", {})
                if k not in unmoved_objects
                and not (self._robot_mode == "link" and self._robot_link_prefix and k.startswith(self._robot_link_prefix))
            ]

        robot_pose_mat = np.array(init_robot_pose)
        robot_position = robot_pose_mat[:3, 3].tolist()
        robot_rotation = euler2quat_wxyz(mat2euler(robot_pose_mat[:3, :3], order="zyx"), order="zyx").tolist()
        if self._robot_mode == "joint":
            init_joint_names = list(init_frame["robot"]["joints"]["joint_name"])
            init_joint_positions = list(init_frame["robot"]["joints"]["joint_position"])
        else:
            init_joint_names = []
            init_joint_positions = []
        from isaacsim.core.cloner import GridCloner
        base_env = "/World/envs"
        cloner = GridCloner(spacing=self.clone_spacing)
        cloner.define_base_env(base_env)
        all_env_paths = cloner.generate_paths(base_env + "/env", self.num_clones)
        env_paths = list(all_env_paths)
        robot_prim_paths = [p + robot_prim_path for p in env_paths]
        add_reference_to_stage(self._resolved_scene_usd, env_paths[0])
        add_reference_to_stage(robot_usd_path, env_paths[0] + robot_prim_path)
        stage = omni.usd.get_context().get_stage()
        robot_root_to_recorded_link = np.eye(4, dtype=np.float64)
        if recorded_pose_prim_path:
            from pxr import UsdGeom

            recorded_prim = stage.GetPrimAtPath(env_paths[0] + recorded_pose_prim_path)
            if not recorded_prim or not recorded_prim.IsValid():
                raise ValueError(
                    f"recorded pose prim not found: {env_paths[0] + recorded_pose_prim_path}"
                )
            recorded_world = np.asarray(
                UsdGeom.Xformable(recorded_prim).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                ),
                dtype=np.float64,
            ).T
            robot_root_prim = stage.GetPrimAtPath(env_paths[0] + robot_prim_path)
            robot_root_world = np.asarray(
                UsdGeom.Xformable(robot_root_prim).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                ),
                dtype=np.float64,
            ).T
            robot_root_to_recorded_link = np.linalg.inv(robot_root_world) @ recorded_world
            robot_root_pose = _robot_root_pose_from_recorded_link_pose(
                np.asarray(init_robot_pose), robot_root_to_recorded_link
            )
            robot_position = robot_root_pose[:3, 3].tolist()
            robot_rotation = euler2quat_wxyz(
                mat2euler(robot_root_pose[:3, :3], order="zyx"), order="zyx"
            ).tolist()
            logger.info(
                "Robot pose frame is %s; root correction translation=%s",
                recorded_pose_prim_path,
                robot_root_to_recorded_link[:3, 3].tolist(),
            )
        set_prim_world_pose(robot_prim_paths[0], robot_position, robot_rotation)
        # Add an optional ground collider when a scene lacks one. It keeps
        # dynamic bodies from shifting during world.reset() and is cloned with
        # the base environment.
        if self.add_default_ground_plane:
            self._add_default_ground_plane(env_paths[0] + "/GroundPlane")
        # Remove cloth physics schemas before world.reset() to avoid repeated
        # deformable cooks. Rendering still updates mesh points directly.
        self._strip_deformable_schemas_global(env_paths[0])
        if self._robot_mode == "joint":
            art = Articulation(prim_path=robot_prim_paths[0], name=robot_name)
            world.scene.add(art)
            world.reset()
            art.initialize()
            articulations_list.append(art)
        else:
            world.reset()
        add_task_objects(env_paths[0])
        add_backgrounds(env_paths[0])
        disable_physics_on_stage(
            robot_prim_paths[0], include_robot=(self._robot_mode == "link")
        )
        cloner.disable_change_listener()
        cloner.clone(
            source_prim_path=env_paths[0],
            prim_paths=env_paths,
            copy_from_source=False,
            replicate_physics=True,
        )
        cloner.enable_change_listener()
        clone_positions = np.array(cloner._positions)
        clone_orientations = np.array(cloner._orientations)

        stage = omni.usd.get_context().get_stage()

        if self._robot_mode == "joint":
            for ei in range(1, len(robot_prim_paths)):
                art_i = Articulation(prim_path=robot_prim_paths[ei], name=f"{robot_name}_{ei}")
                world.scene.add(art_i)
                art_i.initialize()
                articulations_list.append(art_i)
            dof_names = list(articulations_list[0].dof_names)
            joint_name_to_index = {n: i for i, n in enumerate(dof_names)}
            current_joints = np.array(articulations_list[0].get_joint_positions())
            for name, val in zip(init_joint_names, init_joint_positions):
                if name in joint_name_to_index:
                    current_joints[joint_name_to_index[name]] = val
            articulations_list[0].set_joint_positions(current_joints)
        else:
            joint_name_to_index = {}
            self._robot_link_prims_by_key = self._build_robot_link_prims(
                stage, robot_prim_paths[0], init_frame.get("objects") or {}
            )
            logger.info(
                "Link replay: mapped %d/%d state-link keys to robot prims (prefix=%r)",
                len(self._robot_link_prims_by_key),
                sum(1 for k in (init_frame.get("objects") or {}) if k.startswith(self._robot_link_prefix)),
                self._robot_link_prefix,
            )
        world.play()

        def object_path_suffix(key: str) -> str:
            full = scene_objects_mapping.get(key, _base_path + "/" + self._objects_dir_name + "/" + key)
            return full[len(_base_path):].lstrip("/") or self._objects_dir_name + "/" + key

        articulated_articulations = {}
        for key in self._frames[0].get("articulated_object", {}):
            path = env_paths[0] + "/" + self._objects_dir_name + "/" + key
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                try:
                    art = Articulation(prim_path=path, name=key)
                    world.scene.add(art)
                    art.initialize()
                    articulated_articulations[path] = art
                except Exception:
                    pass

        self._frame_durations = transform_utils.frame_durations_from_timestamps(
            self._frames, default_dt=1.0 / self.RENDER_HZ
        )
        self._simulation_app = simulation_app
        self._world = world
        self._set_prim_world_pose = set_prim_world_pose
        self._init_robot_pose = init_robot_pose
        self._robot_root_to_recorded_link = robot_root_to_recorded_link
        self._init_translation_matrix = init_translation_matrix
        self._env_paths = env_paths
        self._robot_prim_paths = robot_prim_paths
        self._articulations_list = articulations_list
        self._clone_positions = clone_positions
        self._clone_orientations = clone_orientations
        self._joint_name_to_index = joint_name_to_index
        self._task_objects = task_objects
        self._object_list = object_list
        self._scene_objects_mapping = scene_objects_mapping
        self._base_path = _base_path
        self._object_path_suffix = object_path_suffix
        self._articulated_articulations = articulated_articulations

        self._unmoved_objects = list(unmoved_objects)
        self._unmoved_object_poses = {}
        if unmoved_objects and init_frame.get("objects"):
            from scipy.spatial.transform import Rotation as R
            cp = clone_positions[0] if len(clone_positions) > 0 else np.zeros(3)
            cq = clone_orientations[0] if len(clone_orientations) > 0 else np.array([1.0, 0.0, 0.0, 0.0])
            for obj_id in unmoved_objects:
                if obj_id not in init_frame["objects"]:
                    continue
                pose = np.array(init_frame["objects"][obj_id]["pose"])
                target_matrix = init_translation_matrix @ (np.linalg.inv(np.array(init_robot_pose)) @ pose)
                t = target_matrix[:3, 3].copy()
                R_mat = target_matrix[:3, :3]
                if obj_id not in scene_objects_mapping and self.enable_object_compensation:
                    prim_path = _base_path + "/" + self._objects_dir_name + "/" + obj_id
                    if prim_path in object_asset_dict:
                        obj_usd = object_asset_dict[prim_path]
                        data_info_dir = os.path.dirname(obj_usd)
                        scale = scene_utils.get_object_scale(data_info_dir, sim_assets)
                        local_center = scene_utils.compute_local_bbox_center(obj_usd)
                        sc = np.array(scale, dtype=np.float64)
                        if sc.size == 1:
                            sc = np.array([sc.item()] * 3)
                        t = t - R_mat @ (local_center * sc)
                rot = euler2quat_wxyz(mat2euler(R_mat, order="zyx"), order="zyx")
                r = R.from_quat([rot[1], rot[2], rot[3], rot[0]])
                q = R.from_quat([cq[1], cq[2], cq[3], cq[0]])
                world_pos = np.array(cp, dtype=np.float64) + q.apply(np.asarray(t, dtype=np.float64))
                world_rot = (q * r).as_quat()
                wp = world_pos.tolist()
                wq = [float(world_rot[3]), float(world_rot[0]), float(world_rot[1]), float(world_rot[2])]
                self._unmoved_object_poses[obj_id] = (wp, wq)

        if self._record_config:
            self._record_config["robot_prim_path"] = robot_prim_path
            self._recording_output_dir_by_env = self._get_recording_dirs_by_env(len(env_paths))
            self._recorder = Recorder(
                self._record_config,
                robot_name,
                source_input_dir=self.input_dir,
            )
            self._recorder.set_output_dirs(self._recording_output_dir_by_env)
            self._recorder.init_cameras(stage, env_paths, robot_cfg)
            self._recording_output_dir = self._recording_output_dir_by_env.get(0, "")
            logger.info(
                "Recording env_id -> path: %s, cameras total: %d",
                self._recording_output_dir_by_env,
                len(self._recorder._cameras),
            )
        else:
            self._recording_output_dir = None
            self._recording_output_dir_by_env = {}

        self._setup_wall_end = time.perf_counter()
        logger.info(
            "Setup finished. setup_wall=%.3fs, wall_since_launch=%.3fs",
            self._setup_wall_end - self._setup_wall_start,
            self._setup_wall_end - self._launch_wall_start,
        )

        # Set up ClothReplayer only when a frame contains deformable entries.
        # It reads NPZ points per frame and writes local-space mesh points.
        self._cloth_replayer = None
        try:
            first_def = None
            for fr in self._frames:
                if fr.get("deformable"):
                    first_def = fr["deformable"]
                    break
            if first_def:
                self._cloth_replayer = ClothReplayer(
                    self.input_dir, base_record_path="/World",
                    subdivision_levels=self.cloth_subdivision_levels)
                import omni  # noqa: PLC0415
                _stage_now = omni.usd.get_context().get_stage()
                self._cloth_replayer.setup(_stage_now, first_def, env_paths)
                logger.info(
                    "[cloth_replay] enabled, %d cloth meshes initialized",
                    len(first_def),
                )
        except Exception as exc:
            logger.warning(f"[cloth_replay] init failed: {exc}")
            self._cloth_replayer = None

        # Runtime-generated meshes are replayed from final artifacts plus each
        # state frame's topology prefixes. Initialization is intentionally
        # strict: a declared but missing/corrupt artifact must stop replay.
        self._generated_geometry_replayer = GeneratedGeometryReplayer.from_state(
            self.input_dir,
            self._state,
        )
        if self._generated_geometry_replayer is not None:
            import omni  # noqa: PLC0415

            _stage_now = omni.usd.get_context().get_stage()
            self._generated_geometry_replayer.setup(_stage_now, env_paths)
            logger.info(
                "[generated_geometry_replay] frame-exact replay enabled"
            )

    def get_env_paths(self) -> list[str]:
        """Return environment paths ordered by ID for Fuzzer."""
        return list(getattr(self, "_env_paths", []) or [])

    def ensure_default_lights(self) -> None:
        """Add the default dome and distant lights."""
        import omni
        from pxr import Sdf, UsdLux
        stage = omni.usd.get_context().get_stage()
        dome_light = UsdLux.DomeLight.Define(stage, "/dome_light")
        dome_light.AddRotateXYZOp().Set((-90.0, -30.0, 0.0))
        dome_light.GetEnableColorTemperatureAttr().Set(True)
        dome_light.GetColorTemperatureAttr().Set(6150.0)
        dome_light.GetIntensityAttr().Set(1.0)
        dome_light.GetExposureAttr().Set(9.0)
        dome_light.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)
        distant_light = UsdLux.DistantLight.Define(stage, "/distant_light")
        distant_light.AddRotateXYZOp().Set((-35.0, 45.0, 0.0))
        distant_light.GetEnableColorTemperatureAttr().Set(True)
        distant_light.GetColorTemperatureAttr().Set(7250.0)
        distant_light.GetIntensityAttr().Set(1.0)
        distant_light.GetExposureAttr().Set(10.0)
        logger.info("Added default dome_light and distant_light")

    def _add_default_ground_plane(self, prim_path: str) -> None:
        """Create an invisible collision plane at Z=0.

        This prevents some dynamic bodies from shifting during world.reset()
        when the scene has no ground collider.
        """
        try:
            import omni  # noqa: PLC0415
            from pxr import Gf, Sdf, UsdGeom, UsdPhysics  # noqa: PLC0415
        except Exception as exc:
            logger.warning(f"[ground_plane] import failed: {exc}")
            return
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        if stage.GetPrimAtPath(prim_path).IsValid():
            logger.info(f"[ground_plane] already exists at {prim_path}, skip")
            return
        gp = UsdGeom.Xform.Define(stage, prim_path)
        gp.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        gp.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        gp.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        plane_path = prim_path + "/CollisionPlane"
        plane = stage.DefinePrim(plane_path, "Plane")
        plane.CreateAttribute("axis", Sdf.ValueTypeNames.Token).Set("Z")
        plane.CreateAttribute("purpose", Sdf.ValueTypeNames.Token).Set("guide")
        UsdPhysics.CollisionAPI.Apply(plane)
        logger.info(f"[ground_plane] added {plane_path} (axis=Z, PhysicsCollisionAPI)")

    def _strip_deformable_schemas_global(self, root_path: str) -> int:
        """Remove deformable schemas below root_path to avoid physics cooking.

        Remove schemas by name because some Isaac Sim versions do not expose
        corresponding Python classes in PhysxSchema.
        """
        try:
            import omni  # noqa: PLC0415
            from pxr import Usd  # noqa: PLC0415
        except Exception as exc:
            logger.warning(f"[cloth_replay] strip schemas: import failed: {exc}")
            return 0
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return 0
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            return 0
        # Cover both OmniPhysics and PhysxSchema deformable APIs.
        candidate_names = [
            "OmniPhysicsDeformableBodyAPI",
            "OmniPhysicsSurfaceDeformableSimAPI",
            "OmniPhysicsSurfaceDeformableMaterialAPI",
            "OmniPhysicsDeformableMaterialAPI",
            "PhysxSurfaceDeformableBodyAPI",
            "PhysxDeformableBodyAPI",
            "PhysxDeformableSurfaceAPI",
            "PhysxAutoAttachmentAPI",
        ]
        # Remove cloth collision APIs as well to prevent repeated cooking.
        collision_names = [
            "PhysicsCollisionAPI",
            "PhysxCollisionAPI",
        ]
        n_removed = 0
        for prim in Usd.PrimRange(root):
            applied = list(prim.GetAppliedSchemas())
            if not applied:
                continue
            has_deformable = any(
                ("Deformable" in n) or ("deformable" in n) for n in applied
            )
            for name in candidate_names:
                if name in applied:
                    try:
                        prim.RemoveAppliedSchema(name)
                        n_removed += 1
                    except Exception as exc:
                        logger.warning(f"[cloth_replay] remove {name} on {prim.GetPath()} failed: {exc}")
            # Preserve collision APIs on ordinary rigid prims.
            if has_deformable:
                for name in collision_names:
                    if name in applied:
                        try:
                            prim.RemoveAppliedSchema(name)
                            n_removed += 1
                        except Exception as exc:
                            logger.warning(f"[cloth_replay] remove {name} on {prim.GetPath()} failed: {exc}")
        if n_removed:
            logger.info(f"[cloth_replay] stripped {n_removed} deformable/collision schemas under {root_path}")
        return n_removed

    def _save_recording_frame(self, frame_id: int) -> tuple[float, float, int]:
        if self._recorder:
            return self._recorder.save_frame(frame_id)
        return 0.0, 0.0, 0

    def _apply_frame(self, idx: int):
        from common.base_utils.transform_utils import euler2quat_wxyz, mat2euler
        frame = self._frames[idx]
        env_count = len(self._env_paths)
        if self._robot_mode == "link":
            inv_init_robot = np.linalg.inv(np.array(self._init_robot_pose))
            objects_in_frame = frame.get("objects") or {}
            for state_key, prim_path in self._robot_link_prims_by_key.items():
                link_data = objects_in_frame.get(state_key)
                if not link_data or "pose" not in link_data:
                    continue
                pose = np.array(link_data["pose"])
                target = self._init_translation_matrix @ (inv_init_robot @ pose)
                R = target[:3, :3]
                t = target[:3, 3]
                rot = euler2quat_wxyz(mat2euler(R, order="zyx"), order="zyx")
                suffix = prim_path[len(self._env_paths[0]):].lstrip("/")
                for ei, ep in enumerate(self._env_paths):
                    wp_i, wq_i = transform_utils.local_to_world_with_clone(t, rot, self._clone_positions[ei], self._clone_orientations[ei])
                    self._set_prim_world_pose(ep + "/" + suffix, wp_i, wq_i)
        else:
            robot_pose_mat = np.array(frame["robot"]["pose"])
            robot_pos = np.array(robot_pose_mat[:3, 3])
            robot_rot = euler2quat_wxyz(mat2euler(robot_pose_mat[:3, :3], order="zyx"), order="zyx")
            if not np.allclose(self._robot_root_to_recorded_link, np.eye(4)):
                robot_pose_mat = _robot_root_pose_from_recorded_link_pose(
                    robot_pose_mat, self._robot_root_to_recorded_link
                )
                robot_pos = robot_pose_mat[:3, 3]
                robot_rot = euler2quat_wxyz(
                    mat2euler(robot_pose_mat[:3, :3], order="zyx"), order="zyx"
                )
            for ei in range(env_count):
                joint_pos = np.array(self._articulations_list[ei].get_joint_positions())
                for n, v in zip(frame["robot"]["joints"]["joint_name"], frame["robot"]["joints"]["joint_position"]):
                    if n in self._joint_name_to_index:
                        joint_pos[self._joint_name_to_index[n]] = v
                wp, wq = transform_utils.local_to_world_with_clone(robot_pos, robot_rot, self._clone_positions[ei], self._clone_orientations[ei])
                self._set_prim_world_pose(self._robot_prim_paths[ei], wp, wq)
                _set_articulation_joint_state(self._articulations_list[ei], joint_pos)
        for key in self._object_list:
            if key not in frame["objects"]:
                continue
            value = frame["objects"][key]
            target_matrix = self._init_translation_matrix @ (
                np.linalg.inv(np.array(self._init_robot_pose)) @ value["pose"]
            )
            R, t = target_matrix[:3, :3], target_matrix[:3, 3].copy()
            if self.enable_object_compensation and key in self._task_objects:
                lc = self._task_objects[key]["local_center"]
                sc = np.array(self._task_objects[key]["scale"], dtype=np.float64)
                if sc.size == 1:
                    sc = np.array([sc.item()] * 3)
                t -= R @ (lc * sc)
            rot = euler2quat_wxyz(mat2euler(R, order="zyx"), order="zyx")
            suffix = self._object_path_suffix(key)
            for ei, ep in enumerate(self._env_paths):
                cp, cq = self._clone_positions[ei], self._clone_orientations[ei]
                wp, wq = transform_utils.local_to_world_with_clone(t, rot, cp, cq)
                prim_path = ep + ("/" + suffix if not suffix.startswith("/") else suffix)
                self._set_prim_world_pose(prim_path, wp, wq)
        for obj_id in getattr(self, "_unmoved_objects", []):
            suffix = self._object_path_suffix(obj_id)
            ep = self._env_paths[0]
            prim_path = (ep + "/" + suffix) if not suffix.startswith("/") else (ep + suffix)
            if self.enable_unmoved_object_disappear:
                self._set_prim_world_pose(prim_path, [0.0, 0.0, -1000.0], [1.0, 0.0, 0.0, 0.0])
            elif obj_id in getattr(self, "_unmoved_object_poses", {}):
                wp, wq = self._unmoved_object_poses[obj_id]
                self._set_prim_world_pose(prim_path, wp, wq)
        for key, val in frame.get("articulated_object", {}).items():
            path = self._env_paths[0] + "/" + self._objects_dir_name + "/" + key
            if path in self._articulated_articulations:
                self._articulated_articulations[path].set_joint_positions(
                    list(val["joints"]["joint_position"])
                )
        # ClothReplayer reads this frame's NPZ points and converts them into
        # mesh-local coordinates before updating the mesh.
        if self._cloth_replayer is not None:
            try:
                import omni  # noqa: PLC0415
                _stage_now = omni.usd.get_context().get_stage()
                self._cloth_replayer.apply_frame(
                    _stage_now, frame.get("deformable"), self._env_paths,
                    env_clone_positions=self._clone_positions,
                    env_clone_orientations=self._clone_orientations,
                )
            except Exception as exc:
                logger.warning(f"[cloth_replay] apply_frame failed at idx={idx}: {exc}")
        # Apply generated geometry after object/deformable state and before the
        # caller renders this frame. Do not swallow errors: partial frames are invalid for
        # frame-exact generated geometry.
        if self._generated_geometry_replayer is not None:
            import omni  # noqa: PLC0415

            _stage_now = omni.usd.get_context().get_stage()
            self._generated_geometry_replayer.apply_frame(
                _stage_now,
                idx,
                self._env_paths,
                env_clone_positions=self._clone_positions,
                env_clone_orientations=self._clone_orientations,
            )

    def get_object_asset_dict(self) -> dict:
        """Map prim paths to USD assets for asset-ID lookup."""
        return self._recording_info.get("object_asset_dict", {})

    def get_recording_dirs_by_env(self):
        out = dict(getattr(self, "_recording_output_dir_by_env", {}))
        ids = getattr(self, "_record_env_ids_for_round", None)
        if ids is not None:
            out = {i: out[i] for i in ids if i in out}
        return out

    def get_recording_camera_image_names(self) -> list[str]:
        if self._recorder:
            return self._recorder.get_camera_image_names()
        return []

    def get_recording_camera_depth_image_names(self) -> list[str]:
        if self._recorder:
            return self._recorder.get_camera_depth_image_names()
        return []

    def release_writers(self) -> set[str]:
        if self._recorder:
            return self._recorder.release_writers()
        return set()

    def log_resource_status(self) -> None:
        if self._recorder:
            self._recorder.log_resource_status()

    @property
    def total_pool_wait_sec(self) -> float:
        if self._recorder:
            return self._recorder.total_pool_wait_sec
        return 0.0

    @property
    def video_preset(self) -> str:
        if self._recorder:
            return self._recorder.video_preset
        return "medium"

    def get_record_depth_mode(self) -> str:
        if self._recorder:
            return self._recorder.get_record_depth_mode()
        return "NONE"

    @property
    def depth_realtime(self) -> bool:
        if self._recorder:
            return self._recorder.depth_realtime
        return False

    def set_recording_dirs_by_env(self, env_id_to_path: dict) -> None:
        self._recording_output_dir_by_env = dict(env_id_to_path)
        for rec_path in self._recording_output_dir_by_env.values():
            os.makedirs(rec_path, exist_ok=True)
        self._recording_output_dir = self._recording_output_dir_by_env.get(0, "")
        if self._recorder:
            self._recorder.set_output_dirs(self._recording_output_dir_by_env)

    def allocate_next_recording_dirs(self) -> dict:
        n = len(getattr(self, "_env_paths", []))
        if n <= 0:
            return {}
        new_dirs = self._get_recording_dirs_by_env(n)
        if new_dirs:
            self.set_recording_dirs_by_env(new_dirs)
        return new_dirs

    def set_record_env_ids_for_round(self, record_env_ids: list[int]) -> None:
        """Record only the selected environments in this round."""
        self._record_env_ids_for_round = list(record_env_ids)
        if not self._recorder:
            return
        all_dirs = getattr(self, "_recording_output_dir_by_env", {})
        filtered = {i: all_dirs[i] for i in record_env_ids if i in all_dirs}
        self._recorder.set_output_dirs(filtered)

    def reset_to_frame0(self) -> None:
        for _ in range(3):
            _apply_advance_render(self._world, self._apply_frame, 0)

    def run_one_round(self) -> int:
        frame_index = 0
        replay_start_wall = None
        frame_apply_times = []
        frame_step_times = []
        frame_save_times = []
        # wait till the rendering is stable after reset
        WAITING_FRAMES = 30
        start_wall = time.perf_counter()
        for i in range(WAITING_FRAMES):
            _apply_advance_render(self._world, self._apply_frame, 0)
            if i % 10 == 0 or i == WAITING_FRAMES - 1:
                logger.info("Waiting for rendering to stabilize... (%d/%d)", i, WAITING_FRAMES)
        end_wall = time.perf_counter()
        logger.info("Rendering stabilized in %.3fs", end_wall - start_wall)
        while self._simulation_app.is_running():
            if not self._world.is_playing():
                self._world.step(render=True)
                continue
            if frame_index >= len(self._frames):
                break
            if replay_start_wall is None:
                replay_start_wall = time.perf_counter()
            frame_wall0 = time.perf_counter()
            t_apply0 = frame_wall0
            self._apply_frame(frame_index)
            t_apply1 = time.perf_counter()
            t_step0 = t_apply1
            self._world.step(render=True)
            t_step1 = time.perf_counter()
            save_dt = 0.0
            save_submit_dt = 0.0
            save_wait_dt = 0.0
            save_inflight = 0
            if self._recorder:
                result = self._save_recording_frame(frame_index)
                save_submit_dt, save_wait_dt, save_inflight = result
                save_dt = save_submit_dt + save_wait_dt
            frame_wall1 = time.perf_counter()
            apply_dt = t_apply1 - t_apply0
            step_dt = t_step1 - t_step0
            frame_apply_times.append(apply_dt)
            frame_step_times.append(step_dt)
            frame_save_times.append(save_dt)
            logger.info(
                "[Frame %d/%d] step=%.1fms save=%.1fms (submit=%.1fms wait=%.1fms inflight=%d) "
                "(apply=%.1fms total=%.1fms) | wall_since_launch=%.3fs | speed=%.3f",
                frame_index,
                len(self._frames),
                step_dt * 1000.0,
                save_dt * 1000.0,
                save_submit_dt * 1000.0,
                save_wait_dt * 1000.0,
                save_inflight,
                apply_dt * 1000.0,
                (frame_wall1 - frame_wall0) * 1000.0,
                frame_wall1 - self._launch_wall_start,
                (1.0 / self.RENDER_HZ) * self.num_clones / max(1e-9, (frame_wall1 - frame_wall0)),
            )
            frame_index += 1
            if frame_index == len(self._frames):
                n = len(frame_apply_times)
                replay_wall = time.perf_counter() - replay_start_wall
                expected_duration = sum(self._frame_durations)
                apply_total = sum(frame_apply_times)
                step_total = sum(frame_step_times)
                save_total = sum(frame_save_times)
                num_record_envs = len(self._record_env_ids_for_round) if self._record_env_ids_for_round is not None else self.num_clones
                logger.info(
                    "[Replay round timing] frames=%d | wall_time=%.2fs | expected_data_duration=%.2fs | "
                    "apply_total=%.2fs | step_total=%.2fs | save_total=%.2fs | "
                    "per_frame_mean apply=%.1fms step=%.1fms save=%.1fms | speed=%.3f (record_envs=%d)",
                    n,
                    replay_wall,
                    expected_duration * max(1, num_record_envs),
                    apply_total,
                    step_total,
                    save_total,
                    apply_total / max(1, n) * 1000.0,
                    step_total / max(1, n) * 1000.0,
                    save_total / max(1, n) * 1000.0,
                    (expected_duration * max(1, num_record_envs)) / max(1e-9, replay_wall),
                    num_record_envs,
                )
                return n
        return 0

    def finish_round_recording(self, num_frames: int) -> None:
        if self._recorder:
            self._recorder.drain()
            if num_frames > 0:
                self._recorder.write_recording_info(num_frames)

    def run(self) -> None:
        logger.info(
            "Replay: %d frames, envs=%d, timing from state.json time_stamp (no free simulation)",
            len(self._frames),
            self.num_clones,
        )
        frame_index = 0
        replay_start_wall = None
        frame_apply_times = []
        frame_step_times = []
        frame_save_times = []
        for _ in range(10):
            _apply_advance_render(self._world, self._apply_frame, 0)
        while self._simulation_app.is_running():
            if not self._world.is_playing():
                self._world.step(render=True)
                continue
            if frame_index >= len(self._frames):
                break
            if replay_start_wall is None:
                replay_start_wall = time.perf_counter()
            frame_wall0 = time.perf_counter()
            t_apply0 = frame_wall0
            self._apply_frame(frame_index)
            t_apply1 = time.perf_counter()
            t_step0 = t_apply1
            self._world.step(render=True)
            t_step1 = time.perf_counter()
            save_dt = 0.0
            save_submit_dt = 0.0
            save_wait_dt = 0.0
            save_inflight = 0
            if self._recorder:
                result = self._save_recording_frame(frame_index)
                save_submit_dt, save_wait_dt, save_inflight = result
                save_dt = save_submit_dt + save_wait_dt
            frame_wall1 = time.perf_counter()
            apply_dt = t_apply1 - t_apply0
            step_dt = t_step1 - t_step0
            frame_apply_times.append(apply_dt)
            frame_step_times.append(step_dt)
            frame_save_times.append(save_dt)
            logger.info(
                "[Frame %d/%d] step=%.1fms save=%.1fms (submit=%.1fms wait=%.1fms inflight=%d) "
                "(apply=%.1fms total=%.1fms) | wall_since_launch=%.3fs | speed=%.3f",
                frame_index,
                len(self._frames),
                step_dt * 1000.0,
                save_dt * 1000.0,
                save_submit_dt * 1000.0,
                save_wait_dt * 1000.0,
                save_inflight,
                apply_dt * 1000.0,
                (frame_wall1 - frame_wall0) * 1000.0,
                frame_wall1 - self._launch_wall_start,
                (1.0 / self.RENDER_HZ) * self.num_clones / max(1e-9, (frame_wall1 - frame_wall0)),
            )
            frame_index += 1
            if frame_index == len(self._frames):
                replay_wall = time.perf_counter() - replay_start_wall
                expected_duration = sum(self._frame_durations)
                apply_total = sum(frame_apply_times)
                step_total = sum(frame_step_times)
                save_total = sum(frame_save_times)
                n = len(frame_apply_times)
                logger.info(
                    "[Replay timing] frames=%d | wall_time=%.2fs | expected_data_duration=%.2fs | "
                    "apply_total=%.2fs | step_total=%.2fs | save_total=%.2fs | "
                    "per_frame_mean apply=%.1fms step=%.1fms save=%.1fms | speed=%.3f",
                    n,
                    replay_wall,
                    expected_duration * self.num_clones,
                    apply_total,
                    step_total,
                    save_total,
                    apply_total / max(1, n) * 1000.0,
                    step_total / max(1, n) * 1000.0,
                    save_total / max(1, n) * 1000.0,
                    (expected_duration * self.num_clones) / max(1e-9, replay_wall),
                )
                if self._recorder:
                    t_flush0 = time.perf_counter()
                    self._recorder.drain()
                    logger.info("Async saver flushed. flush_wall=%.3fs", time.perf_counter() - t_flush0)
                    self._recorder.write_recording_info(n)
                    self._recorder.shutdown()
                    self._recorder = None
                logger.info(
                    "[Launch timing] wall_since_launch=%.3fs (includes setup + replay)",
                    time.perf_counter() - self._launch_wall_start,
                )
                break
        logger.info("Replay exit.")
