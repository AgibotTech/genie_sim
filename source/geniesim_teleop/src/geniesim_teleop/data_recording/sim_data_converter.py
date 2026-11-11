# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Export the data_collection schema with scalar grippers or 20-DOF Wuji hands."""

import datetime
from copy import deepcopy
import tempfile
import json
import logging
import math
import os
import re
import shutil
import uuid
from enum import IntEnum
from pathlib import Path

import h5py
import numpy as np
import pytz
import yaml
from scipy.spatial.transform import Rotation as R

from geniesim_teleop.data_recording.sim_version_constants import SIM_DATA_VERSION
from geniesim_teleop.data_recording.derive_fields import (
    assemble_end_pose,
    chain_transform,
    effector_from_arm_end,
    effector_in_base,
    link3_orientation,
    load_link_chain,
)

# Static sensor models depend on gripper type for arm cameras and camera name
# for head cameras, matching the hardware metadata convention.
_GRIPPER_SENSOR_MODEL = {
    "zhiyuan_gripper_omnipicker": "old_op",
    "zhiyuan_gripper_ctek90d": "new_90d",
    "zhiyuan_gripper_wuji": "sim",
}
_HEAD_SENSOR_MODEL = {
    "head_front_rgbd": "head_front_rgbd_mani",
    "head_left_stereo": "head_left_stereo_mani",
    "head_right_stereo": "head_right_stereo_mani",
}


class ControlMode(IntEnum):
    """Joint and end-effector control modes used in state/*/mode.

    CSP is position control, CST is torque control, and MIT includes velocity
    and torque feed-forward. Automated simulation defaults to CSP.
    """

    CSP = 0
    CST = 1
    MIT = 2


from geniesim_teleop.data_recording.gripper_remap import (
    omnipicker_action_to_rad,
    omnipicker_state_to_rad,
    g2_90d_action_to_rad,
    g2_90d_state_to_rad,
)

log = logging.getLogger("sim_data_converter")

REVIEW_FRAME_GROUPS = int(os.environ.get("PROCESS_FRAME_GROUPS", "30"))

REVIEW_DEFAULT_JOINT_NAMES = json.loads(
    os.environ.get(
        "PROCESS_DEFAULT_JOINT_NAMES",
        '{"joint":["idx21_arm_l_joint1","idx61_arm_r_joint1","idx22_arm_l_joint2",'
        '"idx62_arm_r_joint2","idx23_arm_l_joint3","idx63_arm_r_joint3",'
        '"idx24_arm_l_joint4","idx64_arm_r_joint4","idx25_arm_l_joint5",'
        '"idx65_arm_r_joint5","idx26_arm_l_joint6","idx66_arm_r_joint6",'
        '"idx27_arm_l_joint7","idx67_arm_r_joint7"],'
        '"effector":["idx41_gripper_l_outer_joint1","idx81_gripper_r_outer_joint1"]}',
    )
)

REVIEW_VISUAL_GROUP_CONFIG = json.loads(os.environ.get("PROCESS_VISUAL_GROUP", "[]"))

# NOTE: zhiyuan_gripper_omnipicker is intentionally absent — its h5 values are
# already unified radians [-0.785, 0], so review state.json passes them through
# verbatim (no normalization). ctek_gripper_120s still stores [0,120] -> 1-x/120.
REVIEW_GRIPPER_CAL_MAPPING = json.loads(
    os.environ.get(
        "PROCESS_GRIPPER_CAL_MAPPING",
        '{"ctek_gripper_120s":"ctek_120_to_unit"}',
    )
)


def _build_pose_from_raw_frame(f_in, index):
    """Assemble a 28-value end pose when raw effector fields are available.

    Arm values come from raw state/end fields, while effector values use raw
    precomputed fields. This format layer assembles values without geometry.
    """
    end = f_in["state/end"]
    if "effector_position" not in end or "effector_orientation" not in end:
        return None
    arm_pos = f_in["state/end/position"][index]
    arm_ori = f_in["state/end/orientation"][index]
    eff_pos = f_in["state/end/effector_position"][index]
    eff_ori = f_in["state/end/effector_orientation"][index]
    return assemble_end_pose(arm_pos, arm_ori, eff_pos, eff_ori)


def _get_all_h5_groups(h5_path):
    groups = []
    with h5py.File(h5_path, mode="r") as f:
        for k1 in f.keys():
            groups.append(k1)
            if isinstance(f[k1], h5py.Dataset):
                continue
            for k2 in f[k1].keys():
                if isinstance(f[k1][k2], h5py.Dataset):
                    continue
                groups.append(f"{k1}/{k2}")
                for k3 in f[k1][k2].keys():
                    groups.append(f"{k1}/{k2}/{k3}")
    return groups


def _get_h5_metadata(h5_obj, path):
    parts = [p for p in path.strip("/").split("/") if p]
    if not parts:
        return False, None, None
    current = h5_obj
    try:
        for part in parts[:-1]:
            current = current[part]
        attrs = dict(current.attrs)
        data = current[parts[-1]][()].tolist()
        return True, attrs, data
    except (KeyError, AttributeError):
        return False, None, None


def _get_names_from_group(group):
    return group["name"] if "name" in group else []


def _names_for_visual_path(path, attrs, width):
    """Return display names matching the width of a review data group.

    Prefer the group's name attribute, then configured defaults for its path.
    Pad or truncate names to the data width so later groups stay aligned.
    """
    seg = path.split("/")[1]
    names = list(_get_names_from_group(attrs))
    if not names:
        if seg in REVIEW_DEFAULT_JOINT_NAMES:
            names = list(REVIEW_DEFAULT_JOINT_NAMES[seg])
        elif seg in ("left_effector", "right_effector"):
            eff = list(REVIEW_DEFAULT_JOINT_NAMES.get("effector", []))
            idx = 0 if seg == "left_effector" else 1
            if idx < len(eff):
                names = [eff[idx]]
    if len(names) < width:
        names = names + [f"{seg}_joint{i}" for i in range(len(names), width)]
    else:
        names = names[:width]
    return names


def _combine_arrays(*arrays):
    arrays = [a for a in arrays if a]
    if not arrays:
        return []
    min_len = min(len(a) for a in arrays)
    combined = []
    for i in range(min_len):
        row = []
        for a in arrays:
            if isinstance(a[i], list):
                row.extend(a[i])
            else:
                row.append(a[i])
        combined.append(row)
    return combined


def _convert_gripper_data(data, cal_rule):
    if not cal_rule:
        return data
    if cal_rule not in ("ctek_120_to_unit", "1-{}/120"):
        raise ValueError(f"Unsupported gripper conversion rule: {cal_rule}")

    def _apply(value):
        if isinstance(value, list):
            return [_apply(v) for v in value]
        return 1.0 - float(value) / 120.0

    return _apply(data)


def get_directory_size(directory):
    """Calculate total size of specified directory (bytes)"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            # Skip symbolic links
            if not os.path.islink(filepath):
                total_size += os.path.getsize(filepath)
    return total_size


def canonical_camera_name(name):
    """Normalize G2 camera names from topic/TF/legacy metadata casing."""
    lower = name.lower()
    for tokens, canonical in (
        (("head_front", "head_color", "head_depth"), "head"),
        (("head_stereo_left", "head_left"), "head_stereo_left"),
        (("head_stereo_right", "head_right"), "head_stereo_right"),
        (("hand_left", "left_camera"), "hand_left"),
        (("hand_right", "right_camera"), "hand_right"),
    ):
        if any(token in lower for token in tokens):
            return canonical + ("_depth" if "depth" in lower else "_color")
    return name


def process_camera_parameter_file_name(camera_name):
    if "head_stereo_right" in camera_name.lower():
        camera_name = "head_right_stereo"
    elif "head_stereo_left" in camera_name.lower():
        camera_name = "head_left_stereo"
    elif "head" in camera_name.lower():
        camera_name = "head_front_rgbd"
    elif "hand_left" in camera_name.lower():
        camera_name = "hand_left_rgbd"
    elif "hand_right" in camera_name.lower():
        camera_name = "hand_right_rgbd"
    return camera_name


def _normalize_skill(skill: str) -> str:
    """Normalize data_info skill names to initial uppercase and remaining lowercase.

    Existing skills are single English verbs; preserve an empty value.
    """
    return (skill or "").strip().capitalize()


# Teleop export format revision.
FORMAT_REVISION = 4
MARKER = Path("post_proc/teleop_export.json")
INPUTS = ("metadata.yaml", "state.json", "aligned_joints_all.h5", "recording_info.json")


def _fingerprint(record):
    return {name: [p.stat().st_size, p.stat().st_mtime_ns] for name in INPUTS if (p := record / name).is_file()}


def export_complete(record, output=None):
    record = Path(record)
    output = Path(output) if output else record
    try:
        marker = json.loads((output / MARKER).read_text())
        return (
            marker["revision"] == FORMAT_REVISION
            and marker["inputs"] == _fingerprint(record)
            and bool(marker["files"])
            and all((output / name).is_file() for name in marker["files"])
        )
    except (OSError, ValueError, KeyError):
        return False


def _prepare_teleop_input(record, staging, state, config, grippers, robot):
    """Prepare named measured joints using the FK shipped in this package."""
    import h5py
    import numpy as np
    from scipy.spatial.transform import Rotation
    from geniesim_teleop.data_recording.derive_fields import effector_from_arm_end
    from geniesim_teleop.data_recording.robot_fk import calculate_ee_pose_batch

    frames = state["frames"]
    count = len(frames)
    if not count:
        raise ValueError("Recording contains no frames")
    timestamps = np.asarray([f["time_stamp"] for f in frames], dtype=np.float64)
    # Old extraction could duplicate the terminal frame; retain its video/frame
    # index, but the shared writer will rebuild a strictly increasing timeline.
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0):
        raise ValueError("Recording timestamps must be finite and nondecreasing")
    with h5py.File(record / "aligned_joints_all.h5", "r") as raw:
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in raw["state/joint"].attrs["name"]]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate recorded joint names")
        required = config["arm_joint_names"] + config["head_joint_names"] + config["waist_joint_names"]
        required += config["left_effector_joint_names"] + config["right_effector_joint_names"]
        missing = sorted(set(required) - set(names))
        if missing:
            raise ValueError(f"Missing required joints: {missing}; re-extract the source bag")
        if len(raw["timestamp"]) != count:
            raise ValueError("state.json and raw H5 frame counts differ")
        joint = {}
        for field in ("position", "velocity", "effort"):
            values = np.asarray(raw[f"state/joint/{field}"][:], dtype=np.float32)
            if values.shape != (count, len(names)) or not np.isfinite(values).all():
                raise ValueError(f"Joint names/data mismatch or invalid {field}; re-extract the source bag")
            joint[field] = values
        for i, frame in enumerate(frames):
            recorded = frame["robot"]["joints"]
            if recorded["joint_name"] != names or not np.allclose(
                recorded["joint_position"], joint["position"][i], atol=1e-6
            ):
                raise ValueError(f"state.json and H5 joint layout differ at frame {i}")
    config["joint_state_order"] = names
    arm_indices = [names.index(n) for n in config["arm_joint_names"]]
    arm_pos, arm_quat = calculate_ee_pose_batch(
        joint["position"][:, arm_indices],
        "G2_crsB_wuji" if "wuji" in robot.lower() else ("G2_crsB" if "crsB" in robot else "G2"),
    )
    base_pos, base_quat, ee_pos, ee_quat, eff_pos, eff_quat = [], [], [], [], [], []
    for i, frame in enumerate(frames):
        base = np.asarray(frame["robot"]["pose"], dtype=np.float64)
        arm_base = np.asarray(frame["robot"]["arm_base_pose"], dtype=np.float64)
        for pose in (base, arm_base):
            if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
                raise ValueError(f"Invalid robot transform at frame {i}")
            if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4) or np.linalg.det(pose[:3, :3]) < 0:
                raise ValueError(f"Invalid robot rotation at frame {i}")
        base_pos.append(base[:3, 3])
        base_quat.append(Rotation.from_matrix(base[:3, :3]).as_quat())
        frame["ee"], frame["ee_center"] = {}, {}
        positions, quats, centers, center_quats = [], [], [], []
        for side_idx, side in enumerate(("left", "right")):
            wrist = np.eye(4)
            wrist[:3, :3] = Rotation.from_quat(arm_quat[i, side_idx]).as_matrix()
            wrist[:3, 3] = arm_pos[i, side_idx]
            world_wrist = arm_base @ wrist
            frame["ee"][side] = {"pose": world_wrist.tolist()}
            local_wrist = np.linalg.inv(base) @ world_wrist
            p, q = local_wrist[:3, 3], Rotation.from_matrix(local_wrist[:3, :3]).as_quat()
            positions.append(p)
            quats.append(q)
            if grippers[side_idx] == "zhiyuan_gripper_wuji":
                from geniesim_teleop.data_recording.derive_fields import chain_transform, load_link_chain

                urdf = Path(__file__).resolve().parent / "config/robot_cfg/G2/G2_crsB_wuji.urdf"
                chain = load_link_chain(str(urdf), "arm_base_link", f"hand_{side[0]}_index_link4")
                if not chain:
                    raise ValueError(f"Missing Wuji {side} effector FK chain")
                angles = dict(zip(names, joint["position"][i]))
                center = np.linalg.inv(base) @ arm_base @ chain_transform(chain, angles)
                cp, cq = center[:3, 3], Rotation.from_matrix(center[:3, :3]).as_quat()
            elif grippers[side_idx] == "zhiyuan_gripper_ctek90d":
                cp, cq = effector_from_arm_end(p, q, "ctek90d")
            else:
                cp, cq = effector_from_arm_end(p, q, "omnipicker")
            center = np.eye(4)
            center[:3, :3] = Rotation.from_quat(cq).as_matrix()
            center[:3, 3] = cp
            frame["ee_center"][side] = {"pose": (base @ center).tolist()}
            centers.append(cp)
            center_quats.append(cq)
        ee_pos.append(positions)
        ee_quat.append(quats)
        eff_pos.append(centers)
        eff_quat.append(center_quats)
    # Teleop extraction already rotates camera TFs into OpenCV convention.
    state["camera_convention"] = "opencv"
    (staging / "state.json").write_text(json.dumps(state), encoding="utf-8")
    with h5py.File(staging / "aligned_joints_all.h5", "w") as raw:
        raw.create_dataset("timestamp", data=timestamps)
        for prefix in ("state", "action"):
            group = raw.require_group(f"{prefix}/joint")
            group.attrs["name"] = names
            for field, data in joint.items():
                group.create_dataset(field, data=data)
            end = raw.require_group(f"{prefix}/end")
            for field, data in dict(
                position=ee_pos if prefix == "state" else arm_pos,
                orientation=ee_quat if prefix == "state" else arm_quat,
                arm_position=arm_pos,
                arm_orientation=arm_quat,
                effector_position=eff_pos,
                effector_orientation=eff_quat,
            ).items():
                end.create_dataset(field, data=np.asarray(data, dtype=np.float32))
        raw.create_dataset("state/robot/position", data=np.asarray(base_pos, dtype=np.float32))
        raw.create_dataset("state/robot/orientation", data=np.asarray(base_quat, dtype=np.float32))


class SimDataConverter:
    def __init__(
        self,
        record_path,
        output_path,
        job_id,
        task_id,
        episode_id,
        gripper_names=["zhiyuan_gripper_omnipicker", "zhiyuan_gripper_omnipicker"],
        robot_type="G2",
        skip_data_info=False,
    ):
        self._calibration_issues = []
        self._record_path = record_path
        self._output_path = output_path
        self._skip_data_info = skip_data_info
        if not os.path.exists(self._output_path):
            os.makedirs(self._output_path)
        metadata_file = os.path.join(self._record_path, "metadata.yaml")
        with open(metadata_file, "r") as f:
            self._metadata = yaml.load(f, Loader=yaml.FullLoader)
        self._start_ts = self._metadata["rosbag2_bagfile_information"]["starting_time"]["nanoseconds_since_epoch"]
        state_file = os.path.join(self._record_path, "state.json")
        with open(state_file, "r", encoding="utf-8") as f:
            self._state = json.load(f)
        # Use the median positive timestamp gap as a nominal frame interval.
        # This removes replay gaps and duplicated terminal timestamps.
        _fts = np.array([fr["time_stamp"] for fr in self._state["frames"]], dtype=np.float64)
        _d = np.diff(_fts)
        _pos = _d[_d > 0]
        self._nominal_dt = float(np.median(_pos)) if _pos.size else (1.0 / 30)
        # Duration follows retained frame count, excluding removed replay gaps.
        self._duration = (len(_fts) - 1) * self._nominal_dt if len(_fts) > 1 else 0.0
        self._job_id = job_id
        self._task_id = task_id
        self._episode_id = episode_id
        self.gripper_names = ["zhiyuan_gripper_ctek90d" if name == "gp_90d_gripper" else name for name in gripper_names]
        robot_type = self._state.get("robot", {}).get("name") or robot_type
        self._recorded_robot = robot_type
        if "wuji" in robot_type.lower():
            self.gripper_names = ["zhiyuan_gripper_wuji", "zhiyuan_gripper_wuji"]
        if not self._state["frames"]:
            raise ValueError("Recording contains no frames")
        expected = (
            "zhiyuan_gripper_wuji"
            if self._is_wuji
            else ("zhiyuan_gripper_ctek90d" if "90d" in robot_type.lower() else "zhiyuan_gripper_omnipicker")
        )
        if not robot_type.startswith("G2") or self.gripper_names != [expected, expected]:
            raise ValueError(f"Unsupported teleop robot/gripper combination: {robot_type}, {gripper_names}")
        self._job_id = int(job_id or 1)
        self._task_id = int(task_id or 1)
        self._episode_id = int(episode_id or 1)
        # 90D variant keeps the G2A recording format/SN but reads its own joint
        # config (different gripper joints: idx32/idx72 control vs omnipicker idx41/idx81).
        is_g2_90d_robot = "90d" in str(robot_type or "").lower()
        self._is_g2_90d = is_g2_90d_robot
        # The G2 crsB variant uses crsB motor metadata and a BC serial
        # segment. Legacy G2 uses crs metadata and a B9 segment.
        self._arm_model = "crsB" if ("G2" in str(robot_type) and "crsB" in str(robot_type)) else "crs"
        if "G2" in robot_type:
            self.robot_type = "G2A"
        elif "ARX" in robot_type:
            self.robot_type = "ARX"
        elif "Aloha" in robot_type or "aloha" in robot_type:
            self.robot_type = "Aloha"
        else:
            self.robot_type = "A2D"
        config_path = Path(__file__).resolve().parent / "config/robot_cfg/robot_joint_names.json"
        with open(config_path, "r") as f:
            if self.robot_type == "G2A" and is_g2_90d_robot:
                self.config = json.load(f)["G2_crsB_90d"]
            elif self.robot_type == "G2A":
                self.config = json.load(f)["G2"]
            elif self.robot_type == "ARX":
                self.config = json.load(f)["ARX"]
            elif self.robot_type == "Aloha":
                self.config = json.load(f)["Aloha"]
            else:
                self.config = json.load(f)["G1"]
        if self._is_wuji:
            from geniesim_teleop.data_recording.wuji_layout import JOINT_GROUPS

            for side in ("left", "right"):
                self.config[f"{side}_effector_joint_names"] = list(JOINT_GROUPS[f"{side}_effector"])
        else:
            for side in ("left", "right"):
                self.config[f"{side}_effector_joint_names"] = [self.config[f"{side}_effector_joint_name"]]
        self.joint_state = []
        self.joint_action = []

    @property
    def _is_wuji(self):
        return any("wuji" in name.lower() for name in self.gripper_names)

    def _write_episode(self):
        self.convert_parameters()
        self.convert_hardware()
        self.convert_meta_info()
        self.convert_h5()
        self.h5_post_process()
        self._backfill_derived_fields(os.path.join(self._output_path, "aligned_joints.h5"))
        record_dir = os.path.join(self._output_path, "record")
        os.makedirs(record_dir, exist_ok=True)
        shutil.copy(os.path.join(self._output_path, "aligned_joints.h5"), record_dir)
        self.genrate_data_info()
        self.gen_review()
        self.mkdir()

    def convert(self):
        record, output = Path(self._record_path).resolve(), Path(self._output_path).resolve()
        info_path = record / "recording_info.json"
        info = json.loads(info_path.read_text()) if info_path.is_file() else {}
        inputs = _fingerprint(record)
        record_size = get_directory_size(record)
        original = self._record_path, self._output_path, self._state, self.config
        with tempfile.TemporaryDirectory(prefix=".teleop-export-", dir=output) as work:
            work = Path(work)
            staging, generated = work / "input", work / "output"
            staging.mkdir()
            generated.mkdir()
            shutil.copyfile(record / "metadata.yaml", staging / "metadata.yaml")
            # Human collection has no planner phases; do not invent labels.
            (staging / "frame_state.json").write_text("[]")
            try:
                self._state = deepcopy(self._state)
                self._state["cameras"] = {canonical_camera_name(k): v for k, v in self._state["cameras"].items()}
                for frame in self._state["frames"]:
                    frame["cameras"] = {canonical_camera_name(k): v for k, v in frame.get("cameras", {}).items()}
                self._calibration_issues = []
                self.config = deepcopy(self.config)
                _prepare_teleop_input(
                    record, staging, self._state, self.config, self.gripper_names, self._recorded_robot
                )
                self._record_path, self._output_path = str(staging), str(generated)
                self.joint_state, self.joint_action = [], []
                self._write_episode()
            finally:
                self._record_path, self._output_path, self._state, self.config = original
            # Staging omits videos/bags; report the original episode's size.
            meta_path = generated / "meta_info.json"
            meta = json.loads(meta_path.read_text())
            meta["file_size"] = record_size / 1024.0
            if self._calibration_issues:
                meta["data_validate"]["validate"] = False
            meta_path.write_text(json.dumps(meta, indent=4))
            data_path = generated / "data_info.json"
            if data_path.is_file():
                data = json.loads(data_path.read_text())
                description = info.get("task_description") or {}
                if isinstance(description, dict):
                    for key in ("task_name", "english_task_name", "init_scene_text"):
                        data[key] = description.get(key) or info.get(key) or data[key]
                if self._calibration_issues:
                    data["label_info"]["cloud_post_processing_result"]["data_valid"] = False
                data_path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")
            provenance = {
                "source": "teleop",
                "action_joint_source": "measured_joint_states",
                "end_pose_source": "local_urdf_fk_and_recorded_base_transforms",
                "action_robot_velocity": "unrecorded; canonical six-zero placeholder",
                "vr_buttons": "unrecorded; canonical zero placeholders",
                "operator_event": "canonical placeholder",
                "phase_labels": "unavailable",
            }
            (generated / "post_proc/teleop_source.json").write_text(json.dumps(provenance, indent=2))
            (generated / "post_proc/calibration_status.json").write_text(
                json.dumps(
                    {
                        "valid": not self._calibration_issues,
                        "issues": self._calibration_issues,
                    },
                    indent=2,
                )
            )
            if output != record and (record / "observations").is_dir():
                shutil.copytree(record / "observations", generated / "observations")
            # Legacy Wuji head video names used the USD basename. Preserve the
            # source and publish a canonical alias beside the other final videos.
            video_dir = record / "observations/videos"
            for video in video_dir.glob("*.mp4"):
                name = canonical_camera_name(video.stem) + video.suffix
                if name != video.name:
                    target = generated / "observations/videos" / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(video, target)
            if _fingerprint(record) != inputs:
                raise RuntimeError("Recording changed during conversion; retry after recording finishes")
            files = sorted(str(p.relative_to(generated)) for p in generated.rglob("*") if p.is_file())
            (output / MARKER).unlink(missing_ok=True)
            for name in files:
                destination = output / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(generated / name, destination)
            marker = {"revision": FORMAT_REVISION, "inputs": inputs, "files": files}
            temporary_marker = work / "complete.json"
            temporary_marker.write_text(json.dumps(marker, indent=2))
            os.replace(temporary_marker, output / MARKER)

    def mkdir(self):
        dirs = ["logs", "record", "post_proc"]
        for d in dirs:
            if not os.path.exists(os.path.join(self._output_path, d)):
                os.makedirs(os.path.join(self._output_path, d))
                with open(os.path.join(self._output_path, d, ".keep"), "w"):
                    pass

    def _gripper_model(self, idx):
        """Return the short gripper model name used by hardware and URDF metadata."""
        name = self.gripper_names[idx] if idx < len(self.gripper_names) else ""
        return {
            "zhiyuan_gripper_omnipicker": "omnipicker",
            "zhiyuan_gripper_ctek90d": "ctek90d",
            "zhiyuan_gripper_wuji": "wuji",
        }.get(name, "omnipicker")

    def _urdf_type(self):
        """Return the URDF model family for metadata and calibration."""
        if "G2" in str(self.robot_type):
            return f"G2_t2_{self._arm_model}"
        if self.robot_type == "A2D":
            return "G1"
        return str(self.robot_type)

    def _sn_code(self):
        """Return a fixed simulated serial number shared by data_info and meta_info.

        G2A crsB variants use a BC segment; legacy crs uses B9.
        """
        if self.robot_type == "A2D":
            return "A2D0015AB00061"
        if self.robot_type == "ARX":
            return "ARX0000000000001"
        if self.robot_type == "Aloha":
            return "ALOHA000000000001"
        if self._arm_model == "crsB":
            return "G2A0004BC00170"
        return "G2A0004B900170"  # Default G2A serial placeholder.

    def genrate_data_info(self):
        if self._skip_data_info:
            log.info("Skip genrate data info")
            return
        actions = []
        last_action = ""
        last_error = {}
        start_frame = 0
        end_frame = 0
        error_start_frame = 0
        error_end_frame = 0
        first_frame = -1
        last_error_description = {}
        with open(os.path.join(self._record_path, "frame_state.json"), "r") as f:
            frame_states = json.load(f)
        init_scene_text = ""
        english_task_name = ""
        task_name = ""
        if len(frame_states) > 0:
            if "task_description" in frame_states[0]:
                task_description = frame_states[0]["task_description"]
                init_scene_text = task_description.get("init_scene_text", "")
                english_task_name = task_description.get("english_task_name", "")
                task_name = task_description.get("task_name", "")
        error_descriptions = []
        for i, frame_state in enumerate(frame_states):
            state_ts = frame_state["time_stamp"]
            arm = frame_state["frame_state"]["arm"]
            act = frame_state["frame_state"]["action"]
            active_id = frame_state["frame_state"]["active_id"]
            passive_id = frame_state["frame_state"]["passive_id"]
            action_description = frame_state["frame_state"].get("action_description", {}) or {}
            action_text = action_description.get("action_text", "")
            english_action_text = action_description.get("english_action_text", "")
            current_action_key = " ".join([arm, active_id, act, passive_id, action_text, english_action_text])
            if current_action_key != last_action or i == len(frame_states) - 1:
                found = False
                for idx, fm in enumerate(self._state["frames"]):
                    if fm["time_stamp"] >= state_ts:
                        end_frame = idx
                        found = True
                        break
                if not found and i == len(frame_states) - 1:
                    end_frame = len(self._state["frames"]) - 1
                if len(last_action) and i > 0:
                    last_action.split(" ")[2]
                    last_action_description = frame_states[i - 1]["frame_state"].get(
                        "action_description",
                        {
                            "action_text": "",
                            "english_action_text": "",
                        },
                    )
                    action_end_frame = end_frame - 1 if i != len(frame_states) - 1 else end_frame
                    actions.append(
                        {
                            "start_frame": start_frame,
                            "end_frame": action_end_frame,
                            "action_text": last_action_description["action_text"],
                            "english_action_text": last_action_description["english_action_text"],
                            # Normalize skill names, for example pick to Pick.
                            "skill": _normalize_skill(last_action.split(" ")[2]),
                        }
                    )
                    if first_frame == -1:
                        first_frame = start_frame
                start_frame = end_frame
                last_action = current_action_key
            error_description = frame_state["frame_state"].get("error_description", {})
            error_cause = ""
            error_cause_english = ""
            error_type = error_description.get("type", "")
            description = error_description.get("description", {})
            comment = description.get("comment", "")
            if error_type == "WrongTarget":
                error_cause = "Wrong target object"
                error_cause_english = "Wrong target object"
            elif error_type == "MissGrasp":
                error_cause = "Failed to grasp"
                error_cause_english = "Failed to grasp"
            elif error_type == "Drop":
                error_cause = "Drop"
                error_cause_english = "Drop"
            elif error_type == "KeepClose":
                error_cause = "Grasp without release"
                error_cause_english = "Grasp without release"
            elif error_type == "RandomPerturbations":
                error_cause = "Position perturbation"
                error_cause_english = "Position perturbation"
            if error_cause != last_error.get("error_cause", None) and last_error.get("error_cause", "") != "":
                for idx, fm in enumerate(self._state["frames"]):
                    if fm["time_stamp"] >= state_ts:
                        error_end_frame = idx
                        break
                error_info = {
                    "frame_type_name": "Error Frame",
                    "frame_type_name_english": "Error Frame",
                    "start": max(error_start_frame, first_frame),
                    "end": error_end_frame - 1,
                    "frame_detail": last_error,
                }
                if last_error_description.get("type", "") == "Drop":
                    if len(actions) > 1 and actions[-1]["skill"].lower() == "place":
                        last_action_start = actions[-1]["start_frame"]
                        params = last_error_description.get("params", {})
                        drop_timing = params.get("drop_timing", 0.2)
                        motion_run_ratio = params.get("motion_run_ratio", 1.0)
                        real_start = math.ceil(
                            (error_info["end"] - last_action_start) * (drop_timing / motion_run_ratio) * 0.25
                            + last_action_start
                        )
                        real_start = max(min(real_start, error_info["end"] - 1), last_action_start)
                        error_info["start"] = real_start
                elif last_error_description.get("type", "") in ["MissGrasp", "KeepClose"]:
                    if len(actions) > 0:
                        last_action_start = actions[-1]["start_frame"]
                        error_info["start"] = max(last_action_start, error_info["start"])
                elif last_error_description.get("type", "") == "RandomPerturbations":
                    real_start = math.ceil((error_info["end"] - error_info["start"]) * 0.5 + error_info["start"])
                    error_info["start"] = max(real_start, error_info["start"])
                error_descriptions.append(error_info)
                error_start_frame = error_end_frame
            last_error_description = error_description
            last_error = {
                "error_cause": error_cause,
                "error_cause_english": error_cause_english,
                "restorable": description.get("restorable", True),
                "comment": comment,
            }
        # merge the neighbor action with same action_text
        merged_actions = []
        for action in actions:
            if (
                len(merged_actions) > 0
                and merged_actions[-1]["action_text"] == action["action_text"]
                and action["action_text"] != ""
            ):
                merged_actions[-1]["end_frame"] = action["end_frame"]
            else:
                merged_actions.append(action)
        actions = merged_actions
        data_info = {
            "episode_id": self._episode_id,
            "task_id": self._task_id,
            "task_name": task_name,
            "english_task_name": english_task_name,
            "data_type": "Regular",
            "job_id": self._job_id,
            "init_scene_text": init_scene_text,
            "raw_data_path": "",
            "aligned_data_path": "",
            "visualizing_data_path": "",
            "sn_code": self._sn_code(),
            "label_info": {
                "error_label": "",
                "action_config": actions,
                "key_frame": {"single": [], "dual": error_descriptions},
                "cloud_post_processing_result": {
                    "data_valid": True,
                    "drop_frame_rate": 0.0,
                    "filter_frame_rate": 0.0,
                },
            },
            "sim_data_version": SIM_DATA_VERSION,
        }
        with open(os.path.join(self._output_path, "data_info.json"), "w", encoding="utf-8") as f:
            json.dump(data_info, f, indent=4, ensure_ascii=False)

    def _arm_end_in_base(self, frame, side):
        """Return arm-end pose in base_link coordinates from world poses."""
        Tw_robot = np.asarray(frame["robot"]["pose"], dtype=np.float64)
        Tw_ee = np.asarray(frame["ee"][side]["pose"], dtype=np.float64)
        return np.linalg.inv(Tw_robot) @ Tw_ee

    def _torso_urdf(self):
        """Load the simulation torso/head URDF chain for G2.

        The IK SDK body chain differs from the simulation torso. Return None
        for non-G2 robots, which lack this simulation chain.
        """
        if "G2" not in str(self.robot_type):
            return None
        p = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config", "robot_cfg", "G2", "G2_omnipicker_fixed_dual.urdf"
        )
        return p if os.path.isfile(p) else None

    def _head_link3_in_base(self, frame):
        """Return head_link3 pose in base_link coordinates.

        Prefer forward kinematics through the simulation URDF. If unavailable,
        fall back to the recorded arm base and the shorter IK SDK head chain.
        """
        ang = {
            n: float(v)
            for n, v in zip(frame["robot"]["joints"]["joint_name"], frame["robot"]["joints"]["joint_position"])
        }
        torso = self._torso_urdf()
        if torso is not None:
            chain = load_link_chain(torso, "base_link", "head_link3")
            if chain is not None:
                return chain_transform(chain, ang)
        # Fall back for non-G2 robots or a missing simulation URDF.
        Tw_robot = np.asarray(frame["robot"]["pose"], dtype=np.float64)
        arm_base_base = np.linalg.inv(Tw_robot) @ np.asarray(frame["robot"]["arm_base_pose"], dtype=np.float64)
        urdf = self._urdf_path()
        T_b5_armbase = chain_transform(load_link_chain(urdf, "body_link5", "arm_base_link"), ang)
        T_b5_head3 = chain_transform(load_link_chain(urdf, "body_link5", "head_link3"), ang)
        return arm_base_base @ np.linalg.inv(T_b5_armbase) @ T_b5_head3

    def _static_extrinsic_for(self, cam_file, first_frame, cam_pose0):
        """Derive static end_T_cam from mount and first-frame camera poses.

        Arm cameras mount on arm_end; head cameras mount on head_link3.
        """
        try:
            if "hand_left" in cam_file:
                mount = self._arm_end_in_base(first_frame, "left")
                sensor_model = _GRIPPER_SENSOR_MODEL.get(self.gripper_names[0] if self.gripper_names else "", "old_op")
            elif "hand_right" in cam_file:
                mount = self._arm_end_in_base(first_frame, "right")
                sensor_model = _GRIPPER_SENSOR_MODEL.get(
                    self.gripper_names[1] if len(self.gripper_names) > 1 else "", "old_op"
                )
            else:  # head cameras
                mount = self._head_link3_in_base(first_frame)
                sensor_model = _HEAD_SENSOR_MODEL.get(cam_file, cam_file + "_mani")
            static = np.linalg.inv(mount) @ cam_pose0
            q = R.from_matrix(static[:3, :3]).as_quat()  # xyzw
            return {
                "calib_mode": "offline",
                "rotation": {"w": float(q[3]), "x": float(q[0]), "y": float(q[1]), "z": float(q[2])},
                "sensor_model": sensor_model,
                "translation": {"x": float(static[0, 3]), "y": float(static[1, 3]), "z": float(static[2, 3])},
            }, True
        except Exception as e:
            log.warning("static extrinsic for %s skipped: %s", cam_file, e)
            return None, False

    def convert_parameters(self):
        paramters_path = os.path.join(self._output_path, "parameters")
        if not os.path.exists(paramters_path):
            os.makedirs(paramters_path)
        camera_parameters_path = os.path.join(paramters_path, "sensor")
        if not os.path.exists(camera_parameters_path):
            os.makedirs(camera_parameters_path)

        def _intr_float(intr, *keys, default=0.0):
            for k in keys:
                v = intr.get(k)
                if v is None or v == "None":
                    continue
                try:
                    value = float(v)
                    if np.isfinite(value):
                        return value
                except (TypeError, ValueError):
                    continue
            return default

        # Collect intrinsics and resolutions for camera_resolution.json.
        camera_resolution: dict[str, dict[str, str]] = {}
        for cam in self._state["cameras"]:
            camera = self._state["cameras"][cam]
            intr = camera.get("intrinsic") or {}
            cam_int_params = {}
            fisheye = intr.get("model", intr.get("distortion_model")) == "opencvFisheye"
            if fisheye:
                cam_int_params = {
                    key: _intr_float(intr, source, default=None)
                    for key, source in (
                        ("Fx", "fx"),
                        ("Fy", "fy"),
                        ("Cx", "cx"),
                        ("Cy", "cy"),
                        ("k1", "k1"),
                        ("k2", "k2"),
                        ("k3", "k3"),
                        ("k4", "k4"),
                    )
                }
                cam_int_params["distortion_model"] = "opencvFisheye"
                cam_int_params["SN"] = "sim"
            elif "fisheye" in cam:
                cam_int_params["fu"] = _intr_float(intr, "fx", default=None)
                cam_int_params["fv"] = _intr_float(intr, "fy", default=None)
                cam_int_params["pu"] = _intr_float(intr, "ppx", "cx", default=None)
                cam_int_params["pv"] = _intr_float(intr, "ppy", "cy", default=None)
                cam_int_params["distortion_model"] = "fisheyePolynomial"
                cam = "head_center_fisheye"
            else:
                cam_int_params["Fx"] = _intr_float(intr, "fx", default=None)
                cam_int_params["Fy"] = _intr_float(intr, "fy", default=None)
                cam_int_params["Cx"] = _intr_float(intr, "cx", "ppx", default=None)
                cam_int_params["Cy"] = _intr_float(intr, "cy", "ppy", default=None)
                cam_int_params["k1"] = _intr_float(intr, "k1")
                cam_int_params["k2"] = _intr_float(intr, "k2")
                cam_int_params["p1"] = _intr_float(intr, "p1")
                cam_int_params["p2"] = _intr_float(intr, "p2")
                cam_int_params["k3"] = _intr_float(intr, "k3")
                # Fill unavailable higher-order stereo distortion terms with zero.
                if "stereo" in cam:
                    cam_int_params["k4"] = 0.0
                    cam_int_params["k5"] = 0.0
                    cam_int_params["k6"] = 0.0
                cam_int_params["SN"] = "CPBC853000CC"
            missing = [key for key, value in cam_int_params.items() if value is None]
            for key in ("Fx", "Fy", "fu", "fv"):
                if key in cam_int_params and cam_int_params[key] is not None and cam_int_params[key] <= 0:
                    missing.append(key)
                    cam_int_params[key] = None
            if missing:
                self._calibration_issues.append({"camera": cam, "missing_fields": missing})
                log.warning("Incomplete camera calibration for %s: %s", cam, missing)
            cam = process_camera_parameter_file_name(cam)
            # Parse intrinsic.imageSize and store string dimensions under the
            # hardware camera name. Skip missing or invalid resolutions.
            _isz = re.match(
                r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", str(intr.get("imageSize", "")).replace("[", "(").replace("]", ")")
            )
            if _isz:
                camera_resolution[cam.replace("rgbd", "rgb")] = {
                    "width": _isz.group(1),
                    "height": _isz.group(2),
                }
            if "rgbd" in cam:
                with open(
                    os.path.join(camera_parameters_path, "intrinsic_" + cam.replace("rgbd", "rgb") + ".json"),
                    "w",
                ) as f:
                    json.dump(cam_int_params, f, indent=4)
                with open(
                    os.path.join(
                        camera_parameters_path,
                        "intrinsic_" + cam.replace("rgbd", "depth") + ".json",
                    ),
                    "w",
                ) as f:
                    json.dump(cam_int_params, f, indent=4)
            else:
                with open(
                    os.path.join(camera_parameters_path, "intrinsic_" + cam + ".json"),
                    "w",
                ) as f:
                    json.dump(cam_int_params, f, indent=4)
        # Include only observed cameras in camera_resolution.json.
        with open(os.path.join(camera_parameters_path, "camera_resolution.json"), "w") as f:
            json.dump(camera_resolution, f, indent=4)
        # Detect camera convention (OpenCV vs OpenGL) using the first frame:
        # In OpenCV the Z-axis points forward; in OpenGL it points backward.
        # Hands are normally in front of the head, so if hand cameras lie in the
        # head camera's +Z direction the convention is already OpenCV.
        is_opencv_cam = True
        first_frame = self._state["frames"][0]
        first_frame_cams = first_frame.get("cameras") or {}
        if not first_frame_cams:
            log.warning("frame[0] has no camera poses; skipping extrinsic JSON generation")
            return
        world_to_robot_first = np.linalg.inv(np.array(first_frame["robot"]["pose"]))

        head_cam = None
        hand_cams = []
        for cam_name in self._state["cameras"]:
            if cam_name not in first_frame_cams:
                continue
            if "head" in cam_name and "stereo" not in cam_name:
                head_cam = cam_name
            if "hand_left" in cam_name or "hand_right" in cam_name:
                hand_cams.append(cam_name)

        if self._state.get("camera_convention") == "opencv":
            is_opencv_cam = True
        elif head_cam and hand_cams:
            head_pose = world_to_robot_first @ np.array(first_frame["cameras"][head_cam]["pose"])
            head_z_axis = head_pose[:3, 2]
            head_position = head_pose[:3, 3]

            opencv_votes = sum(
                np.dot(
                    (world_to_robot_first @ np.array(first_frame["cameras"][hc]["pose"]))[:3, 3] - head_position,
                    head_z_axis,
                )
                > 0
                for hc in hand_cams
            )
            is_opencv_cam = opencv_votes > len(hand_cams) / 2
        print("is_opencv_cam", is_opencv_cam)
        # 180° rotation around X-axis (converts OpenGL camera -> OpenCV camera)
        opengl_to_opencv = np.diag([1.0, -1.0, -1.0, 1.0])

        for cam in self._state["cameras"]:
            cam_ext_params = []
            first_frame = None  # First frame with this camera for calibration.
            first_cam_pose = None
            for frame in self._state["frames"]:
                frame_cams = frame.get("cameras") or {}
                if cam not in frame_cams:
                    continue
                cam_pose_world = np.array(frame_cams[cam]["pose"])
                world_to_robot = np.linalg.inv(np.array(frame["robot"]["pose"]))
                cam_pose = world_to_robot @ cam_pose_world
                if not is_opencv_cam:
                    cam_pose = cam_pose @ opengl_to_opencv
                if first_frame is None:
                    first_frame = frame
                    first_cam_pose = cam_pose
                cam_ext_params.append(
                    {
                        "rotation": [
                            [cam_pose[0][0], cam_pose[0][1], cam_pose[0][2]],
                            [cam_pose[1][0], cam_pose[1][1], cam_pose[1][2]],
                            [cam_pose[2][0], cam_pose[2][1], cam_pose[2][2]],
                        ],
                        "translation": [
                            cam_pose[0][3],
                            cam_pose[1][3],
                            cam_pose[2][3],
                        ],
                    }
                )
            if not cam_ext_params:
                continue
            cam_file = process_camera_parameter_file_name(cam)
            with open(
                os.path.join(camera_parameters_path, "extrinsic_end_T_" + cam_file + "_aligned.json"),
                "w",
            ) as f:
                json.dump(cam_ext_params, f, indent=4)
            # Static camera transform relative to arm_end or head_link3,
            # using the hardware calibration metadata format.
            static_ext, ok = self._static_extrinsic_for(cam_file, first_frame, first_cam_pose)
            if ok:
                with open(
                    os.path.join(camera_parameters_path, "extrinsic_end_T_" + cam_file + ".json"),
                    "w",
                ) as f:
                    json.dump(static_ext, f, indent=4)

    def convert_hardware(self):
        """Generate G2 parameters/hardware from in-code templates.

        Fixed firmware and serial values come from G2 references. Gripper
        versions depend on the installed model; unavailable firmware versions
        in JSON manifests use the placeholder "sim".
        """
        if "G2" not in str(self.robot_type):
            return  # Hardware templates are currently defined only for G2.
        from geniesim_teleop.data_recording import hardware_templates as hwt

        hw_dir = os.path.join(self._output_path, "parameters", "hardware")
        os.makedirs(hw_dir, exist_ok=True)

        def _dump_yaml(fname, content):
            with open(os.path.join(hw_dir, fname), "w", encoding="utf-8") as f:
                yaml.safe_dump(content, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        for fname, content in hwt.fixed_g2_yaml_files(self._arm_model).items():
            _dump_yaml(fname, content)
        for side_idx, fname in ((0, "left_hand_version.yaml"), (1, "right_hand_version.yaml")):
            _dump_yaml(fname, hwt.hand_version(self._gripper_model(side_idx)))
        aid = dict(hwt.AID_INFO)
        aid["type"] = self.robot_type
        aid["AID"] = self._sn_code()
        _dump_yaml("aid_info.yaml", aid)

        # JSON manifests use "sim" where no simulated firmware version exists.
        def _dump_json(fname, content):
            with open(os.path.join(hw_dir, fname), "w", encoding="utf-8") as f:
                json.dump(content, f, ensure_ascii=False, indent=2)

        _dump_json("head_manifest.json", {"app_version": "sim"})
        _dump_json("pad_manifest.json", {"app_version": "sim"})
        _dump_json("rgbd_manifest.json", {"app_version": "sim", "fw_version": "sim"})
        _dump_json("soc_manifest.json", {"app_version": "sim", "fw_version": "sim"})

    def convert_meta_info(self):
        meta_info = {
            # Reuse the per-robot serial as AID and follow reference metadata
            # values for the remaining fields.
            "AID": self._sn_code(),
            "data_map_version": "v1.0.0",
            "data_validate": {"validate": True},
            "urdf": {
                "left_end_model": self._gripper_model(0),
                "right_end_model": self._gripper_model(1),
                "urdf_type": self._urdf_type(),
            },
            # The top-level urdf_type identifies the robot family and gripper.
            # ARX and Aloha keep their model names without a gripper suffix.
            "urdf_type": (
                f"{self._urdf_type()}_{self._gripper_model(0)}"
                if self.robot_type in ("G2A", "A2D")
                else self._urdf_type()
            ),
            "author": "agibot",
            # Store clip timestamps as integer Unix seconds.
            "clip_end_time": int(self._start_ts / 1e9 + self._duration),
            "clip_start_time": int(self._start_ts / 1e9),
            "create_time": datetime.datetime.fromtimestamp(
                self._start_ts / 1e9, tz=pytz.timezone("Asia/Shanghai")
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "duration": self._duration,
            "ee_list": [
                {
                    "name": "left_hand",
                    "type": (self.gripper_names[0] if len(self.gripper_names) > 0 else "ctek_gripper_120s"),
                },
                {
                    "name": "right_hand",
                    "type": (self.gripper_names[1] if len(self.gripper_names) > 1 else "ctek_gripper_120s"),
                },
            ],
            # Use positive placeholder IDs when simulation has no platform ID.
            "episode_id": int(self._episode_id or 1),
            "episode_token": str(uuid.uuid4()),
            # Store file_size in KiB, matching the hardware metadata convention.
            "file_size": get_directory_size(self._record_path) / 1024.0,
            "fps_validate": True,
            "integrity": {"integrity": True, "reason": 0},
            "job_id": int(self._job_id or 1),
            "mocap_mapping_status": [],
            # Distinguish simulation metadata with a lowercase _sim robot type;
            # retain uppercase self.robot_type for internal comparisons.
            "robot_type": f"{self.robot_type.lower()}_sim",
            "sw_version": "genie_g02_rb_2.3.5_d7cbd92a_2026-06-12-15-25-57_r1.tar.gz",
            "task_id": int(self._task_id or 1),
            "task_mode": "TDC",
            "text": '{"description":"SIM"}',
            "version": "v0.1.3",
        }
        with open(os.path.join(self._output_path, "meta_info.json"), "w") as f:
            json.dump(meta_info, f, indent=4)

    def convert_h5(self):
        h5_file_path = os.path.join(self._record_path, "aligned_joints_all.h5")
        converted_h5_file_path = os.path.join(self._output_path, "aligned_joints.h5")
        if os.path.isfile(converted_h5_file_path):
            os.remove(converted_h5_file_path)
        with h5py.File(h5_file_path, "r") as f_in:
            with h5py.File(converted_h5_file_path, "w") as f_out:
                data_length = len(f_in["timestamp"])
                for i in range(data_length):
                    group_name = str(i)
                    group = f_out.create_group(group_name)

                    action_group = group.create_group("action")
                    state_group = group.create_group("state")
                    time_group = group.create_group("timestamp")

                    # Rebuild monotonic timestamps from the nominal frame
                    # interval; raw wall-clock values may include replay gaps.
                    timestamp = np.uint64(self._start_ts + round(i * self._nominal_dt * 1e9))
                    group.create_dataset("main_timestamp", data=timestamp)

                    self.process_action_data(f_in, action_group, i)
                    self.process_state_data(f_in, state_group, i)
                    self.process_timestamp_data(timestamp, time_group)

    def process_action_data(self, f_in, action_group, index):
        prefix = "action"
        joint_position = f_in[f"{prefix}/joint/position"][index]
        joint_position_state = f_in["state/joint/position"][index]
        end_group = action_group.create_group("end")
        end_group.create_dataset("orientation", data=f_in[f"{prefix}/end/orientation"][index])
        end_group.create_dataset("position", data=f_in[f"{prefix}/end/position"][index])
        head_index = [self.config["joint_state_order"].index(name) for name in self.config["head_joint_names"]]
        head_group = action_group.create_group("head")
        head_group.attrs["name"] = self.config["head_joint_names"]
        head_group.create_dataset("position", data=joint_position[head_index])
        joint_index = [self.config["joint_state_order"].index(name) for name in self.config["arm_joint_names"]]
        joint_group = action_group.create_group("joint")
        joint_group.attrs["name"] = self.config["arm_joint_names"]
        arm_joints = joint_position[joint_index]  # Fourteen joints: seven per arm.
        joint_group.create_dataset("position", data=arm_joints)
        self.joint_action.append(arm_joints)
        # Copy precomputed reference poses when available; post-processing
        # fills missing action/{l,r}_refpose/orientation values later.
        end_in = f_in["action/end"]
        if "refpose_orientation" in end_in:
            rp = f_in["action/end/refpose_orientation"][index]  # [2,4]
            action_group.create_group("left_refpose").create_dataset(
                "orientation", data=np.asarray(rp[0], dtype=np.float64)
            )
            action_group.create_group("right_refpose").create_dataset(
                "orientation", data=np.asarray(rp[1], dtype=np.float64)
            )
        self._write_effector_data(action_group, joint_position_state, action=True)
        # robot (chassis wheel velocities; ARX/Aloha use 4-wheel placeholder, others 2)
        robot_group = action_group.create_group("robot")
        if self.robot_type in ("ARX", "Aloha"):
            robot_group.create_dataset("velocity", data=np.zeros(4, dtype=np.float32))
        else:
            # Fixed chassis: zero XYZ linear and angular velocities.
            robot_group.create_dataset("velocity", data=np.zeros(6, dtype=np.int64))
        # waist (ARX/Aloha have none; emit a single zero placeholder so the
        # flat layout still has slot [28] = waist position)
        waist_group = action_group.create_group("waist")
        if self.robot_type in ("ARX", "Aloha"):
            waist_group.attrs["name"] = ["waist_placeholder"]
            waist_group.create_dataset("position", data=np.zeros(1, dtype=np.float32))
        else:
            waist_index = [self.config["joint_state_order"].index(name) for name in self.config["waist_joint_names"]]
            waist_group.attrs["name"] = self.config["waist_joint_names"]
            waist_group.create_dataset("position", data=joint_position[waist_index])

    def _write_effector_data(self, group, positions, action):
        for side_idx, side in enumerate(("left", "right")):
            names = self.config[f"{side}_effector_joint_names"]
            indices = [self.config["joint_state_order"].index(name) for name in names]
            values = np.asarray(positions[indices], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite {side} effector positions")
            if not self._is_wuji:
                if self._is_g2_90d:
                    remap = g2_90d_action_to_rad if action else g2_90d_state_to_rad
                else:
                    remap = omnipicker_action_to_rad if action else omnipicker_state_to_rad
                values = np.asarray([remap(values[0])], dtype=np.float64)
            effector = group.create_group(f"{side}_effector")
            effector.attrs["name"] = names
            effector.create_dataset("position", data=values)

    def _urdf_path(self):
        """Use the robot model already shipped with data collection for link3 FK."""
        robot_dir = Path(__file__).resolve().parent / "config" / "robot_cfg"
        if self._is_wuji:
            return str(robot_dir / "G2/G2_crsB_wuji.urdf")
        if "G2" in str(self.robot_type):
            name = (
                "G2_crsB_90D_from_usd.urdf"
                if getattr(self, "_arm_model", "crs") == "crsB"
                else "G2_omnipicker_from_usd.urdf"
            )
            return str(robot_dir / "G2" / name)
        return None  # G1 recordings historically use identity link3 refpose.

    def _effector_block(self, frame, arm_pos, arm_quat, side, side_idx):
        """Return effector-centre pose in base_link coordinates.

        Prefer ee_center from state.json; otherwise use the gripper-specific
        arm-end offset to support older recordings.
        """
        try:
            return effector_in_base(frame["robot"]["pose"], frame["ee_center"][side]["pose"])
        except (KeyError, TypeError):
            log.warning("frame missing ee_center[%s]; fallback to USD gripper-center offset", side)
            model = self._gripper_model(side_idx)
            return effector_from_arm_end(arm_pos, arm_quat, model)

    def _backfill_derived_fields(self, h5_path):
        """Fill missing end poses and link3 reference quaternions using recorded poses and FK."""
        urdf = self._urdf_path()
        arm_names = self.config["arm_joint_names"]
        n_pose, n_refpose = 0, 0
        with h5py.File(h5_path, "r+") as f:
            frame_keys = sorted([k for k in f.keys() if str(k).isdigit()], key=int)
            for fk in frame_keys:
                idx = int(fk)
                g = f[fk]
                # Rebuild any invalid 28-value pose.
                pose_ok = "state/end/pose" in g and tuple(g["state/end/pose"].shape) == (28,)
                if not pose_ok:
                    arm_pos = g["state/end/position"][()]
                    arm_ori = g["state/end/orientation"][()]
                    frame = self._state["frames"][idx]
                    eff_l_p, eff_l_q = self._effector_block(frame, arm_pos[0], arm_ori[0], "left", 0)
                    eff_r_p, eff_r_q = self._effector_block(frame, arm_pos[1], arm_ori[1], "right", 1)
                    pose = assemble_end_pose(
                        arm_pos,
                        arm_ori,
                        np.stack([eff_l_p, eff_r_p]),
                        np.stack([eff_l_q, eff_r_q]),
                    )
                    if "state/end/pose" in g:
                        del g["state/end/pose"]
                    g.create_dataset("state/end/pose", data=pose)
                    n_pose += 1
                # Compute reference pose only when missing.
                if "action/left_refpose/orientation" not in g or "action/right_refpose/orientation" not in g:
                    aj = np.asarray(g["action/joint/position"][()], dtype=np.float64)
                    ang = dict(zip(arm_names, aj))
                    for side in ("left", "right"):
                        ref = g.require_group(f"action/{side}_refpose")
                        if "orientation" not in ref:
                            ref.create_dataset("orientation", data=link3_orientation(ang, side, urdf))
                    n_refpose += 1
        log.info("derive backfill: pose=%d frames, refpose=%d frames", n_pose, n_refpose)

    def process_state_data(self, f_in, state_group, index):
        prefix = "state"
        joint_position = f_in[f"{prefix}/joint/position"][index]
        joint_effort = f_in[f"{prefix}/joint/effort"][index]
        joint_velocity = f_in[f"{prefix}/joint/velocity"][index]
        end_group = state_group.create_group("end")
        position = f_in[f"{prefix}/end/position"][index]  # Two arm-end positions in base_link.
        orientation = f_in[f"{prefix}/end/orientation"][index]  # [2,4] xyzw
        end_group.create_dataset("orientation", data=orientation)
        end_group.create_dataset("position", data=position)
        end_group.create_dataset("arm_orientation", data=f_in[f"{prefix}/end/arm_orientation"][index])
        end_group.create_dataset("arm_position", data=f_in[f"{prefix}/end/arm_position"][index])
        # G2/G1 use the 28-value arm-plus-effector layout. ARX/Aloha retain
        # their 14-value arm pose in arm_base_link coordinates.
        if "ARX" in self.robot_type or "Aloha" in self.robot_type:
            ap = f_in[f"{prefix}/end/arm_position"][index]
            ao = f_in[f"{prefix}/end/arm_orientation"][index]
            end_group.create_dataset("pose", data=np.concatenate([ap[0], ao[0], ap[1], ao[1]], axis=-1))
        else:
            # Assemble G2/G1 pose only when raw effector fields are available;
            # post-processing fills the missing values.
            end_pose = _build_pose_from_raw_frame(f_in, index)
            if end_pose is not None:
                end_group.create_dataset("pose", data=end_pose)
        end_group.create_dataset("errmsg", data=np.zeros(1, dtype=np.int32))
        end_group.create_dataset("errcode", data=np.zeros(1, dtype=np.int32))
        end_group.create_dataset("mode", data=np.array([ControlMode.CSP], dtype=np.uint32))
        # Simulation lacks end velocities and forces; use 24 zero placeholders.
        end_group.create_dataset("velocity", data=np.zeros(24, dtype=np.float64))
        end_group.create_dataset("wrench", data=np.zeros(24, dtype=np.float64))
        # Simulation has no operator event source; set action_src_status to 1.
        operator_event = state_group.create_group("operator_event")
        operator_event.create_dataset("action_src_status", data=np.array([1], dtype=np.int32))
        head_index = [self.config["joint_state_order"].index(name) for name in self.config["head_joint_names"]]
        head_mode = np.full(len(head_index), ControlMode.CSP, dtype=np.uint32)
        head_group = state_group.create_group("head")
        head_group.attrs["name"] = self.config["head_joint_names"]
        head_group.create_dataset("effort", data=joint_effort[head_index])
        head_group.create_dataset("mode", data=head_mode)
        head_group.create_dataset("position", data=joint_position[head_index])
        head_group.create_dataset("velocity", data=joint_velocity[head_index])
        joint_index = [self.config["joint_state_order"].index(name) for name in self.config["arm_joint_names"]]
        joint_mode = np.full(len(joint_index), ControlMode.CSP, dtype=np.uint32)
        joint_group = state_group.create_group("joint")
        joint_group.attrs["name"] = self.config["arm_joint_names"]
        joint_group.create_dataset("effort", data=joint_effort[joint_index])
        joint_group.create_dataset("mode", data=joint_mode)
        joint_group.create_dataset("position", data=joint_position[joint_index])
        joint_group.create_dataset("velocity", data=joint_velocity[joint_index])
        self.joint_state.append(joint_position[joint_index])
        self._write_effector_data(state_group, joint_position, action=False)
        robot_group = state_group.create_group("robot")
        robot_group.create_dataset("orientation", data=f_in[f"{prefix}/robot/orientation"][index])
        robot_group.create_dataset("position", data=f_in[f"{prefix}/robot/position"][index])
        waist_group = state_group.create_group("waist")
        if self.robot_type in ("ARX", "Aloha"):
            waist_group.attrs["name"] = ["waist_placeholder"]
            waist_group.create_dataset("effort", data=np.zeros(1, dtype=np.float32))
            waist_group.create_dataset("mode", data=np.array([ControlMode.CSP], dtype=np.uint32))
            waist_group.create_dataset("position", data=np.zeros(1, dtype=np.float32))
            waist_group.create_dataset("velocity", data=np.zeros(1, dtype=np.float32))
        else:
            waist_index = [self.config["joint_state_order"].index(name) for name in self.config["waist_joint_names"]]
            waist_mode = np.full(len(waist_index), ControlMode.CSP, dtype=np.uint32)
            waist_group.attrs["name"] = self.config["waist_joint_names"]
            waist_group.create_dataset("effort", data=joint_effort[waist_index])
            waist_group.create_dataset("mode", data=waist_mode)
            waist_group.create_dataset("position", data=joint_position[waist_index])
            waist_group.create_dataset("velocity", data=joint_velocity[waist_index])
        # Simulation has no VR controller input; fill button fields with zero.
        # Axes and triggers use float64; digital presses use int64.
        for side in ("left", "right"):
            vr_group = state_group.create_group(f"{side}_vr_button")
            for field in ("axis_x", "axis_y", "hand_trig", "index_trig"):
                vr_group.create_dataset(field, data=np.zeros(1, dtype=np.float64))
            for field in ("axis_click", "key_one", "key_two"):
                vr_group.create_dataset(field, data=np.zeros(1, dtype=np.int64))

    def process_timestamp_data(self, time, group):
        camera_group = group.create_group("camera")
        camera_group.create_dataset("hand_left_color", data=np.array([time]))
        camera_group.create_dataset("hand_right_color", data=np.array([time]))
        camera_group.create_dataset("head_color", data=np.array([time]))
        camera_group.create_dataset("head_depth", data=np.array([time]))
        camera_group.create_dataset("head_stereo_left", data=np.array([time]))
        camera_group.create_dataset("head_stereo_right", data=np.array([time]))

    def h5_post_process(self):
        from geniesim_teleop.data_recording.robot_fk import calculate_ee_pose_batch

        h5_file_path = os.path.join(self._output_path, "aligned_joints.h5")
        if "ARX" in self.robot_type or "Aloha" in self.robot_type:
            # ARX/Aloha end poses come directly from the recorded
            # arm_end_link transform relative to arm_base_link.
            return
        if "G2" in self.robot_type:
            # crsB requires its own URDF rather than the legacy G2 model.
            robot_cfg = "G2_crsB_wuji" if self._is_wuji else ("G2_crsB" if self._arm_model == "crsB" else "G2")
        else:
            robot_cfg = "G1"

        with h5py.File(h5_file_path, "r+") as f:
            frame_keys = sorted([k for k in f.keys() if k.isdigit()], key=int)
            if not frame_keys:
                return

            verify_key = frame_keys[0]
            need_rewrite = False

            def _quat_allclose(a, b, atol_val=1e-3):
                return np.allclose(a, b, atol=atol_val) or np.allclose(a, -b, atol=atol_val)

            state_joint = f[f"{verify_key}/state/joint/position"][:]
            computed_arm_pos, computed_arm_ori = calculate_ee_pose_batch(state_joint[None, :], robot_cfg)
            stored_arm_pos = f[f"{verify_key}/state/end/arm_position"][:]
            stored_arm_ori = f[f"{verify_key}/state/end/arm_orientation"][:]
            computed_arm_pos = computed_arm_pos[0]
            computed_arm_ori = computed_arm_ori[0]
            if not np.allclose(computed_arm_pos, stored_arm_pos, atol=1e-3):
                need_rewrite = True
            for side_idx in range(2):
                if not _quat_allclose(computed_arm_ori[side_idx], stored_arm_ori[side_idx]):
                    need_rewrite = True

            action_joint = f[f"{verify_key}/action/joint/position"][:]
            computed_act_pos, computed_act_ori = calculate_ee_pose_batch(action_joint[None, :], robot_cfg)
            stored_act_pos = f[f"{verify_key}/action/end/position"][:]
            stored_act_ori = f[f"{verify_key}/action/end/orientation"][:]
            computed_act_pos = computed_act_pos[0]
            computed_act_ori = computed_act_ori[0]
            if not np.allclose(computed_act_pos, stored_act_pos, atol=1e-3):
                need_rewrite = True
            for side_idx in range(2):
                if not _quat_allclose(computed_act_ori[side_idx], stored_act_ori[side_idx]):
                    need_rewrite = True

            if not need_rewrite:
                return

            n = len(frame_keys)
            state_joints_all = np.empty((n, 14), dtype=np.float32)
            action_joints_all = np.empty((n, 14), dtype=np.float32)
            for i, fk in enumerate(frame_keys):
                state_joints_all[i] = f[f"{fk}/state/joint/position"][:]
                action_joints_all[i] = f[f"{fk}/action/joint/position"][:]

            s_arm_pos, s_arm_ori = calculate_ee_pose_batch(state_joints_all, robot_cfg)
            a_arm_pos, a_arm_ori = calculate_ee_pose_batch(action_joints_all, robot_cfg)

            for i, fk in enumerate(frame_keys):
                f[f"{fk}/state/end/arm_position"][...] = s_arm_pos[i]
                f[f"{fk}/state/end/arm_orientation"][...] = s_arm_ori[i]
                f[f"{fk}/action/end/position"][...] = a_arm_pos[i]
                f[f"{fk}/action/end/orientation"][...] = a_arm_ori[i]

    def gen_review(self):
        review_path = Path(self._output_path) / "review"
        review_path.mkdir(parents=True, exist_ok=True)
        self.gen_review_meta_json(review_path)
        self.gen_review_state_json(review_path)

    def gen_review_meta_json(self, review_path):
        h5_path = os.path.join(self._output_path, "aligned_joints.h5")
        if not os.path.exists(h5_path):
            log.error(f"H5 file does not exist: {h5_path}")
            return

        with h5py.File(h5_path, mode="r") as oh5:
            root_keys = list(oh5.keys())
            if root_keys and all(k.isdigit() for k in root_keys):
                total_frames = len(root_keys)
            else:
                total_frames = oh5["position"].shape[0]

        meta = {"min": 0, "max": total_frames - 1, "groups": REVIEW_FRAME_GROUPS}
        meta_path = Path(review_path) / "meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        log.info(f"Generated meta.json: {total_frames} frames -> {meta_path}")

    def gen_review_state_json(self, review_path):
        h5_path = os.path.join(self._output_path, "aligned_joints.h5")
        meta_info_path = os.path.join(self._output_path, "meta_info.json")
        state_path = Path(review_path) / "state.json"

        if not os.path.exists(h5_path):
            log.error(f"H5 file does not exist: {h5_path}")
            return

        effector_type = ""
        left_effector_type = ""
        right_effector_type = ""
        if os.path.exists(meta_info_path):
            with open(meta_info_path, "r", encoding="utf-8") as f:
                meta_info = json.load(f)
                effector_type = meta_info.get("ee_type", "")
                for ee in meta_info.get("ee_list", []):
                    if "left" in ee.get("name", ""):
                        left_effector_type = ee.get("type", "")
                    if "right" in ee.get("name", ""):
                        right_effector_type = ee.get("type", "")
        else:
            log.warning("meta_info.json missing; using default configuration")

        visual_group_config = list(REVIEW_VISUAL_GROUP_CONFIG)

        with h5py.File(h5_path, mode="r") as oh5:
            root_keys = list(oh5.keys())
            is_new_fmt = root_keys and all(k.isdigit() for k in root_keys)
            log.info(f"H5 format: {'new' if is_new_fmt else 'old'}")

            all_group = _get_all_h5_groups(h5_path)

            if is_new_fmt:
                frame_keys = sorted([k for k in oh5.keys() if k.isdigit()], key=int)
                has_waist = "state/waist" in oh5[frame_keys[0]] if frame_keys else False
            else:
                has_waist = "state/waist" in all_group

            if "state/effector" in all_group or (
                is_new_fmt and any(f"{k}/state/effector" in all_group for k in oh5.keys() if k.isdigit())
            ):
                visual_groups = [
                    "state/joint/position",
                    "state/effector/position",
                ]
            else:
                visual_groups = [
                    "state/joint/position",
                    "state/left_effector/position",
                    "state/right_effector/position",
                ]

            if has_waist:
                visual_group_config.insert(0, "state/waist/position")
                log.info("Found waist data")

            ee_types = {
                "state/effector/position": effector_type,
                "state/left_effector/position": left_effector_type,
                "state/right_effector/position": right_effector_type,
            }

            if is_new_fmt:
                joint_names, actions = self._process_new_h5_review(oh5, visual_groups, visual_group_config, ee_types)
            else:
                joint_names, actions = self._process_old_h5_review(oh5, visual_groups, visual_group_config, ee_types)

        with state_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"view_config": "", "joint_names": joint_names, "action": actions},
                f,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        log.info(f"Generated state.json: {len(joint_names)} joints, " f"{len(actions)} frames -> {state_path}")

    @staticmethod
    def _process_old_h5_review(oh5, visual_groups, visual_group_config, ee_types):
        joint_names: list = []
        actions: list = []

        for path in visual_groups:
            ee_type = ee_types.get(path, "")
            res, attrs, data = _get_h5_metadata(oh5, path)
            if res:
                cal_rule = REVIEW_GRIPPER_CAL_MAPPING.get(ee_type)
                data = _convert_gripper_data(data, cal_rule)
                width = len(data[0]) if data and isinstance(data[0], list) else 1
                joint_names.extend(_names_for_visual_path(path, attrs, width))
                actions = _combine_arrays(actions, data)

        if not joint_names:
            joint_names.extend(REVIEW_DEFAULT_JOINT_NAMES["joint"])
            joint_names.extend(REVIEW_DEFAULT_JOINT_NAMES["effector"])

        for path in visual_group_config:
            res, attrs, data = _get_h5_metadata(oh5, path)
            if res:
                width = len(data[0]) if data and isinstance(data[0], list) else 1
                joint_names.extend(_names_for_visual_path(path, attrs, width))
                actions = _combine_arrays(actions, data)

        return joint_names, actions

    @staticmethod
    def _process_new_h5_review(oh5, visual_groups, visual_group_config, ee_types):
        joint_names: list = []
        frame_keys = sorted([k for k in oh5.keys() if k.isdigit()], key=int)
        if not frame_keys:
            return [], []

        all_visual_data: list = []
        all_config_data: list = []

        for path in visual_groups:
            ee_type = ee_types.get(path, "")
            path_data = []
            for fk in frame_keys:
                res, attrs, data = _get_h5_metadata(oh5, f"{fk}/{path}")
                if res:
                    cal_rule = REVIEW_GRIPPER_CAL_MAPPING.get(ee_type)
                    data = _convert_gripper_data(data, cal_rule)
                    row = data[0] if data and isinstance(data[0], list) else data
                    if fk == frame_keys[0]:
                        width = len(row) if isinstance(row, list) else 1
                        joint_names.extend(_names_for_visual_path(path, attrs, width))
                    path_data.append(row)
                else:
                    path_data.append([])
            if path_data and any(path_data):
                all_visual_data.append(path_data)

        if not joint_names:
            joint_names.extend(REVIEW_DEFAULT_JOINT_NAMES["joint"])
            joint_names.extend(REVIEW_DEFAULT_JOINT_NAMES["effector"])

        for path in visual_group_config:
            path_data = []
            for fk in frame_keys:
                res, attrs, data = _get_h5_metadata(oh5, f"{fk}/{path}")
                if res:
                    row = data[0] if data and isinstance(data[0], list) else data
                    if fk == frame_keys[0]:
                        width = len(row) if isinstance(row, list) else 1
                        joint_names.extend(_names_for_visual_path(path, attrs, width))
                    path_data.append(row)
                else:
                    path_data.append([])
            if path_data and any(path_data):
                all_config_data.append(path_data)

        all_sources = all_visual_data + all_config_data
        if not all_sources:
            return joint_names, []

        actions = []
        for fi in range(len(frame_keys)):
            row = []
            for src in all_sources:
                if fi < len(src):
                    v = src[fi]
                    row.extend(v) if isinstance(v, list) else row.append(v)
            if row:
                actions.append(row)

        return joint_names, actions
