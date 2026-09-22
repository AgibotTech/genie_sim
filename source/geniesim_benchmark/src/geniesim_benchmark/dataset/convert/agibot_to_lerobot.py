# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Convert agibot episode data to LeRobot v2.1 format.

Public API
----------
- :func:`convert_agibot_to_lerobot` — programmatic entry; takes paths,
  returns a manifest dict.
- :func:`convert_cli` — argv wrapper used by
  ``geniesim dataset convert agibot-to-lerobot``.

The converter auto-detects single-episode vs batch from the input
layout: if ``agibot_dir/aligned_joints.h5`` exists, it's treated as a
single episode; otherwise the directory is scanned for episode
subdirectories.

History
-------
This module was previously ``scripts/convert_agibot_to_lerobot.py`` at
the repo root. Schema constants, dimension offsets, ffmpeg encoder
flags, and the parquet writer settings are preserved verbatim — only
the entry-point shape changed (CLI lives in ``commands/dataset.py``,
the Python API is now importable).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ============================================================
# LeRobot v2.1 parquet schema constants
# ============================================================

ARROW_SCHEMA_HUGGINGFACE_METADATA = {
    "info": {
        "features": {
            "observation.state": {
                "feature": {"dtype": "float32", "_type": "Value"},
                "length": 159,
                "_type": "Sequence",
            },
            "action": {
                "feature": {"dtype": "float32", "_type": "Value"},
                "length": 40,
                "_type": "Sequence",
            },
            "episode_index": {"dtype": "int64", "_type": "Value"},
            "frame_index": {"dtype": "int64", "_type": "Value"},
            "index": {"dtype": "int64", "_type": "Value"},
            "task_index": {"dtype": "int64", "_type": "Value"},
            "timestamp": {"dtype": "float32", "_type": "Value"},
        }
    }
}


# ============================================================
# Data mapping constants (from lerobot meta/info.json)
# ============================================================

# State vector: 159 dimensions
STATE_LEFT_EFFECTOR_POS = 0  # 1 dim
STATE_RIGHT_EFFECTOR_POS = 1  # 1 dim
STATE_END_POSITION = 2  # 6 dims (2 arms x 3)
STATE_END_ORIENTATION = 8  # 8 dims (2 arms x 4)
STATE_END_ARM_ORIENTATION = 16  # 8 dims (2 arms x 4)
STATE_END_ARM_POSITION = 24  # 6 dims (2 arms x 3)
STATE_JOINT_POSITION = 30  # 14 dims
STATE_JOINT_EFFORT = 44  # 14 dims
STATE_JOINT_VELOCITY = 58  # 14 dims
STATE_HEAD_POSITION = 72  # 3 dims
STATE_WAIST_POSITION = 75  # 5 dims
STATE_ROBOT_POSITION = 80  # 3 dims
STATE_ROBOT_ORIENTATION = 83  # 4 dims

# Action vector: 40 dimensions
ACTION_LEFT_EFFECTOR = 0  # 1 dim
ACTION_RIGHT_EFFECTOR = 1  # 1 dim
ACTION_END_POSITION = 2  # 6 dims
ACTION_END_ORIENTATION = 8  # 8 dims
ACTION_JOINT_POSITION = 16  # 14 dims
ACTION_HEAD_POSITION = 30  # 3 dims
ACTION_WAIST_POSITION = 33  # 5 dims
ACTION_ROBOT_VELOCITY = 38  # 2 dims


# ============================================================
# Pre-flight checks
# ============================================================


def _require_ffmpeg() -> None:
    """Verify ffmpeg is on PATH; raise a clear error if not."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is not on PATH. The converter shells out to ffmpeg to "
            "encode RGB (HEVC) and depth (PNG) videos. Install it first:\n"
            "    Ubuntu / Debian:  sudo apt install ffmpeg\n"
            "    macOS (Homebrew): brew install ffmpeg"
        )


def _require_heavy_deps():
    """Lazy import the heavy deps so import-time cost stays low for callers
    that just want to ``from geniesim_benchmark.dataset.convert import …``.
    Returns (h5py, np, pa, pq)."""
    try:
        import h5py
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "agibot→LeRobot conversion requires h5py + numpy + pyarrow. "
            f"Install with: pip install h5py numpy pyarrow\n"
            f"Underlying error: {exc}"
        ) from exc
    return h5py, np, pa, pq


# ============================================================
# Core data building functions
# ============================================================


def build_state(state_dict: dict, joint_all_dict: dict, extrinsic_data: dict, frame_idx: int = 0):
    """Build 159-dim state vector from agibot state data."""
    import numpy as np
    import h5py  # noqa: F401  (needed for downstream type guards)

    state = np.zeros(159, dtype=np.float32)

    state[STATE_LEFT_EFFECTOR_POS] = state_dict["left_effector"]["position"][0]
    state[STATE_RIGHT_EFFECTOR_POS] = state_dict["right_effector"]["position"][0]

    ep = state_dict["end"]["position"]
    eo = state_dict["end"]["orientation"]
    eao = state_dict["end"]["arm_orientation"]
    eap = state_dict["end"]["arm_position"]
    state[STATE_END_POSITION : STATE_END_POSITION + 6] = ep.flatten()
    state[STATE_END_ORIENTATION : STATE_END_ORIENTATION + 8] = eo.flatten()
    state[STATE_END_ARM_ORIENTATION : STATE_END_ARM_ORIENTATION + 8] = eao.flatten()
    state[STATE_END_ARM_POSITION : STATE_END_ARM_POSITION + 6] = eap.flatten()

    state[STATE_JOINT_POSITION : STATE_JOINT_POSITION + 14] = joint_all_dict["joint"]["position"][:14]
    state[STATE_JOINT_EFFORT : STATE_JOINT_EFFORT + 14] = joint_all_dict["joint"]["effort"][:14]
    state[STATE_JOINT_VELOCITY : STATE_JOINT_VELOCITY + 14] = joint_all_dict["joint"]["velocity"][:14]

    state[STATE_HEAD_POSITION : STATE_HEAD_POSITION + 3] = state_dict["head"]["position"]
    state[STATE_WAIST_POSITION : STATE_WAIST_POSITION + 5] = state_dict["waist"]["position"]
    state[STATE_ROBOT_POSITION : STATE_ROBOT_POSITION + 3] = state_dict["robot"]["position"]
    state[STATE_ROBOT_ORIENTATION : STATE_ROBOT_ORIENTATION + 4] = state_dict["robot"]["orientation"]

    # Extrinsics
    if "hand_left_rgbd" in extrinsic_data:
        entry = extrinsic_data["hand_left_rgbd"][frame_idx]
        state[87:96] = np.array(entry["rotation"], dtype=np.float32).flatten()
        state[141:144] = np.array(entry["translation"], dtype=np.float32)

    if "hand_right_rgbd" in extrinsic_data:
        entry = extrinsic_data["hand_right_rgbd"][frame_idx]
        state[96:105] = np.array(entry["rotation"], dtype=np.float32).flatten()
        state[144:147] = np.array(entry["translation"], dtype=np.float32)

    if "head_front_rgbd" in extrinsic_data:
        entry = extrinsic_data["head_front_rgbd"][frame_idx]
        state[123:132] = np.array(entry["rotation"], dtype=np.float32).flatten()
        state[153:156] = np.array(entry["translation"], dtype=np.float32)

    return state


def build_action(action_dict: dict):
    """Build 40-dim action vector from agibot action data."""
    import numpy as np

    action = np.zeros(40, dtype=np.float32)

    action[ACTION_LEFT_EFFECTOR] = action_dict["left_effector"]["position"][0]
    action[ACTION_RIGHT_EFFECTOR] = action_dict["right_effector"]["position"][0]
    action[ACTION_END_POSITION : ACTION_END_POSITION + 6] = action_dict["end"]["position"].flatten()
    action[ACTION_END_ORIENTATION : ACTION_END_ORIENTATION + 8] = action_dict["end"]["orientation"].flatten()
    action[ACTION_JOINT_POSITION : ACTION_JOINT_POSITION + 14] = action_dict["joint"]["position"]
    action[ACTION_HEAD_POSITION : ACTION_HEAD_POSITION + 3] = action_dict["head"]["position"]
    action[ACTION_WAIST_POSITION : ACTION_WAIST_POSITION + 5] = action_dict["waist"]["position"]
    action[ACTION_ROBOT_VELOCITY : ACTION_ROBOT_VELOCITY + 2] = action_dict["robot"]["velocity"]

    return action


def read_h5_frame(h5_file, frame_id: str) -> tuple:
    """Read action and state from a single frame in aligned_joints.h5."""
    import h5py
    import numpy as np

    frame = h5_file[frame_id]

    def read_group(grp):
        result = {}
        for k, item in grp.items():
            if isinstance(item, h5py.Dataset):
                val = item[()]
                if isinstance(val, np.ndarray) and val.ndim == 0:
                    result[k] = val.item()
                elif isinstance(val, np.ndarray):
                    result[k] = val
                else:
                    result[k] = val
            elif isinstance(item, h5py.Group):
                result[k] = read_group(item)
        return result

    return read_group(frame["action"]), read_group(frame["state"]), frame["main_timestamp"][()]


# ============================================================
# File loading utilities
# ============================================================


def load_extrinsics(agibot_dir: Path) -> dict:
    """Load extrinsic calibration files. Returns dict mapping key → list of per-frame data."""
    extrinsics: dict = {}
    extrinsic_files = {
        "hand_left_rgbd": "extrinsic_end_T_hand_left_rgbd_aligned.json",
        "hand_right_rgbd": "extrinsic_end_T_hand_right_rgbd_aligned.json",
        "head_front_rgbd": "extrinsic_end_T_head_front_rgbd_aligned.json",
    }
    sensor_dir = agibot_dir / "parameters" / "sensor"
    for key, filename in extrinsic_files.items():
        filepath = sensor_dir / filename
        if filepath.exists():
            with filepath.open() as f:
                extrinsics[key] = json.load(f)
    return extrinsics


def load_lerobot_reference(lerobot_dir: Path):
    """Load reference state vector from existing lerobot dataset.

    Returns None if the reference parquet isn't present — callers
    silently skip the extrinsic-fill step in that case.
    """
    import numpy as np
    import pyarrow.parquet as pq

    parquet_path = lerobot_dir / "data" / "chunk-000" / "episode_000000.parquet"
    if parquet_path.exists():
        table = pq.read_table(parquet_path)
        return np.array(table.column("observation.state")[0].as_py(), dtype=np.float32)
    return None


def fill_extrinsics_from_lerobot(extrinsic_data: dict, template) -> dict:
    """Fill missing extrinsic fields using lerobot reference data."""
    if template is None:
        return extrinsic_data

    keys_to_fill = {
        "head_left_fisheye": (105, 114, 147, 150),
        "head_right_fisheye": (114, 123, 150, 153),
        "head_back_fisheye": (132, 141, 156, 159),
    }
    for key, (rs, re, ts, te) in keys_to_fill.items():
        if key not in extrinsic_data:
            rot = template[rs:re].reshape(3, 3).tolist()
            trans = template[ts:te].tolist()
            extrinsic_data[key] = [{"rotation": rot, "translation": trans}] * template.shape[0]

    return extrinsic_data


def encode_videos(agibot_dir: Path, output_dir: Path, episode_index: int, fps: float = 30.0):
    """Encode camera images into MP4 videos using ffmpeg.

    RGB videos:   HEVC (libx265), yuv420p, keyframe every 8 frames, CRF 24
    Depth videos: lossless PNG codec, gray16le
    RGB output:   videos/{chunk}/observation.images.{video_key}/episode_{index:06d}.mp4
    Depth output: videos/{chunk}/{video_key}_depth/episode_{index:06d}.mp4
    """
    camera_dir = agibot_dir / "camera"
    chunk_name = f"chunk-{episode_index // 1000:03d}"

    # video_key -> [(image_stem, ext, is_depth), ...]
    camera_map = {
        "top_head": [("head_color", "jpg", False), ("head_depth", "png", True)],
        "hand_left": [("hand_left_color", "jpg", False), ("hand_left_depth", "png", True)],
        "hand_right": [("hand_right_color", "jpg", False), ("hand_right_depth", "png", True)],
    }

    for video_key, image_list in camera_map.items():
        for image_stem, ext, is_depth in image_list:
            # LeRobot formats the video_path template with the full feature
            # key from info.json, so RGB directories must use that same key.
            # Depth streams are not declared as features and keep their
            # existing auxiliary directory names.
            vid_dir = video_key + "_depth" if is_depth else f"observation.images.{video_key}"
            video_path = output_dir / "videos" / chunk_name / vid_dir / f"episode_{episode_index:06d}.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            input_pattern = str(camera_dir / f"%d/{image_stem}.{ext}")

            if is_depth:
                result = subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-framerate",
                        str(int(fps)),
                        "-i",
                        input_pattern,
                        "-c:v",
                        "png",
                        "-pix_fmt",
                        "gray16le",
                        "-r",
                        str(int(fps)),
                        "-movflags",
                        "+faststart",
                        str(video_path),
                    ],
                    capture_output=True,
                    text=True,
                )
            else:
                result = subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "image2",
                        "-threads",
                        "4",
                        "-r",
                        str(int(fps)),
                        "-i",
                        input_pattern,
                        "-vcodec",
                        "libx265",
                        "-pix_fmt",
                        "yuv420p",
                        "-keyint_min",
                        "8",
                        "-sc_threshold",
                        "0",
                        "-vf",
                        f"setpts=N/({int(fps)}*TB)",
                        "-bf",
                        "0",
                        "-crf",
                        "24",
                        "-g",
                        "8",
                        str(video_path),
                    ],
                    capture_output=True,
                    text=True,
                )

            if result.returncode != 0:
                print(f"  ERROR encoding {vid_dir}: {result.stderr[-500:]}")
            else:
                print(f"  Encoded {vid_dir}")


def detect_episodes(agibot_root: Path) -> list:
    """Auto-detect episode directories inside agibot_root."""
    entries = sorted(agibot_root.iterdir(), key=lambda p: p.name)
    episodes = []
    for p in entries:
        if p.is_dir() and (p / "aligned_joints.h5").exists():
            episodes.append(p)
    return episodes


# ============================================================
# Episode conversion
# ============================================================


def convert_episode(
    agibot_dir: Path, output_dir: Path, episode_index: int, lerobot_template, fps: float = 30.0
) -> dict:
    """Convert a single agibot episode. Returns dict with stats."""
    import h5py
    import pyarrow as pa
    import pyarrow.parquet as pq

    print(f"\n{'='*60}")
    print(f"Converting episode {episode_index}: {agibot_dir}")

    extrinsic_data = load_extrinsics(agibot_dir)
    extrinsic_data = fill_extrinsics_from_lerobot(extrinsic_data, lerobot_template)

    h5_path = agibot_dir / "aligned_joints.h5"
    h5_all_path = agibot_dir / "aligned_joints_all.h5"

    camera_dir = agibot_dir / "camera"
    frame_ids = sorted(
        [d.name for d in camera_dir.iterdir() if d.is_dir()],
        key=lambda x: int(x),
    )

    with h5py.File(str(h5_path), "r") as f_h5, h5py.File(str(h5_all_path), "r") as f_h5_all:
        joint_pos_all = f_h5_all["state"]["joint"]["position"][:]
        joint_eff_all = f_h5_all["state"]["joint"]["effort"][:]
        joint_vel_all = f_h5_all["state"]["joint"]["velocity"][:]

        states, actions, timestamps = [], [], []
        for i, fid in enumerate(frame_ids):
            action_dict, state_dict, ts_ns = read_h5_frame(f_h5, fid)
            joint_all_dict = {
                "joint": {"position": joint_pos_all[i], "effort": joint_eff_all[i], "velocity": joint_vel_all[i]}
            }
            states.append(build_state(state_dict, joint_all_dict, extrinsic_data, frame_idx=i))
            actions.append(build_action(action_dict))
            timestamps.append(ts_ns)

    num_frames = len(frame_ids)

    # Use ideal timestamps: frame_index / fps (matches v2.1 reference format)
    ideal_timestamps = [i / fps for i in range(num_frames)]

    # Build parquet using fixed_size_list to match reference schema exactly
    # pa.list_(dtype, length) creates FixedSizeListType (matching reference parquet schema)
    table = pa.table(
        {
            "observation.state": pa.array(states, type=pa.list_(pa.float32(), 159)),
            "action": pa.array(actions, type=pa.list_(pa.float32(), 40)),
            "episode_index": pa.array([episode_index] * num_frames, type=pa.int64()),
            "frame_index": pa.array(list(range(num_frames)), type=pa.int64()),
            "index": pa.array(list(range(num_frames)), type=pa.int64()),
            "task_index": pa.array([0] * num_frames, type=pa.int64()),
            "timestamp": pa.array(ideal_timestamps, type=pa.float32()),
        }
    )

    # Add huggingface schema metadata (present in reference v2.1 parquet files)
    metadata = {b"huggingface": json.dumps(ARROW_SCHEMA_HUGGINGFACE_METADATA).encode()}
    table = table.replace_schema_metadata(metadata)

    chunk_name = f"chunk-{episode_index // 1000:03d}"
    parquet_dir = output_dir / "data" / chunk_name
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = parquet_dir / f"episode_{episode_index:06d}.parquet"

    # Write parquet with row_group_size=1000 to match LeRobot v2.1 format
    pq.write_table(table, str(parquet_path), row_group_size=1000)
    print(f"  Wrote {parquet_path} ({num_frames} frames)")

    # Encode videos
    encode_videos(agibot_dir, output_dir, episode_index, fps=fps)

    return {"episode_index": episode_index, "length": num_frames}


# ============================================================
# Meta file generation
# ============================================================


def load_camera_parameters(agibot_dir: Path) -> dict:
    """Load camera intrinsic/extrinsic parameters from agibot sensor directory."""
    sensor_dir = agibot_dir / "parameters" / "sensor"
    if not sensor_dir.exists():
        return {}

    cam_params: dict = {}
    for fpath in sensor_dir.iterdir():
        if not fpath.name.endswith(".json"):
            continue
        with fpath.open() as f:
            cam_params[fpath.name] = json.load(f)
    return cam_params


def generate_meta(output_dir: Path, episode_info: list, agibot_dirs: list):
    """Generate info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl."""
    import numpy as np
    import pyarrow.parquet as pq

    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # tasks.jsonl
    with (meta_dir / "tasks.jsonl").open("w") as f:
        f.write(json.dumps({"task_index": 0, "task": "Pop the popcorn"}, ensure_ascii=False) + "\n")

    # episodes.jsonl
    with (meta_dir / "episodes.jsonl").open("w") as f:
        for ep in episode_info:
            f.write(
                json.dumps(
                    {
                        "episode_index": ep["episode_index"],
                        "tasks": ["Pop the popcorn"],
                        "length": ep["length"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # episodes_stats.jsonl
    stats_episodes = []
    for ep in episode_info:
        ep_idx = ep["episode_index"]
        chunk_name = f"chunk-{ep_idx // 1000:03d}"
        parquet_path = output_dir / "data" / chunk_name / f"episode_{ep_idx:06d}.parquet"
        if parquet_path.exists():
            table = pq.read_table(str(parquet_path))
            rows = table.to_pylist()
            num_frames = len(rows)
            state_col = np.array([r["observation.state"] for r in rows], dtype=np.float32)
            action_col = np.array([r["action"] for r in rows], dtype=np.float32)
            frame_index_col = np.array([r["frame_index"] for r in rows], dtype=np.int64)
            index_col = np.array([r["index"] for r in rows], dtype=np.int64)
            timestamp_col = np.array([r["timestamp"] for r in rows], dtype=np.float32)

            # Video fields: per-frame stats as nested arrays (matches reference format)
            video_keys = ["top_head", "hand_left", "hand_right"]
            video_stats = {}
            for vk in video_keys:
                key = f"observation.images.{vk}"
                # For videos we don't have pixel data in parquet, use zeros as placeholder
                # (matching the reference which also has zeros for video stats)
                video_stats[key] = {
                    "min": [[[0.0]], [[0.0]], [[0.0]]],
                    "max": [[[0.0]], [[0.0]], [[0.0]]],
                    "mean": [[[0.0]], [[0.0]], [[0.0]]],
                    "std": [[[0.0]], [[0.0]], [[0.0]]],
                    "count": [0],
                }

            stats = {
                "episode_index": ep_idx,
                "stats": {
                    **video_stats,
                    "observation.state": {
                        "mean": state_col.mean(axis=0).tolist(),
                        "std": state_col.std(axis=0).tolist(),
                        "min": state_col.min(axis=0).tolist(),
                        "max": state_col.max(axis=0).tolist(),
                        "count": [num_frames],
                    },
                    "action": {
                        "mean": action_col.mean(axis=0).tolist(),
                        "std": action_col.std(axis=0).tolist(),
                        "min": action_col.min(axis=0).tolist(),
                        "max": action_col.max(axis=0).tolist(),
                        "count": [num_frames],
                    },
                    # Integer fields (episode_index, frame_index, index, task_index)
                    "episode_index": {
                        "min": [ep_idx],
                        "max": [ep_idx],
                        "mean": [float(ep_idx)],
                        "std": [0.0],
                        "count": [num_frames],
                    },
                    "frame_index": {
                        "min": [int(frame_index_col.min())],
                        "max": [int(frame_index_col.max())],
                        "mean": [float(frame_index_col.mean())],
                        "std": [float(frame_index_col.std())],
                        "count": [num_frames],
                    },
                    "index": {
                        "min": [int(index_col.min())],
                        "max": [int(index_col.max())],
                        "mean": [float(index_col.mean())],
                        "std": [float(index_col.std())],
                        "count": [num_frames],
                    },
                    "task_index": {
                        "min": [0],
                        "max": [0],
                        "mean": [0.0],
                        "std": [0.0],
                        "count": [num_frames],
                    },
                    "timestamp": {
                        "min": [float(timestamp_col.min())],
                        "max": [float(timestamp_col.max())],
                        "mean": [float(timestamp_col.mean())],
                        "std": [float(timestamp_col.std())],
                        "count": [num_frames],
                    },
                },
            }
            stats_episodes.append(stats)
        else:
            stats_episodes.append({})

    with (meta_dir / "episodes_stats.jsonl").open("w") as f:
        for s in stats_episodes:
            f.write(json.dumps(s) + "\n")

    # info.json
    total_frames = sum(ep["length"] for ep in episode_info)
    total_episodes = len(episode_info)
    total_chunks = max((ep["episode_index"] // 1000) for ep in episode_info) + 1

    # Build per-episode metadata from agibot dirs
    h5_path: dict = {}
    high_level_instruction: dict = {}
    instruction_segments: dict = {}
    camera_parameters: dict = {}
    intervention_info: dict = {}
    key_frame: dict = {}
    take_over: dict = {}

    for i, ep_dir in enumerate(agibot_dirs):
        h5_path[str(i)] = f"frame://{ep_dir}/aligned_joints.h5"
        high_level_instruction[str(i)] = {"high_level_instruction": ""}
        instruction_segments[str(i)] = []
        intervention_info[str(i)] = {}
        key_frame[str(i)] = {"single": [], "dual": []}
        take_over[str(i)] = []

        # Load camera parameters (intrinsics only for RGB cameras, matching reference)
        sensor_dir = ep_dir / "parameters" / "sensor"
        cam_params: dict = {}
        if sensor_dir.exists():
            # Only load intrinsic_rgb files (not depth), format as {name: content}
            for fpath in sensor_dir.iterdir():
                if fpath.name.startswith("intrinsic_") and fpath.name.endswith("_rgb.json"):
                    cam_name = fpath.name[:-5]  # Remove .json
                    with fpath.open() as f:
                        cam_params[cam_name] = json.load(f)
        camera_parameters[str(i)] = cam_params

    info = {
        "codebase_version": "v2.1",
        "robot_type": "g2a",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_episodes * 3,
        "total_chunks": total_chunks,
        "chunks_size": 1000,
        "fps": 30,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.top_head": {
                "dtype": "video",
                "video_info": {
                    "video.is_depth_map": False,
                    "video.fps": 30.0,
                    "video.codec": "hevc",
                    "video.pix_fmt": "yuv420p",
                    "has_audio": False,
                },
                "shape": [400, 640, 3],
                "names": ["height", "width", "channel"],
            },
            "observation.images.hand_left": {
                "dtype": "video",
                "video_info": {
                    "video.is_depth_map": False,
                    "video.fps": 30.0,
                    "video.codec": "hevc",
                    "video.pix_fmt": "yuv420p",
                    "has_audio": False,
                },
                "shape": [1056, 1280, 3],
                "names": ["height", "width", "channel"],
            },
            "observation.images.hand_right": {
                "dtype": "video",
                "video_info": {
                    "video.is_depth_map": False,
                    "video.fps": 30.0,
                    "video.codec": "hevc",
                    "video.pix_fmt": "yuv420p",
                    "has_audio": False,
                },
                "shape": [1056, 1280, 3],
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [159],
                "field_descriptions": _get_state_field_descriptions(),
            },
            "action": {
                "dtype": "float32",
                "shape": [40],
                "field_descriptions": _get_action_field_descriptions(),
            },
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        },
        "data_version": "v0.1.5",
        "aid_info": {},
        # LeRobot v2.1 additional fields
        "camera_parameters": camera_parameters,
        "h5_path": h5_path,
        "high_level_instruction": high_level_instruction,
        "instruction_segments": instruction_segments,
        "intervention_info": intervention_info,
        "key_frame": key_frame,
        "take_over": take_over,
    }

    with (meta_dir / "info.json").open("w") as f:
        json.dump(info, f, indent=2)

    print(f"\nGenerated meta files in {meta_dir}/")


def _get_state_field_descriptions():
    return {
        "state/left_effector/position": {"description": "", "dimensions": 1, "indices": [0]},
        "state/right_effector/position": {"description": "", "dimensions": 1, "indices": [1]},
        "state/end/wrench": {"description": "", "dimensions": 0, "indices": []},
        "state/end/position": {"description": "", "dimensions": 6, "indices": list(range(2, 8))},
        "state/end/velocity": {"description": "", "dimensions": 0, "indices": []},
        "state/end/orientation": {"description": "", "dimensions": 8, "indices": list(range(8, 16))},
        "state/end/arm_orientation": {"description": "", "dimensions": 8, "indices": list(range(16, 24))},
        "state/end/arm_position": {"description": "", "dimensions": 6, "indices": list(range(24, 30))},
        "state/joint/position": {"description": "", "dimensions": 14, "indices": list(range(30, 44))},
        "state/joint/current_value": {"description": "", "dimensions": 0, "indices": []},
        "state/joint/effort": {"description": "", "dimensions": 14, "indices": list(range(44, 58))},
        "state/joint/velocity": {"description": "", "dimensions": 14, "indices": list(range(58, 72))},
        "state/head/position": {"description": "", "dimensions": 3, "indices": list(range(72, 75))},
        "state/waist/position": {"description": "", "dimensions": 5, "indices": list(range(75, 80))},
        "state/robot/position": {"description": "", "dimensions": 3, "indices": list(range(80, 83))},
        "state/robot/orientation": {"description": "", "dimensions": 4, "indices": list(range(83, 87))},
        "state/operator_event/action_src_status": {"description": "", "dimensions": 0, "indices": []},
        "state/left_ee_force/controlled": {"description": "", "dimensions": 0, "indices": []},
        "state/left_ee_force/rows": {"description": "", "dimensions": 0, "indices": []},
        "state/left_ee_force/cols": {"description": "", "dimensions": 0, "indices": []},
        "state/left_ee_force/resolution_x": {"description": "", "dimensions": 0, "indices": []},
        "state/left_ee_force/resolution_y": {"description": "", "dimensions": 0, "indices": []},
        "state/left_ee_force/normal_force": {"description": "", "dimensions": 0, "indices": []},
        "state/left_ee_force/shear_force_x": {"description": "", "dimensions": 0, "indices": []},
        "state/left_ee_force/shear_force_y": {"description": "", "dimensions": 0, "indices": []},
        "state/left_ee_force/contact": {"description": "", "dimensions": 0, "indices": []},
        "state/left_ee_force/valid": {"description": "", "dimensions": 0, "indices": []},
        "state/left_ee_force/err_code": {"description": "", "dimensions": 0, "indices": []},
        "state/right_ee_force/controlled": {"description": "", "dimensions": 0, "indices": []},
        "state/right_ee_force/rows": {"description": "", "dimensions": 0, "indices": []},
        "state/right_ee_force/cols": {"description": "", "dimensions": 0, "indices": []},
        "state/right_ee_force/resolution_x": {"description": "", "dimensions": 0, "indices": []},
        "state/right_ee_force/resolution_y": {"description": "", "dimensions": 0, "indices": []},
        "state/right_ee_force/normal_force": {"description": "", "dimensions": 0, "indices": []},
        "state/right_ee_force/shear_force_x": {"description": "", "dimensions": 0, "indices": []},
        "state/right_ee_force/shear_force_y": {"description": "", "dimensions": 0, "indices": []},
        "state/right_ee_force/contact": {"description": "", "dimensions": 0, "indices": []},
        "state/right_ee_force/valid": {"description": "", "dimensions": 0, "indices": []},
        "state/right_ee_force/err_code": {"description": "", "dimensions": 0, "indices": []},
        "extrinsic_end_T_hand_left_rgbd_aligned/rotation_matrix": {
            "description": "",
            "dimensions": 9,
            "indices": list(range(87, 96)),
        },
        "extrinsic_end_T_hand_right_rgbd_aligned/rotation_matrix": {
            "description": "",
            "dimensions": 9,
            "indices": list(range(96, 105)),
        },
        "extrinsic_end_T_head_left_fisheye_aligned/rotation_matrix": {
            "description": "",
            "dimensions": 9,
            "indices": list(range(105, 114)),
        },
        "extrinsic_end_T_head_right_fisheye_aligned/rotation_matrix": {
            "description": "",
            "dimensions": 9,
            "indices": list(range(114, 123)),
        },
        "extrinsic_end_T_head_front_rgbd_aligned/rotation_matrix": {
            "description": "",
            "dimensions": 9,
            "indices": list(range(123, 132)),
        },
        "extrinsic_end_T_head_back_fisheye_aligned/rotation_matrix": {
            "description": "",
            "dimensions": 9,
            "indices": list(range(132, 141)),
        },
        "extrinsic_end_T_hand_left_rgbd_aligned/translation_vector": {
            "description": "",
            "dimensions": 3,
            "indices": list(range(141, 144)),
        },
        "extrinsic_end_T_hand_right_rgbd_aligned/translation_vector": {
            "description": "",
            "dimensions": 3,
            "indices": list(range(144, 147)),
        },
        "extrinsic_end_T_head_left_fisheye_aligned/translation_vector": {
            "description": "",
            "dimensions": 3,
            "indices": list(range(147, 150)),
        },
        "extrinsic_end_T_head_right_fisheye_aligned/translation_vector": {
            "description": "",
            "dimensions": 3,
            "indices": list(range(150, 153)),
        },
        "extrinsic_end_T_head_front_rgbd_aligned/translation_vector": {
            "description": "",
            "dimensions": 3,
            "indices": list(range(153, 156)),
        },
        "extrinsic_end_T_head_back_fisheye_aligned/translation_vector": {
            "description": "",
            "dimensions": 3,
            "indices": list(range(156, 159)),
        },
    }


def _get_action_field_descriptions():
    return {
        "action/left_effector/position": {"description": "", "dimensions": 1, "indices": [0]},
        "action/right_effector/position": {"description": "", "dimensions": 1, "indices": [1]},
        "action/end/position": {"description": "", "dimensions": 6, "indices": list(range(2, 8))},
        "action/end/orientation": {"description": "", "dimensions": 8, "indices": list(range(8, 16))},
        "action/joint/position": {"description": "", "dimensions": 14, "indices": list(range(16, 30))},
        "action/head/position": {"description": "", "dimensions": 3, "indices": list(range(30, 33))},
        "action/waist/position": {"description": "", "dimensions": 5, "indices": list(range(33, 38))},
        "action/robot/velocity": {"description": "", "dimensions": 2, "indices": list(range(38, 40))},
    }


# ============================================================
# Public API
# ============================================================


def convert_agibot_to_lerobot(
    agibot_dir: Path,
    output_dir: Path,
    lerobot_ref_dir: Optional[Path] = None,
    fps: float = 30.0,
) -> dict:
    """Convert one or more agibot episodes to LeRobot v2.1 format.

    Auto-detects single vs batch from the ``agibot_dir`` layout: if
    ``agibot_dir/aligned_joints.h5`` exists, it's treated as a single
    episode; otherwise the directory is scanned for episode
    subdirectories (each containing ``aligned_joints.h5``).

    Parameters
    ----------
    agibot_dir
        Path to an episode dir (single) or its parent (batch).
    output_dir
        Where the LeRobot dataset is written (``data/``, ``videos/``,
        ``meta/`` are created here).
    lerobot_ref_dir
        Optional reference LeRobot dataset (must contain
        ``data/chunk-000/episode_000000.parquet``). When provided, the
        converter fills missing fisheye / head_back extrinsic columns
        from the reference. When ``None`` (the default), those
        columns are left empty.
    fps
        Video frame rate (default ``30.0``).

    Returns
    -------
    Manifest dict::

        {
            "episodes": [{"episode_index": int, "length": int}, ...],
            "total_episodes": int,
            "total_frames": int,
            "output_dir": str,
        }

    Raises
    ------
    RuntimeError
        If ffmpeg is not on ``PATH``, or heavy deps
        (``h5py``/``pyarrow``/``numpy``) are missing, or no episode
        directories are found.
    """
    # Pre-flight checks (fail fast with friendly messages).
    _require_ffmpeg()
    _require_heavy_deps()

    agibot_dir = Path(agibot_dir).resolve()
    output_dir = Path(output_dir).resolve()

    lerobot_template = None
    if lerobot_ref_dir is not None:
        lerobot_template = load_lerobot_reference(Path(lerobot_ref_dir).resolve())

    # Auto-detect episodes
    if (agibot_dir / "aligned_joints.h5").exists():
        episodes = [agibot_dir]
    else:
        episodes = detect_episodes(agibot_dir)

    if not episodes:
        raise RuntimeError(
            f"No episode directories found in {agibot_dir}. " "Each episode dir must contain aligned_joints.h5."
        )

    print(f"Found {len(episodes)} episode(s) to convert:")
    for ep in episodes:
        print(f"  - {ep}")

    episode_info = []
    for i, ep_dir in enumerate(episodes):
        result = convert_episode(ep_dir, output_dir, i, lerobot_template, fps=fps)
        episode_info.append(result)

    generate_meta(output_dir, episode_info, episodes)

    total_frames = sum(ep["length"] for ep in episode_info)
    print(f"\nConversion complete! {len(episode_info)} episode(s), {total_frames} total frames.")

    return {
        "episodes": episode_info,
        "total_episodes": len(episode_info),
        "total_frames": total_frames,
        "output_dir": str(output_dir),
    }


# ============================================================
# CLI wrapper (called by `geniesim dataset convert agibot-to-lerobot`)
# ============================================================


def convert_cli(argv: list) -> int:
    """argparse front-end for :func:`convert_agibot_to_lerobot`.

    Returns a process exit code (``0`` on success, non-zero on
    failure). The ``geniesim dataset convert agibot-to-lerobot``
    dispatcher in ``geniesim_cli/commands/dataset.py`` is the only
    caller; the script is no longer invocable directly.
    """
    parser = argparse.ArgumentParser(
        prog="geniesim dataset convert agibot-to-lerobot",
        description="Convert agibot episode data to LeRobot v2.1 format. "
        "Supports single episode (--agibot-dir <one-episode>) "
        "and batch (--agibot-dir <parent-dir-of-episodes>) modes.",
    )
    parser.add_argument(
        "--agibot-dir",
        type=Path,
        required=True,
        help="agibot episode dir, or parent dir containing multiple episode subdirs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for LeRobot-format data",
    )
    parser.add_argument(
        "--lerobot-ref-dir",
        type=Path,
        default=None,
        help="Optional reference LeRobot dataset (must contain "
        "data/chunk-000/episode_000000.parquet) — used to fill missing "
        "fisheye / head_back extrinsic columns. Omit to leave them empty.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Video FPS (default 30.0)",
    )
    args = parser.parse_args(argv)

    try:
        convert_agibot_to_lerobot(
            agibot_dir=args.agibot_dir,
            output_dir=args.output_dir,
            lerobot_ref_dir=args.lerobot_ref_dir,
            fps=args.fps,
        )
    except RuntimeError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1
    return 0
