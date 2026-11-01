# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Inspect asset grasp annotations without importing the collection runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle


def resolve_asset_id(asset_id, scene_info=None, object_id=None):
    if asset_id:
        return asset_id
    if scene_info is None or not object_id:
        raise ValueError("Provide --asset-id, or both --scene-info and --object-id")
    scene = json.loads(Path(scene_info).read_text(encoding="utf-8"))
    value = scene.get("layout", {}).get(object_id, {}).get("usd")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Object {object_id!r} has no asset ID in scene_info.layout")
    return value


def _load_pose_file(path):
    import numpy as np

    with path.open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, dict) or "grasp_pose" not in data:
        raise ValueError("Expected a dictionary with grasp_pose")
    poses = np.asarray(data["grasp_pose"], dtype=float)
    if poses.size == 0:
        return {"status": "empty", "pose_count": 0}
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("grasp_pose must have shape (N, 4, 4)")
    if not np.isfinite(poses).all():
        raise ValueError("grasp_pose contains non-finite values")
    if not np.allclose(poses[:, 3, :], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("Invalid homogeneous transform bottom row")
    rotations = poses[:, :3, :3]
    if (not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=2e-3)
            or not np.allclose(np.linalg.det(rotations), 1.0, atol=2e-3)):
        raise ValueError("Invalid rotation matrix")
    source = "file" if "width" in data else "runtime_fallback"
    widths = np.asarray(data.get("width", np.full(len(poses), 0.02)), dtype=float)
    if widths.shape != (len(poses),) or not np.isfinite(widths).all() or (widths <= 0).any():
        raise ValueError("width must contain one positive finite value per grasp_pose")
    translations = poses[:, :3, 3]
    return {
        "status": "available", "pose_count": len(poses), "width_source": source,
        "width_range_m": [float(widths.min()), float(widths.max())],
        "translation_bounds": [translations.min(axis=0).tolist(), translations.max(axis=0).tolist()],
        "frame": "asset annotation frame; apply asset alignment/scale and gripper conversion before reuse",
    }


def inspect_grasps(assets_root, asset_id, load_poses=False):
    if not asset_id or Path(asset_id).name != asset_id or asset_id in {".", ".."}:
        raise ValueError("asset_id must be a single asset directory name")
    directory = Path(assets_root).expanduser().resolve() / "interaction" / asset_id
    annotation = directory / "interaction.json"
    result = {"asset_id": asset_id, "annotation": str(annotation), "labels": {}, "pose_count": None}
    if not annotation.is_file():
        return {**result, "status": "missing_annotation"}
    data = json.loads(annotation.read_text(encoding="utf-8"))
    grasp = data.get("interaction", data).get("passive", {}).get("grasp", {})
    if not isinstance(grasp, dict):
        raise ValueError("passive.grasp must be a label-to-file mapping")
    if not grasp:
        return {**result, "status": "no_grasp_labels"}
    statuses = []
    count = 0
    for label, names in grasp.items():
        names = [names] if isinstance(names, str) else names
        entries = []
        if not isinstance(names, list):
            names = [names]
        for name in names:
            entry = {"declared_path": name}
            try:
                if not isinstance(name, str):
                    raise ValueError("Expected a grasp pickle filename")
                path = (directory / name).resolve()
                if not path.is_relative_to(directory.resolve()):
                    raise ValueError("Declared path is outside the asset annotation directory")
                entry["path"] = str(path)
                if not path.is_file():
                    entry["status"] = "missing_file"
                elif load_poses:
                    entry.update(_load_pose_file(path))
                    count += entry.get("pose_count", 0)
                else:
                    entry["status"] = "files_present_unchecked"
            except Exception as exc:
                entry.update(status="invalid", error=str(exc))
            entries.append(entry)
            statuses.append(entry["status"])
        result["labels"][label] = entries
    if "available" in statuses:
        status = "available"
    elif "files_present_unchecked" in statuses:
        status = "files_present_unchecked"
    elif "invalid" in statuses:
        status = "invalid"
    elif "empty" in statuses:
        status = "empty"
    else:
        status = "missing_files"
    result.update(status=status, pose_count=count if load_poses else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", default=os.environ.get("GENIESIM_ASSETS") or os.environ.get("SIM_ASSETS"))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--asset-id")
    group.add_argument("--scene-info", type=Path)
    parser.add_argument("--object-id")
    parser.add_argument("--load-poses", action="store_true", help="Read trusted asset pickles and validate arrays")
    args = parser.parse_args()
    if not args.assets_root:
        parser.error("Set GENIESIM_ASSETS or provide --assets-root")
    try:
        asset_id = resolve_asset_id(args.asset_id, args.scene_info, args.object_id)
        result = inspect_grasps(args.assets_root, asset_id, args.load_poses)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
