"""Scene and object helpers for USD paths, motion filtering, bounds, and scale."""
import json
import os
from pathlib import Path

import numpy as np


def resolve_scene_usd(recording_dir: str, scene_usd_raw: str) -> str:
    """Resolve the scene USD path from a recording directory."""
    recording_path = Path(recording_dir)
    raw_path = Path(scene_usd_raw)
    if raw_path.is_absolute() and raw_path.is_file():
        return scene_usd_raw
    candidate = (recording_path / scene_usd_raw).resolve()
    if candidate.is_file():
        return str(candidate)
    basename = raw_path.name
    matches = list(recording_path.rglob(basename))
    if len(matches) == 1:
        return str(matches[0].resolve())
    if len(matches) > 1:
        raw_parts = raw_path.parts
        for p in matches:
            try:
                rel = p.relative_to(recording_path)
                if rel.parts[-len(raw_parts) :] == raw_parts:
                    return str(p.resolve())
            except ValueError:
                continue
        return str(matches[0].resolve())
    return scene_usd_raw


def get_scene_objects(scene_usd: str, objects_dir_name: str = "Objects") -> list:
    """Collect object prim paths below /World/{objects_dir_name}."""
    out = []
    root_path = f"/World"
    from pxr import Usd
    stage = Usd.Stage.Open(scene_usd)
    obj_prim = stage.GetPrimAtPath(root_path)
    if not obj_prim or not obj_prim.IsValid():
        return out

    def collect(prim):
        p = prim.GetPath().pathString
        if len(p) > len(root_path) and p.startswith(root_path):
            out.append(p)
        for c in prim.GetChildren():
            collect(c)

    for c in obj_prim.GetChildren():
        collect(c)
    return out


def filter_unmoved_objects(frames: list, pos_th: float = 0.01, rot_th: float = 0.05) -> list:
    """List objects whose pose changed less than the threshold."""
    if len(frames) < 2:
        return []
    first, last = frames[0], frames[-1]
    unmoved = []
    for name in first.get("objects", {}):
        if name not in last.get("objects", {}):
            continue
        p0 = np.array(first["objects"][name]["pose"])
        p1 = np.array(last["objects"][name]["pose"])
        if np.linalg.norm(p1[:3, 3] - p0[:3, 3]) < pos_th and np.linalg.norm(
            p1[:3, :3] - p0[:3, :3], "fro"
        ) < rot_th:
            unmoved.append(name)
    return unmoved


def compute_local_bbox_center(usd_path: str) -> np.ndarray:
    """Compute the local bounding-box centre of the default USD prim."""
    try:
        from pxr import Usd, UsdGeom
        stage = Usd.Stage.Open(usd_path)
        prim = stage.GetDefaultPrim()
        if not prim or not prim.IsValid():
            children = stage.GetPseudoRoot().GetChildren()
            prim = children[0] if children else None
        if not prim:
            return np.zeros(3)
        bbox = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
        ).ComputeLocalBound(prim)
        c = bbox.GetRange().GetCenter()
        return np.array([float(c[0]), float(c[1]), float(c[2])])
    except Exception:
        return np.zeros(3)


def get_object_scale(data_info_dir: str, sim_assets: str = "") -> list:
    """Read the scale from object_parameters.json."""
    assets = sim_assets or os.environ.get("SIM_ASSETS", "")
    path = os.path.join(assets, data_info_dir, "object_parameters.json")
    if not os.path.isfile(path):
        return [1.0, 1.0, 1.0]
    with open(path, "r", encoding="utf-8") as f:
        s = json.load(f).get("scale", 1.0)
    return [s] * 3 if not isinstance(s, list) else s
