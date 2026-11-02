# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Replay verified raw IPC state through the pinned Wuji ADER checkers.

Only the archive reader is imported from the independent libuipc backend.
Coordinates remain world metres; raw body_q is XYZ + XYZW, converted here
once to the COLUMN matrices consumed by the original geometry predicates.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

PACKAGE = Path(__file__).resolve().parents[1]
TASK_ALIASES = {"fold_cloth": "fold", "cloth_hanger": "hanger", "drawstring_bag": "bag"}
PROFILE_HASHES = {
    "fold_cloth": "32c2256e6ab9f831a1d622d8111f93fbde6269ef2729c39d72ccb011c8ecfd5f",
    "cloth_hanger": "c230c6c0fc6e742a41acce09ee22af11b0db79291ce474216dcac098ea3257cc",
    "drawstring_bag": "8ccf257c9314ab720d03b9288b462dac4ceaccda816589201f05eb55a1eb4703",
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def profile_config(task):
    if task not in TASK_ALIASES:
        raise ValueError(f"Unknown Wuji scoring task: {task!r}")
    return PACKAGE / "benchmark/config/llm_task/wuji_ipc" / task


def load_profile(task):
    path = profile_config(task) / "checker_profile.json"
    if digest(path) != PROFILE_HASHES[task]:
        raise ValueError("Canonical Wuji checker profile differs from its approved identity")
    return json.loads(path.read_text())


def check_scoring_cancelled(cancel):
    if cancel is not None and cancel():
        raise InterruptedError("Wuji scoring cancelled")


def load_archive(episode, repo_root=None, *, follow=False, cancel=None, timeout_s=None):
    from genie_sim_engine_libuipc.physical_state_recording import (
        PhysicalStateArchive,
        PhysicalStateArchiveFollower,
    )

    directory = Path(episode) / "physical_state"
    if follow:
        return PhysicalStateArchiveFollower(directory, cancel=cancel, timeout_s=timeout_s).initialize()
    return PhysicalStateArchive(directory)


def validate_fold_identity(identity, profile):
    matches = [mesh for mesh in identity.get("meshes", []) if mesh.get("name") == "tshirt"]
    if len(matches) != 1:
        raise ValueError("Wuji fold requires one unambiguous cloth identity")
    mesh, expected = matches[0], profile["mesh"]
    required = {
        "prim_path": expected["path"],
        "vertices": expected["vertex_count"],
        "face_counts_sha256": expected["face_counts_sha256"],
        "face_indices_sha256": expected["face_indices_sha256"],
    }
    if any(mesh.get(key) != value for key, value in required.items()):
        raise ValueError("Wuji cloth topology / material identity differs from checker profile")


def body_state_to_pose(value):
    """Decode raw archive [x,y,z,qx,qy,qz,qw], allowing only float roundoff."""
    q = np.asarray(value, dtype=float)
    if q.shape != (7,) or not np.isfinite(q).all():
        raise ValueError("Hanger rigid state must be finite [x,y,z,qx,qy,qz,qw]")
    norm = np.linalg.norm(q[3:])
    if not math.isclose(float(norm), 1.0, rel_tol=0, abs_tol=1e-5):
        raise ValueError("Hanger rigid quaternion must be normalized")
    x, y, z, w = q[3:] / norm
    pose = np.eye(4)
    pose[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    pose[:3, 3] = q[:3]
    return pose


def validate_hanger_identity(identity, profile):
    for name, field in [("tshirt", "mesh"), ("hanger", "hanger"), ("rack", "rack")]:
        matches = [m for m in identity.get("meshes", []) if m.get("name") == name]
        if len(matches) != 1:
            raise ValueError(f"Hanger scoring requires unambiguous {name} identity")
        mesh, expected = matches[0], profile[field]
        required = {
            "prim_path": expected.get("collider_path", expected["path"]),
            "vertices": expected["vertex_count"],
            "face_counts_sha256": expected["face_counts_sha256"],
            "face_indices_sha256": expected["face_indices_sha256"],
            "points_sha256": expected["points_sha256"],
        }
        if any(mesh.get(k) != v for k, v in required.items()):
            raise ValueError(f"{name} topology / source identity differs from checker profile")
        if name == "rack":
            matrix = np.asarray(mesh.get("world_transform"), dtype=float)
            if (
                matrix.shape != (4, 4)
                or not np.isfinite(matrix).all()
                or not np.allclose(matrix.T, expected["world_transform"], rtol=0, atol=1e-8)
            ):
                raise ValueError("Static rack transform differs from scoring geometry")
        if name == "hanger":
            matrix = np.asarray(mesh.get("world_transform"), dtype=float)
            # body_q encodes rotation/translation only. Source mesh scaling must
            # match the profile's hanger-local probes.
            if (
                matrix.shape != (4, 4)
                or not np.isfinite(matrix).all()
                or not np.allclose(
                    np.linalg.norm(matrix[:3, :3], axis=1),
                    np.linalg.norm(np.asarray(expected["world_transform"])[:3, :3], axis=0),
                    rtol=0,
                    atol=1e-8,
                )
            ):
                raise ValueError("Hanger scale differs from checker local probes")
            rotation = matrix[:3, :3] / np.linalg.norm(matrix[:3, :3], axis=1)[:, None]
            if (
                not np.allclose(matrix[:, 3], [0, 0, 0, 1], rtol=0, atol=1e-8)
                or not np.allclose(rotation @ rotation.T, np.eye(3), rtol=0, atol=1e-8)
                or not np.isclose(np.linalg.det(rotation), 1.0, rtol=0, atol=1e-8)
            ):
                raise ValueError("Hanger asset must have a proper rigid transform without shear or reflection")


def _rigid_asset_transform(value, expected):
    """Asset identity stores USD ROW matrices; poses/profiles use COLUMN."""
    matrix = np.asarray(value, dtype=float)
    target = np.asarray(expected, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Invalid bottle source transform")
    scales = np.linalg.norm(matrix[:3, :3], axis=1)
    if np.any(scales <= 0) or not np.allclose(scales, np.linalg.norm(target[:3, :3], axis=0), rtol=0, atol=1e-8):
        raise ValueError("Bottle scale differs from local scoring probes")
    rotation = matrix[:3, :3] / scales[:, None]
    if (
        not np.allclose(matrix[:, 3], [0, 0, 0, 1], rtol=0, atol=1e-8)
        or not np.allclose(rotation @ rotation.T, np.eye(3), rtol=0, atol=1e-8)
        or not np.isclose(np.linalg.det(rotation), 1.0, rtol=0, atol=1e-8)
    ):
        raise ValueError("Bottle source transform must not contain shear or reflection")


def validate_bag_identity(identity, profile):
    items = [("drawstring_bag", profile["mesh"]), ("trash_can", profile["bin"])]
    items += [(b["name"], b) for b in profile["bottles"]]
    for name, expected in items:
        matches = [m for m in identity.get("meshes", []) if m.get("name") == name]
        if len(matches) != 1:
            raise ValueError(f"Bag scoring requires unique {name} identity")
        mesh = matches[0]
        required = {
            "prim_path": expected.get("collider_path", expected["path"]),
            "vertices": expected["vertex_count"],
            "points_sha256": expected["points_sha256"],
            "face_counts_sha256": expected["face_counts_sha256"],
            "face_indices_sha256": expected["face_indices_sha256"],
        }
        if any(mesh.get(k) != v for k, v in required.items()):
            raise ValueError(f"{name} topology / source identity differs from checker profile")
        if name == "trash_can":
            matrix = np.asarray(mesh.get("world_transform"), dtype=float)
            if (
                matrix.shape != (4, 4)
                or not np.isfinite(matrix).all()
                or not np.allclose(matrix.T, expected["world_transform"], rtol=0, atol=1e-8)
            ):
                raise ValueError("Static bin transform differs from scoring cavity")
        elif name != "drawstring_bag":
            _rigid_asset_transform(mesh.get("world_transform"), expected["collider_world_transform"])


def _objects(scene):
    block = scene.get("libuipc")
    objects = (block or {}).get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("Scene has no IPC object geometry")
    names = [obj.get("name") for obj in objects]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Scene object names must be nonempty and unique")
    return objects


def mesh_identity(mesh):
    from pxr import Usd, UsdGeom

    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError(f"Invalid geometry points: {mesh.GetPath()}")
    if (
        counts.ndim != 1
        or not len(counts)
        or np.any(counts < 3)
        or indices.ndim != 1
        or counts.sum() != len(indices)
        or indices.min() < 0
        or indices.max() >= len(points)
    ):
        raise ValueError(f"Invalid geometry face topology: {mesh.GetPath()}")
    transform = np.asarray(UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=float)
    if not np.isfinite(transform).all() or abs(np.linalg.det(transform[:3, :3])) < 1e-15:
        raise ValueError(f"Invalid geometry transform: {mesh.GetPath()}")
    result = dict(
        vertices=len(points),
        faces=len(counts),
        points_sha256=hashlib.sha256(points.tobytes()).hexdigest(),
        face_counts_sha256=hashlib.sha256(counts.tobytes()).hexdigest(),
        face_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
        world_transform=transform.tolist(),
    )
    initial = mesh.GetPrim().GetAttribute("geniesim:initialPoints").Get()
    if initial is not None:
        initial = np.asarray(initial, dtype=np.float32)
        if initial.shape != points.shape or not np.isfinite(initial).all():
            raise ValueError(f"Invalid initial geometry points: {mesh.GetPath()}")
        result["initial_points_sha256"] = hashlib.sha256(initial.tobytes()).hexdigest()
    return result


def scene_geometry(scene, assets_root, *, stage_path=None):
    from pxr import Usd, UsdGeom

    root = Path(assets_root).resolve(strict=True)
    background = Path(stage_path) if stage_path is not None else root / scene["scene"]
    background = background.resolve(strict=True)
    if not background.is_relative_to(root):
        raise ValueError("Scene geometry references a background outside the selected asset root")
    stage = Usd.Stage.Open(str(background))
    if stage is None or stage.GetCompositionErrors():
        raise ValueError(f"Invalid composed scene geometry: {background}")
    meshes = []
    for obj in _objects(scene):
        if obj["kind"] == "box":
            # Preserve object order: backend deformable mesh IDs use YAML indices.
            meshes.append(dict(name=obj["name"], kind="box", specification=obj))
            continue
        prim = stage.GetPrimAtPath(obj["mesh_prim_path"])
        if not prim or not prim.IsA(UsdGeom.Mesh):
            raise ValueError(f'Missing task geometry mesh: {obj["mesh_prim_path"]}')
        entry = dict(
            name=obj["name"],
            kind=obj["kind"],
            prim_path=obj["mesh_prim_path"],
            **mesh_identity(UsdGeom.Mesh(prim)),
        )
        expected = obj.get("expected_particle_count")
        if expected is not None and (type(expected) is not int or entry["vertices"] != expected):
            raise ValueError(f'Task geometry point count differs from scene: {obj["name"]}')
        if obj["kind"] == "rigid_mesh":
            body = obj.get("body_prim_path")
            if not isinstance(body, str) or not (
                obj["mesh_prim_path"] == body or obj["mesh_prim_path"].startswith(body + "/")
            ):
                raise ValueError(f'Rigid body binding must name a mesh ancestor: {obj["name"]}')
            body_prim = stage.GetPrimAtPath(body)
            if not body_prim or not body_prim.IsA(UsdGeom.Xformable):
                raise ValueError(f"Missing rigid body binding: {body}")
            entry["body_prim_path"] = body
            entry["body_world_transform"] = np.asarray(
                UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            ).tolist()
        meshes.append(entry)
    return dict(background=scene["scene"], meshes=meshes)


def validate_geometry(task, identity):
    validators = {
        "fold_cloth": validate_fold_identity,
        "cloth_hanger": validate_hanger_identity,
        "drawstring_bag": validate_bag_identity,
    }
    if task not in validators:
        raise ValueError("Only the three registered Wuji IPC tasks are supported")
    validators[task](identity, load_profile(task))


def prepare_assets(config, repo_root, assets_root=None):
    """Read current task geometry from the normal bringup asset mount."""
    from .config import resolve_assets_root

    root = resolve_assets_root(repo_root, assets_root)
    settings = config["libuipc"]
    scene = yaml.safe_load(Path(settings["evaluation_scene"]).read_text())
    identity = scene_geometry(scene, root)
    validate_geometry(settings["task"], identity)
    return {
        "assets_root": str(root),
        **identity,
        "process_environment": {"SIM_ASSETS": str(root), "GENIESIM_ASSETS_PATH": str(root)},
    }


def assembled_geometry(manifest_path, expected_root, evaluation_scene, *, expected_geometry=None):
    """Bind the actual assembled scene and robot frames before inference."""
    from .adapter import policy_frame_bindings

    root = Path(expected_root).resolve(strict=True)
    path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(path.read_text())
    base = Path(manifest.get("base_path", path.parent))
    base = (base if base.is_absolute() else path.parent / base).resolve(strict=True)
    if base != root:
        raise ValueError("Assembled scene base_path differs from the selected asset root")
    stage_path = manifest.get("scene_usda") or manifest.get("usd_path")
    if not isinstance(stage_path, str) or not stage_path:
        raise ValueError("Assembled scene has no geometry path")
    stage_path = Path(stage_path)
    stage_path = stage_path if stage_path.is_absolute() else base / stage_path
    scene = yaml.safe_load(Path(evaluation_scene).read_text())
    expected = expected_geometry if expected_geometry is not None else scene_geometry(scene, root)
    actual = scene_geometry(scene, root, stage_path=stage_path)
    if expected["meshes"] != actual["meshes"]:
        raise ValueError("Assembled task geometry differs from the evaluation scene")
    return {
        "assets_root": str(root),
        "render_scene": str(stage_path.resolve()),
        "meshes": actual["meshes"],
        "policy_frame_bindings": policy_frame_bindings(path),
    }


def checker_sources(task, repo_root=None):
    """Hash the selected checker, shared combinators, profile and strict reader."""
    from genie_sim_engine_libuipc import physical_state_recording

    alias = TASK_ALIASES[task]
    paths = [
        Path(__file__),
        Path(physical_state_recording.__file__),
        profile_config(task) / "checker_profile.json",
        profile_config(task) / "problems.json",
        PACKAGE / f"plugins/ader/geometry/wuji_{alias}.py",
        PACKAGE / f"plugins/ader/action/custom/wuji_{alias}.py",
        PACKAGE / f"plugins/ader/wuji_{alias}_runner.py",
        PACKAGE / "plugins/ader/action/custom/__init__.py",
        PACKAGE / "plugins/ader/ader_base.py",
        PACKAGE / "plugins/ader/action/common_actions.py",
        PACKAGE / "plugins/ader/action/action_parsing.py",
        PACKAGE / "plugins/ader/action/action_manager.py",
        PACKAGE / "plugins/output_system/eval_utils.py",
    ]
    if task == "drawstring_bag":
        paths.append(PACKAGE / "plugins/ader/geometry/wuji_hanger.py")
    root = Path(repo_root).resolve() if repo_root is not None else PACKAGE
    return {str(path.relative_to(root)) if path.is_relative_to(root) else str(path): digest(path) for path in paths}


def physical_provenance(
    episode, identity_path, identity_sha256, archive, *, follow=False, cancel=None, archive_identity=None, **extra
):
    """Resolve mutable evidence only after every committed frame was verified."""
    check_scoring_cancelled(cancel)
    if digest(identity_path) != identity_sha256:
        raise ValueError("Asset identity changed during physical replay")
    path = episode / "physical_state/manifest.json"
    current_manifest = json.loads(path.read_text())
    if current_manifest != archive.manifest:
        raise ValueError("Physical archive manifest changed after replay")
    if (
        current_manifest.get("recording_closed") is not True
        or current_manifest.get("solver_shutdown_clean") is not True
    ):
        raise ValueError("Physical archive did not finish with clean solver shutdown")
    manifest_hash, index_hash = digest(path), digest(episode / "physical_state/frames.jsonl")
    if archive_identity is not None and (manifest_hash, index_hash) != archive_identity:
        raise ValueError("Physical archive index or manifest changed after reader initialization")
    if follow:
        verification = archive.verification
        if (
            not archive.verified
            or verification.get("all_physical_frames_verified") is not True
            or verification.get("solver_shutdown_clean") is not True
            or verification.get("physical_frames") != current_manifest["frame_count"]
            or manifest_hash != verification.get("physical_manifest_sha256")
            or index_hash != verification.get("physical_index_sha256")
        ):
            raise ValueError("Physical archive changed after final streaming verification")
    else:
        # The same reader snapshot verifies final bytes again. An attacker
        # cannot replace hashes and content after a frame was already scored.
        if sum(1 for _ in archive.iter_frames()) != current_manifest["frame_count"]:
            raise ValueError("Physical archive frame count differs from final verification")
    check_scoring_cancelled(cancel)
    return {
        "scoring_mode": "physical_archive_replay",
        "input_directory": str(episode),
        "asset_identity_sha256": identity_sha256,
        "physical_manifest_sha256": manifest_hash,
        "physical_index_sha256": index_hash,
        **extra,
    }


def named_frames(archive, task, profile):
    """Return all physical frames with manifest-bound cloth and named bodies."""
    metadata = archive.manifest.get("metadata", {})
    name = "drawstring_bag" if task == "drawstring_bag" else "tshirt"
    matches = [obj for obj in metadata.get("objects", []) if obj.get("name") == name]
    if len(matches) != 1:
        raise ValueError("Missing or ambiguous deformable particle range")
    cloth = matches[0]
    start, count = cloth.get("particle_start"), cloth.get("particle_count")
    if (
        type(start) is not int
        or start < 0
        or type(count) is not int
        or count != profile["mesh"]["vertex_count"]
        or cloth.get("mesh_prim_path") != profile["mesh"]["path"]
    ):
        raise ValueError("Deformable material range / identity differs from checker profile")
    labels = metadata.get("body_labels")
    names = (
        ["hanger"]
        if task == "cloth_hanger"
        else [b["name"] for b in profile["bottles"]] if task == "drawstring_bag" else []
    )
    if names and (not isinstance(labels, list) or any(labels.count(name) != 1 for name in names)):
        raise ValueError("Rigid labels must identify exactly one instance of each scoring body")
    indices = {name: labels.index(name) for name in names}

    def frames():
        for frame in archive.iter_frames():
            arrays = frame["state"]
            particles = arrays["particle_q"]
            if particles.ndim != 2 or particles.shape[1] != 3 or start + count > len(particles):
                raise ValueError("Deformable particle range exceeds archived array")
            values = (frame["tick"], frame["sim_time"], particles[start : start + count])
            if names:
                bodies = arrays["body_q"]
                if bodies.shape != (len(labels), 7):
                    raise ValueError("Rigid body array does not match declared labels")
                poses = [body_state_to_pose(bodies[indices[name]]) for name in names]
                values += (poses[0] if task == "cloth_hanger" else np.stack(poses),)
            yield values

    extra = (
        {"hanger_body_index": indices["hanger"]}
        if task == "cloth_hanger"
        else {"bottle_body_indices": indices, "clock_source": "physical_sim_time"} if task == "drawstring_bag" else {}
    )
    return frames(), extra


def score_physical_episode(
    task, episode, output, repo_root=None, *, follow=False, cancel=None, timeout_s=None, on_success=None
):
    """Score a closed archive, or follow committed frames until verified close.

    ``on_success`` runs at most once and conveys provisional private early-stop
    evidence. A public final result is written only after clean verification.
    The original per-task call signature is also available through the three
    ``score_*_physical_episode`` wrappers below.
    """
    profile = load_profile(task)
    sources = checker_sources(task, repo_root)
    episode, output = Path(episode).resolve(), Path(output).resolve()
    identity_path = episode / "asset_identity.json"
    identity_bytes = identity_path.read_bytes()
    validator = {
        "fold_cloth": validate_fold_identity,
        "cloth_hanger": validate_hanger_identity,
        "drawstring_bag": validate_bag_identity,
    }[task]
    validator(json.loads(identity_bytes), profile)
    identity_hash = hashlib.sha256(identity_bytes).hexdigest()
    archive_paths = (episode / "physical_state/manifest.json", episode / "physical_state/frames.jsonl")
    archive_identity = None if follow else tuple(digest(path) for path in archive_paths)
    archive = load_archive(episode, follow=follow, cancel=cancel, timeout_s=timeout_s)
    if not follow and tuple(digest(path) for path in archive_paths) != archive_identity:
        raise ValueError("Physical archive changed during reader initialization")
    frames, extra = named_frames(archive, task, profile)
    alias = TASK_ALIASES[task]
    module = importlib.import_module(f"geniesim_benchmark.plugins.ader.wuji_{alias}_runner")
    runner = getattr(module, f"Wuji{alias.title()}Runner")(profile)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "checker_profile.json", profile)
    write_json(output / "checker_problem.json", json.loads((profile_config(task) / "problems.json").read_text()))
    if checker_sources(task, repo_root) != sources:
        raise ValueError("Checker sources changed before physical replay")
    write_json(output / "checker_sources.json", sources)
    count, trajectory = 0, hashlib.sha256()
    success_notified = False
    with (output / "checker_frames.jsonl").open("w") as stream:
        for tick, sim_time, *values in frames:
            check_scoring_cancelled(cancel)
            runner.update(tick, sim_time, *values)
            count += 1
            trajectory.update(np.asarray(values[0], dtype="<f4").tobytes())
            for poses in values[1:]:
                trajectory.update(np.asarray(poses, dtype="<f8").tobytes())
            result = runner.result()
            stream.write(
                json.dumps(
                    {
                        "tick": tick,
                        "sim_time_s": sim_time,
                        "score": result["score"],
                        "stages": result["stages"],
                        "diagnostics": result["diagnostics"],
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            if follow:
                stream.flush()
            if (
                on_success is not None
                and not success_notified
                and result.get("success") is True
                and result.get("score") == 1
            ):
                success_notified = True
                on_success(
                    {
                        "tick": tick,
                        "sim_time_s": sim_time,
                        "observed_frames": count,
                        "score": result["score"],
                        "success": True,
                        "profile_id": profile["profile_id"],
                    }
                )
    if count < 2:
        raise ValueError("Wuji scoring requires initialized and advanced physical frames")
    if checker_sources(task, repo_root) != sources:
        raise ValueError("Checker sources changed during replay")
    provenance = physical_provenance(
        episode,
        identity_path,
        identity_hash,
        archive,
        follow=follow,
        cancel=cancel,
        archive_identity=archive_identity,
        **extra,
    )
    if checker_sources(task, repo_root) != sources:
        raise ValueError("Checker sources changed during final archive verification")
    if count != archive.manifest["frame_count"]:
        raise ValueError("Scored frame count differs from verified physical archive")
    result = runner.result()
    trajectory_key = "trajectory_points_sha256" if task == "fold_cloth" else "trajectory_sha256"
    result.update(
        scoring_status="scored",
        scoring_framework="ader",
        observed_frames=count,
        profile_id=profile["profile_id"],
        checker_sources=sources,
        task_tree_finished=runner.env.execute_action.is_finished(),
        **{trajectory_key: trajectory.hexdigest()},
        **provenance,
    )
    check_scoring_cancelled(cancel)
    write_json(output / "checker_result.json", result)
    return result


def score_fold_physical_episode(episode, output, repo_root=None, **options):
    return score_physical_episode("fold_cloth", episode, output, repo_root, **options)


def score_hanger_physical_episode(episode, output, repo_root=None, **options):
    return score_physical_episode("cloth_hanger", episode, output, repo_root, **options)


def score_bag_physical_episode(episode, output, repo_root=None, **options):
    return score_physical_episode("drawstring_bag", episode, output, repo_root, **options)
