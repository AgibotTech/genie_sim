# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Plan visual-only USD bindings and require a live renderer application receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from geniesim_benchmark.material_selection import select_materials

IDENTITY_FIELDS = ("schema_version", "episode_id", "episode_index", "base_seed", "derived_seed", "enabled", "plan_sha256")


def validate_material_settings(config):
    seed = config.get("benchmark", {}).get("seed", 1)
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("benchmark.seed must be an integer in [0, 4294967295]")
    material = config.get("generalization", {}).get("material", {})
    if not isinstance(material, dict) or type(material.get("enable", False)) is not bool:
        raise ValueError("generalization.material.enable must be a boolean")
    if type(material.get("num", 0)) is not int or material.get("num", 0) != 0:
        raise ValueError("episode material generalization requires num: 0")
    if not material.get("enable", False):
        return
    targets = material.get("targets")
    allowed = material.get("allowed_materials")
    if (not isinstance(targets, list) or not targets
            or any(not isinstance(p, str) or not p.startswith("/World/Objects/") for p in targets)
            or len(targets) != len(set(targets))):
        raise ValueError("material targets must be distinct absolute /World/Objects mesh paths")
    if (not isinstance(allowed, list) or not allowed
            or any(not isinstance(p, str) or not p or "/" in p for p in allowed)
            or len(allowed) != len(set(allowed))):
        raise ValueError("allowed_materials must be distinct material names")
    if type(material.get("share_by_parent", False)) is not bool:
        raise ValueError("material share_by_parent must be a boolean")
    if material.get("selection", "random") not in ("random", "enumerate"):
        raise ValueError("material selection must be random or enumerate")
    if material.get("selection") == "enumerate" and (not material.get("enable") or not allowed):
        raise ValueError("enumerated materials require enabled generalization and allowed_materials")


def create_material_plan(config, scene, assets_root, *, episode_index, episode_id):
    """Discover candidates in each target's own bound-material scope, without writes."""
    from pxr import Usd, UsdGeom, UsdShade

    validate_material_settings(config)
    if type(episode_index) is not int or episode_index < 0 or not isinstance(episode_id, str) or not episode_id:
        raise ValueError("material plan requires a nonnegative episode index and a unique episode identity")
    material = config.get("generalization", {}).get("material", {})
    seed = config.get("benchmark", {}).get("seed", 1)
    derived = int(np.random.SeedSequence([seed, episode_index]).generate_state(1, dtype=np.uint32)[0])
    root = Path(assets_root).resolve(strict=True)
    source = (root / scene["scene"]).resolve(strict=True)
    if not source.is_relative_to(root):
        raise ValueError("material source scene must be inside the selected asset root")
    enabled = material.get("enable", False)
    plan = dict(schema_version=1, episode_id=episode_id, episode_index=episode_index,
                base_seed=seed, derived_seed=derived, enabled=enabled, source_scene=str(source),
                targets=list(material.get("targets", [])), allowed_materials=list(material.get("allowed_materials", [])),
                bindings=[])
    if enabled:
        stage = Usd.Stage.Open(str(source))
        if not stage:
            raise ValueError(f"invalid material source USD: {source}")
        info = {}
        for target in plan["targets"]:
            prim = stage.GetPrimAtPath(target)
            if not prim or not prim.IsA(UsdGeom.Mesh) or prim.IsInstanceProxy():
                raise ValueError(f"material target must be an editable Mesh: {target}")
            bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if not bound:
                raise ValueError(f"material target has no existing visual binding: {target}")
            candidates = sorted(str(p.GetPath()) for p in Usd.PrimRange(bound.GetPrim().GetParent())
                                if p.IsA(UsdShade.Material))
            missing = set(plan["allowed_materials"]) - {p.rsplit("/", 1)[-1] for p in candidates}
            if missing:
                raise ValueError(f"missing allowed materials for {target}: {sorted(missing)}")
            info[target] = candidates
        selection = dict(material)
        if material.get("selection") == "enumerate":
            material_index = episode_index
            if config.get("libuipc", {}).get("instance_base_xy_offsets") is not None:
                material_index %= len(plan["allowed_materials"])
            if material_index >= len(plan["allowed_materials"]):
                raise ValueError("material instance index exceeds allowed_materials")
            selection["allowed_materials"] = [plan["allowed_materials"][material_index]]
        plan["bindings"] = select_materials(info, selection, rng=np.random.RandomState(derived))
    plan["plan_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return plan


def validate_material_receipt(plan, receipt, *, manifest_path=None):
    """Reject stale, incomplete or unverified applications before model inference."""
    if not isinstance(receipt, dict) or any(receipt.get(key) != plan[key] for key in IDENTITY_FIELDS):
        raise ValueError("material receipt episode identity or plan hash differs")
    if type(receipt.get("renderer_pid")) is not int or receipt["renderer_pid"] <= 0:
        raise ValueError("material receipt has no live renderer PID")
    if manifest_path is not None and receipt.get("manifest_path") != str(Path(manifest_path).resolve()):
        raise ValueError("material receipt manifest differs from this episode")
    actual = receipt.get("bindings")
    if not isinstance(actual, list) or len(actual) != len(plan["bindings"]):
        raise ValueError("material receipt binding inventory differs")
    for expected, applied in zip(plan["bindings"], actual):
        if (applied.get("mesh_path") != expected["mesh_path"]
                or applied.get("material_path") != expected["material_path"]
                or applied.get("computed_material_path") != expected["material_path"]
                or not applied.get("material_outputs") or not applied.get("shader_inputs")):
            raise ValueError("material receipt has an unverified visual material binding")
    return receipt


SNAPSHOT_FIELDS = ("source_id", "epoch", "request_id", "snapshot_id", "tick", "cloth_tick", "sim_time")
RENDER_FIELDS = ("render_frame", "render_reference_time", "swh_frame_number", "render_complete_wall_time_ns")


def validate_material_capture(plan, receipt_path, packet, *, manifest_path):
    receipt = validate_material_receipt(plan, json.loads(Path(receipt_path).read_text()), manifest_path=manifest_path)
    if receipt.get("phase") != "captured":
        raise ValueError("material receipt does not confirm a completed camera capture")
    applied, verified = receipt.get("applied_wall_time_ns"), receipt.get("capture_verified_wall_time_ns")
    complete = packet.get("render_complete_wall_time_ns")
    if any(type(value) is not int for value in (applied, verified, complete)) or not 0 < applied <= complete <= verified:
        raise ValueError("material receipt application/capture timing is invalid")
    if receipt.get("snapshot") != {key: packet["snapshot"][key] for key in SNAPSHOT_FIELDS}:
        raise ValueError("material receipt snapshot identity differs from model observation")
    render = {key: packet[key] for key in RENDER_FIELDS}
    # IPCClient freezes JSON arrays to tuples; compare the same wire values.
    render["render_reference_time"] = list(render["render_reference_time"])
    if receipt.get("render") != render:
        raise ValueError("material receipt render batch differs from model observation")
    from .adapter import wire_images

    hashes = {name: hashlib.sha256(row["image_data"]).hexdigest() for name, row in wire_images(packet["images"]).items()}
    if receipt.get("image_sha256") != hashes:
        raise ValueError("material receipt camera images differ from model observation")
    return receipt


def wait_for_material_receipt(plan, receipt_path, *, manifest_path, process, members, timeout_s):
    deadline = time.monotonic() + timeout_s
    path = Path(receipt_path)
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"bringup exited {process.returncode} before material verification")
        if path.is_file():
            receipt = validate_material_receipt(plan, json.loads(path.read_text()), manifest_path=manifest_path)
            if receipt["renderer_pid"] not in members(process):
                raise ValueError("material receipt renderer is not a live member of this episode bringup session")
            return receipt
        if time.monotonic() >= deadline:
            raise TimeoutError("live renderer did not confirm this episode's visual materials")
        time.sleep(0.05)
