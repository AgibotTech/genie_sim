# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Apply an already selected visual material plan on the renderer's own stage."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from pathlib import Path

IDENTITY_FIELDS = (
    "schema_version",
    "episode_id",
    "episode_index",
    "base_seed",
    "derived_seed",
    "enabled",
    "plan_sha256",
)
SNAPSHOT_FIELDS = ("source_id", "epoch", "request_id", "snapshot_id", "tick", "cloth_tick", "sim_time")
RENDER_FIELDS = ("render_frame", "render_reference_time", "swh_frame_number", "render_complete_wall_time_ns")


def load_episode_material_config(manifest_path, scene_loader):
    """Legacy assembled stages may outlive their source YAML; selection is opt-in.

    Other consumers retain scene_loader's strict explicit-scene contract. An
    identifiable episode_plan is returned unchanged for strict application.
    """
    import yaml

    try:
        scene = scene_loader(manifest_path)
    except (OSError, ValueError, yaml.YAMLError):
        return None
    generalization = scene.get("generalization", {})
    material = generalization.get("material", {}) if isinstance(generalization, dict) else {}
    if not isinstance(material, dict) or "episode_plan" not in material:
        return None
    sequence = scene.get("libuipc", {}).get("episode_sequence")
    if sequence is not None:
        if not isinstance(sequence, list) or not sequence:
            raise ValueError("invalid configured material episode sequence")
        material = dict(material, episode_sequence=[entry["material"] for entry in sequence])
    return material


def _write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _read_bindings(stage, plan):
    from pxr import Usd, UsdGeom, UsdShade

    rows = []
    for selected in plan["bindings"]:
        path, material_path = selected["mesh_path"], selected["material_path"]
        prim = stage.GetPrimAtPath(path)
        material = UsdShade.Material.Get(stage, material_path)
        if not prim or not prim.IsA(UsdGeom.Mesh) or prim.IsInstanceProxy() or not material:
            raise ValueError(f"visual material target or Material is missing: {path} -> {material_path}")
        computed, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not computed or str(computed.GetPath()) != material_path:
            raise ValueError(f"visual material did not resolve on renderer stage: {path}")
        outputs = {}
        for output in material.GetOutputs():
            connections = output.GetAttr().GetConnections()
            if not connections:
                continue
            for connection in connections:
                if not stage.GetPropertyAtPath(connection):
                    raise ValueError(f"visual material output is unresolved: {connection}")
            outputs[output.GetFullName()] = [str(p) for p in connections]
        # MDL materials must have all three resolved contexts, not just a
        # relationship pointing at an arbitrary valid Material prim.
        if any(name.startswith("outputs:mdl:") for name in outputs):
            if any("outputs:mdl:" + name not in outputs for name in ("surface", "displacement", "volume")):
                raise ValueError(f"visual Material has incomplete MDL outputs: {material_path}")
        inputs = {}
        for shader in Usd.PrimRange(material.GetPrim()):
            for attr in shader.GetAttributes():
                value = attr.Get()
                if value is None:
                    continue
                name = attr.GetName()
                if name.startswith("info:") or (
                    name.startswith("inputs:")
                    and ("color" in str(attr.GetTypeName()) or str(attr.GetTypeName()) == "asset")
                ):
                    inputs[str(attr.GetPath())] = (
                        list(map(float, value)) if "color" in str(attr.GetTypeName()) else str(value)
                    )
        if not outputs or not inputs:
            raise ValueError(f"visual material has no resolved shader outputs/inputs: {material_path}")
        rows.append(
            dict(
                mesh_path=path,
                material_path=material_path,
                computed_material_path=str(computed.GetPath()),
                material_outputs=outputs,
                shader_inputs=inputs,
            )
        )
    return rows


def apply_material_plan(stage, plan, receipt_path, *, manifest_path=""):
    """Bind once in the private worker layer, leaving physics-purpose bindings intact."""
    from pxr import Usd, UsdGeom, UsdShade

    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise ValueError("unsupported visual material plan schema")
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    expected_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if expected_hash != plan.get("plan_sha256"):
        raise ValueError("visual material plan hash differs")
    if not plan["enabled"] and plan["bindings"]:
        raise ValueError("disabled visual material plan has bindings")
    targets = []
    for row in plan["bindings"]:
        prim = stage.GetPrimAtPath(row["mesh_path"])
        material = UsdShade.Material.Get(stage, row["material_path"])
        if (
            not prim
            or not prim.IsA(UsdGeom.Mesh)
            or prim.IsInstanceProxy()
            or not material
            or row["material_path"] not in row["candidates"]
        ):
            raise ValueError("visual material plan target/Material/candidate is invalid")
        targets.append((prim, material))
    # The renderer references source assets into its new root layer. Never
    # Save/Export that layer or mutate any source file.
    with Usd.EditContext(stage, stage.GetRootLayer()):
        for prim, material in targets:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    receipt = {key: plan[key] for key in IDENTITY_FIELDS}
    receipt.update(
        renderer_pid=os.getpid(),
        manifest_path=str(Path(manifest_path).resolve()) if manifest_path else "",
        phase="initialized",
        bindings=_read_bindings(stage, plan),
        applied_wall_time_ns=time.time_ns(),
    )
    _write_json(receipt_path, receipt)
    return receipt


def record_material_capture(stage, plan, receipt_path, response, *, manifest_path):
    """Read bindings after deformation and bind evidence to the completed RGB batch."""
    previous = json.loads(Path(receipt_path).read_text())
    if any(previous.get(key) != plan[key] for key in IDENTITY_FIELDS):
        raise ValueError("material capture receipt lost its episode identity")
    receipt = {key: plan[key] for key in IDENTITY_FIELDS}
    receipt.update(
        renderer_pid=os.getpid(),
        manifest_path=str(Path(manifest_path).resolve()),
        phase="captured",
        applied_wall_time_ns=previous["applied_wall_time_ns"],
        capture_verified_wall_time_ns=time.time_ns(),
        bindings=_read_bindings(stage, plan),
        snapshot={key: response["snapshot"][key] for key in SNAPSHOT_FIELDS},
        render={key: response[key] for key in RENDER_FIELDS},
        image_sha256={
            name: hashlib.sha256(base64.b64decode(row["image_data"], validate=True)).hexdigest()
            for name, row in response["images"].items()
        },
    )
    _write_json(receipt_path, receipt)
    return receipt


class MaterialEpisodes:
    """Select only preloaded YAML plans at a correlated physical epoch boundary."""

    def __init__(self, config):
        self.entries = config.get("episode_sequence", [config])
        self.index = 0

    def select(self, snapshot):
        index = snapshot.get("episode_index", 0)
        if type(index) is not int or not 0 <= index < len(self.entries):
            raise ValueError("snapshot episode_index is outside the configured material sequence")
        if len(self.entries) > 1 and snapshot["epoch"] != index:
            raise ValueError("material episode does not match physical reset epoch")
        if index < self.index or index > self.index + 1:
            raise ValueError("material sequence cannot skip or replay an episode")
        changed = index != self.index
        self.index = index
        entry = self.entries[index]
        return entry["episode_plan"], entry["receipt_path"], changed
