# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""UrdfToUsdStage — URDF → ``robot.usda`` via Isaac Sim's URDF importer.

Runs :mod:`genie_sim_engine_isaacsim.assemble._robot_impl` in a **fresh Python
subprocess**. Isaac Sim's ``SimulationApp`` is a per-process singleton
and its shutdown can terminate the interpreter (``os._exit``-style),
so running this stage inline would kill the pipeline runner before
:class:`IsaacSceneStage` gets to fire. Subprocessing restores the
pre-migration behaviour where each ``assemble_*.py`` was its own
Python process.

Skip semantics: the legacy script exited cleanly when
``robot.robot_source.urdf`` was absent (pre-baked-USD scene, nothing to
convert). This stage replicates that via :meth:`skip_reason`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

from genie_sim_engine_core.assemble import AssembleContext, AssembleStage, StageResult


class UrdfToUsdStage(AssembleStage):
    """Stage: URDF → ``robot.usda`` (runs as a subprocess).

    Reads the URDF from the ``/robot_description`` topic (latched,
    TRANSIENT_LOCAL QoS) unless ``newton.assemble.urdf_file`` in the
    scene YAML overrides with an explicit path (debug convenience —
    mirrors the legacy ``--urdf`` flag).
    """

    name = "urdf_to_usd"
    produces = frozenset({"robot.usda", "robot.urdf", "robot_raw.urdf", "urdf.sha256"})

    def skip_reason(self, ctx: AssembleContext) -> Optional[str]:
        source = (ctx.inputs.scene_cfg.get("robot") or {}).get("robot_source") or {}
        if not isinstance(source, dict):
            return "robot.robot_source is not a mapping"
        if "urdf" not in source:
            return "robot.robot_source.urdf absent (scene targets pre-baked robot.usda)"
        return None

    def run(self, ctx: AssembleContext) -> StageResult:
        assemble_extras = (ctx.inputs.scene_cfg.get("assemble") or {}).get("robot") or {}
        urdf_override = assemble_extras.get("urdf_file") or None
        timeout_s = float(assemble_extras.get("robot_description_timeout_s", 30.0))

        cmd = [
            sys.executable,
            "-m",
            "genie_sim_engine_isaacsim.assemble._robot_impl",
            "--scene",
            str(ctx.inputs.scene_path),
            "--output-dir",
            str(ctx.inputs.output_dir),
            "--robot-description-timeout",
            str(timeout_s),
        ]
        if urdf_override:
            cmd += ["--urdf", str(urdf_override)]
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"urdf_to_usd subprocess exited with {result.returncode}; see stdout above")

        out = Path(ctx.inputs.output_dir)
        files = {
            "robot.usda": out / "robot.usda",
            "robot.urdf": out / "robot.urdf",
            "robot_raw.urdf": out / "robot_raw.urdf",
            "urdf.sha256": out / "urdf.sha256",
        }
        # v2 data-exchange protocol: emit ``robot.body_paths`` — the
        # tf-frame → absolute-USD-prim-path table that ``render_node``
        # uses on every render tick to look up where each URDF link's
        # world pose (arriving from RSP on ``/tf``) should be written on
        # the OVRTX stage. Keys are bare URDF link names; values are
        # absolute prim paths inside ``robot.usda``.
        # See DESIGN.DATA_EXCHANGE_PROTOCOL.md §Manifest additions.
        robot_prefix = (ctx.inputs.scene_cfg.get("robot") or {}).get("robot_prefix") or "genie"
        body_paths = _extract_robot_body_paths(
            urdf_path=files["robot.urdf"],
            robot_usda=files["robot.usda"],
            robot_prefix=robot_prefix,
        )
        # v2: also emit ``robot.joint_names`` so the common publish
        # loop can split ``/joint_states`` (robot-only, RSP fans out
        # onto /tf) from ``/joint_states_scene`` (articulated-scene
        # DoFs: drawers, doors, cart wheels).  Fixed joints don't
        # produce a scalar DoF and are omitted.
        joint_names = _extract_robot_joint_names(urdf_path=files["robot.urdf"])
        return StageResult(
            files_added=files,
            manifest_slice={
                "robot": {
                    "source": "urdf",
                    "usd": "robot.usda",
                    "urdf": "robot.urdf",
                    "urdf_raw": "robot_raw.urdf",
                    "urdf_sha256_file": "urdf.sha256",
                    "body_paths": body_paths,
                    "joint_names": joint_names,
                }
            },
        )


def _extract_robot_joint_names(*, urdf_path: Path) -> list:
    """Return every non-fixed URDF joint name in document order.

    Consumed by the common publish loop to split ``/joint_states``
    (robot-only, RSP-fanned) from ``/joint_states_scene`` (articulated
    scene DoFs). Fixed joints have no scalar state so they're excluded.
    """
    import xml.etree.ElementTree as ET

    if not urdf_path.is_file():
        return []
    try:
        root = ET.parse(str(urdf_path)).getroot()
    except ET.ParseError:
        return []
    return [
        j.get("name", "") for j in root.findall("joint") if j.get("name") and (j.get("type") or "").lower() != "fixed"
    ]


def _extract_robot_body_paths(*, urdf_path: Path, robot_usda: Path, robot_prefix: str = "genie") -> dict:
    """Return ``{urdf_link_name: absolute_prim_path}`` for every URDF link.

    Reads the URDF via stdlib ``ElementTree`` (no ROS message parse
    required). AS3 links are identified by RigidBodyAPI after composing
    their physics variant in a private session layer. Flat assets retain
    the shared :func:`build_body_paths` walk.

    ``robot.usda`` is authored with its own ``defaultPrim`` (e.g.
    ``/robot``), but the runtime OVRTX stage loads it as a reference
    under ``/<robot_prefix>`` (``/genie``). The composed prim paths on
    the wire are therefore ``/genie/...``, not ``/robot/...``. We
    rewrite the defaultPrim scope to ``/<robot_prefix>`` here so the
    manifest entries match what the renderer will resolve at runtime.

    An empty result is fine — the renderer treats a missing entry as
    "no OVRTX write for this frame" and downstream consumers get a
    startup log line.
    """
    import xml.etree.ElementTree as ET

    from genie_sim_engine_core.common.usd_walk import build_body_paths
    from pxr import Sdf, Usd, UsdPhysics

    from .physics_variant import select_physics_variant

    if not urdf_path.is_file():
        return {}
    try:
        root = ET.parse(str(urdf_path)).getroot()
    except ET.ParseError:
        return {}
    link_names = [link.get("name", "") for link in root.findall("link") if link.get("name")]
    stage = Usd.Stage.Open(Sdf.Layer.FindOrOpen(str(robot_usda)), Sdf.Layer.CreateAnonymous())
    stage.SetEditTarget(stage.GetSessionLayer())
    if select_physics_variant(stage) == "physx":
        # Geometry alone has same-named mesh wrappers (including gripper
        # base_link meshes). Longest-path matching can drop the real base
        # or move only a visual child. Physics identifies the link owner.
        raw = {}
        names = set(link_names)
        for prim in stage.Traverse():
            name = prim.GetName()
            if name not in names or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            if name in raw:
                raise RuntimeError(f"Robot body mapping: duplicate rigid link {name!r} in {robot_usda}")
            raw[name] = str(prim.GetPath())
    else:
        raw = build_body_paths(robot_usda, link_names)
    if not raw:
        return {}

    # Rewrite ``<defaultPrim>/…`` → ``/<robot_prefix>/…`` so the
    # manifest entries name the composed OVRTX stage prims. This
    # mirrors the mujoco assembler's suffix-transplant at
    # ``_assemble.py::_build_v2_body_paths``.
    default_prefix = ""
    try:
        if stage is not None:
            default_prim = stage.GetDefaultPrim()
            if default_prim and default_prim.IsValid():
                default_prefix = str(default_prim.GetPath())
    except Exception:
        default_prefix = ""

    if not default_prefix:
        return raw  # No rewrite possible; caller decides whether to accept.

    prefix_slash = default_prefix + "/"
    remapped: dict = {}
    for name, path in raw.items():
        if path == default_prefix:
            remapped[name] = f"/{robot_prefix}"
        elif path.startswith(prefix_slash):
            suffix = path[len(default_prefix) :]
            remapped[name] = f"/{robot_prefix}{suffix}"
        else:
            # Prim exists outside the defaultPrim scope; can't compose
            # cleanly onto ``/<robot_prefix>``. Skip so a bad entry
            # doesn't feed the renderer.
            continue
    return remapped


__all__ = ["UrdfToUsdStage"]
