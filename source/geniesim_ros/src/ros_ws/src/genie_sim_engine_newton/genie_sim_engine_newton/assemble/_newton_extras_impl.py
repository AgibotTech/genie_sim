# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Newton-extras assemble body — Newton-only scene extras (cloth, soft, …).

The body :class:`NewtonExtrasStage` calls with already-resolved paths.

What it does:

1. Validates the scene yaml's ``newton:`` block.
2. Writes ``<output-dir>/newton_solvers.json`` — prim paths, solver
   preference, contact tuning, physics params. No geometry.
3. Cloth USDs remain referenced from the stage scene USD; this sidecar
   only points at them by absolute path.

Sidecar shape (``version: 2``)::

    {
      "version": 2,
      "solver": {"prefer": "vbd" | "xpbd" | "style3d" | "avbd" | "none"},
      "contact": {"soft_ke": float, "soft_kd": float, "soft_mu": float},
      "entries": [
        {
          "kind": "cloth" | "box",
          "name": "<str>",
          "usd_prim_path": "/World/<name>",
          "mesh_usd_abs": "<absolute path>",   # cloth only
          "pose": [tx,ty,tz,qx,qy,qz,qw],
          "vel":  [vx,vy,vz],
          "params": {...}
        }
      ]
    }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

_KIND_REQUIRED_PARAMS: Dict[str, Tuple[str, ...]] = {
    "cloth": (
        "density",
        "tri_ke",
        "tri_ka",
        "tri_kd",
        "edge_ke",
        "edge_kd",
        "particle_radius",
    ),
    # ``box`` is a static rigid collider authored as ``UsdGeom.Cube`` with
    # ``UsdPhysics.CollisionAPI``. Newton's ``add_usd`` picks it up as a
    # world-static shape; cloth and articulations collide against it.
    "box": ("half_extents",),
}
# Particle solvers supported by the newton-standalone engine. Picking one
# of these in ``newton.solver.prefer`` is REQUIRED when the scene declares
# any cloth entry — the engine refuses to start with ``auto`` or a
# missing/empty value. ``auto`` is rejected because the resolution
# differs across code paths, so what's in the yaml would have nothing to
# do with what actually simulates.
_PARTICLE_SOLVERS = {"vbd", "xpbd", "style3d", "avbd"}
_KIND_REQUIRES_PARTICLE_SOLVER = {"cloth"}
# Geometry kinds that come from an external mesh USD file (referenced
# into the staged ``newton_scene.usda``). Primitive kinds (``box`` and
# other inline shapes) are authored inline and don't need a mesh_usd.
_KIND_REQUIRES_MESH_USD = {"cloth"}

#: Sidecar filename inside the stage dir. Logical name used by the
#: :class:`NewtonExtrasStage.produces` contract.
SIDECAR_NAME = "newton_solvers.json"


class NewtonExtrasError(RuntimeError):
    """Raised on any scene-yaml validation failure.

    Callers that want the legacy ``sys.exit(1)`` behaviour catch this
    at the CLI boundary; the stage lets it propagate to the runner.
    """


def _die(msg: str) -> None:
    raise NewtonExtrasError(msg)


def load_scene_yaml(scene_path: Path) -> Tuple[Path, Dict[str, Any]]:
    """Load and parse a scene YAML. Same suffix + mapping-shape checks
    the legacy script performed. Returns the resolved path and parsed
    mapping.
    """
    p = Path(scene_path).resolve()
    if p.suffix not in (".yaml", ".yml"):
        _die(f"--scene must be a YAML file, got {p}")
    with p.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        _die(f"scene YAML did not parse to a mapping: {p}")
    return p, cfg


def check_scene_has_newton_extras(scene_cfg: Dict[str, Any]) -> Optional[str]:
    """Return a non-None skip reason when the scene has nothing to do.

    Preserves the legacy script's two skip paths:

    * no ``newton:`` block at all
    * ``newton:`` block present but empty ``entries`` list

    Both were "print skipping / return 0" in the legacy script. In the
    stage world, they're :meth:`AssembleStage.skip_reason` returns.
    """
    newton_cfg = scene_cfg.get("newton")
    if not isinstance(newton_cfg, dict):
        return "no newton block in scene yaml"
    if not newton_cfg.get("entries"):
        # scene_plugin may still run at engine build time — that's
        # engine-time behaviour, not assemble-time. The sidecar is
        # not needed here.
        plugin_ref = newton_cfg.get("scene_plugin")
        if plugin_ref:
            return f"no newton.entries; scene_plugin={plugin_ref!r} runs at engine build"
        return "no newton.entries"
    return None


def _validate_newton_block(newton_cfg: Dict[str, Any], scene_path: Path) -> None:
    if not isinstance(newton_cfg.get("entries"), list):
        _die(f"newton.entries must be a list in {scene_path}")
    seen: set = set()
    for i, entry in enumerate(newton_cfg["entries"]):
        ctx = f"newton.entries[{i}]"
        if not isinstance(entry, dict):
            _die(f"{ctx}: must be a mapping")
        kind = entry.get("kind")
        if kind not in _KIND_REQUIRED_PARAMS:
            _die(f"{ctx}: unknown kind={kind!r}. Supported: {sorted(_KIND_REQUIRED_PARAMS)}")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            _die(f"{ctx}: 'name' required and must be a non-empty string")
        if name in seen:
            _die(f"{ctx}: duplicate name={name!r}")
        seen.add(name)
        if kind in _KIND_REQUIRES_MESH_USD and "mesh_usd" not in entry:
            _die(f"{ctx} ({name!r}): 'mesh_usd' is required for kind={kind!r}")
        pose = entry.get("pose")
        if pose is not None and (not isinstance(pose, list) or len(pose) != 7):
            _die(f"{ctx} ({name!r}): 'pose' must be 7 floats [tx,ty,tz,qx,qy,qz,qw]")
        vel = entry.get("vel")
        if vel is not None and (not isinstance(vel, list) or len(vel) != 3):
            _die(f"{ctx} ({name!r}): 'vel' must be 3 floats [vx,vy,vz]")
        params = entry.get("params") or {}
        missing = [k for k in _KIND_REQUIRED_PARAMS[kind] if k not in params]
        if missing:
            _die(f"{ctx} ({name!r}): params missing required keys: {sorted(missing)}")
        if kind == "box":
            he = params.get("half_extents")
            if (
                not isinstance(he, list)
                or len(he) != 3
                or not all(isinstance(x, (int, float)) and float(x) > 0 for x in he)
            ):
                _die(f"{ctx} ({name!r}): params.half_extents must be 3 positive floats")


def _validate_scene_plugin(newton_cfg: Dict[str, Any], scene_path: Path) -> None:
    plugin_ref = newton_cfg.get("scene_plugin")
    if plugin_ref is not None and not isinstance(plugin_ref, str):
        _die(f"newton.scene_plugin must be a string path (got {type(plugin_ref).__name__}) " f"in {scene_path}")


def _resolve_solver(newton_cfg: Dict[str, Any], entries: List[Dict[str, Any]]) -> str:
    solver_block = newton_cfg.get("solver")
    needs_particles = any(e["kind"] in _KIND_REQUIRES_PARTICLE_SOLVER for e in entries)

    raw = ""
    if isinstance(solver_block, dict):
        raw = str(solver_block.get("prefer") or "").strip().lower()

    if not needs_particles:
        if raw and raw not in _PARTICLE_SOLVERS:
            _die(
                f"newton.solver.prefer={raw!r} is not a valid particle solver. "
                f"Valid: {sorted(_PARTICLE_SOLVERS)}. (Also: the field is "
                f"optional when no cloth entry is declared — leave it out.)"
            )
        return raw or "none"

    if not raw:
        _die(
            "newton.solver.prefer is REQUIRED when newton.entries contains a "
            f"cloth/particle entry. Set it to one of: {sorted(_PARTICLE_SOLVERS)}. "
            "'auto' is not supported — every path must declare the solver "
            "explicitly so the yaml matches the runtime."
        )
    if raw == "auto":
        _die("newton.solver.prefer='auto' is not supported. " f"Pick one explicitly: {sorted(_PARTICLE_SOLVERS)}.")
    if raw not in _PARTICLE_SOLVERS:
        _die(f"newton.solver.prefer={raw!r} is not supported. " f"Valid options: {sorted(_PARTICLE_SOLVERS)}.")
    return raw


def _resolve_mesh_path(mesh_ref: str, base_path: str, scene_path: Path) -> Path:
    p = Path(mesh_ref)
    if p.is_absolute() and p.is_file():
        return p
    if base_path:
        cand = Path(base_path) / p
        if cand.is_file():
            return cand.resolve()
    cand = scene_path.parent / p
    if cand.is_file():
        return cand.resolve()
    tried_base = str(Path(base_path) / p) if base_path else "(no --base-path)"
    _die(f"mesh_usd={mesh_ref!r} not found. Tried:\n" f"  {tried_base}\n" f"  {scene_path.parent / p}")


def _build_sidecar(newton_cfg: Dict[str, Any], scene_path: Path, base_path: str) -> Dict[str, Any]:
    entries_in = newton_cfg["entries"]
    solver_str = _resolve_solver(newton_cfg, entries_in)

    contact_in = newton_cfg.get("contact") or {}
    contact_out = {
        "soft_ke": float(contact_in.get("soft_ke", 1.0e4)),
        "soft_kd": float(contact_in.get("soft_kd", 1.0e-2)),
        "soft_mu": float(contact_in.get("soft_mu", 0.25)),
    }

    entries_out: List[Dict[str, Any]] = []
    for entry in entries_in:
        kind = entry["kind"]
        name = entry["name"]
        pose = [float(x) for x in (entry.get("pose") or [0, 0, 0, 0, 0, 0, 1])]
        vel = [float(x) for x in (entry.get("vel") or [0, 0, 0])]
        # params may contain nested lists (e.g. box.half_extents) — only
        # scalar values get coerced to float here. Downstream consumers
        # already accept the mixed shape.
        params_in = entry.get("params") or {}
        params: Dict[str, Any] = {}
        for k, v in params_in.items():
            if isinstance(v, (int, float)):
                params[k] = float(v)
            elif isinstance(v, list):
                params[k] = [float(x) for x in v]
            else:
                params[k] = v

        out_entry: Dict[str, Any] = {
            "kind": kind,
            "name": name,
            "usd_prim_path": f"/World/{name}",
            "pose": pose,
            "vel": vel,
            "params": params,
        }
        if kind in _KIND_REQUIRES_MESH_USD:
            mesh_p = _resolve_mesh_path(entry["mesh_usd"], base_path, scene_path)
            print(f"[assemble_newton]   {name}: {kind} @ {mesh_p.name}", flush=True)
            out_entry["mesh_usd_abs"] = str(mesh_p)
        else:
            print(
                f"[assemble_newton]   {name}: {kind} (primitive, " f"params={ {k: v for k, v in params.items()} })",
                flush=True,
            )
        entries_out.append(out_entry)

    return {
        "version": 2,
        "solver": {"prefer": solver_str},
        "contact": contact_out,
        "entries": entries_out,
    }


def build_and_write(
    scene_path: Path,
    output_dir: Path,
    base_path: str,
    scene_cfg: Optional[Dict[str, Any]] = None,
) -> Path:
    """Validate, build, and write the sidecar. Return its resolved path.

    ``scene_cfg`` may be provided pre-parsed by the caller (the stage
    reuses the runner's parse); otherwise the YAML is loaded here so
    the legacy CLI shim can call this function directly.
    """
    scene_path = Path(scene_path).resolve()
    if scene_cfg is None:
        scene_path, scene_cfg = load_scene_yaml(scene_path)

    newton_cfg = scene_cfg.get("newton")
    if not isinstance(newton_cfg, dict):
        # Caller should have consulted check_scene_has_newton_extras
        # first. If we got here anyway, treat it as a hard error to
        # surface the caller bug.
        _die(f"internal: build_and_write called with no newton block in {scene_path}")

    _validate_scene_plugin(newton_cfg, scene_path)
    if not newton_cfg.get("entries"):
        _die(
            f"internal: build_and_write called with empty newton.entries in {scene_path}; "
            "caller must consult check_scene_has_newton_extras first"
        )

    print(f"[assemble_newton] scene:     {scene_path}", flush=True)
    print(f"[assemble_newton] output:    {output_dir}", flush=True)

    _validate_newton_block(newton_cfg, scene_path)
    sidecar = _build_sidecar(newton_cfg, scene_path, base_path)

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sidecar_path = out / SIDECAR_NAME
    # Match legacy formatting exactly (indent=2, no sort_keys) so any
    # downstream diffing / caching stays byte-identical.
    with sidecar_path.open("w") as f:
        json.dump(sidecar, f, indent=2)

    n = len(sidecar["entries"])
    size = sidecar_path.stat().st_size
    print(
        f"[assemble_newton] OK: {n} entr(ies), solver={sidecar['solver']['prefer']}, " f"sidecar={size} bytes",
        flush=True,
    )
    return sidecar_path


def run_from_cli(scene_path: str, output_dir: str, base_path: str = "", physics_engine: str = "") -> int:
    """Legacy CLI entry — preserves the pre-migration exit-code shape.

    Returns 0 on skip or success, 1 on validation failure (matching
    the previous ``sys.exit(1)`` from ``_die``). ``--physics-engine``
    is accepted for CLI compatibility but ignored; the sidecar is
    written whenever the scene declares newton entries, regardless of
    the active engine (the PhysX / isaac_newton runtimes don't read
    it, so writing it is harmless).
    """
    try:
        _, scene_cfg = load_scene_yaml(Path(scene_path))
        skip = check_scene_has_newton_extras(scene_cfg)
        if skip is not None:
            scene_p = Path(scene_path).resolve()
            print(f"[assemble_newton] {skip} in {scene_p} → skipping.", flush=True)
            return 0
        build_and_write(Path(scene_path), Path(output_dir), base_path, scene_cfg)
        return 0
    except NewtonExtrasError as err:
        print(f"[assemble_newton] ERROR: {err}", file=sys.stderr)
        return 1


__all__ = [
    "NewtonExtrasError",
    "SIDECAR_NAME",
    "build_and_write",
    "check_scene_has_newton_extras",
    "load_scene_yaml",
    "run_from_cli",
]
