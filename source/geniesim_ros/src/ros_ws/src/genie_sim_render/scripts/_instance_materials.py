# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Instance material bindings shared by tiled and fused render nodes."""
from __future__ import annotations


def apply_instance_materials(stage, table: dict, env_ids, instances) -> None:
    """Bind a selected instance before its reset image is published."""
    mesh = table.get("mesh_rel")
    if not mesh:
        return
    if len(env_ids) != len(instances):
        raise ValueError("instance materials require one instance per reset env")
    by_instance = table.get("by_instance") or {}
    for env, instance in zip(env_ids, instances):
        material = by_instance[str(int(instance))]
        applied, requested, missing = _apply_material_bindings(stage, int(env), {mesh: material})
        if applied != requested:
            raise RuntimeError(f"instance {instance} material failed for env {env}: {missing}")

def _apply_material_bindings(stg, env_id: int, bindings: dict) -> "tuple[int, int, Optional[str]]":
    """The application side of material randomization — applies the
    decision the bridge process has **already fully resolved**
    (``{mesh relative path: material relative path}``, computed in the
    bridge process by ``randomize.material.scan_material_candidates`` /
    ``randomize.sampler._draw_material_plan`` and carried over as-is via
    randomize_cmd.json — see those two modules' docs) to this one env's
    subtree, ``/World/envs/env_<env_id>``.

    This function **does no scanning or selection of its own** — what the
    candidates are and which one got picked is already decided by the
    bridge process (it has a static scene USD it can open directly with
    pxr, no live stage needed); this function just does
    ``UnbindAllBindings()`` + ``Bind()`` verbatim, the same way the lights
    side's ``_apply_light_perturbation`` does: a single binding whose prim
    cannot be found only gets a WARN and is skipped, it does not take down
    the whole command (``_apply_randomize_cmd``'s try/except is the real
    backstop for genuine exceptions). **Only ever changes things inside
    this one env's subtree** — ``mesh_rel``/``material_rel`` are both
    relative paths, so prefixing them naturally confines them under
    ``/World/envs/env_<env_id>/``; they can never touch another env, and
    can never touch the robot either (the candidates are scanned from
    ``foreground_usda``/``background_usda``, which never include the robot
    USD, so a robot path structurally cannot appear in ``bindings`` —
    ``exclude_prims`` filtering has already happened on the bridge-process
    side, this function does not re-check it).

    Returns ``(applied, requested, first_missing_mesh_path)`` — the
    material readback that lets the caller (``_apply_randomize_cmd``) know
    how many bindings actually landed, rather than unconditionally acking
    ``ok=True`` and letting the plan report a whole list of "materials
    used" when not a single one was actually bound.
    ``requested`` is the number of bindings this call received;
    ``first_missing_mesh`` is the **absolute** path of the first binding
    that failed (whether the mesh could not be found, the mesh was found
    but the material could not, or the independent readback below did not
    match — all three cases record the mesh path, not the material path,
    so it is easy to go check on the stage whether that path actually
    exists and track down a mount-point mapping problem); it is ``None``
    when everything succeeds.

    ``applied`` does not treat "``Bind()`` did not raise" as evidence: that
    is self-reporting, and self-reported success is not trustworthy here —
    using self-reporting to implement the readback would be circular.
    Right after ``Bind()``, this function uses
    ``UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()`` to
    independently re-query the stage and confirm what actually resolves is
    indeed the material it just bound, and only then counts it in
    ``applied`` — this is not hypothetical paranoia: ``Bind()`` only writes
    one opinion onto "whichever layer is the current edit target"; if a
    stronger layer (for example the session layer that a lot of Isaac Sim
    runtime edits land on) already has an explicit binding on the same
    mesh, the composed result still comes out to that stronger layer's
    value, and ``Bind()`` itself reports no error at all — a unit test
    builds this exact scenario with a real, stronger session layer, and
    ``ComputeBoundMaterial()`` is able to tell it apart from "did not
    raise"; see ``test_material_render_node.py`` for details."""
    from pxr import Sdf, UsdShade  # noqa: PLC0415

    env_root = f"/World/envs/env_{int(env_id)}"
    requested = 0
    applied = 0
    first_missing_mesh = None
    for mesh_rel, material_rel in (bindings or {}).items():
        requested += 1
        mesh_path = f"{env_root}/{mesh_rel}"
        mesh_prim = stg.GetPrimAtPath(mesh_path)
        if not (mesh_prim and mesh_prim.IsValid()):
            print(f"[tiled] WARN: randomize material — mesh prim "
                  f"{mesh_path} not found (env {env_id}); skipping this "
                  f"binding", flush=True)
            if first_missing_mesh is None:
                first_missing_mesh = mesh_path
            continue
        material_path = f"{env_root}/{material_rel}"
        material = UsdShade.Material.Get(stg, Sdf.Path(material_path))
        if not material:
            print(f"[tiled] WARN: randomize material — material prim "
                  f"{material_path} not found (env {env_id}); skipping "
                  f"this binding", flush=True)
            if first_missing_mesh is None:
                first_missing_mesh = mesh_path
            continue
        binding_api = UsdShade.MaterialBindingAPI(mesh_prim)
        # Only replace the visual (all-purpose) direct binding. In fused,
        # clearing every purpose would also remove the PhysX material.
        binding_api.UnbindDirectBinding()
        binding_api.Bind(material)
        # Independent readback: do not trust Bind()'s return value or
        # whether it raised — recompute from the stage what material this
        # mesh actually resolves to.
        bound, _ = UsdShade.MaterialBindingAPI(mesh_prim).ComputeBoundMaterial()
        if bound and bound.GetPrim().GetPath() == material.GetPrim().GetPath():
            applied += 1
        else:
            resolved = bound.GetPrim().GetPath() if bound else None
            print(f"[tiled] WARN: randomize material — Bind() on {mesh_path} "
                  f"did not raise, but ComputeBoundMaterial() resolves to "
                  f"{resolved!r} instead of the intended {material_path!r} "
                  f"(env {env_id}); a stronger opinion elsewhere in the "
                  f"stage is winning composition — not counting this "
                  f"binding as applied", flush=True)
            if first_missing_mesh is None:
                first_missing_mesh = mesh_path
    return applied, requested, first_missing_mesh
