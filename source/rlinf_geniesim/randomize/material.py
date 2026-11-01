# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Scan material variants for deterministic sampling by render bridges.

Paths are relative to the assembled environment root. Prefix the source file's
paths with its mount scope (World or Background) after stripping defaultPrim.
The render node applies the sampled bindings without performing its own scan."""

from __future__ import annotations

from typing import Sequence

_VALID_MODES = ("ml_prefix", "keyword", "both")


def scan_material_candidates(
    usd_path: str,
    *,
    mesh_scope: str,
    mode: str = "ml_prefix",
    keywords: Sequence[str] = (),
) -> dict[str, list[str]]:
    """Return mesh/subset paths mapped to sorted candidate material paths.

    Open the generated single-env USD without Isaac/Kit. mesh_scope identifies its
    mount in the assembled stage. ml_prefix restricts material names, keyword
    restricts mesh paths, and both applies both filters. Include only bindings with
    at least two candidates. Stable sorting preserves seeded draw order."""
    if mode not in _VALID_MODES:
        raise ValueError(f"scan_material_candidates: mode must be one of " f"{_VALID_MODES!r}, got {mode!r}")

    from pxr import Usd, UsdGeom, UsdShade  # noqa: PLC0415

    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        return {}

    default_prim = stage.GetDefaultPrim()
    root = str(default_prim.GetPath()) if default_prim and default_prim.IsValid() else ""

    def _rel(abs_path: str) -> str:
        if root and abs_path.startswith(root):
            abs_path = abs_path[len(root) :]
        return f"{mesh_scope}{abs_path}"

    mesh_keyword_filter = mode in ("keyword", "both")
    material_ml_filter = mode in ("ml_prefix", "both")

    out: dict[str, list[str]] = {}
    for prim in stage.Traverse():
        if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Subset)):
            continue
        mesh_rel = _rel(str(prim.GetPath()))
        if mesh_keyword_filter and not any(kw in mesh_rel for kw in keywords):
            continue
        binding_api = UsdShade.MaterialBindingAPI(prim)
        material, _binding_rel = binding_api.ComputeBoundMaterial()
        if not material:
            continue  # no material bound -> not a variant candidate
        material_prim = stage.GetPrimAtPath(material.GetPath())
        if not (material_prim and material_prim.IsValid()):
            continue
        parent = material_prim.GetParent()
        if not (parent and parent.IsValid()):
            continue
        siblings: set[str] = set()
        for sib in parent.GetChildren():
            if not sib.IsA(UsdShade.Material):
                continue
            if material_ml_filter and not sib.GetName().startswith("ml_"):
                continue
            siblings.add(_rel(str(sib.GetPath())))
        if len(siblings) > 1:
            out[mesh_rel] = sorted(siblings)
    return out


__all__ = ["scan_material_candidates"]
