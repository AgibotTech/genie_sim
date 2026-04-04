# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Merge per-face Gaussians3D PLY outputs into one PLY."""

from __future__ import annotations

from pathlib import Path

import torch
from sharp.utils.gaussians import Gaussians3D, load_ply, save_ply


def merge_gaussians_from_ply_paths(ply_paths: list[Path], output_ply: Path) -> None:
    """Concatenate Gaussians from multiple per-view PLY files."""
    gaussians: list[Gaussians3D] = []
    meta = None
    for ply_path in ply_paths:
        g, m = load_ply(ply_path)
        gaussians.append(g)
        if meta is None:
            meta = m
    if not gaussians or meta is None:
        raise ValueError("No Gaussians loaded")

    merged = Gaussians3D(
        mean_vectors=torch.cat([g.mean_vectors for g in gaussians], dim=1),
        colors=torch.cat([g.colors for g in gaussians], dim=1),
        singular_values=torch.cat([g.singular_values for g in gaussians], dim=1),
        quaternions=torch.cat([g.quaternions for g in gaussians], dim=1),
        opacities=torch.cat([g.opacities for g in gaussians], dim=1),
    )
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    save_ply(merged, meta.focal_length_px, meta.resolution_px, output_ply)
