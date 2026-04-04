# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Helpers for Gaussians3D without modifying sharp source."""

from __future__ import annotations

import logging
import math

import torch

from sharp.utils.gaussians import Gaussians3D

LOGGER = logging.getLogger(__name__)


def mask_gaussians(gaussians: Gaussians3D, mask: torch.Tensor) -> Gaussians3D:
    """Keep Gaussians where mask is True (per-point, batch dim preserved)."""
    m = mask
    if m.dim() == 2:
        m = m.squeeze(0)
    op = gaussians.opacities
    # ml-sharp composer flattens opacities to [B, N]; other fields stay [B, N, C].
    if op.dim() == 2:
        op_sel = op[:, m]
    else:
        op_sel = op[:, m, :]
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors[:, m, :],
        singular_values=gaussians.singular_values[:, m, :],
        quaternions=gaussians.quaternions[:, m, :],
        colors=gaussians.colors[:, m, :],
        opacities=op_sel,
    )


def apply_ndc_frustum_mask(
    gaussians: Gaussians3D,
    x_margin: float = 2.0,
    y_margin: float = 2.0,
) -> Gaussians3D:
    """Remove Gaussians whose NDC xy/z falls outside [-margin, margin]."""
    z = gaussians.mean_vectors[:, :, 2]
    eps = 1e-6
    ndc_x = gaussians.mean_vectors[:, :, 0] / (z + eps)
    ndc_y = gaussians.mean_vectors[:, :, 1] / (z + eps)
    ok = (ndc_x > -x_margin) & (ndc_x < x_margin) & (ndc_y > -y_margin) & (ndc_y < y_margin)
    n_kept = int(ok[0].sum().item())
    n_tot = int(ok[0].numel())
    if n_kept == 0:
        # Keep this lightweight: only ranges for debug.
        nx_min = float(torch.nan_to_num(ndc_x[0]).min().item())
        nx_max = float(torch.nan_to_num(ndc_x[0]).max().item())
        ny_min = float(torch.nan_to_num(ndc_y[0]).min().item())
        ny_max = float(torch.nan_to_num(ndc_y[0]).max().item())
        LOGGER.warning(
            "apply_ndc_frustum_mask kept=0/%d (x_range=%.3f..%.3f y_range=%.3f..%.3f margin=%.2f/%.2f).",
            n_tot,
            nx_min,
            nx_max,
            ny_min,
            ny_max,
            float(x_margin),
            float(y_margin),
        )
    return mask_gaussians(gaussians, ok[0])


def cull_gaussians_outside_pinhole_frustum(
    gaussians: Gaussians3D,
    *,
    extrinsics_cam_to_world: torch.Tensor,
    intrinsics_4x4: torch.Tensor,
    image_wh: tuple[int, int],
    frustum_fov_deg: float = 91.0,
    reference_fov_deg: float = 90.0,
    margin_px: float = 0.0,
    z_eps: float = 1e-3,
) -> Gaussians3D:
    """Remove Gaussians whose mean lies outside a slightly widened cube-face frustum.

    World positions use the same convention as :func:`sharp.utils.gaussians.unproject_gaussians`
    with ``torch.linalg.inv(extr)``. Culling uses **camera-space** bounds
    ``|X/Z| ≤ lim_x``, ``|Y/Z| ≤ lim_y``. Tangents at the image edge (optionally inset by
    ``margin_px``) define the nominal ``reference_fov_deg`` cone (cube faces use 90°); limits are
    scaled by ``tan(frustum_fov_deg/2) / tan(reference_fov_deg/2)`` (default **91° vs 90°**) so
    seam-adjacent splats are kept and merged views look smoother than a hard 90° crop.
    """
    means = gaussians.mean_vectors
    device, dtype = means.device, means.dtype
    extr = extrinsics_cam_to_world.to(device=device, dtype=dtype)
    intr = intrinsics_4x4.to(device=device, dtype=dtype)
    world_to_cam = torch.linalg.inv(extr)

    b, n, _ = means.shape
    ones = torch.ones(b, n, 1, device=device, dtype=dtype)
    p_w = torch.cat([means, ones], dim=-1)
    p_cam = torch.matmul(p_w, world_to_cam.transpose(0, 1))
    x, y, z = p_cam[..., 0], p_cam[..., 1], p_cam[..., 2]

    fx = float(intr[0, 0])
    fy = float(intr[1, 1])
    cx = float(intr[0, 2])
    cy = float(intr[1, 2])
    im_w, im_h = int(image_wh[0]), int(image_wh[1])
    m = float(margin_px)
    u0, u1 = m, float(im_w - 1) - m
    v0, v1 = m, float(im_h - 1) - m
    if u1 <= u0 or v1 <= v0:
        LOGGER.warning(
            "📎 Pinhole frustum cull: margin_px=%.2f too large for %dx%d; skipping cull.",
            m,
            im_w,
            im_h,
        )
        return gaussians
    tan_half_x = max(abs(u1 - cx), abs(cx - u0)) / max(fx, 1e-12)
    tan_half_y = max(abs(v1 - cy), abs(cy - v0)) / max(fy, 1e-12)
    half_ref = math.radians(float(reference_fov_deg) * 0.5)
    half_f = math.radians(float(frustum_fov_deg) * 0.5)
    angular_scale = math.tan(half_f) / max(math.tan(half_ref), 1e-12)
    lim_x = tan_half_x * angular_scale
    lim_y = tan_half_y * angular_scale

    zc = torch.clamp(z, min=z_eps)
    in_front = z > z_eps
    in_pyramid = (torch.abs(x / zc) <= lim_x) & (torch.abs(y / zc) <= lim_y)
    ok = in_front & in_pyramid
    if not ok[0].any():
        LOGGER.warning("📎 Pinhole frustum cull removed all Gaussians (check camera convention); keeping original set.")
        return gaussians
    n_kept = int(ok[0].sum().item())
    n_tot = int(ok[0].numel())
    LOGGER.info(
        "📎 Pinhole frustum cull: kept %d / %d Gaussians (fov=%.2f° ref=%.2f° margin_px=%.2f, lim_tan xy=%.4f/%.4f)",
        n_kept,
        n_tot,
        float(frustum_fov_deg),
        float(reference_fov_deg),
        m,
        lim_x,
        lim_y,
    )
    return mask_gaussians(gaussians, ok[0])
