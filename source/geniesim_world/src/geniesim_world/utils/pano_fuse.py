# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Panorama depth fusion; inputs/outputs are metric depth in meters (SI)."""

from __future__ import annotations

import logging

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


def _masked_gaussian(x: np.ndarray, valid: np.ndarray, sigma: float, eps: float) -> np.ndarray:
    if sigma <= 0:
        out = np.zeros_like(x, dtype=np.float32)
        out[valid] = x[valid]
        return out
    xf = x.astype(np.float32)
    vf = valid.astype(np.float32)
    num = cv2.GaussianBlur(xf * vf, (0, 0), sigmaX=sigma, sigmaY=sigma)
    den = cv2.GaussianBlur(vf, (0, 0), sigmaX=sigma, sigmaY=sigma)
    out = np.zeros_like(xf, dtype=np.float32)
    np.divide(num, np.maximum(den, eps), out=out, where=den > eps)
    return out


def _build_laplacian_pyramid(x: np.ndarray, levels: int) -> list[np.ndarray]:
    g = [x.astype(np.float32)]
    for _ in range(max(1, levels) - 1):
        if min(g[-1].shape[:2]) <= 16:
            break
        g.append(cv2.pyrDown(g[-1]))
    lap: list[np.ndarray] = []
    for i in range(len(g) - 1):
        up = cv2.pyrUp(g[i + 1], dstsize=(g[i].shape[1], g[i].shape[0]))
        lap.append((g[i] - up).astype(np.float32))
    lap.append(g[-1].astype(np.float32))
    return lap


def _reconstruct_laplacian(lap: list[np.ndarray]) -> np.ndarray:
    x = lap[-1].astype(np.float32)
    for i in range(len(lap) - 2, -1, -1):
        x = cv2.pyrUp(x, dstsize=(lap[i].shape[1], lap[i].shape[0])) + lap[i]
    return x.astype(np.float32)


def fuse_pano_depth(
    da360_depth: np.ndarray,
    sharp_depth: np.ndarray,
    *,
    method: str = "laplacian",
    sharp_weight: float = 0.6,
    levels: int = 4,
    sigma: float = 2.0,
    depth_max: float | None = None,
    eps: float = 5e-5,
) -> np.ndarray:
    """Fuse DA360 and SHARP pano depth in inverse-depth domain."""
    da = np.asarray(da360_depth, dtype=np.float32)
    sh = np.asarray(sharp_depth, dtype=np.float32)
    if da.shape != sh.shape:
        raise ValueError(f"Depth shapes must match, got {da.shape} vs {sh.shape}")

    finite_da = np.isfinite(da)
    finite_sh = np.isfinite(sh)
    if depth_max is not None:
        dmv = float(depth_max)
        valid_da = (da > eps) & (da < dmv) & finite_da
        valid_sh = (sh > eps) & (sh < dmv) & finite_sh
    else:
        valid_da = (da > eps) & finite_da
        valid_sh = (sh > eps) & finite_sh
    valid = valid_da | valid_sh
    da_valid_ratio = float(valid_da.mean())
    sh_valid_ratio = float(valid_sh.mean())
    LOGGER.info(
        "🧬 Pano fuse start: method=%s sharp_weight=%.3f levels=%d sigma=%.3f da_valid=%.2f%% sharp_valid=%.2f%%",
        method,
        float(sharp_weight),
        int(levels),
        float(sigma),
        100.0 * da_valid_ratio,
        100.0 * sh_valid_ratio,
    )

    inv_da = np.zeros_like(da, dtype=np.float32)
    inv_sh = np.zeros_like(sh, dtype=np.float32)
    np.divide(1.0, da, out=inv_da, where=valid_da)
    np.divide(1.0, sh, out=inv_sh, where=valid_sh)

    a = float(np.clip(sharp_weight, 0.0, 1.0))

    if method == "gaussian":
        da_s = _masked_gaussian(inv_da, valid_da, sigma, eps)
        sh_s = _masked_gaussian(inv_sh, valid_sh, sigma, eps)
        fused_inv = (1.0 - a) * da_s + a * sh_s
    elif method == "laplacian":
        lap_da = _build_laplacian_pyramid(inv_da, levels=max(2, int(levels)))
        lap_sh = _build_laplacian_pyramid(inv_sh, levels=max(2, int(levels)))
        n = min(len(lap_da), len(lap_sh))
        fused_lap: list[np.ndarray] = []
        for i in range(n):
            # Keep coarsest structure close to DA360, inject SHARP detail at finer bands.
            alpha = 0.15 if i == n - 1 else a
            fused_lap.append((1.0 - alpha) * lap_da[i] + alpha * lap_sh[i])
        fused_inv = _reconstruct_laplacian(fused_lap)
    else:
        raise ValueError(f"Unknown fuse method: {method}")

    # Invalid areas are represented by NaN (not 0) so downstream can identify holes.
    out = np.full_like(da, np.nan, dtype=np.float32)
    np.divide(
        1.0,
        np.maximum(fused_inv, eps),
        out=out,
        where=valid & np.isfinite(fused_inv),
    )
    out = np.maximum(out, 0.0).astype(np.float32)
    if depth_max is not None:
        out = np.clip(out, 0.0, float(depth_max)).astype(np.float32)
    finite_out = np.isfinite(out)
    if np.any(finite_out):
        p = np.percentile(out[finite_out], [1, 50, 99]).astype(np.float32)
        LOGGER.info(
            "✨ Pano fuse done: depth p01/p50/p99=%.4f/%.4f/%.4f min=%.4f max=%.4f",
            float(p[0]),
            float(p[1]),
            float(p[2]),
            float(np.min(out[finite_out])),
            float(np.max(out[finite_out])),
        )
    else:
        LOGGER.warning("⚠️ Pano fuse done: no valid fused pixels.")
    return out
