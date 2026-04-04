# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Shared SHARP inference used by ``geniesim_world create`` (per-face cubemap)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sharp.models import PredictorParams, create_predictor
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import save_ply, unproject_gaussians

from geniesim_world.gaussian_ops import cull_gaussians_outside_pinhole_frustum
from geniesim_world.predictor import wrap_predictor
from geniesim_world.utils.cubes import _well_defined_depth_np

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
_DEPTH_EPS = 1e-3
_MIN_VALID_DEPTH_FRACTION = 0.03
_ROBUST_CLIP_PERCENTILE = 99.5
_ROBUST_CLIP_EXPAND = 1.35
_SAT_RATIO_FALLBACK = 0.02


def _nonneg_depth_np(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(x, dtype=np.float32), 0.0)


def _clip_depth_np(x: np.ndarray, depth_max: float | None) -> np.ndarray:
    if depth_max is None:
        return _nonneg_depth_np(x)
    return np.clip(np.asarray(x, dtype=np.float32), 0.0, float(depth_max))


def _valid_depth_mask_torch(d: torch.Tensor, depth_max: float | None, eps: float) -> torch.Tensor:
    ok = (d > eps) & torch.isfinite(d)
    if depth_max is not None:
        ok = ok & (d < float(depth_max) * 0.98)
    return ok.float()


def resolve_device(device: str) -> torch.device:
    if device == "default":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    return torch.device(device)


def load_sharp_state_dict(checkpoint_path: Path | None):
    if checkpoint_path is None:
        LOGGER.info("⬇️ Downloading default checkpoint")
        return torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def build_predictor(checkpoint_path: Path | None, device: str):
    """Load SHARP weights once and move to *device* (for multi-view pipelines)."""
    dev = resolve_device(device)
    state_dict = load_sharp_state_dict(checkpoint_path)
    pred = create_predictor(PredictorParams())
    pred.load_state_dict(state_dict)
    pred = wrap_predictor(pred)
    pred.eval()
    pred.to(dev)
    p0 = next(pred.parameters())
    LOGGER.info("🤖 SHARP on device %s (dtype=%s)", p0.device, p0.dtype)
    return pred


def predict_one_image(
    *,
    input_path: Path,
    depth_path: Path,
    intr_path: Path,
    extr_path: Path,
    output_dir: Path,
    checkpoint_path: Path | None,
    depth_max: float | None,
    device: str,
    no_depth_gt_init: bool,
    predictor: torch.nn.Module | None = None,
    ndc_frustum_mask: bool = True,
    ply_frustum_cull: bool = True,
    ply_frustum_fov_deg: float = 91.0,
    ply_frustum_margin_px: float = 0.0,
    sharp_ply_radius_cap: float = 2000.0,
) -> Path:
    """Run depth-guided SHARP on one RGB + depth + intr/extr; write under *output_dir*:

    - ``<stem>.ply`` — Gaussian export
    - ``<stem>_sharp_depth.exr`` — SHARP aligned monodepth (H×W matches the face RGB / DA360 EXR)

    Pass *predictor* from :func:`build_predictor` to avoid reloading weights (e.g. six cubemap faces).

    When *ply_frustum_cull* is true (default), Gaussians outside a pyramid slightly wider than the
    nominal 90° cube face (default *ply_frustum_fov_deg* = 91° vs *reference* 90°) are removed
    before saving the PLY. Optional *ply_frustum_margin_px* tightens the reference edge tangents.
    """
    import cv2

    if predictor is None:
        pred = build_predictor(checkpoint_path, device)
        dev = next(pred.parameters()).device
    else:
        pred = predictor
        dev = next(pred.parameters()).device

    image_bgr = cv2.imread(str(input_path))
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {input_path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        raise RuntimeError(f"Failed to read depth: {depth_path}")
    if depth_path.suffix.lower() == ".exr":
        if depth_img.ndim == 3:
            depth_img = depth_img[:, :, 0]
        depth = _clip_depth_np(depth_img.astype(np.float32), depth_max)
    else:
        if depth_max is None:
            raise ValueError(
                "Legacy uint16 PNG depth requires --depth-max (meters full-scale). "
                "Use float EXR cube depth or pass --depth-max."
            )
        depth = depth_img.astype(np.float32) / 65535.0 * float(depth_max)
        depth = np.clip(np.asarray(depth, dtype=np.float32), 0.0, float(depth_max))

    intr = np.load(intr_path)
    extr = np.load(extr_path)
    f_px = float(intr[0, 0])
    height, width = image.shape[:2]

    # Robust clamp to suppress heavy long-tail depth outliers that often cause
    # SHARP depth alignment to collapse into near-flat planes.
    valid_np = _well_defined_depth_np(depth, eps=_DEPTH_EPS, depth_max=depth_max)
    if np.any(valid_np):
        p = np.percentile(depth[valid_np], [50, _ROBUST_CLIP_PERCENTILE]).astype(np.float32)
        p50, p_hi = float(p[0]), float(p[1])
        if depth_max is not None:
            clip_hi = min(float(depth_max), max(p_hi * _ROBUST_CLIP_EXPAND, p50 * 3.0))
            clip_hi = min(clip_hi, float(depth_max) * 0.99)
        else:
            clip_hi = max(p_hi * _ROBUST_CLIP_EXPAND, p50 * 3.0)
        clipped = np.clip(depth, 0.0, clip_hi).astype(np.float32)
        depth = np.where(valid_np, clipped, depth)
        LOGGER.info(
            "✂️ Face %s robust clip: p50=%.4f p%.1f=%.4f -> clip_hi=%.4f",
            input_path.stem,
            p50,
            _ROBUST_CLIP_PERCENTILE,
            p_hi,
            clip_hi,
        )

    image_pt = torch.from_numpy(image.copy()).float().to(dev).permute(2, 0, 1) / 255.0
    depth_pt = torch.from_numpy(depth.copy()).float().to(dev)
    extr_pt = torch.from_numpy(extr.copy()).float().to(dev)

    internal_shape = (1536, 1536)
    image_resized = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )
    depth_resized = F.interpolate(
        depth_pt[None, None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )
    valid_depth_mask = _valid_depth_mask_torch(depth_resized, depth_max, _DEPTH_EPS)
    valid_ratio = float(valid_depth_mask.mean().item())
    if depth_max is not None:
        dm = float(depth_max)
        not_sentinel = depth_resized < dm * 0.995
        sat_ratio = float(((depth_resized >= dm * 0.98) & not_sentinel).float().mean().item())
    else:
        sat_ratio = 0.0
    valid_stat = _well_defined_depth_np(depth, eps=_DEPTH_EPS, depth_max=depth_max)
    if np.any(valid_stat):
        p = np.percentile(depth[valid_stat], [1, 50, 99]).astype(np.float32)
        LOGGER.info(
            "📊 Face %s input depth stats: valid=%.2f%% p01/p50/p99=%.4f/%.4f/%.4f",
            input_path.stem,
            100.0 * valid_ratio,
            float(p[0]),
            float(p[1]),
            float(p[2]),
        )
    else:
        LOGGER.warning(
            "⚠️ Face %s input depth has no well-defined positive values.",
            input_path.stem,
        )
    disparity_factor = torch.tensor([f_px / width], dtype=torch.float32, device=dev)

    with torch.no_grad():
        # If depth is almost empty for this face (e.g. sky-heavy), forcing depth-guided
        # init can still collapse geometry to near-flat planes. Fall back to monodepth-only.
        effective_no_depth_gt = (
            no_depth_gt_init or (valid_ratio < _MIN_VALID_DEPTH_FRACTION) or (sat_ratio > _SAT_RATIO_FALLBACK)
        )
        if valid_ratio < _MIN_VALID_DEPTH_FRACTION:
            LOGGER.warning(
                "⚠️ Face %s has low valid depth ratio %.2f%% (< %.2f%%); falling back to monodepth-only SHARP init",
                input_path.stem,
                100.0 * valid_ratio,
                100.0 * _MIN_VALID_DEPTH_FRACTION,
            )
        if sat_ratio > _SAT_RATIO_FALLBACK:
            LOGGER.warning(
                "⚠️ Face %s has saturated depth ratio %.2f%% (> %.2f%%); forcing monodepth-only SHARP init",
                input_path.stem,
                100.0 * sat_ratio,
                100.0 * _SAT_RATIO_FALLBACK,
            )

        if effective_no_depth_gt:
            gaussians_ndc = pred(image_resized, disparity_factor, None)
        else:
            gaussians_ndc = pred(
                image_resized,
                disparity_factor,
                depth_resized,
                depth_gt=depth_resized,
                use_depth_gt_for_init=True,
                ndc_frustum_mask=ndc_frustum_mask,
            )

        # Dense SHARP depth (monodepth + same depth_alignment as RGBGaussianPredictor.forward).
        # Exported at face resolution (H, W) to match DA360 cube EXR and RGB (`*_depth.exr`, `*_image.png`).
        # Note: runs monodepth once more than `pred()` alone (same graph as SHARP's depth branch).
        md_out = pred.monodepth_model(image_resized)
        md_disp = md_out.disparity
        df_exp = disparity_factor[:, None, None, None]
        monodepth = df_exp / md_disp.clamp(min=1e-4, max=1e4)
        # depth_alignment expects single-channel metric depth [B,1,H,W].
        if effective_no_depth_gt:
            depth_for_alignment = monodepth[:, 0:1]
        else:
            depth_for_alignment = torch.where(valid_depth_mask > 0.5, depth_resized, monodepth[:, 0:1])
        monodepth_aligned, _ = pred.depth_alignment(
            monodepth,
            depth_for_alignment,
            md_out.decoder_features,
        )
        sharp_depth_hw = F.interpolate(
            monodepth_aligned,
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        )

    intrinsics = torch.tensor(
        [
            [f_px, 0, width / 2, 0],
            [0, f_px, height / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=dev,
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    gaussians = unproject_gaussians(gaussians_ndc, torch.linalg.inv(extr_pt), intrinsics_resized, internal_shape)
    n0 = int(gaussians.mean_vectors.shape[1])
    assert n0 > 0, f"[{input_path.stem}] unproject produced 0 Gaussians; check init depth_gt & masks."
    if ply_frustum_cull:
        gaussians = cull_gaussians_outside_pinhole_frustum(
            gaussians,
            extrinsics_cam_to_world=extr_pt,
            intrinsics_4x4=intrinsics_resized,
            image_wh=internal_shape,
            frustum_fov_deg=float(ply_frustum_fov_deg),
            reference_fov_deg=90.0,
            margin_px=float(ply_frustum_margin_px),
        )
    n1 = int(gaussians.mean_vectors.shape[1])
    assert n1 > 0, f"[{input_path.stem}] post-cull produced 0 Gaussians; likely ndc_frustum_mask emptied them."

    # Optionally reduce far-point drift on exported per-face PLY.
    # When |xyz| exceeds the threshold, we apply a log compression on the
    # distance so far points move inward without a hard cutoff sphere.
    if float(sharp_ply_radius_cap) > 0.0:
        thr = float(sharp_ply_radius_cap)
        xyz = gaussians.mean_vectors
        r = torch.linalg.norm(xyz, dim=-1, keepdim=True).clamp(min=1e-9)
        mask = r > thr
        if torch.any(mask):
            scale = torch.ones_like(r)
            # log compression: r' = thr * log1p(r/thr) / log(2)
            # => r'=thr when r=thr, and r' grows sub-linearly afterwards.
            denom = float(np.log(2.0))
            r_m = r[mask]
            r_scaled = thr * torch.log1p(r_m / thr) / denom
            scale[mask] = r_scaled / r_m
            # singular_values shape is typically [N, 3]; keep scale as [N, 1]
            # so broadcasting applies on the 3 channels.
            scale1 = scale.squeeze(-1).unsqueeze(-1)  # [N, 1]
            gaussians = gaussians._replace(
                mean_vectors=xyz * scale,
                singular_values=gaussians.singular_values * scale1,
            )
            LOGGER.info(
                "🔧 sharp ply radius log-compress: thr=%.2f compressed=%d/%d",
                thr,
                int(mask.sum().item()),
                int(mask.numel()),
            )

    # Final sanity before save_ply(): save_ply computes quantiles on disparity,
    # which requires at least one Gaussian.
    n2 = int(gaussians.mean_vectors.shape[1])
    assert n2 > 0, f"[{input_path.stem}] before save_ply got 0 Gaussians."
    finite_ratio = float(torch.isfinite(gaussians.mean_vectors).all(dim=-1).float().mean().item()) if n2 > 0 else 0.0
    LOGGER.info(
        "🧪 [%s] Gaussians N=%d finite=%.2f%% z_range=%.4f..%.4f",
        input_path.stem,
        n2,
        100.0 * finite_ratio,
        float(gaussians.mean_vectors[0, :, 2].min().item()),
        float(gaussians.mean_vectors[0, :, 2].max().item()),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_ply = output_dir / f"{input_path.stem}.ply"
    save_ply(gaussians, f_px, (height, width), out_ply)
    LOGGER.info("💾 Saved %s", out_ply)

    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    sharp_depth_np = sharp_depth_hw[0, 0].detach().cpu().numpy().astype(np.float32)
    sharp_depth_np = _clip_depth_np(sharp_depth_np, depth_max)
    valid_sh = sharp_depth_np > _DEPTH_EPS
    if np.any(valid_sh):
        ps = np.percentile(sharp_depth_np[valid_sh], [1, 50, 99]).astype(np.float32)
        LOGGER.info(
            "📏 Face %s SHARP depth stats: valid=%.2f%% p01/p50/p99=%.4f/%.4f/%.4f",
            input_path.stem,
            100.0 * float(valid_sh.mean()),
            float(ps[0]),
            float(ps[1]),
            float(ps[2]),
        )
    else:
        LOGGER.warning("⚠️ Face %s SHARP depth output is empty/invalid.", input_path.stem)
    out_exr = output_dir / f"{input_path.stem}_sharp_depth.exr"
    if not cv2.imwrite(str(out_exr), sharp_depth_np):
        raise RuntimeError(f"Failed to write SHARP depth EXR: {out_exr}")
    LOGGER.info("💾 Saved %s", out_exr)
    return out_ply


def predict_pano_depth_only(
    *,
    image_rgb: np.ndarray,
    depth_hint: np.ndarray | None,
    depth_max: float | None,
    predictor: torch.nn.Module | None = None,
    checkpoint_path: Path | None = None,
    device: str = "default",
) -> np.ndarray:
    """Predict SHARP aligned depth for a panorama-sized RGB+depth pair.

    This runs SHARP monodepth + depth_alignment branch only (no PLY/unprojection).
    Output shape matches input ``image_rgb`` spatial size. Values are **metric depth in meters**.
    """
    if predictor is None:
        pred = build_predictor(checkpoint_path, device)
        dev = next(pred.parameters()).device
    else:
        pred = predictor
        dev = next(pred.parameters()).device

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"image_rgb must be HxWx3, got shape {image_rgb.shape}")
    if depth_hint is not None and depth_hint.ndim != 2:
        raise ValueError(f"depth_hint must be HxW when provided, got shape {depth_hint.shape}")

    h, w = image_rgb.shape[:2]
    image_pt = torch.from_numpy(image_rgb.astype(np.float32)).to(dev).permute(2, 0, 1) / 255.0
    depth_pt = None
    if depth_hint is not None:
        depth_pt = torch.from_numpy(_clip_depth_np(depth_hint.astype(np.float32), depth_max)).to(dev)

    internal_shape = (1536, 1536)
    image_resized = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )
    # Panorama has no single perspective intrinsics; use a stable nominal factor.
    disparity_factor = torch.tensor([0.5], dtype=torch.float32, device=dev)

    with torch.no_grad():
        md_out = pred.monodepth_model(image_resized)
        md_disp = md_out.disparity
        df_exp = disparity_factor[:, None, None, None]
        monodepth = df_exp / md_disp.clamp(min=1e-4, max=1e4)
        depth_for_alignment = None
        if depth_pt is not None:
            depth_resized = F.interpolate(
                depth_pt[None, None],
                size=(internal_shape[1], internal_shape[0]),
                mode="bilinear",
                align_corners=True,
            )
            valid_depth_mask = _valid_depth_mask_torch(depth_resized, depth_max, _DEPTH_EPS)
            depth_for_alignment = torch.where(valid_depth_mask > 0.5, depth_resized, monodepth[:, 0:1])
        monodepth_aligned, _ = pred.depth_alignment(
            monodepth,
            depth_for_alignment,
            md_out.decoder_features,
        )
        sharp_depth_hw = F.interpolate(
            monodepth_aligned,
            size=(h, w),
            mode="bilinear",
            align_corners=True,
        )
    out = sharp_depth_hw[0, 0].detach().cpu().numpy().astype(np.float32)
    return _clip_depth_np(out, depth_max)
