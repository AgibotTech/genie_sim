# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""CLI: ``geniesim_world create`` / ``geniesim_world debug``."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import click
import cv2
import numpy as np
import torch
from PIL import Image
from sharp.utils import logging as logging_utils

from geniesim_world.utils.cubes import (
    FACE_NAMES,
    _well_defined_depth_np,
    gen_cubes,
    resize_depth_map,
    save_cube_data,
)
from geniesim_world.utils.da360_depth import (
    clamp_da360_depth,
    estimate_depth_with_da360,
    export_da360_pred_results,
)
from geniesim_world.utils.pano_fuse import fuse_pano_depth as fuse_pano_depth_lap_gauss
from geniesim_world.utils.merge import merge_gaussians_from_ply_paths
from geniesim_world.predict_run import build_predictor, predict_one_image, predict_pano_depth_only
from geniesim_world.cli_debug import debug_cli

LOGGER = logging.getLogger(__name__)

_PANO_DEPTH_INTERP = {
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos4": cv2.INTER_LANCZOS4,
}


def _depth_viz_scale(d: np.ndarray, depth_max: float | None) -> float:
    """Upper bound for uint16 debug PNG scaling; p99/max when ``depth_max`` is unset."""
    if depth_max is not None:
        return float(depth_max)
    x = np.asarray(d, dtype=np.float32)
    m = x[np.isfinite(x) & (x > 1e-9)]
    if m.size == 0:
        return 1.0
    return max(float(np.percentile(m, 99.0)), float(np.max(m)), 1e-6)


def _log_depth_stats(name: str, d: np.ndarray, depth_max: float | None) -> None:
    x = np.asarray(d, dtype=np.float32)
    valid = _well_defined_depth_np(x, eps=1e-3, depth_max=depth_max)
    if np.any(valid):
        p = np.percentile(x[valid], [1, 50, 99]).astype(np.float32)
        LOGGER.info(
            "📊 %s stats: shape=%s valid=%.2f%% p01/p50/p99=%.4f/%.4f/%.4f min=%.4f max=%.4f",
            name,
            tuple(x.shape),
            100.0 * float(valid.mean()),
            float(p[0]),
            float(p[1]),
            float(p[2]),
            float(np.min(x[valid])),
            float(np.max(x[valid])),
        )
    else:
        LOGGER.warning("⚠️ %s stats: shape=%s has no valid depth pixels.", name, tuple(x.shape))


def _robust_clip_depth_map(
    d: np.ndarray, depth_max: float | None, *, p_hi: float = 99.5, expand: float = 1.25
) -> np.ndarray:
    x = np.asarray(d, dtype=np.float32)
    valid = _well_defined_depth_np(x, eps=1e-3, depth_max=depth_max)
    if not np.any(valid):
        return x
    p50, ph = np.percentile(x[valid], [50, p_hi]).astype(np.float32)
    if depth_max is not None:
        clip_hi = min(float(depth_max), max(float(ph) * float(expand), float(p50) * 3.0))
        clip_hi = min(clip_hi, float(depth_max) * 0.99)
    else:
        clip_hi = max(float(ph) * float(expand), float(p50) * 3.0)
    out = x.copy()
    out[valid] = np.clip(x[valid], 0.0, clip_hi).astype(np.float32)
    LOGGER.info(
        "✂️ Pano robust clip: p50=%.4f p%.1f=%.4f -> clip_hi=%.4f",
        float(p50),
        float(p_hi),
        float(ph),
        float(clip_hi),
    )
    return out


def _align_da360_relative_for_fuse(
    da360_rel: np.ndarray,
    sharp_m: np.ndarray,
    depth_max: float | None,
    *,
    eps: float = 5e-5,
) -> np.ndarray:
    """Scale DA360 relative pano depth into SHARP's meter range (median ratio) so laplacian fuse is defined."""
    da = np.asarray(da360_rel, dtype=np.float32)
    sh = np.asarray(sharp_m, dtype=np.float32)
    ok = _well_defined_depth_np(sh, eps=eps, depth_max=depth_max) & (da > eps)
    if not np.any(ok):
        LOGGER.warning("Fuse: no valid overlap for DA360/SHARP median alignment; fusing may be ill-conditioned.")
        out0 = np.maximum(da, 0.0).astype(np.float32)
        return np.clip(out0, 0.0, float(depth_max)).astype(np.float32) if depth_max is not None else out0
    ratios = sh[ok] / np.maximum(da[ok], eps)
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if ratios.size == 0:
        out0 = np.maximum(da, 0.0).astype(np.float32)
        return np.clip(out0, 0.0, float(depth_max)).astype(np.float32) if depth_max is not None else out0
    scale = float(np.median(ratios))
    out = np.maximum(da * scale, 0.0).astype(np.float32)
    if depth_max is not None:
        out = np.clip(out, 0.0, float(depth_max)).astype(np.float32)
    LOGGER.info(
        "🧲 Fuse: DA360 relative aligned to SHARP meters via median(ratio) scale=%.6f",
        scale,
    )
    return out


def _world_model_repo_root() -> Path:
    """Parent of ``geniesim_world/`` (the ``world_model`` checkout that contains ``external/``)."""
    # geniesim_world/src/geniesim_world/cli_pano.py -> parents[3] == world_model
    return Path(__file__).resolve().parents[3]


def _default_realesrgan_bin() -> Path | None:
    """Locate the bundled realesrgan-ncnn-vulkan binary."""
    repo_root = _world_model_repo_root()
    cand = repo_root / "external" / "realesrgan-ncnn-vulkan-20220424-ubuntu" / "realesrgan-ncnn-vulkan"
    if cand.is_file() and os.access(cand, os.X_OK):
        return cand.resolve()
    env = os.environ.get("GENIESIM_REALESRGAN_BIN")
    if env:
        p = Path(env).expanduser()
        return p.resolve() if p.is_file() else None
    return None


def _run_realesrgan_x4plus(input_path: Path, output_path: Path, *, bin_path: Path) -> None:
    """Run: realesrgan-ncnn-vulkan -n realesrgan-x4plus -i <in> -o <out>."""
    cmd = [
        str(bin_path),
        "-n",
        "realesrgan-x4plus",
        "-i",
        str(input_path),
        "-o",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, cwd=str(bin_path.parent))


def _default_da360_root() -> Path | None:
    """Resolve DA360 checkout: env, then ``<world_model>/external/DA360``, then ``cwd/external/DA360``."""
    env = os.environ.get("GENIESIM_DA360_ROOT")
    if env:
        p = Path(env).expanduser()
        return p.resolve() if p.is_dir() else None
    repo_cand = _world_model_repo_root() / "external" / "DA360"
    if repo_cand.is_dir():
        return repo_cand.resolve()
    cwd_cand = Path.cwd() / "external" / "DA360"
    if cwd_cand.is_dir():
        return cwd_cand.resolve()
    return None


@click.group()
def cli() -> None:
    """Panorama → DA360 depth → cubemap → SHARP per face → merged PLY."""


cli.add_command(debug_cli)


@cli.command("create")
@click.option(
    "-p",
    "--panorama",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Equirectangular RGB panorama (e.g. PNG/JPG).",
)
@click.option(
    "--da360-root",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="DA360 repo root (contains networks/). Default: $GENIESIM_DA360_ROOT or <repo>/external/DA360 (see README).",
)
@click.option(
    "--da360-checkpoint",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="DA360 weights (.pth). Default: <da360-root>/DA360_large.pth or $GENIESIM_DA360_CHECKPOINT.",
)
@click.option(
    "-o",
    "--work-dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Output base directory; run folder is <work-dir>/<input-stem>/ (e.g. -o . -p foo/bar.png → ./bar/).",
)
@click.option(
    "-c",
    "--checkpoint-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
)
@click.option(
    "--depth-max",
    type=float,
    default=None,
    show_default=False,
    help=(
        "Optional upper clip (meters) for metric depth, fuse/align, and debug PNG scaling. "
        "Default: no global cap (avoids piling geometry at a fixed far plane). "
        "Required for legacy uint16 PNG cube depth. DA360 uses relative units regardless."
    ),
)
@click.option("--device", type=str, default="default", help="Torch device for DA360 and SHARP.")
@click.option("--no-depth-gt-init", is_flag=True, help="Stock SHARP (ignore depth_gt override).")
@click.option(
    "--no-ndc-frustum-mask",
    is_flag=True,
    help=(
        "Do not cull Gaussians outside NDC xy in [-margin,margin] after depth-guided SHARP. "
        "Sky-heavy cube faces can lose most splats to this mask; try this flag if a face PLY looks empty."
    ),
)
@click.option(
    "--no-ply-frustum-cull",
    is_flag=True,
    help=(
        "Do not remove SHARP PLY Gaussians outside the widened cube-face frustum (see --ply-frustum-fov-deg); "
        "default cull reduces neighbor-face leakage from far-out-of-FOV outliers."
    ),
)
@click.option(
    "--ply-frustum-fov-deg",
    type=float,
    default=91.0,
    show_default=True,
    help=(
        "Full horizontal/vertical FOV (deg) used for PLY frustum cull vs nominal 90° cube face; "
        "slightly >90 (default 91) softens seams. Ignored with --no-ply-frustum-cull."
    ),
)
@click.option(
    "--ply-frustum-margin-px",
    type=float,
    default=0.0,
    show_default=True,
    help="Optional pixel inset on reference image edges before frustum tangents (tightens cull).",
)
@click.option("--skip-sharp", is_flag=True, help="Only DA360 + cubemap export (no SHARP).")
@click.option("--skip-merge", is_flag=True, help="Run SHARP per face but skip merged PLY.")
@click.option(
    "--pano-depth-engine",
    type=click.Choice(["da360", "sharp", "both", "fuse"]),
    default="da360",
    show_default=True,
    help="Pano depth engine: da360(old), sharp(DA360 skipped), both(da360 source + sharp pano diff), fuse(DA360+SHARP fusion).",
)
@click.option(
    "--pano-depth-fuse-method",
    type=click.Choice(["laplacian", "gaussian"]),
    default="laplacian",
    show_default=True,
    help="Fuse engine only: fusion method for DA360 + SHARP pano depth.",
)
@click.option(
    "--pano-depth-fuse-sharp-weight",
    type=float,
    default=0.6,
    show_default=True,
    help="Fuse engine only: SHARP contribution weight (0~1).",
)
@click.option(
    "--pano-depth-fuse-levels",
    type=int,
    default=4,
    show_default=True,
    help="Fuse engine only: pyramid levels for laplacian fusion.",
)
@click.option(
    "--pano-depth-engine-debug",
    is_flag=True,
    help="Save extra engine debug artifacts (e.g. DA360 exports for da360/both).",
)
@click.option(
    "--sharp-ply-radius-cap",
    type=float,
    default=40.0,
    show_default=True,
    help=(
        "Reduce SHARP per-face PLY far-point drift: when |xyz| exceeds threshold "
        "(= this value), log-compress the distance (no hard cutoff) and scale splat sizes accordingly. "
        "Set 0 to disable."
    ),
)
@click.option(
    "--super-sample",
    type=click.Choice(["realesrgan-x4plus"]),
    default=None,
    show_default=False,
    help="Upsample panorama with realesrgan-x4plus (x4) before DA360 / cubemap / SHARP.",
)
@click.option(
    "--pano-depth-interpolation",
    type=click.Choice(["linear", "cubic", "lanczos4"]),
    default="linear",
    show_default=True,
    help="OpenCV interpolation for resizing depth/disparity (see --pano-depth-resize-space).",
)
@click.option(
    "--pano-depth-resize-space",
    type=click.Choice(["inverse", "depth"]),
    default="inverse",
    show_default=True,
    help="inverse: resize float depth in 1/depth (disparity) space for metric consistency; depth: resize depth values directly.",
)
@click.option(
    "--cube-face-size",
    type=click.IntRange(min=64, max=4096),
    default=1536,
    show_default=True,
    help="Cube face width/height (px); must match SHARP internal face size used on disk.",
)
@click.option(
    "--cube-depth-refine",
    type=click.Choice(["none", "bilateral"]),
    default="bilateral",
    show_default=True,
    help="Light inverse-depth bilateral on each cube face before writing EXR (reduces jagged alias).",
)
@click.option("-v", "--verbose", is_flag=True)
def create(
    panorama: Path,
    da360_root: Path | None,
    da360_checkpoint: Path | None,
    work_dir: Path,
    checkpoint_path: Path | None,
    depth_max: float | None,
    device: str,
    no_depth_gt_init: bool,
    no_ndc_frustum_mask: bool,
    no_ply_frustum_cull: bool,
    ply_frustum_fov_deg: float,
    ply_frustum_margin_px: float,
    skip_sharp: bool,
    skip_merge: bool,
    pano_depth_engine: str,
    pano_depth_fuse_method: str,
    pano_depth_fuse_sharp_weight: float,
    pano_depth_fuse_levels: int,
    pano_depth_engine_debug: bool,
    sharp_ply_radius_cap: float,
    super_sample: str | None,
    pano_depth_interpolation: str,
    pano_depth_resize_space: str,
    cube_face_size: int,
    cube_depth_refine: str,
    verbose: bool,
) -> None:
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    da360_repo = None
    if pano_depth_engine in ("da360", "both", "fuse"):
        da360_repo = Path(da360_root).resolve() if da360_root else _default_da360_root()
        if da360_repo is None:
            raise click.ClickException(
                "Set --da360-root to your DA360 clone (e.g. .../external/DA360) or export GENIESIM_DA360_ROOT."
            )
        if da360_checkpoint is None:
            env_ckpt = os.environ.get("GENIESIM_DA360_CHECKPOINT")
            if env_ckpt:
                da360_checkpoint = Path(env_ckpt).expanduser()
            else:
                da360_checkpoint = da360_repo / "DA360_large.pth"
        else:
            da360_checkpoint = Path(da360_checkpoint).resolve()

    panorama = Path(panorama).resolve()
    output_base = Path(work_dir).resolve()
    work_dir = output_base / panorama.stem
    work_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("📂 Output directory: %s", work_dir)

    cube_dir = work_dir / "cubes"
    sharp_dir = work_dir / "sharp"
    merged_ply = work_dir / "merged_gaussians.ply"

    panorama_img = Image.open(panorama).convert("RGB")
    panorama_img.save(work_dir / "input.png", format="PNG")
    LOGGER.info("🖼️ Saved input copy as %s", work_dir / "input.png")

    if super_sample == "realesrgan-x4plus":
        ss_path = work_dir / f"{panorama.stem}.4x.png"
        bin_path = _default_realesrgan_bin()
        if bin_path is None:
            raise click.ClickException(
                "Could not find realesrgan-ncnn-vulkan binary. Expected "
                "`external/realesrgan-ncnn-vulkan-20220424-ubuntu/realesrgan-ncnn-vulkan`, "
                "or set GENIESIM_REALESRGAN_BIN."
            )
        _run_realesrgan_x4plus(panorama, ss_path, bin_path=bin_path)
        panorama_img = Image.open(ss_path).convert("RGB")
        LOGGER.info("✨ Saved super-sampled panorama (realesrgan) as %s", ss_path)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "default" else torch.device(device)

    input_w, input_h = panorama_img.size
    pano_interp = _PANO_DEPTH_INTERP[pano_depth_interpolation]
    predictor: torch.nn.Module | None = None
    da360_depth_np: np.ndarray | None = None

    if pano_depth_engine in ("da360", "both", "fuse"):
        assert da360_repo is not None
        LOGGER.info("🧠 DA360 depth: repo=%s ckpt=%s", da360_repo, da360_checkpoint)
        # DA360 outputs one continuous ERP relative depth map; resize to pano resolution only.
        pred_depth_native = estimate_depth_with_da360(
            panorama_img,
            da360_repo_root=da360_repo,
            checkpoint_path=da360_checkpoint,
            device=dev,
        )
        pred_depth_input = resize_depth_map(
            pred_depth_native,
            (input_w, input_h),
            interpolation=pano_interp,
            space=pano_depth_resize_space,
            anti_alias=True,
            depth_max=None,
        )
        pred_depth_input_np = clamp_da360_depth(pred_depth_input)
        LOGGER.info("DA360 depth: resized min-normalized output, clamped to [1e-4, 1e4] like SHARP disparity band.")
        da360_depth_np = pred_depth_input_np
        _log_depth_stats("DA360 pano depth (resized)", da360_depth_np, depth_max)
    else:
        predictor = build_predictor(checkpoint_path, device)
        pred_depth_native = np.empty((0, 0), dtype=np.float32)
        pred_depth_input_np = predict_pano_depth_only(
            image_rgb=np.array(panorama_img),
            depth_hint=None,
            depth_max=depth_max,
            predictor=predictor,
            checkpoint_path=None,
            device=device,
        ).astype(np.float32)
        LOGGER.info("🤖 SHARP pano depth engine enabled (DA360 skipped).")
        _log_depth_stats("SHARP pano depth (engine output)", pred_depth_input_np, depth_max)
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    input_depth_exr = work_dir / "input_depth.exr"
    if not cv2.imwrite(str(input_depth_exr), pred_depth_input_np):
        raise RuntimeError(f"Failed to write resized pano depth EXR: {input_depth_exr}")
    LOGGER.info(
        "💾 Saved input depth %s at input resolution %sx%s (space=%s interpolation=%s anti_alias=on)",
        input_depth_exr,
        input_w,
        input_h,
        pano_depth_resize_space,
        pano_depth_interpolation,
    )

    # Build cubemap directly from the full current panorama resolution
    # (original input or Real-ESRGAN super-sampled image).
    W, H = panorama_img.size
    pred_depth = pred_depth_input_np
    LOGGER.info(
        "🌍 Using full pano ERP for cubemap projection: %sx%s (depth native %s space=%s interpolation=%s)",
        W,
        H,
        pred_depth_native.shape if pred_depth_native.size else (input_h, input_w),
        pano_depth_resize_space,
        pano_depth_interpolation,
    )

    pred_depth_np = pred_depth_input_np
    sharp_pano_depth_np: np.ndarray | None = None

    if pano_depth_engine in ("sharp", "both", "fuse"):
        if predictor is None:
            predictor = build_predictor(checkpoint_path, device)
        sharp_pano_depth_np = predict_pano_depth_only(
            image_rgb=np.array(panorama_img),
            depth_hint=da360_depth_np,
            depth_max=depth_max,
            predictor=predictor,
            checkpoint_path=None,
            device=device,
        ).astype(np.float32)
        sharp_pano_exr = work_dir / "sharp_pano_depth.exr"
        if not cv2.imwrite(str(sharp_pano_exr), sharp_pano_depth_np):
            raise RuntimeError(f"Failed to write SHARP pano depth EXR: {sharp_pano_exr}")
        _log_depth_stats("SHARP pano depth (comparison)", sharp_pano_depth_np, depth_max)
        if da360_depth_np is not None:
            diff_abs = np.abs(sharp_pano_depth_np - da360_depth_np).astype(np.float32)
            diff_exr = work_dir / "sharp_vs_da360_pano_absdiff.exr"
            if not cv2.imwrite(str(diff_exr), diff_abs):
                raise RuntimeError(f"Failed to write pano depth absdiff EXR: {diff_exr}")
            diff_scale = _depth_viz_scale(diff_abs, depth_max)
            diff_u16 = np.clip(diff_abs / diff_scale * 65535.0, 0, 65535).astype(np.uint16)
            diff_png = work_dir / "sharp_vs_da360_pano_absdiff.png"
            if not cv2.imwrite(str(diff_png), diff_u16):
                raise RuntimeError(f"Failed to write pano depth absdiff PNG: {diff_png}")
            LOGGER.info(
                "✅ Saved SHARP pano depth %s and absdiff %s / %s",
                sharp_pano_exr,
                diff_exr,
                diff_png,
            )
            _log_depth_stats("SHARP vs DA360 pano absdiff", diff_abs, depth_max)
        else:
            LOGGER.info("✅ Saved SHARP pano depth %s (DA360 skipped, absdiff not generated)", sharp_pano_exr)

    if pano_depth_engine == "fuse":
        if da360_depth_np is None or sharp_pano_depth_np is None:
            raise RuntimeError("Fuse engine requires both DA360 and SHARP pano depth")
        da360_for_fuse = _align_da360_relative_for_fuse(da360_depth_np, sharp_pano_depth_np, depth_max)
        fused_pano_depth_np = fuse_pano_depth_lap_gauss(
            da360_for_fuse,
            sharp_pano_depth_np,
            method=pano_depth_fuse_method,
            sharp_weight=float(pano_depth_fuse_sharp_weight),
            levels=int(pano_depth_fuse_levels),
            sigma=2.0,
            depth_max=depth_max,
        ).astype(np.float32)
        # Sanity checks around robust clip: we expect fuse invalids to be NaN (not 0/Inf).
        before_nan_mask = np.isnan(fused_pano_depth_np)
        non_finite = ~np.isfinite(fused_pano_depth_np)
        if np.any(non_finite):
            assert np.all(np.isnan(fused_pano_depth_np[non_finite])), "fused_pano_depth has Inf/other non-finite values"
        fused_pano_depth_np = _robust_clip_depth_map(fused_pano_depth_np, depth_max, p_hi=99.5, expand=1.20)
        after_non_finite = ~np.isfinite(fused_pano_depth_np)
        if np.any(after_non_finite):
            assert np.all(
                np.isnan(fused_pano_depth_np[after_non_finite])
            ), "robust clip produced Inf/other non-finite values"
        assert np.array_equal(
            before_nan_mask, np.isnan(fused_pano_depth_np)
        ), "robust clip should not change NaN locations"
        fused_exr = work_dir / "fused_pano_depth.exr"
        if not cv2.imwrite(str(fused_exr), fused_pano_depth_np):
            raise RuntimeError(f"Failed to write fused pano depth EXR: {fused_exr}")
        _log_depth_stats("FUSED pano depth", fused_pano_depth_np, depth_max)
        pred_depth_np = fused_pano_depth_np
        pred_depth = pred_depth_np
        if not cv2.imwrite(str(input_depth_exr), pred_depth_np):
            raise RuntimeError(f"Failed to overwrite selected input depth EXR: {input_depth_exr}")
        LOGGER.info(
            "🧩 Using pano depth engine: fuse (%s, sharp_weight=%.3f, levels=%d)",
            pano_depth_fuse_method,
            float(pano_depth_fuse_sharp_weight),
            int(pano_depth_fuse_levels),
        )
    elif pano_depth_engine == "sharp":
        pred_depth_np = pred_depth_input_np
        pred_depth = pred_depth_np
        if not cv2.imwrite(str(input_depth_exr), pred_depth_np):
            raise RuntimeError(f"Failed to overwrite selected input depth EXR: {input_depth_exr}")
        LOGGER.info("🎯 Using pano depth engine: sharp")
    elif pano_depth_engine == "both":
        pred_depth_np = da360_depth_np if da360_depth_np is not None else pred_depth_input_np
        pred_depth = pred_depth_np
        if not cv2.imwrite(str(input_depth_exr), pred_depth_np):
            raise RuntimeError(f"Failed to overwrite selected input depth EXR: {input_depth_exr}")
        LOGGER.info("🔀 Using pano depth engine: both (DA360 as source, SHARP pano exported for diff)")
    else:
        LOGGER.info("🧭 Using pano depth engine: da360")

    pred_depth = np.maximum(np.asarray(pred_depth, dtype=np.float32), 0.0)
    if depth_max is not None:
        pred_depth = np.clip(pred_depth, 0.0, float(depth_max))

    if pano_depth_engine_debug and da360_depth_np is not None:
        # Persist DA360 panorama depth for debugging / downstream usage.
        # - .npy keeps float precision (no scaling).
        # - .png uses uint16 scaling convention for quick visualization.
        np.save(work_dir / "da360_depth.npy", da360_depth_np)
        da_scale = _depth_viz_scale(da360_depth_np, depth_max)
        depth_u16 = np.clip(da360_depth_np / da_scale * 65535.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(work_dir / "da360_depth.png"), depth_u16)
        LOGGER.info("📌 Saved DA360 depth to %s/da360_depth.npy and da360_depth.png", work_dir)

        # Also export DA360-style visualization / artifacts.
        # - <stem>_depth_pred_DA360.exr (float depth)
        # - <stem>_depth_pred_DA360.jpg (KITTI colormap visualization)
        # - <stem>_pc_pred_DA360.ply (spherical point cloud)
        rgb01 = np.asarray(panorama_img, dtype=np.float32) / 255.0
        export_da360_pred_results(work_dir, panorama.stem, rgb01, da360_depth_np)
    elif pano_depth_engine_debug and da360_depth_np is None:
        LOGGER.warning("⚠️ Ignoring --pano-depth-engine-debug because DA360 is skipped in sharp mode.")

    pano_mask = np.ones((H, W), dtype=np.uint8)
    original_ext = np.eye(4, dtype=np.float32)

    cube_images, cube_depths, cube_masks, cube_intr, cube_extr = gen_cubes(
        np.array(panorama_img),
        pred_depth,
        pano_mask,
        original_ext,
        out_size=cube_face_size,
        depth_max=depth_max,
    )
    cube_depth_np = cube_depths.numpy().squeeze(-1).astype(np.float32)
    for i, face in enumerate(FACE_NAMES):
        _log_depth_stats(f"Cube depth before save [{face}]", cube_depth_np[i], depth_max)
    cube_dir.mkdir(parents=True, exist_ok=True)
    save_cube_data(
        cube_images,
        cube_depths,
        cube_masks,
        cube_intr,
        cube_extr,
        str(cube_dir),
        depth_max=depth_max,
        cube_depth_refine=cube_depth_refine,
    )
    LOGGER.info("📦 Wrote cubemap data under %s", cube_dir)

    if skip_sharp:
        LOGGER.info("🏁 Done (--skip-sharp).")
        return

    sharp_dir.mkdir(parents=True, exist_ok=True)
    if predictor is None:
        predictor = build_predictor(checkpoint_path, device)

    def _run_sharp_on_cube_dir() -> list[Path]:
        out: list[Path] = []
        for face in FACE_NAMES:
            face_dir = cube_dir / face
            img_path = face_dir / f"{face}_image.png"
            depth_path = face_dir / f"{face}_depth.exr"
            intr_path = face_dir / f"{face}_intr.npy"
            extr_path = face_dir / f"{face}_extr.npy"
            ply = predict_one_image(
                input_path=img_path,
                depth_path=depth_path,
                intr_path=intr_path,
                extr_path=extr_path,
                output_dir=sharp_dir,
                checkpoint_path=None,
                depth_max=depth_max,
                device=device,
                no_depth_gt_init=no_depth_gt_init,
                predictor=predictor,
                ndc_frustum_mask=not no_ndc_frustum_mask,
                ply_frustum_cull=not no_ply_frustum_cull,
                ply_frustum_fov_deg=float(ply_frustum_fov_deg),
                ply_frustum_margin_px=float(ply_frustum_margin_px),
                sharp_ply_radius_cap=float(sharp_ply_radius_cap),
            )
            out.append(ply)
        return out

    ply_paths = _run_sharp_on_cube_dir()

    if skip_merge:
        LOGGER.info("🧱 Per-face PLY under %s (--skip-merge).", sharp_dir)
        return

    merge_gaussians_from_ply_paths(ply_paths, merged_ply)
    LOGGER.info("🧬 Merged PLY: %s", merged_ply)


if __name__ == "__main__":
    cli()
