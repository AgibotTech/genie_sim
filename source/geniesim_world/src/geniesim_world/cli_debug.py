# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``geniesim_world debug`` — visualize EXR depth maps and compare DA360 vs SHARP."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import click
import cv2
import numpy as np
import torch
from sharp.utils import logging as logging_utils
from sharp.utils.vis import METRIC_DEPTH_MAX_CLAMP_METER, colorize_depth

from geniesim_world.utils.cubes import FACE_NAMES

LOGGER = logging.getLogger(__name__)


def _read_exr_float(path: Path) -> np.ndarray:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read EXR: {path}")
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.float32)


def _auto_depth_scale(d: np.ndarray, eps: float = 1e-9) -> float:
    x = np.asarray(d, dtype=np.float32)
    m = x[np.isfinite(x) & (x > eps)]
    if m.size == 0:
        return 1.0
    return max(float(np.percentile(m, 99.0)), float(np.max(m)), eps)


def _depth_to_png_u16(depth: np.ndarray, depth_max: float | None) -> np.ndarray:
    d = np.maximum(depth.astype(np.float32), 0.0)
    if depth_max is not None:
        d = np.clip(d, 0.0, float(depth_max))
        s = float(depth_max)
    else:
        s = _auto_depth_scale(d)
    return np.clip(d / s * 65535.0, 0, 65535).astype(np.uint16)


def _colorize_depth_meters_bgr(depth: np.ndarray, val_max: float) -> np.ndarray:
    """Same turbo colormap as ``sharp/cli/predict.py`` (`colorize_depth` + `val_max`)."""
    d = np.maximum(depth.astype(np.float32), 0.0)
    depth_pt = torch.from_numpy(d)[None, None]
    colorized = colorize_depth(depth_pt, float(val_max))
    rgb = colorized[0].permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _color_val_max(depth_max: float | None) -> float:
    """SHARP predict uses ``METRIC_DEPTH_MAX_CLAMP_METER`` when no custom cap; else CLI ``--depth-max``."""
    if depth_max is not None:
        return float(depth_max)
    return float(METRIC_DEPTH_MAX_CLAMP_METER)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _write_exr_as_png(exr_path: Path, png_path: Path, depth_max: float | None) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    arr = _read_exr_float(exr_path)
    u16 = _depth_to_png_u16(arr, depth_max)
    if not cv2.imwrite(str(png_path), u16):
        raise RuntimeError(f"Failed to write PNG: {png_path}")
    color_path = png_path.with_name(f"{png_path.stem}_color.png")
    v_max = _color_val_max(depth_max)
    color_bgr = _colorize_depth_meters_bgr(arr, v_max)
    if not cv2.imwrite(str(color_path), color_bgr):
        raise RuntimeError(f"Failed to write color PNG: {color_path}")
    LOGGER.info(
        "  📎 %s  (min=%.4f max=%.4f)",
        png_path,
        float(np.min(arr)),
        float(np.max(arr)),
    )
    LOGGER.info(
        "  🎨 %s  (SHARP turbo, val_max=%.2f m, same as sharp predict)",
        color_path,
        v_max,
    )


def _resize_to_match(a: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if a.shape[0] == h and a.shape[1] == w:
        return a
    return cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)


def _flat_png_basename_from_exr(run_dir: Path, exr_path: Path) -> str:
    """Flatten ``run_dir``-relative EXR path to a single filename for ``debug/*.png``."""
    rel = exr_path.relative_to(run_dir)
    return "_".join(rel.with_suffix("").parts) + ".png"


def run_debug(run_dir: Path, depth_max: float | None, *, verbose: bool) -> None:
    """Analyze a ``geniesim_world create`` output folder; write under ``<run_dir>/debug/``."""
    t0 = time.perf_counter()
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise click.ClickException(f"Not a directory: {run_dir}")

    debug_root = run_dir / "debug"
    debug_root.mkdir(parents=True, exist_ok=True)

    LOGGER.info("📂 Run directory: %s", run_dir)
    LOGGER.info("📁 Debug output (flat): %s", debug_root)
    if depth_max is not None:
        LOGGER.info(
            "📏 Depth uint16 PNG scale: --depth-max=%s m  |  🎨 turbo val_max: %s m",
            depth_max,
            depth_max,
        )
    else:
        LOGGER.info(
            "📏 Depth uint16 PNG: auto per EXR (p99 vs max)  |  🎨 turbo val_max: %.1f m (METRIC_DEPTH_MAX_CLAMP_METER, like sharp predict)",
            METRIC_DEPTH_MAX_CLAMP_METER,
        )

    sharp_dir = run_dir / "sharp"
    cubes_dir = run_dir / "cubes"

    # --- Per-face |DA360 − SHARP| → debug/*.exr + *.png ---
    summary_lines: list[str] = []
    for face in FACE_NAMES:
        da360_exr = cubes_dir / face / f"{face}_depth.exr"
        sharp_exr = sharp_dir / f"{face}_image_sharp_depth.exr"
        if not da360_exr.is_file():
            LOGGER.warning("⚠️ Missing DA360 depth (skip diff): %s", da360_exr)
            continue
        if not sharp_exr.is_file():
            LOGGER.warning("⚠️ Missing SHARP depth (skip diff): %s", sharp_exr)
            continue

        LOGGER.info("🔍 Diff [%s] 📥 DA360: %s", face, da360_exr)
        LOGGER.info("🔍 Diff [%s] 📥 SHARP: %s", face, sharp_exr)

        d_da = _read_exr_float(da360_exr)
        d_sh = _read_exr_float(sharp_exr)
        if verbose:
            LOGGER.debug(
                "  DA360 shape=%s dtype=%s min=%.6f max=%.6f",
                d_da.shape,
                d_da.dtype,
                float(np.min(d_da)),
                float(np.max(d_da)),
            )
            LOGGER.debug(
                "  SHARP shape=%s dtype=%s min=%.6f max=%.6f",
                d_sh.shape,
                d_sh.dtype,
                float(np.min(d_sh)),
                float(np.max(d_sh)),
            )

        if d_da.shape != d_sh.shape:
            LOGGER.info(
                "  📐 Resizing DA360 %s → %s to match SHARP",
                d_da.shape,
                d_sh.shape,
            )
            d_da = _resize_to_match(d_da, (d_sh.shape[0], d_sh.shape[1]))
        absdiff = np.abs(d_da - d_sh).astype(np.float32)

        stem = f"diff_da360_vs_sharp_{face}"
        diff_exr = debug_root / f"{stem}.exr"
        if not cv2.imwrite(str(diff_exr), absdiff):
            raise RuntimeError(f"Failed to write {diff_exr}")
        LOGGER.info("  💾 Wrote %s", diff_exr)

        diff_png = debug_root / f"{stem}.png"
        _write_exr_as_png(diff_exr, diff_png, depth_max)

        mae = float(np.mean(absdiff))
        mx = float(np.max(absdiff))
        rmse = float(np.sqrt(np.mean((d_da - d_sh) ** 2)))
        scale_note = "meters" if depth_max is not None else "EXR native units"
        summary_lines.append(f"{face}: mae={mae:.6f} max_abs={mx:.6f} rmse={rmse:.6f} ({scale_note})")
        LOGGER.info("  📊 Stats: MAE=%.6f  max|Δ|=%.6f  RMSE=%.6f", mae, mx, rmse)

    summary_path = debug_root / "depth_diff_summary.txt"
    summary_path.write_text(
        "\n".join(summary_lines) + ("\n" if summary_lines else ""),
        encoding="utf-8",
    )
    LOGGER.info("📝 Wrote %s (%d faces compared)", summary_path, len(summary_lines))

    # --- Pipeline EXR → debug/<flattened>.png (skip run_dir/debug/) ---
    exr_list = [p for p in sorted(run_dir.rglob("*.exr")) if not _is_under(p, debug_root)]
    LOGGER.info(
        "🖼️ Converting %d pipeline EXR(s) (excluding %s) → flat PNGs in %s",
        len(exr_list),
        debug_root,
        debug_root,
    )

    for i, exr_path in enumerate(exr_list, start=1):
        rel = exr_path.relative_to(run_dir)
        flat_name = _flat_png_basename_from_exr(run_dir, exr_path)
        out_png = debug_root / flat_name
        LOGGER.info("🔄 [%d/%d] EXR→PNG %s → %s", i, len(exr_list), rel, flat_name)
        try:
            _write_exr_as_png(exr_path, out_png, depth_max)
        except Exception as e:
            LOGGER.warning("  ⚠️ Skip %s: %s", exr_path, e)

    elapsed = time.perf_counter() - t0
    LOGGER.info(
        "✅ Done in %.2fs — all artifacts under %s (see depth_diff_summary.txt)",
        elapsed,
        debug_root,
    )


@click.command("debug")
@click.option(
    "--dir",
    "run_dir",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    required=True,
    help="Run output directory from geniesim_world create (e.g. ./output/mansion_01).",
)
@click.option(
    "--depth-max",
    type=float,
    default=None,
    show_default=False,
    help=(
        "Optional upper depth (meters) for linear uint16 PNG; also used as turbo val_max for "
        f"*_color.png (default val_max={METRIC_DEPTH_MAX_CLAMP_METER} m when omitted, matching sharp predict)."
    ),
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Verbose logging (per-array min/max and debug-level detail).",
)
def debug_cli(run_dir: Path, depth_max: float | None, verbose: bool) -> None:
    """Write EXR visualizations and DA360 vs SHARP |diff| under <run>/debug/."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)
    if verbose:
        logging.getLogger("geniesim_world").setLevel(logging.DEBUG)
    run_debug(run_dir, depth_max, verbose=verbose)
