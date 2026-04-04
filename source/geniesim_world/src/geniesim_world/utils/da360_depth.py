# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""DA360 panorama depth (requires ``external/DA360`` on ``sys.path``).

``geniesim_world create`` uses DA360 output (min-normalized relative depth), resized to the
panorama. Depth values are **clamped** to ``[1e-4, 1e4]``, matching SHARP’s
``monodepth_disparity.clamp(min=1e-4, max=1e4)`` numeric band (applied here on depth).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Same bounds as SHARP ``md_disp.clamp(min=1e-4, max=1e4)`` (used on depth after DA360 inference).
_DA360_DEPTH_CLAMP_LO = 1e-4
_DA360_DEPTH_CLAMP_HI = 1e4


def clamp_da360_depth(d: np.ndarray) -> np.ndarray:
    """Clamp DA360 depth to ``[1e-4, 1e4]`` (SHARP disparity clamp band)."""
    return np.clip(
        np.asarray(d, dtype=np.float32),
        _DA360_DEPTH_CLAMP_LO,
        _DA360_DEPTH_CLAMP_HI,
    )


def estimate_depth_with_da360(
    panorama_img: Image.Image,
    *,
    da360_repo_root: Path,
    checkpoint_path: Path,
    device: torch.device | None = None,
) -> np.ndarray:
    """Run DA360 on an equirectangular RGB panorama.

    Returns H×W **min-normalized relative depth** ``(1/pred_disp) / min``, matching common DA360
    usage. The pipeline resamples this field to the ERP size and passes it through without a
    separate global metric rescale.
    """
    repo = Path(da360_repo_root).resolve()
    ckpt = Path(checkpoint_path).resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"DA360 repo not found: {repo}")
    if not ckpt.is_file():
        raise FileNotFoundError(f"DA360 checkpoint not found: {ckpt}")

    root_str = str(repo)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    os_environ_backup = os.environ.get("OPENCV_IO_ENABLE_OPENEXR")
    try:
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_dict = torch.load(ckpt, map_location=device)
        if "net" not in model_dict:
            model_dict["net"] = "DA360"
        if "dinov2_encoder" not in model_dict:
            model_dict["dinov2_encoder"] = "vits"
        if "height" not in model_dict:
            model_dict["height"] = 518
        if "width" not in model_dict:
            model_dict["width"] = 1036

        import networks

        Net = getattr(networks, model_dict["net"])
        model = Net(
            model_dict["height"],
            model_dict["width"],
            dinov2_encoder=model_dict["dinov2_encoder"],
        )
        model.to(device)
        model_state_dict = model.state_dict()
        model.load_state_dict({k: v for k, v in model_dict.items() if k in model_state_dict}, strict=False)
        model.eval()

        input_tensor = (
            torch.tensor(
                np.array(panorama_img.resize((model_dict["width"], model_dict["height"]))) / 255.0,
                dtype=torch.float32,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
        )

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        normalized_input = (input_tensor - mean) / std

        with torch.no_grad():
            outputs = model(normalized_input)

        pred_disp = outputs["pred_disp"].detach().cpu()
        pred_depth = 1.0 / (pred_disp + 1e-6)
        pred_depth = pred_depth[0, 0].numpy().astype(np.float32)
        pred_depth = pred_depth / float(np.maximum(np.min(pred_depth), 1e-12))
        return clamp_da360_depth(pred_depth)
    finally:
        if os_environ_backup is None:
            os.environ.pop("OPENCV_IO_ENABLE_OPENEXR", None)
        else:
            os.environ["OPENCV_IO_ENABLE_OPENEXR"] = os_environ_backup


def kitti_colormap(disparity: np.ndarray, maxval: float = -1.0) -> np.ndarray:
    """Reproduce KITTI fake colormap (copied/adapted from DA360/saver.py)."""
    if maxval < 0:
        maxval = float(np.max(disparity))

    # 8-color LUT with alpha in 4th channel (kept from original implementation).
    colormap = np.asarray(
        [
            [0, 0, 0, 114],
            [0, 0, 1, 185],
            [1, 0, 0, 114],
            [1, 0, 1, 174],
            [0, 1, 0, 114],
            [0, 1, 1, 185],
            [1, 1, 0, 114],
            [1, 1, 1, 0],
        ],
        dtype=np.float32,
    )
    weights = np.asarray(
        [
            8.771929824561404,
            5.405405405405405,
            8.771929824561404,
            5.747126436781609,
            8.771929824561404,
            5.405405405405405,
            8.771929824561404,
            0,
        ],
        dtype=np.float32,
    )
    cumsum = np.asarray(
        [0, 0.114, 0.299, 0.413, 0.587, 0.701, 0.8859999999999999, 0.9999999999999999],
        dtype=np.float32,
    )

    values = np.expand_dims(np.minimum(np.maximum(disparity / maxval, 0.0), 1.0), -1)
    bins = np.repeat(
        np.repeat(
            np.expand_dims(np.expand_dims(cumsum, axis=0), axis=0),
            disparity.shape[1],
            axis=1,
        ),
        disparity.shape[0],
        axis=0,
    )
    diffs = np.where(
        (np.repeat(values, 8, axis=-1) - bins) > 0,
        -1000,
        (np.repeat(values, 8, axis=-1) - bins),
    )
    index = np.argmax(diffs, axis=-1) - 1

    w = 1.0 - (values[:, :, 0] - cumsum[index]) * weights[index]
    colored = np.zeros((disparity.shape[0], disparity.shape[1], 3), dtype=np.float32)
    colored[:, :, 2] = w * colormap[index][:, :, 0] + (1.0 - w) * colormap[index + 1][:, :, 0]
    colored[:, :, 1] = w * colormap[index][:, :, 1] + (1.0 - w) * colormap[index + 1][:, :, 1]
    colored[:, :, 0] = w * colormap[index][:, :, 2] + (1.0 - w) * colormap[index + 1][:, :, 2]

    return (colored * np.expand_dims((disparity > 0), -1) * 255.0).astype(np.uint8)


def _write_ply_ascii(
    ply_path: Path,
    points_xyz: np.ndarray,
    colors_rgb_u8: np.ndarray,
) -> None:
    """Write a minimal ASCII PLY with float XYZ and uchar RGB."""
    n = points_xyz.shape[0]
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ply_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = points_xyz[i]
            r, g, b = colors_rgb_u8[i]
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")


def export_da360_pred_results(
    work_dir: Path,
    stem: str,
    rgb01: np.ndarray,
    pred_depth: np.ndarray,
    *,
    model_name: str = "DA360",
    ply_depth_threshold: float = 200.0,
    max_points: int | None = None,
) -> None:
    """Save DA360-like outputs: exr depth, depth_pred jpg, and pc_pred ply."""
    work_dir = Path(work_dir)
    pred_depth_np = clamp_da360_depth(pred_depth)
    h, w = pred_depth_np.shape[:2]

    if rgb01.shape[:2] != (h, w):
        raise ValueError(f"rgb01 shape {rgb01.shape} does not match depth {(h, w)}")

    # Match Saver.save_pred_samples -> depth_pred_jpg.
    depth = pred_depth_np.copy()
    valid = np.isfinite(depth) & (depth >= _DA360_DEPTH_CLAMP_LO)
    if np.any(valid):
        depth_norm = depth / float(depth[valid].max())
    else:
        depth_norm = depth
    disp = depth_norm
    disp[valid] = 1.0 / np.clip(depth_norm[valid], 1e-12, None)
    disp[~valid] = 0.0
    depth_pred_jpg = kitti_colormap(disp)

    depth_pred_jpg_path = work_dir / f"{stem}_depth_pred_{model_name}.jpg"
    import cv2

    cv2.imwrite(str(depth_pred_jpg_path), depth_pred_jpg[:, :, ::-1])  # RGB->BGR for OpenCV

    # Match Saver.save_pred_samples -> depth_pred.exr (float).
    exr_path = work_dir / f"{stem}_depth_pred_{model_name}.exr"
    old = os.environ.get("OPENCV_IO_ENABLE_OPENEXR")
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    try:
        cv2.imwrite(str(exr_path), pred_depth_np)
    finally:
        if old is None:
            os.environ.pop("OPENCV_IO_ENABLE_OPENEXR", None)
        else:
            os.environ["OPENCV_IO_ENABLE_OPENEXR"] = old

    # Match Saver.save_pred_samples -> pc_pred ply (with mask pred_depth<200).
    mask = pred_depth_np < ply_depth_threshold
    if max_points is not None:
        # Keep first subset to cap size; deterministic.
        mask_flat = mask.reshape(-1)
        idx = np.flatnonzero(mask_flat)
        if idx.shape[0] > max_points:
            mask_flat[:] = False
            mask_flat[idx[:max_points]] = True
            mask = mask_flat.reshape(h, w)

    theta = np.pi - (np.arange(h).reshape(h, 1) * np.pi / h) - (np.pi / h / 2.0)
    phi = np.arange(w).reshape(1, w) * 2.0 * np.pi / w + np.pi / w - np.pi
    # theta/phi -> [H, W]
    theta = np.repeat(theta, w, axis=1).astype(np.float32)
    phi = np.repeat(phi, h, axis=0).astype(np.float32)

    x = pred_depth_np * np.sin(theta) * np.sin(phi)
    y = pred_depth_np * np.cos(theta)
    z = pred_depth_np * np.sin(theta) * np.cos(phi)

    if mask is None:
        points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        colors = rgb01.reshape(-1, 3)
    else:
        points = np.stack([x[mask], y[mask], z[mask]], axis=1)
        colors = rgb01[mask]

    colors_u8 = (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)
    ply_path = work_dir / f"{stem}_pc_pred_{model_name}.ply"
    _write_ply_ascii(ply_path, points.astype(np.float32), colors_u8)
