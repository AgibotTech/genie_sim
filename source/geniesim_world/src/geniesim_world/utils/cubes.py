# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Equirectangular → cubemap split and disk I/O helpers."""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from einops import rearrange, repeat
from scipy.ndimage import map_coordinates

if TYPE_CHECKING:
    pass

import logging

LOGGER = logging.getLogger(__name__)


def _well_defined_depth_np(
    d: np.ndarray,
    *,
    eps: float = 1e-3,
    depth_max: float | None = None,
) -> np.ndarray:
    """Pixels usable for inverse-depth resize / bilateral refine (positive, optional upper bound)."""
    x = np.asarray(d, dtype=np.float32)
    m = x > float(eps)
    if depth_max is not None:
        m &= x < float(depth_max)
    return m


def resize_depth_map(
    depth: np.ndarray,
    dsize: tuple[int, int],
    *,
    interpolation: int = cv2.INTER_LINEAR,
    space: str = "inverse",
    eps: float = 1e-6,
    anti_alias: bool = True,
    depth_max: float | None = None,
) -> np.ndarray:
    """Resize a single-channel float depth map (e.g. from EXR or a depth network).

    Plain ``cv2.resize`` interpolates **depth** values, which is not
    perspective-consistent at discontinuities. For metric float depth, a common
    approach is to upsample **inverse depth (disparity)** and convert back.

    - ``space="inverse"`` (default): blend ``1/depth`` with a mask (``depth > eps``;
      if ``depth_max`` is set, also ``depth < depth_max``).
      Pixels with no valid inverse-depth support after resize are set to **0** (unknown).
    - ``space="depth"``: resize depth directly (for comparison or special cases).

    Output is always ``float32`` (``CV_32F``-compatible for EXR I/O).
    """
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim != 2:
        raise ValueError(f"resize_depth_map expects HxW depth, got shape {d.shape}")
    h, w = d.shape
    tw, th = dsize
    if (w, h) == (tw, th):
        return d.copy()

    sx = tw / float(w)
    sy = th / float(h)
    downsample = sx < 1.0 or sy < 1.0

    def _resize_float(img: np.ndarray, *, interp: int) -> np.ndarray:
        work = img.astype(np.float32)
        if anti_alias and downsample:
            sigma_x = max(0.0, 0.5 * (1.0 / max(sx, 1e-6) - 1.0))
            sigma_y = max(0.0, 0.5 * (1.0 / max(sy, 1e-6) - 1.0))
            if sigma_x > 1e-6 or sigma_y > 1e-6:
                work = cv2.GaussianBlur(work, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
        eff_interp = cv2.INTER_AREA if downsample else interp
        return cv2.resize(work, dsize, interpolation=eff_interp).astype(np.float32)

    if space == "depth":
        return _resize_float(d, interp=interpolation)

    if space != "inverse":
        raise ValueError(f"Unknown resize space: {space!r} (use 'inverse' or 'depth')")

    valid_m = _well_defined_depth_np(d, eps=eps, depth_max=depth_max)
    valid = valid_m.astype(np.float32)
    inv = np.zeros_like(d, dtype=np.float32)
    np.divide(1.0, d, out=inv, where=valid_m)
    inv_r = _resize_float(inv, interp=interpolation)
    w_r = _resize_float(valid, interp=cv2.INTER_LINEAR)
    inv_norm = inv_r / np.maximum(w_r, eps)
    out = np.zeros((th, tw), dtype=np.float32)
    np.divide(1.0, inv_norm, out=out, where=w_r > eps)
    out[w_r <= eps] = 0.0
    return out


class Equirec2Cube:
    def __init__(self, equ_h: int, equ_w: int, face_w: int):
        self.equ_h = equ_h
        self.equ_w = equ_w
        self.face_w = face_w

        self._xyzcube()
        self._xyz2coor()

        cosmap = 1 / np.sqrt((2 * self.grid[..., 0]) ** 2 + (2 * self.grid[..., 1]) ** 2 + 1)
        self.cosmaps = np.concatenate(6 * [cosmap], axis=1)[..., np.newaxis]

    def _xyzcube(self):
        self.xyz = np.zeros((self.face_w, self.face_w * 6, 3), np.float32)
        rng = np.linspace(-0.5, 0.5, num=self.face_w, dtype=np.float32)
        self.grid = np.stack(np.meshgrid(rng, -rng), -1)

        self.xyz[:, 0 * self.face_w : 1 * self.face_w, [0, 1]] = self.grid
        self.xyz[:, 0 * self.face_w : 1 * self.face_w, 2] = 0.5

        self.xyz[:, 1 * self.face_w : 2 * self.face_w, [2, 1]] = self.grid[:, ::-1]
        self.xyz[:, 1 * self.face_w : 2 * self.face_w, 0] = 0.5

        self.xyz[:, 2 * self.face_w : 3 * self.face_w, [0, 1]] = self.grid[:, ::-1]
        self.xyz[:, 2 * self.face_w : 3 * self.face_w, 2] = -0.5

        self.xyz[:, 3 * self.face_w : 4 * self.face_w, [2, 1]] = self.grid
        self.xyz[:, 3 * self.face_w : 4 * self.face_w, 0] = -0.5

        self.xyz[:, 4 * self.face_w : 5 * self.face_w, [0, 2]] = self.grid[::-1, :]
        self.xyz[:, 4 * self.face_w : 5 * self.face_w, 1] = 0.5

        self.xyz[:, 5 * self.face_w : 6 * self.face_w, [0, 2]] = self.grid
        self.xyz[:, 5 * self.face_w : 6 * self.face_w, 1] = -0.5

    def _xyz2coor(self):
        x, y, z = np.split(self.xyz, 3, axis=-1)
        lon = np.arctan2(x, z)
        c = np.sqrt(x**2 + z**2)
        lat = np.arctan2(y, c)

        self.coor_x = (lon / (2 * np.pi) + 0.5) * self.equ_w - 0.5
        self.coor_y = (-lat / np.pi + 0.5) * self.equ_h - 0.5

    def sample_equirec(self, e_img, order=0):
        pad_u = np.roll(e_img[[0]], self.equ_w // 2, 1)
        pad_d = np.roll(e_img[[-1]], self.equ_w // 2, 1)
        e_img = np.concatenate([e_img, pad_d, pad_u], 0)
        return map_coordinates(e_img, [self.coor_y, self.coor_x], order=order, mode="wrap")[..., 0]

    def run(self, equ_img, equ_dep, equ_mask, depth_max: float | None = None):
        h, w = equ_img.shape[:2]
        equ_mask = equ_mask.astype(np.float32)
        if h != self.equ_h or w != self.equ_w:
            equ_img = cv2.resize(equ_img, (self.equ_w, self.equ_h))
            # Float depth: resize in inverse-depth space (see ``resize_depth_map``).
            equ_dep = resize_depth_map(
                equ_dep,
                (self.equ_w, self.equ_h),
                interpolation=cv2.INTER_LINEAR,
                space="inverse",
                depth_max=depth_max,
            )
            equ_mask = cv2.resize(equ_mask, (self.equ_w, self.equ_h), interpolation=cv2.INTER_NEAREST)

        cube_img = np.stack([self.sample_equirec(equ_img[..., i], order=1) for i in range(equ_img.shape[2])], axis=-1)
        # order=1 (linear) sampling for depth reduces jagged alias vs order=0 (nearest) on the sphere.
        cube_dep = np.stack([self.sample_equirec(equ_dep, order=1)], axis=-1)
        cube_dep = cube_dep * self.cosmaps

        cube_mask = np.stack([self.sample_equirec(equ_mask, order=0)], axis=-1)
        cube_mask = cube_mask > 0.5

        return cube_img, cube_dep, cube_mask


def gen_cubes(
    pano_image: np.ndarray,
    pano_depth: np.ndarray,
    pano_mask: np.ndarray,
    original_ext: np.ndarray,
    *,
    # SHARP predictor uses internal_shape=(1536, 1536). Generating cubes at the
    # same resolution avoids an extra resize step and keeps intrinsics consistent.
    out_size: int = 1536,
    fov_deg: float = 90.0,
    depth_max: float | None = None,
):
    """Split equirect RGB + depth into 6 cube faces with intrinsics/extrinsics.

    For stable depth across face boundaries, resize pano RGB + depth to this ERP
    before calling (see ``cli_pano`` / ``--pano-before-cube sharp``) so sampling
    does not mix an extra ERP resize inside :class:`Equirec2Cube`.
    """
    H, W = pano_image.shape[:2]

    fov = fov_deg
    fx = fy = out_size / (2.0 * math.tan(math.radians(fov / 2.0)))
    cx = out_size / 2.0
    cy = out_size / 2.0

    e2c_mono = Equirec2Cube(pano_image.shape[0], pano_image.shape[1], out_size)
    cubemap_Rs = torch.eye(4, dtype=torch.float32)
    cubemap_Rs = repeat(cubemap_Rs, "... -> f ...", f=6).clone()
    cubemap_Rs[:, :3, :3] = torch.tensor(
        [
            [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
            [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
            [[-1, 0, 0], [0, 0, -1], [0, -1, 0]],
            [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
        ],
        dtype=torch.float32,
    ).inverse()

    pano_image = pano_image / 255.0
    cube_images, cube_depths, cube_masks = e2c_mono.run(
        pano_image,
        pano_depth.reshape(H, W),
        pano_mask.reshape(H, W),
        depth_max=depth_max,
    )
    cube_images = rearrange(cube_images, "h (v w) c -> v c h w", v=6)
    cube_depths = rearrange(cube_depths, "h (v w) c -> v h w c", v=6)
    cube_masks = rearrange(cube_masks, "h (v w) c -> v h w c", v=6)

    intrinsic = np.array(
        [
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ]
        ],
        dtype=np.float32,
    )
    cube_intr = np.repeat(intrinsic, 6, axis=0)

    cube_images = torch.from_numpy(cube_images).float()
    cube_depths = torch.from_numpy(cube_depths).float()
    cube_masks = torch.from_numpy(cube_masks).bool()
    cube_intr = torch.from_numpy(cube_intr).float()
    cube_extr = torch.einsum("ij,mjk -> mik", torch.from_numpy(original_ext), cubemap_Rs).float()

    return cube_images, cube_depths, cube_masks, cube_intr, cube_extr


FACE_NAMES = ("front", "right", "back", "left", "up", "down")


def refine_cube_face_depth(
    depth_hw: np.ndarray,
    *,
    mode: str = "bilateral",
    eps: float = 5e-5,
    depth_max: float | None = None,
) -> np.ndarray:
    """Reduce alias / stair-steps on cube depth after equirect warp.

    Operates in **inverse depth (disparity)** with a small bilateral filter so
    flat regions smooth while stronger gradients are mostly preserved.
    """
    if mode == "none":
        d = depth_hw.astype(np.float32)
        valid = _well_defined_depth_np(d, eps=eps, depth_max=depth_max) & np.isfinite(d)
        out = d.copy()
        out[~valid] = np.nan
        return out.astype(np.float32)
    if mode != "bilateral":
        raise ValueError(f"Unknown cube depth refine mode: {mode}")

    d = depth_hw.astype(np.float32)
    valid = _well_defined_depth_np(d, eps=eps, depth_max=depth_max) & np.isfinite(d)
    if not np.any(valid):
        return np.full_like(d, np.nan, dtype=np.float32)

    inv = np.zeros_like(d, dtype=np.float32)
    np.divide(1.0, d, out=inv, where=valid)
    inv_valid = inv[valid]
    inv_rng = float(np.percentile(inv_valid, 99) - np.percentile(inv_valid, 1)) + 1e-6
    sigma_color = 0.08 * inv_rng
    inv_f = cv2.bilateralFilter(inv, d=5, sigmaColor=sigma_color, sigmaSpace=5)
    out = np.full_like(d, np.nan, dtype=np.float32)
    np.divide(1.0, np.clip(inv_f, 1e-6, None), out=out, where=valid)
    # Sanity: for invalid pixels we must keep NaN (not 0) so EXR holes are identifiable.
    if np.any(~valid):
        assert np.all(np.isnan(out[~valid])), "refine_cube_face_depth must output NaN for invalid pixels"
    if np.any(valid):
        assert np.all(np.isfinite(out[valid])), "refine_cube_face_depth produced non-finite values on valid pixels"
    return out.astype(np.float32)


def save_cube_data(
    cube_images: torch.Tensor,
    cube_depths: torch.Tensor,
    cube_masks: torch.Tensor,
    cube_intr: torch.Tensor,
    cube_extr: torch.Tensor,
    output_dir: str,
    *,
    depth_max: float | None = None,
    cube_depth_refine: str = "bilateral",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for i, face_name in enumerate(FACE_NAMES):
        face_dir = os.path.join(output_dir, face_name)
        os.makedirs(face_dir, exist_ok=True)

        img_np = (cube_images[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        cv2.imwrite(
            os.path.join(face_dir, f"{face_name}_image.png"),
            cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR),
        )

        # Write float depth EXR for SHARP depth-guided initialization.
        # This avoids uint16 quantization artifacts when `depth_max` is large.
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        depth_np = cube_depths[i].numpy().squeeze().astype(np.float32)
        eps_refine = 5e-5
        dm = float(depth_max) if depth_max is not None else None
        depth_np = refine_cube_face_depth(depth_np, mode=cube_depth_refine, eps=eps_refine, depth_max=dm)

        # Sanity: invalid pixels must be NaN (not 0) so EXR holes can be detected downstream.
        valid_ref = _well_defined_depth_np(depth_np, eps=eps_refine, depth_max=depth_max) & np.isfinite(depth_np)
        if np.any(~valid_ref):
            assert np.all(np.isnan(depth_np[~valid_ref])), "refine_cube_face_depth must encode invalid as NaN"
        if np.any(valid_ref):
            assert np.all(np.isfinite(depth_np[valid_ref])), "refine_cube_face_depth produced non-finite values"
        mask_bool = cube_masks[i].numpy().squeeze().astype(bool)
        depth_np = np.where(mask_bool, depth_np, np.nan).astype(np.float32)
        if not np.all(mask_bool):
            assert np.all(np.isnan(depth_np[~mask_bool])), "masked-out cube depth must be NaN"
        if depth_max is not None:
            depth_exr = np.clip(depth_np, 0.0, float(depth_max)).astype(np.float32)
        else:
            depth_exr = np.maximum(depth_np, 0.0).astype(np.float32)
        cv2.imwrite(os.path.join(face_dir, f"{face_name}_depth.exr"), depth_exr)

        mask_np = cube_masks[i].numpy().squeeze().astype(np.uint8) * 255
        cv2.imwrite(os.path.join(face_dir, f"{face_name}_mask.png"), mask_np)

        np.save(os.path.join(face_dir, f"{face_name}_intr.npy"), cube_intr[i].numpy())
        np.save(os.path.join(face_dir, f"{face_name}_extr.npy"), cube_extr[i].numpy())


def load_cube_data(input_dir: str, *, depth_max: float = 10.0):
    cube_images = []
    cube_depths = []
    cube_masks = []
    cube_intr = []
    cube_extr = []

    for face_name in FACE_NAMES:
        face_dir = os.path.join(input_dir, face_name)

        img_path = os.path.join(face_dir, f"{face_name}_image.png")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float() / 255.0
        img = img.permute(2, 0, 1)
        cube_images.append(img)

        depth_path = os.path.join(face_dir, f"{face_name}_depth.exr")
        depth_exr = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_exr is None:
            raise FileNotFoundError(f"Depth EXR not found: {depth_path}")
        if depth_exr.ndim == 3:
            depth_exr = depth_exr[:, :, 0]
        depth = torch.from_numpy(depth_exr.astype(np.float32)).unsqueeze(-1)
        depth = depth.clamp(min=0.0, max=float(depth_max))
        cube_depths.append(depth)

        mask_path = os.path.join(face_dir, f"{face_name}_mask.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = torch.from_numpy(mask).bool().unsqueeze(-1)
        cube_masks.append(mask)

        intr_path = os.path.join(face_dir, f"{face_name}_intr.npy")
        intr = torch.from_numpy(np.load(intr_path))
        cube_intr.append(intr)

        extr_path = os.path.join(face_dir, f"{face_name}_extr.npy")
        extr = torch.from_numpy(np.load(extr_path))
        cube_extr.append(extr)

    return (
        torch.stack(cube_images, dim=0),
        torch.stack(cube_depths, dim=0),
        torch.stack(cube_masks, dim=0),
        torch.stack(cube_intr, dim=0),
        torch.stack(cube_extr, dim=0),
    )
