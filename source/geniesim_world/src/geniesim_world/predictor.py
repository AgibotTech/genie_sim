# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Subclass RGBGaussianPredictor for optional depth_gt-guided initialization."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from sharp.models.predictor import RGBGaussianPredictor
from sharp.utils.gaussians import Gaussians3D

from geniesim_world.gaussian_ops import apply_ndc_frustum_mask

LOGGER = logging.getLogger(__name__)

# Near-zero depth is treated as invalid (holes) for alignment to avoid
# blow-ups. During init, `depth_gt` may be used directly depending on
# `use_depth_gt_for_init` (use with care when depth_gt contains holes/zeros).
_DEPTH_INVALID_EPS = 1e-3


class DepthGuidedRGBGaussianPredictor(RGBGaussianPredictor):
    """Same weights as RGBGaussianPredictor; extends forward with optional depth_gt."""

    def forward(
        self,
        image: torch.Tensor,
        disparity_factor: torch.Tensor,
        depth: torch.Tensor | None = None,
        *,
        depth_gt: torch.Tensor | None = None,
        use_depth_gt_for_init: bool = False,
        ndc_frustum_mask: bool = True,
    ) -> Gaussians3D:
        if not use_depth_gt_for_init or depth_gt is None:
            return super().forward(image, disparity_factor, depth)

        monodepth_output = self.monodepth_model(image)
        monodepth_disparity = monodepth_output.disparity

        disparity_factor_exp = disparity_factor[:, None, None, None]
        monodepth = disparity_factor_exp / monodepth_disparity.clamp(min=1e-4, max=1e4)

        # Invalid DA360 depth (0 / sky) must not be passed to depth_alignment or init:
        # initializer uses disparity_factor / depth and 1/depth, which blows up; alignment
        # also scales the whole view to match bad zeros → flat Gaussians on some faces.
        d_ref = depth[:, 0:1]
        d_ok = d_ref > _DEPTH_INVALID_EPS
        depth_for_align = torch.where(d_ok, d_ref, monodepth[:, 0:1])

        monodepth, _ = self.depth_alignment(
            monodepth,
            depth_for_align,
            monodepth_output.decoder_features,
        )

        h, w = monodepth.shape[-2:]
        if depth_gt.dim() == 2:
            depth_gt = depth_gt[None, None, ...]
        elif depth_gt.dim() == 3:
            depth_gt = depth_gt[:, None, ...]
        elif depth_gt.dim() == 4:
            pass
        else:
            raise ValueError(f"depth_gt must be 2D/3D/4D, got shape {tuple(depth_gt.shape)}")
        depth_gt = F.interpolate(depth_gt.float(), size=(h, w), mode="nearest")
        # Init: use depth_gt directly for init (no invalid-depth fallback).
        monodepth = torch.cat([depth_gt, depth_gt], dim=1)

        init_output = self.init_model(image, monodepth)
        image_features = self.feature_model(init_output.feature_input, encodings=monodepth_output.output_features)
        delta_values = self.prediction_head(image_features)
        gaussians = self.gaussian_composer(
            delta=delta_values,
            base_values=init_output.gaussian_base_values,
            global_scale=init_output.global_scale,
        )
        if ndc_frustum_mask:
            gaussians = apply_ndc_frustum_mask(gaussians)
        return gaussians


def wrap_predictor(model: RGBGaussianPredictor) -> DepthGuidedRGBGaussianPredictor:
    """Cast an existing predictor to DepthGuidedRGBGaussianPredictor (same state_dict)."""
    model.__class__ = DepthGuidedRGBGaussianPredictor
    return model  # type: ignore[return-value]
