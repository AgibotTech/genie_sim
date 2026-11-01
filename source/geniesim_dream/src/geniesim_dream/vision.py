from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from PIL import Image

VIEW_ROLES = ("head", "left_wrist", "right_wrist")
HEAD_SIZE = (640, 400)
WRIST_SIZE = (320, 200)
CONCAT_SIZE = (640, 600)
CONCAT_LAYOUT = "head_full_width_top__left_right_wrists_bottom"
RAW_FD_SIZE = (640, 592)
CANONICAL_EXPORT_VERSION = "dream-canonical-export-v1"

# Empirically verified on Dream session outputs (initial 640x600 → FD frames 640x592):
# MAE is minimized by keeping the top 592 rows (drop bottom 8), not center-crop.
# Head/wrist split therefore remains at y=400; wrists become 320x192 in raw frames.
# CropToMultiple in Framework documents center-crop, but live Dream AR outputs match
# top-aligned drop-bottom-8; export recovery follows the measured transform.
RAW_GEOMETRY_KIND = "drop_bottom_rows_h600_to_h592"
RAW_TOP_OFFSET = 0
RAW_BOTTOM_PAD = 8
HEAD_WRIST_SPLIT_Y = 400


def compose_concat_view(views: Mapping[str, Image.Image]) -> Image.Image:
    if set(views) != set(VIEW_ROLES):
        raise ValueError("Exactly head, left_wrist, and right_wrist views are required")

    head = views["head"].convert("RGB")
    if head.size != HEAD_SIZE:
        raise ValueError(f"Head view must be {HEAD_SIZE[0]}x{HEAD_SIZE[1]}")

    canvas = Image.new("RGB", CONCAT_SIZE)
    canvas.paste(head, (0, 0))
    for role, position in (("left_wrist", (0, 400)), ("right_wrist", (320, 400))):
        wrist = views[role].convert("RGB")
        if wrist.size != WRIST_SIZE:
            wrist = wrist.resize(WRIST_SIZE, Image.Resampling.BILINEAR)
        canvas.paste(wrist, position)
    return canvas


def split_concat_view(image: Image.Image) -> dict[str, Image.Image]:
    source = image.convert("RGB")
    if source.size != CONCAT_SIZE:
        raise ValueError(f"concat_view must be {CONCAT_SIZE[0]}x{CONCAT_SIZE[1]}")
    return {
        "head": source.crop((0, 0, 640, 400)),
        "left_wrist": source.crop((0, 400, 320, 600)),
        "right_wrist": source.crop((320, 400, 640, 600)),
    }


def describe_raw_geometry(width: int, height: int) -> dict[str, Any]:
    size = (int(width), int(height))
    if size == CONCAT_SIZE:
        kind = "native_canonical"
        bottom_pad = 0
    elif size == RAW_FD_SIZE:
        kind = RAW_GEOMETRY_KIND
        bottom_pad = RAW_BOTTOM_PAD
    else:
        kind = "unsupported"
        bottom_pad = 0
    return {
        "raw_width": size[0],
        "raw_height": size[1],
        "canonical_width": CONCAT_SIZE[0],
        "canonical_height": CONCAT_SIZE[1],
        "layout": CONCAT_LAYOUT,
        "geometry_kind": kind,
        "top_offset": RAW_TOP_OFFSET if kind == RAW_GEOMETRY_KIND else 0,
        "bottom_pad": bottom_pad,
        "head_wrist_split_y": HEAD_WRIST_SPLIT_Y,
        "export_version": CANONICAL_EXPORT_VERSION,
        "content_unrecoverable_rows": bottom_pad,
        "fill_mode": "edge_replicate_bottom" if bottom_pad else "none",
    }


def validate_raw_output_geometry(frames: np.ndarray) -> dict[str, Any]:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("FD output must have shape [T,H,W,3]")
    height, width = int(frames.shape[1]), int(frames.shape[2])
    meta = describe_raw_geometry(width, height)
    if meta["geometry_kind"] == "unsupported":
        raise ValueError(
            f"unsupported_output_geometry: got {width}x{height}; "
            f"expected {CONCAT_SIZE[0]}x{CONCAT_SIZE[1]} or {RAW_FD_SIZE[0]}x{RAW_FD_SIZE[1]}"
        )
    return meta


def to_canonical_frame(frame: np.ndarray | Image.Image) -> tuple[np.ndarray, dict[str, Any]]:
    """Restore Dream export contract 640x600 without mutating AR native tensors in place."""
    if isinstance(frame, Image.Image):
        array = np.array(frame.convert("RGB"))
    else:
        array = np.asarray(frame)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("Frame must be HxWx3 RGB")
    if array.dtype != np.uint8:
        raise ValueError("Frame must be uint8")
    height, width = int(array.shape[0]), int(array.shape[1])
    meta = describe_raw_geometry(width, height)
    if meta["geometry_kind"] == "native_canonical":
        return array.copy(), meta
    if meta["geometry_kind"] != RAW_GEOMETRY_KIND:
        raise ValueError(
            f"unsupported_output_geometry: cannot canonicalize {width}x{height}"
        )
    pad = np.repeat(array[-1:, :, :], RAW_BOTTOM_PAD, axis=0)
    restored = np.concatenate([array, pad], axis=0)
    if restored.shape[0] != CONCAT_SIZE[1] or restored.shape[1] != CONCAT_SIZE[0]:
        raise ValueError(f"Canonical restore produced {restored.shape[1]}x{restored.shape[0]}")
    return restored, meta


def ensure_session_geometry_consistency(
    previous: tuple[int, int] | None, width: int, height: int
) -> tuple[int, int]:
    current = (int(width), int(height))
    if previous is None:
        return current
    if previous != current:
        raise ValueError(
            f"Session output geometry changed from {previous[0]}x{previous[1]} "
            f"to {current[0]}x{current[1]}"
        )
    return current
