# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from typing import Protocol

import numpy as np


class ImageMessage(Protocol):
    height: int
    width: int
    encoding: str
    is_bigendian: int
    step: int
    data: bytes


def copy_rgba8_to_image_message(pixels: np.ndarray, message: ImageMessage) -> None:
    """Copy one RGBA uint8 frame into a sensor_msgs/Image-compatible object."""

    frame = np.asarray(pixels)
    if frame.dtype != np.uint8:
        raise ValueError(f"RGBA image dtype must be uint8, got {frame.dtype}")
    if frame.ndim != 3 or frame.shape[2] != 4:
        raise ValueError(f"RGBA image shape must be HxWx4, got {frame.shape}")

    frame = np.ascontiguousarray(frame)
    message.height = int(frame.shape[0])
    message.width = int(frame.shape[1])
    message.encoding = "rgba8"
    message.is_bigendian = 0
    message.step = int(frame.strides[0])
    message.data = frame.tobytes()
