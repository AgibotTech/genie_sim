"""Sample 5 frames per video: first, 3 middle, last (paper appendix).

Release path: always sample in memory; no sidecar ``{stem}_vlm_frames/``.
"""

from typing import List, Tuple

import cv2
from PIL import Image


def _five_indices(n: int) -> List[int]:
    """Return 5 frame indices for a video with n frames."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    if n <= 5:
        return list(range(n))
    last = n - 1
    return [
        0,
        last // 4,
        last // 2,
        (3 * last) // 4,
        last,
    ]


def sample_five_frames(video_path: str) -> List[Image.Image]:
    """Load a video and return up to 5 RGB PIL frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = _five_indices(total)
    frames: List[Image.Image] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))

    cap.release()
    return frames


def crop_start_end_top(
    start_image: Image.Image,
    end_image: Image.Image,
) -> Tuple[Image.Image, Image.Image]:
    """Crop left-third (top-head) from start/end triplets in memory."""
    return _crop_left_third(start_image), _crop_left_third(end_image)


def _crop_left_third(img: Image.Image) -> Image.Image:
    rgb = img.convert("RGB")
    width, height = rgb.size
    if width < 3:
        raise ValueError(f"image width {width} is too small to crop left third")
    return rgb.crop((0, 0, width // 3, height))


def split_triplet_views(
    triplet: Image.Image,
) -> Tuple[
    Image.Image,
    Image.Image,
    Image.Image,
]:
    """
    Split a horizontal triplet:

        HEAD/TOP | LEFT ARM | RIGHT ARM

    into three independent full-resolution crops.
    """
    if triplet is None:
        raise ValueError("triplet is None")

    img = triplet.convert("RGB")

    width, height = img.size

    if width < 3:
        raise ValueError(
            f"triplet width too small: {width}"
        )

    x1 = width // 3
    x2 = (2 * width) // 3

    head = img.crop(
        (0, 0, x1, height)
    )

    left_arm = img.crop(
        (x1, 0, x2, height)
    )

    right_arm = img.crop(
        (x2, 0, width, height)
    )

    return (
        head,
        left_arm,
        right_arm,
    )
