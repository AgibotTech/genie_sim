"""Shared data types and array-layout helpers.

Layout conventions:
- Frames: ``(T, 3, V, H, W)`` float32 in ``[0, 1]``; views ordered
  head, left_wrist, right_wrist.
- World-model action/state layout (16-D): ``[L7_arm, L_grip, R7_arm, R_grip]``.
- Policy input state layout (16-D): ``[L7_arm, R7_arm, L_grip, R_grip]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VIEW_NAMES: tuple[str, ...] = ("head", "left_wrist", "right_wrist")

STATE_DIM = 16
ACTION_DIM = 16


@dataclass(frozen=True)
class Observation:
    """A single multi-camera observation, the contract between env, policies and rewards.

    Attributes:
        images: per-view uint8 ``(H, W, 3)`` RGB arrays keyed by ``VIEW_NAMES``.
        state: ``(16,)`` float32 proprioception in policy layout
            ``[L7_arm, R7_arm, L_grip, R_grip]``.
        task: natural-language task instruction.
    """

    images: dict[str, np.ndarray]
    state: np.ndarray
    task: str


@dataclass(frozen=True)
class StepInfo:
    """Extra outputs of ``WorldModelEnv.step``.

    Attributes:
        frames: ``(T, 3, V, H, W)`` float32 ``[0, 1]`` frames generated this step.
        progress: ``(T,)`` float32 task progress from the reward model, or None.
    """

    frames: np.ndarray
    progress: np.ndarray | None


def frame_to_view_images(frame: np.ndarray) -> dict[str, np.ndarray]:
    """Split one ``(3, V, H, W)`` float ``[0, 1]`` frame into per-view uint8 HWC images."""
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim != 4 or frame.shape[0] != 3 or frame.shape[1] < len(VIEW_NAMES):
        raise ValueError(f"expected frame (3, V>={len(VIEW_NAMES)}, H, W), got {frame.shape}")
    images: dict[str, np.ndarray] = {}
    for idx, name in enumerate(VIEW_NAMES):
        view = np.clip(frame[:, idx], 0.0, 1.0)
        images[name] = (view * 255.0).round().astype(np.uint8).transpose(1, 2, 0)
    return images


def head_view_frames(frames: np.ndarray) -> np.ndarray:
    """Extract the head view of ``(T, 3, V, H, W)`` frames as ``(T, H, W, 3)`` uint8."""
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 5 or frames.shape[1] != 3 or frames.shape[2] < len(VIEW_NAMES):
        raise ValueError(f"expected frames (T, 3, V>={len(VIEW_NAMES)}, H, W), got {frames.shape}")
    head = np.clip(frames[:, :, 0], 0.0, 1.0)
    return (head.transpose(0, 2, 3, 1) * 255.0).round().astype(np.uint8)


def wm_state_to_policy_state(state: np.ndarray) -> np.ndarray:
    """Reorder a 16-D vector from WM layout ``[L7, Lg, R7, Rg]`` to policy ``[L7, R7, Lg, Rg]``."""
    v = np.asarray(state, dtype=np.float32).reshape(-1)
    if v.shape[0] < STATE_DIM:
        raise ValueError(f"expected at least {STATE_DIM} dims, got {v.shape[0]}")
    out = np.empty(STATE_DIM, dtype=np.float32)
    out[0:7] = v[0:7]
    out[7:14] = v[8:15]
    out[14] = v[7]
    out[15] = v[15]
    return out
