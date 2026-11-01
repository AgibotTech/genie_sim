# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pose preparation and IK candidate selection for follow-trajectory actions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation


def normalize_waypoints(waypoints: Sequence[Sequence[float]]) -> np.ndarray:
    """Return waypoint candidates as finite ``(M, N, 7)`` xyz+wxyz values."""
    try:
        normalized = np.asarray(waypoints, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("waypoints must contain numeric xyz+wxyz values") from exc

    if normalized.ndim == 2:
        normalized = normalized[np.newaxis, ...]
    if normalized.ndim != 3 or normalized.shape[-1] != 7:
        raise ValueError("waypoints must have shape (N, 7) or (M, N, 7)")
    if normalized.shape[0] < 1 or normalized.shape[1] < 1:
        raise ValueError("waypoints must contain at least one trajectory and waypoint")
    if not np.isfinite(normalized).all():
        raise ValueError("waypoints must contain only finite values")

    normalized = normalized.copy()
    quaternion_norms = np.linalg.norm(normalized[..., 3:], axis=-1, keepdims=True)
    if np.any(quaternion_norms <= np.finfo(np.float64).eps):
        raise ValueError("waypoint quaternions must be non-zero")
    normalized[..., 3:] /= quaternion_norms
    return normalized


def validate_trajectory_options(
    noise_range: Sequence[float], noise_frame: str, sample_count: int
) -> np.ndarray:
    """Validate sampling options and return the six-dimensional range array."""
    try:
        ranges = np.asarray(noise_range, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("noise_range must contain six numeric values") from exc
    if ranges.shape != (6,):
        raise ValueError("noise_range must contain six values")
    if not np.isfinite(ranges).all() or np.any(ranges < 0):
        raise ValueError("noise_range values must be finite and non-negative")
    if noise_frame not in ("local", "world"):
        raise ValueError("noise_frame must be 'local' or 'world'")
    if (
        isinstance(sample_count, (bool, np.bool_))
        or not isinstance(sample_count, (int, np.integer))
        or sample_count < 1
    ):
        raise ValueError("sample_count must be an integer greater than or equal to 1")
    return ranges


def waypoints_to_poses(waypoints: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert xyz+wxyz waypoint candidates to ``(M, N, 4, 4)`` matrices."""
    normalized = normalize_waypoints(waypoints)
    poses = np.broadcast_to(
        np.eye(4, dtype=np.float64), (*normalized.shape[:2], 4, 4)
    ).copy()
    quaternions_xyzw = normalized[..., [4, 5, 6, 3]].reshape(-1, 4)
    poses[..., :3, :3] = Rotation.from_quat(quaternions_xyzw).as_matrix().reshape(
        *normalized.shape[:2], 3, 3
    )
    poses[..., :3, 3] = normalized[..., :3]
    return poses


def apply_pose_noise(
    pose: np.ndarray, delta: Sequence[float], noise_frame: str
) -> np.ndarray:
    """Apply xyz + XYZ-Euler noise, with angular components expressed in radians."""
    pose_array = np.asarray(pose, dtype=np.float64)
    delta_array = np.asarray(delta, dtype=np.float64)
    if pose_array.shape != (4, 4) or not np.isfinite(pose_array).all():
        raise ValueError("pose must be a finite 4x4 matrix")
    if delta_array.shape != (6,) or not np.isfinite(delta_array).all():
        raise ValueError("delta must contain six finite values")
    if noise_frame not in ("local", "world"):
        raise ValueError("noise_frame must be 'local' or 'world'")

    noisy_pose = pose_array.copy()
    rotation = pose_array[:3, :3]
    delta_rotation = Rotation.from_euler(
        "xyz", delta_array[3:], degrees=False
    ).as_matrix()
    if noise_frame == "local":
        noisy_pose[:3, 3] = pose_array[:3, 3] + rotation @ delta_array[:3]
        noisy_pose[:3, :3] = rotation @ delta_rotation
    else:
        noisy_pose[:3, 3] = pose_array[:3, 3] + delta_array[:3]
        noisy_pose[:3, :3] = delta_rotation @ rotation
    return noisy_pose


def _normalize_pose_candidates(actor_pose_candidates: np.ndarray) -> np.ndarray:
    candidates = np.asarray(actor_pose_candidates, dtype=np.float64)
    if candidates.ndim == 3:
        candidates = candidates[np.newaxis, ...]
    if candidates.ndim != 4 or candidates.shape[-2:] != (4, 4):
        raise ValueError("actor_pose_candidates must have shape (N,4,4) or (M,N,4,4)")
    if candidates.shape[0] < 1 or candidates.shape[1] < 1:
        raise ValueError("actor_pose_candidates must not be empty")
    if not np.isfinite(candidates).all():
        raise ValueError("actor_pose_candidates must contain only finite values")
    return candidates


def _solve_mask(
    solve_ik: Callable[[np.ndarray], np.ndarray], poses: np.ndarray
) -> np.ndarray:
    mask = np.asarray(solve_ik(poses), dtype=bool).reshape(-1)
    if mask.shape != (len(poses),):
        raise ValueError("solve_ik must return one boolean result per pose")
    return mask


def resolve_trajectory_candidates(
    actor_pose_candidates: np.ndarray,
    actor_to_ee: np.ndarray,
    solve_ik: Callable[[np.ndarray], np.ndarray],
    noise_range: Sequence[float],
    noise_frame: str,
    sample_count: int,
    rng=None,
) -> Optional[np.ndarray]:
    """Choose the fully solvable gripper trajectory with the least pose noise."""
    candidates = _normalize_pose_candidates(actor_pose_candidates)
    relative_pose = np.asarray(actor_to_ee, dtype=np.float64)
    if relative_pose.shape != (4, 4) or not np.isfinite(relative_pose).all():
        raise ValueError("actor_to_ee must be a finite 4x4 matrix")
    ranges = validate_trajectory_options(noise_range, noise_frame, sample_count)
    rng = np.random.default_rng() if rng is None else rng

    nonzero_ranges = ranges > 0
    best_trajectory = None
    best_score = np.inf

    for actor_trajectory in candidates:
        original_gripper_poses = actor_trajectory @ relative_pose
        original_solvable = _solve_mask(solve_ik, original_gripper_poses)
        resolved_gripper_poses = []
        trajectory_score = 0.0
        complete = True

        for waypoint_index, actor_pose in enumerate(actor_trajectory):
            gripper_pose = original_gripper_poses[waypoint_index]
            if original_solvable[waypoint_index]:
                resolved_gripper_poses.append(gripper_pose)
                continue

            deltas = np.asarray(
                rng.uniform(-ranges, ranges, size=(sample_count, 6)),
                dtype=np.float64,
            )
            if deltas.shape != (sample_count, 6) or not np.isfinite(deltas).all():
                raise ValueError("random sampler returned invalid pose deltas")
            noisy_actor_poses = np.stack(
                [apply_pose_noise(actor_pose, delta, noise_frame) for delta in deltas]
            )
            noisy_gripper_poses = noisy_actor_poses @ relative_pose
            solvable = _solve_mask(solve_ik, noisy_gripper_poses)
            valid_indices = np.flatnonzero(solvable)
            if len(valid_indices) == 0:
                complete = False
                break

            normalized_deltas = np.zeros_like(deltas)
            normalized_deltas[:, nonzero_ranges] = (
                deltas[:, nonzero_ranges] / ranges[nonzero_ranges]
            )
            distances = np.linalg.norm(normalized_deltas, axis=1)
            selected_index = int(valid_indices[np.argmin(distances[valid_indices])])
            resolved_gripper_poses.append(noisy_gripper_poses[selected_index])
            trajectory_score += float(distances[selected_index])

        if not complete:
            continue

        resolved = np.stack(resolved_gripper_poses)
        if trajectory_score < best_score:
            best_trajectory = resolved
            best_score = trajectory_score
            if best_score == 0.0:
                break

    return best_trajectory


__all__ = [
    "apply_pose_noise",
    "normalize_waypoints",
    "resolve_trajectory_candidates",
    "validate_trajectory_options",
    "waypoints_to_poses",
]
