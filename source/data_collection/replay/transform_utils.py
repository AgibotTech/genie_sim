"""Pose and timestamp helpers."""
import numpy as np


def local_to_world_with_clone(
    local_pos, local_quat_wxyz, clone_pos, clone_quat_wxyz
) -> tuple[list, list]:
    """Transform a local pose into world coordinates using the clone pose."""
    from scipy.spatial.transform import Rotation as R
    r = R.from_quat(
        [local_quat_wxyz[1], local_quat_wxyz[2], local_quat_wxyz[3], local_quat_wxyz[0]]
    )
    q = R.from_quat(
        [clone_quat_wxyz[1], clone_quat_wxyz[2], clone_quat_wxyz[3], clone_quat_wxyz[0]]
    )
    world_pos = np.array(clone_pos, dtype=np.float64) + q.apply(
        np.asarray(local_pos, dtype=np.float64)
    )
    world_rot = (q * r).as_quat()
    return world_pos.tolist(), [
        float(world_rot[3]),
        float(world_rot[0]),
        float(world_rot[1]),
        float(world_rot[2]),
    ]


def frame_durations_from_timestamps(
    frames: list, default_dt: float = 1.0 / 30.0
) -> list:
    """Calculate frame durations from timestamps, falling back to default_dt."""
    n = len(frames)
    if n == 0:
        return []
    if n == 1:
        return [default_dt]
    out = []
    for i in range(n - 1):
        t0 = frames[i].get("time_stamp")
        t1 = frames[i + 1].get("time_stamp")
        if t0 is not None and t1 is not None:
            dt = float(t1) - float(t0)
            if dt <= 0:
                dt = default_dt
            out.append(dt)
        else:
            out.append(default_dt)
    out.append(out[-1] if out else default_dt)
    return out
