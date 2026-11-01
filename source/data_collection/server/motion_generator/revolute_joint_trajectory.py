# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.

"""Find USD revolute joints and sample world pose trajectories towards their limits.
Angles use the axial offset transform from the Isaac Sim joint viewport.
Use world_point_trajectory_with_body1_revolute for positions and wxyz orientations,
or get_current_revolute_joint_info for state and limits. Requires pxr;
optional debug markers require Isaac Sim."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from common.base_utils.transform_utils import mat2quat_wxyz, quat2mat_wxyz

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
except ImportError as e:  # pragma: no cover
    raise ImportError("pxr (OpenUSD Python bindings) is required") from e


TargetLimit = Literal["min", "max"]

float64 = np.float64


def _gf_mat3_to_np(m: Gf.Matrix3d) -> np.ndarray:
    return np.array([[m[r][c] for c in range(3)] for r in range(3)], dtype=float64)


def find_revolute_joints_under(stage: Usd.Stage, root_path: str) -> list[Sdf.Path]:
    """Return revolute joint paths under root in depth-first tree order."""
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise ValueError(f"Invalid prim path: {root_path}")

    out: list[Sdf.Path] = []

    def _is_revolute(prim: Usd.Prim) -> bool:
        if prim.IsA(UsdPhysics.RevoluteJoint):
            return True
        return prim.GetTypeName() == "PhysicsRevoluteJoint"

    def walk(prim: Usd.Prim) -> None:
        if _is_revolute(prim):
            out.append(prim.GetPath())
        for ch in prim.GetChildren():
            walk(ch)

    walk(root)
    return out


def _axis_token_to_index(axis_tok: str) -> int:
    m = {"X": 0, "Y": 1, "Z": 2, "x": 0, "y": 1, "z": 2}
    if axis_tok not in m:
        raise ValueError(f"Unsupported joint axis token: {axis_tok!r}; expected X/Y/Z")
    return m[axis_tok]


def _read_quat_attr_to_R(attr) -> np.ndarray:
    if not attr or not attr.HasAuthoredValue():
        return np.eye(3, dtype=float64)
    q = attr.Get()
    w = float(q.GetReal())
    im = q.GetImaginary()
    return quat2mat_wxyz(np.array([w, float(im[0]), float(im[1]), float(im[2])], dtype=float64))


@dataclass
class RevoluteJointExtraction:
    joint_path: str
    body0_path: str
    body1_path: str
    local_pos0: np.ndarray
    R_local0: np.ndarray
    local_pos1: np.ndarray
    R_local1: np.ndarray
    axis_index: int
    lower_limit_deg: float
    upper_limit_deg: float


@dataclass
class RevoluteJointInfo:
    """Current revolute state, using the trajectory helper's joint selection rules."""

    joint_path: str
    joint_prim_name: str
    body0_path: str
    body1_path: str
    axis: str
    current_angle_deg: float
    lower_limit_deg: float
    upper_limit_deg: float


def extract_revolute_joint(
    stage: Usd.Stage,
    root_prim_path: str,
    joint_prim_path: str | None = None,
) -> RevoluteJointExtraction:
    """Find a revolute joint under root; validate the explicit path when provided."""
    joint_candidates = find_revolute_joints_under(stage, root_prim_path)
    if joint_prim_path is not None:
        jp = Sdf.Path(joint_prim_path)
        if jp not in joint_candidates:
            raise ValueError(
                f"Path {joint_prim_path} is not a RevoluteJoint under {root_prim_path}; "
                f"candidates: {[str(p) for p in joint_candidates]}"
            )
        joint_path = joint_prim_path
    else:
        if len(joint_candidates) != 1:
            raise ValueError(
                f"Under {root_prim_path}, found {len(joint_candidates)} RevoluteJoint prims; "
                f"specify joint_prim_path explicitly. Candidates: {[str(p) for p in joint_candidates]}"
            )
        joint_path = str(joint_candidates[0])

    prim = stage.GetPrimAtPath(joint_path)
    joint = UsdPhysics.RevoluteJoint(prim)

    b0 = joint.GetBody0Rel().GetTargets()
    b1 = joint.GetBody1Rel().GetTargets()
    if len(b0) != 1 or len(b1) != 1:
        raise ValueError(f"Joint {joint_path} must have exactly one target for each body0/body1 relationship")
    body0_path = str(b0[0])
    body1_path = str(b1[0])

    _lp0 = joint.GetLocalPos0Attr().Get()
    _lp1 = joint.GetLocalPos1Attr().Get()
    p0 = np.array(_lp0 if _lp0 is not None else (0.0, 0.0, 0.0), dtype=float64)
    p1 = np.array(_lp1 if _lp1 is not None else (0.0, 0.0, 0.0), dtype=float64)
    R0 = _read_quat_attr_to_R(joint.GetLocalRot0Attr())
    R1 = _read_quat_attr_to_R(joint.GetLocalRot1Attr())

    axis_attr = joint.GetAxisAttr()
    axis_index = 1
    if axis_attr and axis_attr.HasAuthoredValue():
        axis_index = _axis_token_to_index(axis_attr.Get())

    lo_attr = joint.GetLowerLimitAttr()
    hi_attr = joint.GetUpperLimitAttr()
    lower_limit_deg = float(lo_attr.Get()) if lo_attr and lo_attr.HasAuthoredValue() else -360.0
    upper_limit_deg = float(hi_attr.Get()) if hi_attr and hi_attr.HasAuthoredValue() else 360.0

    return RevoluteJointExtraction(
        joint_path=joint_path,
        body0_path=body0_path,
        body1_path=body1_path,
        local_pos0=p0,
        R_local0=R0,
        local_pos1=p1,
        R_local1=R1,
        axis_index=axis_index,
        lower_limit_deg=lower_limit_deg,
        upper_limit_deg=upper_limit_deg,
    )


def prim_world_rotation_translation(
    stage: Usd.Stage, prim_path: str, time: Usd.TimeCode | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return the prim world rotation matrix (3x3) and translation."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim_path}")
    tc = Usd.TimeCode.Default() if time is None else time
    xf = UsdGeom.Xformable(prim)
    M = xf.ComputeLocalToWorldTransform(tc)
    Rm = _gf_mat3_to_np(M.ExtractRotationMatrix())
    t = np.array([M.ExtractTranslation()[i] for i in range(3)], dtype=float64)
    return Rm, t


def rodrigues(axis: np.ndarray, theta_rad: float) -> np.ndarray:
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=float64)
    return np.eye(3) + np.sin(theta_rad) * K + (1 - np.cos(theta_rad)) * (K @ K)


def R_joint_from_axis_index(axis_index: int, theta_rad: float) -> np.ndarray:
    e = np.zeros(3, dtype=float64)
    e[axis_index] = 1.0
    return rodrigues(e, theta_rad)


def _joint_attachment_world_matrix_gf(
    stage: Usd.Stage,
    body_path: str,
    local_pos_attr,
    local_rot_attr,
    time: Usd.TimeCode,
) -> Gf.Matrix4d:
    """Match JointValidator.Attachment.compute_transform:
    R_local * (T_local * W_body).RemoveScaleShear()."""
    prim = stage.GetPrimAtPath(body_path)
    if not prim.IsValid():
        raise ValueError(f"Invalid body prim: {body_path}")
    W = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time)
    p_get = local_pos_attr.Get() if local_pos_attr else None
    if p_get is None:
        p_get = Gf.Vec3f(0.0, 0.0, 0.0)
    q_get = local_rot_attr.Get() if local_rot_attr else None
    if q_get is None:
        q_get = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    t = Gf.Matrix4d().SetTranslate(Gf.Vec3d(float(p_get[0]), float(p_get[1]), float(p_get[2])))
    r = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Quatd(q_get)))
    return r * (t * W).RemoveScaleShear()


def compute_axial_orientation_offsets_deg(delta: Gf.Matrix4d) -> tuple[float, float, float]:
    """Match viewport joint orientation offsets using row-vector atan2.
    This computes axial offsets rather than an Euler decomposition."""
    orientation_offset = [0.0, 0.0, 0.0]
    for axis in range(3):
        row = delta.GetRow3((axis + 1) % 3)
        orientation_offset[axis] = math.degrees(
            math.atan2(row[(axis + 2) % 3], row[(axis + 1) % 3])
        )
    return (orientation_offset[0], orientation_offset[1], orientation_offset[2])


def joint_angle_deg_from_offset_transform_like_viewport(
    stage: Usd.Stage,
    info: RevoluteJointExtraction,
    *,
    time: Usd.TimeCode | None = None,
) -> float:
    """Read the axial angle in degrees from current world attachment transforms.
    Compute delta = M1 * inverse(M0), with Mi = R_local_i * (T_local_i * W_body_i).
    Select the joint axis component and apply the viewport large-angle correction."""
    tc = Usd.TimeCode.Default() if time is None else time
    joint_prim = stage.GetPrimAtPath(info.joint_path)
    if not joint_prim.IsValid():
        raise ValueError(info.joint_path)
    joint = UsdPhysics.Joint(joint_prim)

    m0 = _joint_attachment_world_matrix_gf(
        stage, info.body0_path, joint.GetLocalPos0Attr(), joint.GetLocalRot0Attr(), tc
    )
    m1 = _joint_attachment_world_matrix_gf(
        stage, info.body1_path, joint.GetLocalPos1Attr(), joint.GetLocalRot1Attr(), tc
    )
    delta = m1 * m0.GetInverse()

    orientation_offset = list(compute_axial_orientation_offsets_deg(delta))
    ax = info.axis_index
    if orientation_offset[ax] > 90.0 or orientation_offset[ax] < -90.0:
        oi = (ax + 1) % 3
        if orientation_offset[oi] < 0.0:
            orientation_offset[oi] += 180.0
        else:
            orientation_offset[oi] -= 180.0

    return float(orientation_offset[ax])


def get_current_joint_angle_deg(
    stage: Usd.Stage,
    info: RevoluteJointExtraction,
    *,
    time: Usd.TimeCode | None = None,
) -> float:
    """Return the axial offset angle in degrees, matching viewport Local Offsets."""
    tc = Usd.TimeCode.Default() if time is None else time
    return joint_angle_deg_from_offset_transform_like_viewport(stage, info, time=tc)


def body1_world_pose_at_joint_angle(
    info: RevoluteJointExtraction,
    R_wb0: np.ndarray,
    t_wb0: np.ndarray,
    theta_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute body1 world pose from body0 and the absolute joint angle in degrees.
    Uses R_wb1 = R_wb0 * R0 * R_joint(theta) * inverse(R1)."""
    hinge_w = R_wb0 @ info.local_pos0 + t_wb0
    theta_rad = np.deg2rad(theta_deg)
    Rj = R_joint_from_axis_index(info.axis_index, theta_rad)
    R_wb1 = R_wb0 @ info.R_local0 @ Rj @ np.linalg.inv(info.R_local1)
    t_wb1 = hinge_w - R_wb1 @ info.local_pos1
    return R_wb1, t_wb1


def joint_axis_unit_world(info: RevoluteJointExtraction, R_wb0: np.ndarray) -> np.ndarray:
    """Return the joint axis as a unit vector in world coordinates."""
    e = np.zeros(3, dtype=float64)
    e[info.axis_index] = 1.0
    u = R_wb0 @ info.R_local0 @ e
    n = np.linalg.norm(u)
    if n < 1e-12:
        raise ValueError("World-frame joint axis length is too close to zero")
    return u / n


def hinge_point_world_blended(
    info: RevoluteJointExtraction,
    R_wb0: np.ndarray,
    t_wb0: np.ndarray,
    R_wb1: np.ndarray,
    t_wb1: np.ndarray,
) -> np.ndarray:
    """Average the two world attachment points to reduce numerical error."""
    h0 = R_wb0 @ info.local_pos0 + t_wb0
    h1 = R_wb1 @ info.local_pos1 + t_wb1
    return 0.5 * (h0 + h1)


def _arc_positions_and_tangents_world(
    thetas_deg: np.ndarray,
    theta_start_deg: float,
    hinge_H: np.ndarray,
    axis_u: np.ndarray,
    v0_at_start: np.ndarray,
    theta_target_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate around axis_u through hinge_H, preserving the hinge distance.
    P(theta) = H + R(u, theta - theta_start) * v0."""
    n_s = len(thetas_deg)
    pos_w = np.zeros((n_s, 3), dtype=float64)
    tan_w = np.zeros((n_s, 3), dtype=float64)
    path_sign = 1.0 if theta_target_deg >= theta_start_deg else -1.0
    for i, th in enumerate(thetas_deg):
        d_rad = np.deg2rad(float(th) - theta_start_deg)
        pos_w[i] = hinge_H + rodrigues(axis_u, d_rad) @ v0_at_start
        rvec = pos_w[i] - hinge_H
        dpdth = np.cross(axis_u, rvec)
        norm = np.linalg.norm(dpdth)
        if norm < 1e-12:
            tan_w[i] = np.zeros(3, dtype=float64)
        else:
            tan_w[i] = path_sign * (dpdth / norm)
    return pos_w, tan_w


@dataclass
class ArcTrajectoryResult:
    """Samples of a point fixed in body1 from the current angle to a joint limit."""

    theta_start_deg: float
    theta_end_deg: float
    thetas_deg: np.ndarray
    positions_world: np.ndarray
    positions_body0: np.ndarray
    tangents_world: np.ndarray
    point_local_body1: np.ndarray


def sample_point_trajectory_to_limit(
    stage: Usd.Stage,
    root_prim_path: str,
    point_local_body1: np.ndarray,
    target_limit: TargetLimit,
    step_deg: float,
    *,
    joint_prim_path: str | None = None,
) -> ArcTrajectoryResult:
    """Sample from the current viewport angle to the selected joint limit.
    step_deg bounds angular spacing; the final sample reaches the limit.
    positions_world and positions_body0 describe the same point fixed in body1.
    tangents_world are unit vectors in the direction of travel.
    Use P = H + R(u, theta - theta_cur) * (P0 - H) with the current stage pose
    to avoid drift from discrepancies between absolute angles and stage transforms."""
    if step_deg <= 0:
        raise ValueError("step_deg must be positive (degrees)")

    info = extract_revolute_joint(stage, root_prim_path, joint_prim_path)
    R_wb0, t_wb0 = prim_world_rotation_translation(stage, info.body0_path)
    R_wb1_now, t_wb1_now = prim_world_rotation_translation(stage, info.body1_path)
    theta_cur = get_current_joint_angle_deg(stage, info)
    theta_tgt = (
        info.lower_limit_deg if target_limit == "min" else info.upper_limit_deg
    )

    if abs(theta_tgt - theta_cur) < 1e-9:
        thetas = np.array([theta_cur], dtype=float64)
    else:
        n = max(1, int(np.ceil(abs(theta_tgt - theta_cur) / step_deg)))
        thetas = np.linspace(theta_cur, theta_tgt, n + 1, dtype=float64)

    point_local_body1 = np.asarray(point_local_body1, dtype=float64).reshape(3)
    p0_world = t_wb1_now + R_wb1_now @ point_local_body1
    H = hinge_point_world_blended(info, R_wb0, t_wb0, R_wb1_now, t_wb1_now)
    u_w = joint_axis_unit_world(info, R_wb0)
    v0 = p0_world - H

    pos_w, tan_w = _arc_positions_and_tangents_world(
        thetas, float(theta_cur), H, u_w, v0, float(theta_tgt)
    )
    pos_b0 = np.array([R_wb0.T @ (pw - t_wb0) for pw in pos_w], dtype=float64)

    return ArcTrajectoryResult(
        theta_start_deg=float(theta_cur),
        theta_end_deg=float(theta_tgt),
        thetas_deg=thetas,
        positions_world=pos_w,
        positions_body0=pos_b0,
        tangents_world=tan_w,
        point_local_body1=point_local_body1.copy(),
    )


def _joint_angles_every_deg(theta_start: float, theta_end: float, every_deg: float) -> list[float]:
    """Sample from theta_start to theta_end at every_deg, including both endpoints."""
    if every_deg <= 0:
        raise ValueError("every_deg must be positive")
    if abs(theta_end - theta_start) < 1e-9:
        return [float(theta_start)]
    sign = 1.0 if theta_end >= theta_start else -1.0
    out: list[float] = [float(theta_start)]
    t = float(theta_start)
    while True:
        nxt = t + sign * every_deg
        if sign * (nxt - theta_end) >= -1e-9:
            if abs(out[-1] - theta_end) > 1e-6:
                out.append(float(theta_end))
            break
        out.append(float(nxt))
        t = float(nxt)
    return out


def _debug_remove_prefix(stage: Usd.Stage, path_str: str) -> None:
    p = stage.GetPrimAtPath(path_str)
    if p.IsValid():
        stage.RemovePrim(path_str)


def get_current_revolute_joint_info(
    stage: Usd.Stage,
    root_prim_path: str,
    *,
    joint_prim_path: str | None = None,
) -> RevoluteJointInfo:
    """Return joint path/name, bodies, axis, current viewport angle, and USD limits.
    Angles and limits are in degrees. Joint selection matches the trajectory helper."""
    info = extract_revolute_joint(stage, root_prim_path, joint_prim_path)
    theta = float(get_current_joint_angle_deg(stage, info))
    jp = stage.GetPrimAtPath(info.joint_path)
    prim_name = jp.GetName() if jp.IsValid() else info.joint_path.split("/")[-1]
    axis_tok = ("X", "Y", "Z")[info.axis_index]
    return RevoluteJointInfo(
        joint_path=info.joint_path,
        joint_prim_name=str(prim_name),
        body0_path=info.body0_path,
        body1_path=info.body1_path,
        axis=axis_tok,
        current_angle_deg=theta,
        lower_limit_deg=float(info.lower_limit_deg),
        upper_limit_deg=float(info.upper_limit_deg),
    )


def world_point_trajectory_with_body1_revolute(
    stage: Usd.Stage,
    root_prim_path: str,
    world_position: np.ndarray,
    world_orientation_wxyz: np.ndarray,
    *,
    target_limit: TargetLimit = "min",
    trajectory_step_deg: float = 1.0,
    joint_prim_path: str | None = None,
    debug: bool = False,
    debug_angle_step_deg: float = 10.0,
    debug_cuboid_size: float = 0.1,
    debug_prim_path_prefix: str = "/Debug/RevoluteTrajMarker",
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a world pose rigidly attached to body1 towards a revolute joint limit.
    The starting pose must match body1 at the current stage time and joint angle.
    Positions and orientations use incremental rotation around the hinge axis:
    P = H + R(u, delta_theta) * (P0 - H), avoiding absolute-angle FK drift.

    Args:
        stage: Current USD stage.
        root_prim_path: Subtree containing the revolute joint and its bodies.
        world_position: Current world position (3,), in scene units.
        world_orientation_wxyz: Current quaternion (w, x, y, z), shape (4,).
            Normalized internally; maps the attached point's local frame to world.
        target_limit: "min" for lowerLimit or "max" for upperLimit.
        trajectory_step_deg: Maximum angular spacing in degrees; includes the limit.
        joint_prim_path: Explicit path, or None when the subtree contains one joint.
        debug: Create VisualCuboid markers in the Isaac Sim stage.
        debug_angle_step_deg: Marker spacing in degrees, independent of trajectory spacing.
        debug_cuboid_size: Marker scale along each axis, in meters.
        debug_prim_path_prefix: Marker parent path, removed before creating new markers.

    Returns:
        World positions (N, 3) and corresponding wxyz quaternions (N, 4),
        starting at the current angle and ending at the selected limit.
        Quaternion signs may differ while representing the same rotation."""
    world_position = np.asarray(world_position, dtype=float64).reshape(3)
    q_in = np.asarray(world_orientation_wxyz, dtype=float64).reshape(4)
    nq = np.linalg.norm(q_in)
    if nq < 1e-12:
        raise ValueError("world_orientation_wxyz length is too close to zero")
    q_in = q_in / nq

    info = extract_revolute_joint(stage, root_prim_path, joint_prim_path)
    R_wb1_now, t_wb1_now = prim_world_rotation_translation(stage, info.body1_path)
    p_body1 = R_wb1_now.T @ (world_position - t_wb1_now)

    arc = sample_point_trajectory_to_limit(
        stage,
        root_prim_path,
        p_body1,
        target_limit,
        trajectory_step_deg,
        joint_prim_path=joint_prim_path,
    )

    R_wb0, t_wb0 = prim_world_rotation_translation(stage, info.body0_path)
    u_w = joint_axis_unit_world(info, R_wb0)
    R0 = quat2mat_wxyz(q_in)
    theta0 = float(arc.theta_start_deg)
    n = len(arc.thetas_deg)
    quats_wxyz = np.zeros((n, 4), dtype=float64)
    for i, th in enumerate(arc.thetas_deg):
        d_rad = np.deg2rad(float(th) - theta0)
        quats_wxyz[i] = mat2quat_wxyz(rodrigues(u_w, d_rad) @ R0)

    if debug:
        try:
            from isaacsim.core.api.objects import cuboid  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError("debug=True requires an Isaac Sim environment (isaacsim.core.api.objects)") from e

        _debug_remove_prefix(stage, debug_prim_path_prefix)
        thetas_dbg = _joint_angles_every_deg(
            arc.theta_start_deg, arc.theta_end_deg, debug_angle_step_deg
        )
        s = float(debug_cuboid_size)
        scale = np.array([s, s, s], dtype=float64)
        color = np.array([1.0, 0.2, 0.2])
        H_dbg = hinge_point_world_blended(
            info, R_wb0, t_wb0, R_wb1_now, t_wb1_now
        )
        v_dbg = world_position - H_dbg
        for i, th in enumerate(thetas_dbg):
            d_rad = np.deg2rad(float(th) - theta0)
            pw = (H_dbg + rodrigues(u_w, d_rad) @ v_dbg).astype(float)
            orientwxyz = mat2quat_wxyz(rodrigues(u_w, d_rad) @ R0)
            prim_path = f"{debug_prim_path_prefix}/pt_{i:04d}"
            cuboid.VisualCuboid(
                prim_path=prim_path,
                position=pw.tolist(),
                orientation=tuple(float(x) for x in orientwxyz),
                scale=scale.tolist(),
                color=color,
            )

    return arc.positions_world, quats_wxyz
