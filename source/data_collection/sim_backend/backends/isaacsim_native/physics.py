# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Native PhysX and SingleArticulation wrappers.
Runtime-specific imports are deferred until methods are called after app startup."""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from sim_backend import backend as _bk


# ---------------------------------------------------------------------------
# Read joint state
# ---------------------------------------------------------------------------


def get_joint_positions(
    art, *, joint_indices: Any = None, as_torch: Optional[bool] = None
):
    """Read positions with backend-compatible indices and configurable output type."""
    indices = _bk.pipeline_aware_input(joint_indices) if joint_indices is not None else None
    raw = art.get_joint_positions(joint_indices=indices)
    return _to_out(raw, as_torch)


def get_joint_velocities(
    art, *, joint_indices: Any = None, as_torch: Optional[bool] = None
):
    indices = _bk.pipeline_aware_input(joint_indices) if joint_indices is not None else None
    raw = art.get_joint_velocities(joint_indices=indices)
    return _to_out(raw, as_torch)


def get_dof_names(art) -> List[str]:
    """Return DOF names, or an empty list when unavailable before initialization."""
    names = getattr(art, "dof_names", None)
    return list(names) if names else []


def get_joints_state(art, *, as_torch: Optional[bool] = None):
    """Read positions and velocities with the requested output type; efforts remain None."""
    from isaacsim.core.utils.types import JointsState  # noqa: PLC0415

    state = art.get_joints_state()
    return JointsState(
        positions=_to_out(state.positions, as_torch),
        velocities=_to_out(state.velocities, as_torch),
        efforts=None,
    )


def get_measured_joint_efforts(art, *, as_torch: Optional[bool] = None):
    """Read efforts; callers handle articulations without effort measurement support."""
    raw = art.get_measured_joint_efforts()
    return _to_out(raw, as_torch)


def get_prim_world_pose(prim_path: str):
    """Return current world position (3,) and wxyz orientation (4,) as float64 arrays.
    With torch, read live Fabric attributes without creating tensor views, falling
    back to USD when unavailable. With numpy, read the USD world transform.
    Use this for object poses and obstacle updates that require current physics state."""
    if _bk.tensors_as() == "torch":
        result = _read_prim_pose_via_fabric(prim_path)
        if result is not None:
            return result
    return _read_prim_pose_via_usd(prim_path)


def _read_prim_pose_via_fabric(prim_path: str):
    """Read live rigid-body Fabric attributes as float64 arrays, or return None.
    Only rigid bodies/articulation links have PhysX-updated world attributes.
    Nonrigid containers and markers may retain spawn snapshots, so reject them
    using a USD RigidBodyAPI check and let callers use the USD fallback."""
    try:
        import omni.usd  # noqa: PLC0415
        import usdrt  # noqa: PLC0415
        from pxr import UsdUtils, UsdPhysics  # noqa: PLC0415
    except ImportError:
        return None

    usd_stage = omni.usd.get_context().get_stage()
    if usd_stage is None:
        return None

    # Nonrigid containers, markers, and FEM entities can have stale Fabric
    # snapshots; use USD instead.
    usd_prim = usd_stage.GetPrimAtPath(prim_path)
    if not usd_prim or not usd_prim.IsValid():
        return None
    if not usd_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        return None

    stage_cache = UsdUtils.StageCache.Get()
    stage_id = stage_cache.GetId(usd_stage).ToLongInt()
    if stage_id < 0:
        stage_id = stage_cache.Insert(usd_stage).ToLongInt()
    fabric_stage = usdrt.Usd.Stage.Attach(stage_id)
    fabric_prim = fabric_stage.GetPrimAtPath(usdrt.Sdf.Path(prim_path))
    if not fabric_prim:
        return None
    xform = usdrt.Rt.Xformable(fabric_prim)
    pos_attr = xform.GetWorldPositionAttr()
    orient_attr = xform.GetWorldOrientationAttr()
    if not pos_attr.IsValid() or not orient_attr.IsValid():
        return None
    pos_v = pos_attr.Get()
    orient_q = orient_attr.Get()
    if pos_v is None or orient_q is None:
        return None
    pos = np.asarray([pos_v[0], pos_v[1], pos_v[2]], dtype=np.float64)
    q_imag = orient_q.GetImaginary()
    quat_wxyz = np.asarray(
        [orient_q.GetReal(), q_imag[0], q_imag[1], q_imag[2]],
        dtype=np.float64,
    )
    return pos, quat_wxyz


def _read_prim_pose_via_usd(prim_path: str):
    """Read a USD XformCache pose: numpy default and torch fallback."""
    import omni.usd  # noqa: PLC0415
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"prim not found: {prim_path}")
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_xf = xform_cache.GetLocalToWorldTransform(prim)
    t_mat = world_xf.RemoveScaleShear()
    translation = t_mat.ExtractTranslation()
    rotation = t_mat.ExtractRotation()
    q = rotation.GetQuaternion()
    pos = np.asarray([translation[0], translation[1], translation[2]], dtype=np.float64)
    q_imag = q.GetImaginary()
    quat_wxyz = np.asarray(
        [q.GetReal(), q_imag[0], q_imag[1], q_imag[2]],
        dtype=np.float64,
    )
    return pos, quat_wxyz


# ---------------------------------------------------------------------------
# Write joint state
# ---------------------------------------------------------------------------


def set_joint_positions(art, positions, joint_indices: Any = None) -> None:
    """Convert input types and dtypes before calling SingleArticulation."""
    positions = _coerce_float(_bk.pipeline_aware_input(positions))
    indices = (
        _coerce_index(_bk.pipeline_aware_input(joint_indices))
        if joint_indices is not None
        else None
    )
    art.set_joint_positions(positions, joint_indices=indices)


def set_joint_velocities(art, velocities, joint_indices: Any = None) -> None:
    velocities = _coerce_float(_bk.pipeline_aware_input(velocities))
    indices = (
        _coerce_index(_bk.pipeline_aware_input(joint_indices))
        if joint_indices is not None
        else None
    )
    art.set_joint_velocities(velocities, joint_indices=indices)


# ---------------------------------------------------------------------------
# apply_action
# ---------------------------------------------------------------------------


def apply_action(art, action) -> None:
    """Apply dense or sparse-with-None articulation actions.
    CPU uses one SingleArticulation call with backend-compatible inputs.
    GPU normalizes sparse fields to dense values and explicit indices, then sends
    positions/efforts through the controller and velocities through the view.
    The velocity view path avoids the controller's numpy operation on CUDA tensors."""
    from isaacsim.core.utils.types import (  # noqa: PLC0415
        ArticulationAction,
        ArticulationActions,
    )

    if _bk.tensors_as() != "torch":
        # CPU: apply all fields in one call after converting input types.
        converted = ArticulationAction(
            joint_positions=_bk.pipeline_aware_input(action.joint_positions),
            joint_velocities=_bk.pipeline_aware_input(action.joint_velocities),
            joint_efforts=_bk.pipeline_aware_input(action.joint_efforts),
            joint_indices=_bk.pipeline_aware_input(action.joint_indices),
        )
        art.apply_action(converted)
        return

    # Torch: normalize sparse fields and dtypes, then split controller/view calls.
    pos = action.joint_positions
    vel = action.joint_velocities
    eff = action.joint_efforts
    idx = action.joint_indices

    pos, vel, eff, idx = _normalize_sparse_action(pos, vel, eff, idx)

    pos_p = _coerce_float(_bk.pipeline_aware_input(pos))
    vel_p = _coerce_float(_bk.pipeline_aware_input(vel))
    eff_p = _coerce_float(_bk.pipeline_aware_input(eff))
    idx_p = _coerce_index(_bk.pipeline_aware_input(idx))

    # Position and effort use the controller, which converts its numpy checks safely.
    if pos_p is not None or eff_p is not None:
        art.apply_action(
            ArticulationAction(
                joint_positions=pos_p,
                joint_velocities=None,  # Send velocities directly to the view to avoid CUDA numpy checks.
                joint_efforts=eff_p,
                joint_indices=idx_p,
            )
        )

    # The velocity controller path calls np.isnan on CUDA tensors.
    # Bypass it with a dense view action shaped (1, K).
    if vel_p is not None:
        vel_b = _add_batch(vel_p)
        art._articulation_view.apply_action(
            ArticulationActions(
                joint_positions=None,
                joint_velocities=vel_b,
                joint_efforts=None,
                joint_indices=idx_p,
            )
        )


# ---------------------------------------------------------------------------
# Articulation parameters accessed through the private view.
# ---------------------------------------------------------------------------


def initialize_view(art) -> None:
    """Initialize the underlying articulation view, including multi-instance views."""
    art._articulation_view.initialize()


def set_rigid_velocity(rigid_prim, linear=None, angular=None) -> None:
    """Set linear and angular velocity through the CPU/GPU-compatible rigid view.
    Combine them into a 6D vector; require at least one component and preserve
    the current value of the component omitted by the caller."""
    if linear is None and angular is None:
        raise ValueError("set_rigid_velocity: at least one of linear/angular must be given")
    view = rigid_prim._rigid_prim_view
    if linear is None:
        linear = view.get_linear_velocities()[0]
    if angular is None:
        angular = view.get_angular_velocities()[0]
    lin = _coerce_float(_bk.pipeline_aware_input(linear))
    ang = _coerce_float(_bk.pipeline_aware_input(angular))
    # Inputs now match the backend; combine them using torch or numpy accordingly.
    if _bk.tensors_as() == "torch":
        import torch  # noqa: PLC0415

        combined = torch.cat([lin.flatten(), ang.flatten()]).unsqueeze(0)
    else:
        combined = np.concatenate(
            [np.asarray(lin).flatten(), np.asarray(ang).flatten()]
        )[None, :]
    view.set_velocities(velocities=combined)


def get_dof_limits(art, *, as_torch: Optional[bool] = None):
    """Return lower and upper limits, each shaped (num_dof,), in the requested type.
    Read the view directly to avoid CUDA tensors entering dof_properties numpy
    structured arrays. Select the first articulation from (M, num_dof, 2)."""
    raw = art._articulation_view.get_dof_limits()
    # Shape (M, num_dof, 2); select the first articulation.
    single = raw[0]
    lower = single[..., 0]
    upper = single[..., 1]
    return _to_out(lower, as_torch), _to_out(upper, as_torch)


def set_max_efforts(art, efforts, *, joint_indices: Any = None) -> None:
    """Set maximum efforts through the articulation view with compatible types.
    Pass (N,) or (M, N) values and optional joint_indices through without adding
    a batch dimension; the view handles indexing and broadcasting."""
    efforts = _coerce_float(_bk.pipeline_aware_input(efforts))
    indices = (
        _coerce_index(_bk.pipeline_aware_input(joint_indices))
        if joint_indices is not None
        else None
    )
    art._articulation_view.set_max_efforts(values=efforts, joint_indices=indices)


def set_gains(art, *, kps=None, kds=None, joint_indices: Any = None) -> None:
    """Set view gains with compatible types; kps and kds may independently be None."""
    kps_c = _coerce_float(_bk.pipeline_aware_input(kps)) if kps is not None else None
    kds_c = _coerce_float(_bk.pipeline_aware_input(kds)) if kds is not None else None
    indices = (
        _coerce_index(_bk.pipeline_aware_input(joint_indices))
        if joint_indices is not None
        else None
    )
    art._articulation_view.set_gains(kps=kps_c, kds=kds_c, joint_indices=indices)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_out(raw, as_torch: Optional[bool]):
    """Choose the output tensor type from as_torch or backend configuration."""
    if raw is None:
        return None
    if _bk.resolve_as_torch(as_torch):
        return _bk.to_torch(raw)
    return _bk.to_numpy(raw)


def _is_sparse_with_none(field) -> bool:
    """Detect a list or tuple with None placeholders."""
    return isinstance(field, (list, tuple)) and any(v is None for v in field)


def _add_batch(x):
    """Expand a 1D (K,) array/tensor to (1, K) for view actions; pass None through."""
    if x is None:
        return None
    try:
        import torch  # noqa: PLC0415

        if isinstance(x, torch.Tensor):
            if x.dim() == 1:
                return x.unsqueeze(0)
            return x
    except ImportError:
        pass
    if isinstance(x, np.ndarray):
        if x.ndim == 1:
            return np.expand_dims(x, axis=0)
        return x
    return x


def _normalize_sparse_action(pos, vel, eff, idx):
    """Convert sparse-with-None action fields to dense values and explicit indices.
    When indices are omitted, all sparse fields must share their non-None slots.
    Reject differing patterns; pass dense fields or explicit indices through."""
    if idx is not None:
        return pos, vel, eff, idx
    sparse_fields = {name: f for name, f in (("pos", pos), ("vel", vel), ("eff", eff)) if _is_sparse_with_none(f)}
    if not sparse_fields:
        return pos, vel, eff, idx
    # Derive non-None indices from one sparse field, cross-check the rest.
    first_name, first_field = next(iter(sparse_fields.items()))
    derived_idx = [i for i, v in enumerate(first_field) if v is not None]
    for name, f in sparse_fields.items():
        if name == first_name:
            continue
        check = [i for i, v in enumerate(f) if v is not None]
        if check != derived_idx:
            raise ValueError(
                "sparse ArticulationAction with divergent None patterns across "
                f"({first_name}, {name}) is not supported; pass explicit "
                "joint_indices or split into multiple apply_action calls."
            )
    new_pos = [pos[i] for i in derived_idx] if "pos" in sparse_fields else pos
    new_vel = [vel[i] for i in derived_idx] if "vel" in sparse_fields else vel
    new_eff = [eff[i] for i in derived_idx] if "eff" in sparse_fields else eff
    return new_pos, new_vel, new_eff, derived_idx


def _coerce_float(x):
    """On torch backends, coerce numeric arrays/tensors to PhysX float32 buffers.
    Setters for positions, velocities, efforts, and gains require matching dtypes.
    CPU inputs pass through for Isaac Sim's internal conversion."""
    if x is None:
        return x
    if _bk.tensors_as() != "torch":
        return x  # CPU/numpy: let Isaac Sim convert internally.
    try:
        import torch  # noqa: PLC0415

        if isinstance(x, torch.Tensor):
            if x.dtype != torch.float32:
                return x.to(dtype=torch.float32)
            return x
    except ImportError:
        pass
    if isinstance(x, np.ndarray) and x.dtype != np.float32:
        return x.astype(np.float32, copy=False)
    return x


def _coerce_index(x):
    """On torch backends, coerce integer index arrays/tensors to int64.
    PhysX indexing requires matching integer dtypes. CPU inputs pass through."""
    if x is None:
        return x
    if _bk.tensors_as() != "torch":
        return x  # CPU/numpy:passthrough
    try:
        import torch  # noqa: PLC0415

        if isinstance(x, torch.Tensor):
            if x.dtype != torch.int64 and not x.is_floating_point():
                return x.to(dtype=torch.int64)
            return x
    except ImportError:
        pass
    if isinstance(x, np.ndarray) and x.dtype != np.int64 and np.issubdtype(x.dtype, np.integer):
        return x.astype(np.int64, copy=False)
    return x
