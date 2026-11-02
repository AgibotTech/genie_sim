# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Resolve separate configuration (q) and velocity (qd) slices for SDK joints.

Copied from the Newton backend and reduced to the metadata consumed by libuipc.
A free joint has seven coordinates and six velocities, so scalar joints after
it must retain their distinct q_start and qd_start indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, List


@dataclass(frozen=True)
class JointSlice:
    name: str
    label: str
    joint_index: int
    joint_type: int
    q_start: int
    q_count: int
    qd_start: int
    dof_count: int


class JointIndex:
    """Cache model-order slice metadata without importing the Newton backend."""

    def __init__(self, model: Any) -> None:
        labels = list(getattr(model, "joint_label", []) or [])
        types_arr = getattr(model, "joint_type", None)
        q_start_arr = getattr(model, "joint_q_start", None)
        qd_start_arr = getattr(model, "joint_qd_start", None)
        dof_dim_arr = getattr(model, "joint_dof_dim", None)
        joint_q_arr = getattr(model, "joint_q", None)
        joint_qd_arr = getattr(model, "joint_qd", None)

        # Total q / qd vector lengths — only needed to compute the last
        # joint's slice extent. We do not read the actual values.
        self._n_q = int(joint_q_arr.size) if joint_q_arr is not None else 0
        self._n_qd = int(joint_qd_arr.size) if joint_qd_arr is not None else 0

        if types_arr is None or q_start_arr is None or not labels:
            self._slices: List[JointSlice] = []
            return

        types_np = types_arr.numpy()
        q_starts = q_start_arr.numpy()
        # joint_qd_start is present on Newton ≥ 1.x. In its absence
        # (exotic test stubs without FREE joints) the chain has q == qd
        # joint-for-joint, so falling back is correct *for those builds*.
        # Builds that support FREE joints always expose qd_start.
        qd_starts = qd_start_arr.numpy() if qd_start_arr is not None else q_starts
        # joint_dof_dim can be a scalar (per-joint dof count) or a
        # 2-vector (linear, angular). sum() handles both.
        dof_dim = dof_dim_arr.numpy() if dof_dim_arr is not None else None

        n_joints = len(labels)
        slices: List[JointSlice] = []
        for ji in range(n_joints):
            label = labels[ji] or ""
            short = label.rsplit("/", 1)[-1] if "/" in label else label
            jtype = int(types_np[ji])
            q_start = int(q_starts[ji])
            qd_start = int(qd_starts[ji])
            q_end = int(q_starts[ji + 1]) if ji + 1 < n_joints else self._n_q
            q_count = q_end - q_start
            # Prefer the authoritative dof_dim when present — using
            # "next qd_start − this qd_start" would falsely report
            # 1 DoF for a fixed joint sandwiched between two revolute
            # joints (qd_start doesn't advance through fixed joints).
            if dof_dim is not None:
                try:
                    dof_count = int(sum(int(x) for x in dof_dim[ji]))
                except TypeError:
                    dof_count = int(dof_dim[ji])
            else:
                dof_count = (int(qd_starts[ji + 1]) if ji + 1 < n_joints else self._n_qd) - qd_start
            slices.append(
                JointSlice(
                    name=short,
                    label=label,
                    joint_index=ji,
                    joint_type=jtype,
                    q_start=q_start,
                    q_count=q_count,
                    qd_start=qd_start,
                    dof_count=dof_count,
                )
            )

        self._slices = slices

    def slices(self) -> Iterator[JointSlice]:
        """Yield every joint's slice metadata in model order."""
        return iter(self._slices)
