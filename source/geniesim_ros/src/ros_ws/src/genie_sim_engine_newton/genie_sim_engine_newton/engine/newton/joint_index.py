# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Per-joint q / qd index resolution for a Newton model.

Newton stores articulation state in two arrays of *different* per-joint
width:

  * ``model.joint_q``  — the **configuration** vector. FREE joints
    occupy 7 entries (x, y, z + qw, qx, qy, qz); BALL joints occupy 4
    (a quaternion); D6 joints carry one extra entry per angular axis
    they expose.
  * ``model.joint_qd`` — the **velocity** vector. Same FREE joint
    occupies 6 entries (3 linear + 3 angular velocities); BALL takes 3;
    D6 has one slot per angular *velocity* axis.

For the common joints in a robot articulation (revolute / prismatic /
fixed) ``q_count == dof_count``, so ``joint_q_start[ji] ==
joint_qd_start[ji]`` and you can index either array with the same
value. The standalone engine selects Newton's DOF target layout during model
construction. Every downstream array sized one-DOF-per-slot
(``joint_qd``, ``control.joint_target_q``,
``control.joint_target_qd``, ``model.joint_target_ke``, the
controlled-DOF mask in the featherstone / AVBD adapters) must be
indexed with the **qd** value. ``model.joint_q`` is the only array
that wants the **q** value.

Mixing them is silent until a single FREE joint shows up at the head
of the articulation — then every subsequent joint's qd_start is
``q_start − 1``, and every controller would address the next joint's
slot.  This module makes that asymmetry impossible to ignore at
call-sites: callers ask for either ``q_idx`` or ``dof_idx`` explicitly,
the two never appear in the same dict.

Use ``JointIndex(model)`` once at lifecycle-build time; cache the
result. All lookups thereafter are dict-fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

# Newton joint-type enum values. Centralised here so a new joint type
# only adds one line; module-level constants beat magic numbers
# sprinkled through three different call-sites.
JT_PRISMATIC = 0
JT_REVOLUTE = 1
JT_BALL = 2
JT_FIXED = 3
JT_FREE = 4
JT_DISTANCE = 5
JT_D6 = 6
JT_CABLE = 7


@dataclass(frozen=True)
class JointSlice:
    """Everything the engine needs to address one joint in either vector.

    ``q_start`` / ``q_count`` index ``model.joint_q``.
    ``qd_start`` / ``dof_count`` index ``model.joint_qd``,
    ``control.joint_target_q``, ``control.joint_target_qd``,
    ``model.joint_target_ke / _kd``, the controlled-DOF mask, and any
    other dof-sized array.

    ``name`` is the short leaf name (``"idx01_body_joint1"``) — what
    ROS topics use.  ``label`` is the full Newton label (typically
    ``"/genie/Physics/idx01_body_joint1"``) — what diagnostics print.
    """

    name: str
    label: str
    joint_index: int
    joint_type: int
    q_start: int
    q_count: int
    qd_start: int
    dof_count: int

    @property
    def is_widthless(self) -> bool:
        """True for joint types whose q-slice and qd-slice differ in
        width (FREE, BALL, D6 with quat). A naive 1-to-1 element copy
        across this boundary is meaningless — there's no qd counterpart
        for a quaternion's scalar component.
        """
        return self.q_count != self.dof_count

    @property
    def is_actuated(self) -> bool:
        """True if the joint contributes at least one DOF that can hold
        a target. Fixed joints (dof_count=0) cannot.
        """
        return self.dof_count > 0

    @property
    def is_free(self) -> bool:
        """True for the 6-DOF floating-base joint type."""
        return self.joint_type == JT_FREE

    @property
    def is_fixed(self) -> bool:
        return self.joint_type == JT_FIXED


class JointIndex:
    """Resolved q / qd slices for every joint in a Newton model.

    Construct once after ``add_usd`` has populated ``model.joint_*``;
    subsequent reads (``name_to_dof()``, ``name_to_q_idx()``,
    ``slices()``, ``copy_q_to_qd()``) are pure-Python dict / list
    walks with no further numpy round-trips.

    Tolerant of Newton builds that don't expose
    ``model.joint_qd_start`` — falls back to ``joint_q_start`` (which
    is correct in any chain without a FREE / BALL / D6 joint).
    """

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
            self._by_name: Dict[str, JointSlice] = {}
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
        # Build the name map last so a duplicated short name (unlikely
        # but possible if the URDF importer ever reuses one) yields the
        # later-built entry rather than crashing.
        self._by_name = {s.name: s for s in slices}

    # ----- structural queries -----------------------------------------

    def slices(self) -> Iterator[JointSlice]:
        """Yield every joint's slice metadata in model order."""
        return iter(self._slices)

    def __len__(self) -> int:
        return len(self._slices)

    def get(self, name: str) -> Optional[JointSlice]:
        """Look up one joint's slice by short name. None if absent."""
        return self._by_name.get(name)

    @property
    def n_q(self) -> int:
        """Length of ``model.joint_q``."""
        return self._n_q

    @property
    def n_qd(self) -> int:
        """Length of ``model.joint_qd`` (= ``joint_target_q.size``,
        = ``joint_target_ke.size``, etc.)."""
        return self._n_qd

    # ----- canonical name → index maps --------------------------------
    #
    # Returning fresh dicts (not views) is deliberate: callers cache
    # them on engine state and the engine outlives this JointIndex.

    def name_to_q_idx(self) -> Dict[str, int]:
        """Map: short joint name → ``q_start`` (index into ``joint_q``).

        Use this only when reading from ``state.joint_q`` —
        specifically, the position field of ``/joint_states``.
        """
        return {s.name: s.q_start for s in self._slices}

    def name_to_dof(self) -> Dict[str, int]:
        """Map: short joint name → ``qd_start`` (index into the DOF /
        velocity vector and every dof-sized array).

        Use this for every dof-indexed array:
        ``control.joint_target_q`` (the engine selects DOF target layout), ``control.joint_target_qd``,
        ``state.joint_qd``, ``model.joint_target_ke / _kd``, the
        controlled-DOF mask in featherstone / AVBD adapters, and any
        other per-DOF buffer.
        """
        return {s.name: s.qd_start for s in self._slices}

    # ----- bulk operations --------------------------------------------

    def copy_q_to_qd(
        self,
        src_q: np.ndarray,
        dst_qd: np.ndarray,
        *,
        skip_widthless: bool = True,
    ) -> int:
        """Copy each joint's q-slice into its qd-slice in ``dst_qd``.

        Used to seed ``control.joint_target_q`` from ``model.joint_q``
        at init time. The naive equivalent —
        ``dst_qd[:] = src_q[: len(dst_qd)]`` — shifts every joint
        downstream of a FREE joint by one slot. The per-joint loop
        here lands each value in the right place.

        For width-mismatch joints (FREE / BALL / D6) there's no
        meaningful element-wise mapping between q and qd: q carries a
        quaternion, qd carries an angular velocity vector. With
        ``skip_widthless=True`` (the default) we leave their qd slots
        unwritten — they're typically classified ``JK_PASSIVE`` so the
        actuator ignores their target value anyway. Pass
        ``skip_widthless=False`` to do a partial copy of the leading
        ``min(q_count, dof_count)`` entries (useful only for narrow
        Cartesian-only subsets — not generally meaningful).

        Returns the number of qd slots written.
        """
        n_copied = 0
        for s in self._slices:
            if s.q_count == s.dof_count:
                if s.q_count == 0:
                    continue  # fixed joint
                if s.q_start + s.q_count <= len(src_q) and s.qd_start + s.dof_count <= len(dst_qd):
                    dst_qd[s.qd_start : s.qd_start + s.dof_count] = src_q[s.q_start : s.q_start + s.q_count]
                    n_copied += s.dof_count
            elif not skip_widthless:
                k = min(s.q_count, s.dof_count)
                if k > 0 and s.q_start + k <= len(src_q) and s.qd_start + k <= len(dst_qd):
                    dst_qd[s.qd_start : s.qd_start + k] = src_q[s.q_start : s.q_start + k]
                    n_copied += k
            # else: skip (FREE / BALL widthless joints; qd slot ignored).
        return n_copied
