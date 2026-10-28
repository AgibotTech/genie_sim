# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Module-level Warp kernels used by ``NewtonHeadlessEngine``.

Defined here (rather than as attributes of the engine class) so callers
can ``from engine.newton.kernels import _kernel_X`` without importing
the engine itself.
"""

from __future__ import annotations

import warp as wp


@wp.kernel(enable_backward=False)
def _kernel_velocity_inject_masked(
    joint_q: wp.array(dtype=float),
    target_pos: wp.array(dtype=float),
    controlled_mask: wp.array(dtype=int),
    joint_qd_out: wp.array(dtype=float),
):
    """Velocity-injection over a controlled-DOF mask only.

    For each DOF ``i``:
      * ``controlled_mask[i] != 0`` → set ``joint_qd_out[i] = target_pos[i] - joint_q[i]``
                                       (so the integrator drives q toward target).
      * ``controlled_mask[i] == 0`` → leave ``joint_qd_out[i]`` untouched, so
                                       Newton's solver-produced velocity (from
                                       gravity, contacts, joint reactions) is
                                       preserved.

    The mask keeps passive dynamic bodies (hanger, dropped object,
    ragdoll) moving under physics: their DOFs are uncontrolled, so the
    kernel doesn't overwrite the integrator's velocity.

    Mask is built once at ``post_joint_map`` time from ``joint_name_to_dof``
    — every DOF the controller addresses by name is "controlled"; everything
    else (FREE-joint twists for passive bodies, dynamic ragdoll DOFs,
    anything Newton parsed but no ROS topic commands) is "uncontrolled"
    and Newton handles it as a normal dynamic DOF.
    """
    i = wp.tid()
    if controlled_mask[i] != 0:
        joint_qd_out[i] = target_pos[i] - joint_q[i]
