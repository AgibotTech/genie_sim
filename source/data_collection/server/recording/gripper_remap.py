# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Gripper values written by the data-collection converter.

Target range: ``[-0.785, 0]`` rad — ``-0.785`` (≈ -π/4) fully open, ``0`` fully closed.

The raw omnipicker joint is already an angle (URDF
``idx41_gripper_l_outer_joint1`` limit ``[0, π/4]``). The state conversion
negates that angle; the action conversion maps it to an open/close command.
"""

GRIPPER_OPEN_RAD = -0.785    # fully open (≈ -π/4)
GRIPPER_CLOSED_RAD = 0.0     # fully closed

# Action transition band on the raw joint angle: g <= 0.45 closed, g >= 0.75 open.
_ACTION_CLOSE_THR = 0.45
_ACTION_OPEN_THR = 0.75


def omnipicker_state_to_rad(g_pos):
    """Faithful joint angle [0, 0.785] (0=closed, 0.785=open) -> [-0.785, 0]."""
    g = min(max(float(g_pos), 0.0), -GRIPPER_OPEN_RAD)   # clamp to [0, 0.785]
    return -g


def omnipicker_action_to_rad(g_pos):
    """Saturating open/close command from the joint angle -> [-0.785, 0].

    Monotone, continuous: closed (0) for g <= 0.45, open (-0.785) for g >= 0.75,
    linear in between.
    """
    g = float(g_pos)
    if g >= _ACTION_OPEN_THR:
        return GRIPPER_OPEN_RAD
    if g <= _ACTION_CLOSE_THR:
        return GRIPPER_CLOSED_RAD
    return GRIPPER_OPEN_RAD * (g - _ACTION_CLOSE_THR) / (_ACTION_OPEN_THR - _ACTION_CLOSE_THR)


# ---------------------------------------------------------------------------
# ctek90d (G2 90D gripper)
# ---------------------------------------------------------------------------
# Faithful joint angle: raw 0 = OPEN, |angle| = CLOSED. Both hands close toward
# -52.13916° (≈ -0.910 rad), so the raw value is negative-closing on both sides;
# we take abs() to be sign-agnostic. Unified sign convention matches omnipicker
# (0 = closed, negative = open), but the 90D physical span is 0.910 rad.
G2_90D_CLOSED_ABS = 0.910084   # |closed angle| rad (= radians(52.13916))

# Action transition band on |raw| (fraction-of-span mirrors omnipicker's 0.45/0.75):
_G2_90D_ACTION_OPEN_THR = G2_90D_CLOSED_ABS * 0.045   # |g| <= ~0.041 -> open
_G2_90D_ACTION_CLOSE_THR = G2_90D_CLOSED_ABS * 0.427  # |g| >= ~0.389 -> closed


def g2_90d_state_to_rad(g_pos):
    """ctek90d joint (0=open, |0.910|=closed, either sign) -> faithful [-0.910, 0].

    0 = open, -0.910 = closed. Both hands close toward |0.910|; we take abs()
    then negate so the output is sign-agnostic on the raw joint.
    """
    a = min(max(abs(float(g_pos)), 0.0), G2_90D_CLOSED_ABS)   # |g| in [0, 0.910]
    return -a


def g2_90d_action_to_rad(g_pos):
    """Saturating open/close command for ctek90d -> [-0.910, 0].

    open (0) for |g| <= ~0.041, closed (-0.910) for |g| >= ~0.389, linear between.
    """
    a = abs(float(g_pos))
    if a <= _G2_90D_ACTION_OPEN_THR:
        return 0.0
    if a >= _G2_90D_ACTION_CLOSE_THR:
        return -G2_90D_CLOSED_ABS
    # linear: open_thr -> 0 ; close_thr -> -0.910
    frac = (a - _G2_90D_ACTION_OPEN_THR) / (_G2_90D_ACTION_CLOSE_THR - _G2_90D_ACTION_OPEN_THR)
    return -G2_90D_CLOSED_ABS * frac
