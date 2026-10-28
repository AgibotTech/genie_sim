#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Lock-down test for Task 2 — ownership-based scene-joint classification
+ class-level passive params + joint-map ownership filter (see
``docs/superpowers/plans/2026-07-23-articulated-scene-objects.md`` and the
T1 measurements in ``.superpowers/port/door-probe-facts.md``).

Problem T1 measured: a door hinge joint whose USD leaf name happens to be
``RevoluteJoint_00`` classified ``JK_PASSIVE`` by NAME luck (the
``_RE_PASSIVE_DEFAULT`` regex) — but a scene prop authored with a CUSTOM
joint name (``door_hinge_left``) would classify ``JK_OTHER`` and trip the
``unknown_joint_names`` hard-fail.  And regardless of name, the door
revolute joints LEAKED into ``/joint_states`` (the T1 probe measured
34→36) because ``topology.py::_build_joint_map`` filters only by Newton
joint TYPE (FREE/FIXED), with zero ownership awareness.

Task 2 closes both gaps by keying on body OWNERSHIP (does the joint's
parent/child body sit under the robot prefix), NOT on joint name:

  * ``object_classification.is_scene_joint`` — the pure ownership
    predicate (mirrors ``classify_shape``'s OK_ROBOT prefix test, at the
    joint level: a joint whose parent AND child bodies are both NOT under
    the robot prefix is a SCENE joint).
  * ``joint_classification.classify_scene_joint`` — routes a scene joint
    to ``JK_SCENE_ARTICULATION`` (non-free hinge/slide: zero PD gains PLUS
    the ``scene_articulation`` passive damping so it settles) or
    ``JK_PASSIVE`` (a free-floating scene body: zero PD, natural
    free-fall, no artificial damping), and returns ``None`` for
    robot-owned / undecidable joints so the caller keeps the byte-
    identical name-based path.

The classifiers are dependency-free (stdlib only), so this file imports
them directly.  The adapter per-class mapping test genuinely imports the
mjwarp adapter (``newton`` / ``warp``) and is guarded by ``importorskip``.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

_HERE = pathlib.Path(__file__).resolve()
# .../genie_sim_engine_newton/genie_sim_engine_newton/tools/<this>
_NEWTON_SRC_ROOT = _HERE.parents[2]  # dir containing the newton package
_SRC = _HERE.parents[3]  # ros_ws/.../src (holds both packages)
_CORE_SRC_ROOT = _SRC / "genie_sim_engine_core"  # dir containing the core package

for _p in (str(_NEWTON_SRC_ROOT), str(_CORE_SRC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Pure classifiers — no newton/warp/pxr dependency.
from genie_sim_engine_newton.common.object_classification import (  # noqa: E402
    body_is_robot,
    is_scene_joint,
)
from genie_sim_engine_newton.common.joint_classification import (  # noqa: E402
    JK_ARM_SHOULDER,
    JK_BODY,
    JK_GRIPPER,
    JK_OTHER,
    JK_PASSIVE,
    JK_SCENE_ARTICULATION,
    ALL_JOINT_KINDS,
    classify_joint_by_name,
    classify_scene_joint,
)

_ROBOT_PREFIX = "genie"

# ----------------------------------------------------------------------
# Body-label fixtures (Newton ``model.body_label`` strings).
# ----------------------------------------------------------------------
# Robot bodies sit under ``/<robot_prefix>/`` (or equal ``/<robot_prefix>``).
_ROBOT_ROOT = "/genie"
_ROBOT_BASE = "/genie/base_link"
_ROBOT_ARM = "/genie/arm_l_link1"
_ROBOT_GRIP = "/genie/gripper_l_inner_link1"
# Scene (door) bodies sit under /World — NOT the robot prefix.
_DOOR_FRAME = "/World/Objects/benchmark_door_000/body"
_DOOR_PANEL = "/World/Objects/benchmark_door_000/part_00"
_DOOR_KNOB = "/World/Objects/benchmark_door_000/part_07"
# World body (Newton joint parent == -1 → no label).
_WORLD = None


# ======================================================================
# body_is_robot — the shared prefix predicate
# ======================================================================
def test_body_is_robot_matches_prefix_and_root():
    assert body_is_robot(_ROBOT_BASE, _ROBOT_PREFIX)
    assert body_is_robot(_ROBOT_ARM, _ROBOT_PREFIX)
    assert body_is_robot(_ROBOT_ROOT, _ROBOT_PREFIX)  # exact ``/genie``


def test_body_is_robot_rejects_scene_world_and_empty():
    assert not body_is_robot(_DOOR_PANEL, _ROBOT_PREFIX)
    assert not body_is_robot(_WORLD, _ROBOT_PREFIX)  # world body
    assert not body_is_robot(_ROBOT_BASE, "")  # no prefix → undecidable
    # A body whose path merely CONTAINS "genie" as a non-prefix must not match.
    assert not body_is_robot("/World/genie_lookalike/body", _ROBOT_PREFIX)


# ======================================================================
# is_scene_joint — joint-level ownership
# ======================================================================
def test_scene_joint_both_bodies_non_robot():
    # Door hinge: frame → panel, both scene bodies.
    assert is_scene_joint(parent_body_label=_DOOR_FRAME, child_body_label=_DOOR_PANEL, robot_prefix=_ROBOT_PREFIX)
    # Door free-body frame: world → frame (parent is world, child is scene).
    assert is_scene_joint(parent_body_label=_WORLD, child_body_label=_DOOR_FRAME, robot_prefix=_ROBOT_PREFIX)


def test_robot_joint_is_not_scene():
    # Robot internal joint: both bodies robot.
    assert not is_scene_joint(parent_body_label=_ROBOT_BASE, child_body_label=_ROBOT_ARM, robot_prefix=_ROBOT_PREFIX)
    # Robot floating base: world → robot base (child robot ⇒ robot-owned).
    assert not is_scene_joint(parent_body_label=_WORLD, child_body_label=_ROBOT_BASE, robot_prefix=_ROBOT_PREFIX)
    # Robot gripper coupler: both bodies robot.
    assert not is_scene_joint(parent_body_label=_ROBOT_GRIP, child_body_label=_ROBOT_GRIP, robot_prefix=_ROBOT_PREFIX)


def test_empty_prefix_is_never_scene():
    # No robot_prefix ⇒ ownership undecidable ⇒ fall through to name path.
    assert not is_scene_joint(parent_body_label=_DOOR_FRAME, child_body_label=_DOOR_PANEL, robot_prefix="")


# ======================================================================
# classify_scene_joint — the JK routing (①: custom-named door exempt)
# ======================================================================
def test_custom_named_scene_hinge_routes_to_scene_articulation():
    """A door hinge with a CUSTOM name that matches NO joint regex would
    trip the ``unknown_joint_names`` hard-fail on the name path — proving
    the ownership route (returning a non-None kind) is what exempts it."""
    # Sanity: the custom name really is unclassifiable by name alone.
    assert classify_joint_by_name("door_hinge_left") == JK_OTHER
    # Ownership routes it to the scene-articulation class (non-free hinge).
    kind = classify_scene_joint(
        is_free=False,
        parent_body_label=_DOOR_FRAME,
        child_body_label=_DOOR_PANEL,
        robot_prefix=_ROBOT_PREFIX,
    )
    assert kind == JK_SCENE_ARTICULATION
    assert kind is not None  # ⇒ never appended to unknown_joint_names


def test_free_scene_body_routes_to_passive_not_articulation():
    """A free-floating scene body (single 6-DOF FREE joint to world) stays
    JK_PASSIVE — natural free-fall, NO artificial scene damping (free table
    and block bodies must keep their pre-Task-2 behaviour)."""
    kind = classify_scene_joint(
        is_free=True,
        parent_body_label=_WORLD,
        child_body_label=_DOOR_FRAME,
        robot_prefix=_ROBOT_PREFIX,
    )
    assert kind == JK_PASSIVE


def test_robot_joint_routes_to_none():
    """②: robot-owned joints get None from the scene router ⇒ the adapter
    keeps the byte-identical name-based classification for them."""
    for is_free in (False, True):
        assert (
            classify_scene_joint(
                is_free=is_free,
                parent_body_label=_ROBOT_BASE,
                child_body_label=_ROBOT_ARM,
                robot_prefix=_ROBOT_PREFIX,
            )
            is None
        )
    # Robot floating base (world → robot base) is robot-owned, not scene.
    assert (
        classify_scene_joint(
            is_free=True,
            parent_body_label=_WORLD,
            child_body_label=_ROBOT_BASE,
            robot_prefix=_ROBOT_PREFIX,
        )
        is None
    )


def test_scene_articulation_kind_registered():
    assert JK_SCENE_ARTICULATION in ALL_JOINT_KINDS
    assert JK_SCENE_ARTICULATION == "scene_articulation"


# ======================================================================
# ②: robot-joint NAME classification is byte-identical (ownership never
# perturbs the name path for the canonical G2 joint set).
# ======================================================================
# A representative slice of the G2 joint set + expected name-based kind.
_G2_NAME_EXPECT = {
    "idx01_body_joint1": JK_BODY,
    "idx21_arm_l_joint1": JK_ARM_SHOULDER,
    "idx31_gripper_l_inner_joint1": JK_GRIPPER,
    "idx41_gripper_l_outer_joint1": JK_GRIPPER,
}


def test_robot_joint_names_classify_unchanged():
    for name, expect in _G2_NAME_EXPECT.items():
        assert classify_joint_by_name(name) == expect
    # And every one of them is robot-owned (both bodies under /genie/),
    # so the scene router returns None and leaves the name result standing.
    for name in _G2_NAME_EXPECT:
        assert (
            classify_scene_joint(
                is_free=False,
                parent_body_label=f"/genie/{name}_parent",
                child_body_label=f"/genie/{name}_child",
                robot_prefix=_ROBOT_PREFIX,
            )
            is None
        )


# ======================================================================
# ③: joint-map ownership filter — scene joints excluded, robot kept.
# The filter's decision IS ``is_scene_joint``; assert it directly on the
# T1 door-probe body pairs.  (The live 36→34 wiring is re-validated by
# the empirical door-probe re-run recorded in door-probe-facts.md.)
# ======================================================================
def test_joint_map_filter_excludes_only_scene_joints():
    # The two door revolute joints the T1 probe measured leaking (36).
    assert is_scene_joint(
        parent_body_label=_DOOR_FRAME, child_body_label=_DOOR_PANEL, robot_prefix=_ROBOT_PREFIX
    )  # RevoluteJoint_00 → excluded
    assert is_scene_joint(
        parent_body_label=_DOOR_FRAME, child_body_label=_DOOR_KNOB, robot_prefix=_ROBOT_PREFIX
    )  # RevoluteJoint_05 → excluded
    # Robot joints stay in the map (at least one body under the prefix).
    assert not is_scene_joint(parent_body_label=_ROBOT_BASE, child_body_label=_ROBOT_ARM, robot_prefix=_ROBOT_PREFIX)
    assert not is_scene_joint(parent_body_label=_ROBOT_GRIP, child_body_label=_ROBOT_GRIP, robot_prefix=_ROBOT_PREFIX)


# ======================================================================
# Passive params — the ``scene_articulation`` YAML class block.
# ======================================================================
def test_scene_articulation_default_params():
    from genie_sim_engine_core.common.params import default_physics_params

    p = default_physics_params()
    assert hasattr(p, "drive_scene_articulation")
    assert p.drive_scene_articulation.dof_damping == pytest.approx(0.5)
    assert p.drive_scene_articulation.dof_frictionloss == pytest.approx(0.1)


def test_scene_articulation_yaml_block_parses():
    yaml = pytest.importorskip("yaml")  # noqa: F841 — presence gate
    from genie_sim_engine_core.common.params import load_physics_params

    cfg = _CORE_SRC_ROOT / "config" / "physics_params.yaml"
    assert cfg.is_file(), f"physics_params.yaml missing at {cfg}"
    p = load_physics_params(str(cfg))
    # The block must exist and carry the Task-2 starting values.
    assert p.drive_scene_articulation.dof_damping == pytest.approx(0.5)
    assert p.drive_scene_articulation.dof_frictionloss == pytest.approx(0.1)


# ======================================================================
# ① mapping — the mjwarp adapter routes JK_SCENE_ARTICULATION to zero PD
# gains + the scene_articulation passive params.  Genuinely imports the
# adapter (newton/warp); guarded.
# ======================================================================
def test_adapter_maps_scene_articulation_zero_gains_and_passive():
    pytest.importorskip("newton")
    pytest.importorskip("warp")
    from genie_sim_engine_core.common.params import default_physics_params
    from genie_sim_engine_newton.engine.newton.adapters.mujoco_warp_multi_env import (
        MuJoCoWarpMultiEnvAdapter,
    )

    adapter = MuJoCoWarpMultiEnvAdapter(physics_params=default_physics_params())

    gains = adapter._per_class_gains()
    assert gains is not None
    # Zero PD gains — a scene hinge is passive, no ghost drive (matches
    # JK_PASSIVE's zero-gain path, per T1 Fact 2/4).
    assert gains[JK_SCENE_ARTICULATION] == (0.0, 0.0, 0.0)

    passive = adapter._per_class_passive_joint()
    assert passive is not None
    dof_damping, dof_frictionloss, _armature = passive[JK_SCENE_ARTICULATION]
    assert dof_damping == pytest.approx(0.5)
    assert dof_frictionloss == pytest.approx(0.1)


def test_none_none_labels_read_as_scene_hence_call_sites_must_guard():
    """Documents WHY both classification call sites carry an arrays-None guard.

    ``is_scene_joint(None, None, prefix)`` is True by design (a world-anchored
    scene joint has no robot-owned side).  The dangerous corollary: a model
    with NO ``joint_parent``/``joint_child`` arrays resolves EVERY joint's
    labels to (None, None) — without the call-site guard (mujoco_warp
    ``_jp is not None and _jc is not None``; topology ``_is_scene_owned``'s
    early False) every actuated ROBOT joint would silently route to the
    zero-gain scene path and the robot would go limp.  If this test's premise
    ever changes, re-examine both guards before relying on it.
    """
    from genie_sim_engine_newton.common.object_classification import is_scene_joint

    # The premise the guards protect against:
    assert is_scene_joint(parent_body_label=None, child_body_label=None, robot_prefix="genie") is True
    # And the normal robot case stays non-scene even one-sided:
    assert (
        is_scene_joint(
            parent_body_label="/genie/base_link",
            child_body_label=None,
            robot_prefix="genie",
        )
        is False
    )
