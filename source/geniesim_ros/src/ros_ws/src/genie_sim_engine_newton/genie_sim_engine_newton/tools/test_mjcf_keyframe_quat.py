#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pure host test (no GPU, no Newton) for ``add_init_pose_keyframe``'s
FREE-joint quaternion reordering.

Newton stores a FREE joint's orientation in ``model.joint_q`` as
(x, y, z, w); a MuJoCo keyframe's ``qpos`` expects (w, x, y, z).
Writing the Newton vector verbatim silently rotates every free body
about the wrong axis, so a pure-MuJoCo replay of the dumped MJCF starts
from a different state than the live engine.

These tests assert the round trip through a real ``mj_resetDataKeyframe``
rather than the byte layout, so they stay honest if the reorder moves.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

import pytest

_HERE = pathlib.Path(__file__).resolve()
# .../genie_sim_engine_newton/genie_sim_engine_newton/tools/<this>
_NEWTON_SRC_ROOT = _HERE.parents[2]  # dir containing the newton package
if str(_NEWTON_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_NEWTON_SRC_ROOT))

mujoco = pytest.importorskip("mujoco", reason="keyframe quat check needs the mujoco package")

import numpy as np  # noqa: E402

from genie_sim_engine_newton.common.mjcf_postprocess import add_init_pose_keyframe  # noqa: E402

# One free body (7 qpos) followed by one hinge (1 qpos) — enough to prove
# the reorder hits the free joint's quat slice and nothing else.
_XML = """<mujoco><worldbody>
  <body name="table" pos="0 0 1"><freejoint/><geom type="box" size=".3 .3 .02"/></body>
  <body name="arm" pos="1 0 0">
    <joint name="j0" type="hinge" axis="0 0 1"/>
    <geom type="capsule" fromto="0 0 0 .2 0 0" size=".02"/>
  </body>
</worldbody></mujoco>"""

_S = 0.7071067811865476  # sin/cos of 45 deg — the 90-deg quat's two live components


def _write_and_restore(qpos, tmp_path):
    """Author the keyframe, reload, restore it, return the MjData."""
    path = str(tmp_path / "scene.xml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_XML)
    add_init_pose_keyframe(mjcf_path=path, qpos=qpos, name="home", logger=None)
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return model, data


def test_free_joint_quat_survives_the_keyframe_round_trip(tmp_path):
    """A 90-deg-about-X pose must still be about X after the round trip.

    Passing the Newton XYZW vector through verbatim turns this into
    90 deg about Z — the failure that spawned the benchmark table in
    deep penetration and exploded pure-MuJoCo replay at t=0.
    """
    # Newton-side layout: free joint (xyz + quat XYZW), then the hinge.
    newton_qpos = np.array([0.0, 0.0, 1.0, _S, 0.0, 0.0, _S, 0.3], dtype=float)
    model, data = _write_and_restore(newton_qpos, tmp_path)

    # A rotation about +X maps the body's +Y axis onto world +Z.
    y_axis = data.xmat[model.body("table").id].reshape(3, 3)[:, 1]
    assert np.allclose(
        y_axis, [0.0, 0.0, 1.0], atol=1e-4
    ), f"free-joint quat garbled: expected +Y -> +Z (rotation about X), got {y_axis}"


def test_non_free_dofs_are_untouched(tmp_path):
    """The reorder must not disturb hinge/slide qpos entries."""
    newton_qpos = np.array([0.0, 0.0, 1.0, _S, 0.0, 0.0, _S, 0.3], dtype=float)
    _model, data = _write_and_restore(newton_qpos, tmp_path)
    assert data.qpos[7] == pytest.approx(0.3, abs=1e-9)


def test_translation_is_untouched(tmp_path):
    """Only the quat slice moves; the free joint's xyz stays put."""
    newton_qpos = np.array([0.1, -0.2, 1.0, _S, 0.0, 0.0, _S, 0.0], dtype=float)
    _model, data = _write_and_restore(newton_qpos, tmp_path)
    assert np.allclose(data.qpos[0:3], [0.1, -0.2, 1.0], atol=1e-9)


def test_length_mismatch_leaves_qpos_alone(tmp_path):
    """A wrong-length qpos must warn and write verbatim, not corrupt.

    ``nq`` is 8 here; hand it 4 values.  The reorder can't know which
    slot is which, so it must bail rather than guess.
    """
    path = str(tmp_path / "scene.xml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_XML)
    add_init_pose_keyframe(mjcf_path=path, qpos=np.array([1.0, 2.0, 3.0, 4.0]), name="home", logger=None)
    # MuJoCo pads a short keyframe with zeros; the point is that the four
    # authored values arrive in the order given, unreordered.
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    assert "1 2 3 4" in text, "short qpos should be written verbatim after the reorder bails"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
