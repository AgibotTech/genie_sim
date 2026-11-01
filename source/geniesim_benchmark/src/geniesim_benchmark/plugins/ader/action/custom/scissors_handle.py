# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Scissors handle checkers: two-finger insertion + actuation under grip.

Both checkers are driven by ONE structured parameter in problems.json::

    {"ScissorsHandleInserted": {"object": "benchmark_scissors_006"}}
    {"ScissorsHandleActuated": {"object": "benchmark_scissors_006",
                                "hold_ticks": 2, "min_travel_ratio": 0.15}}

``object`` locates the scene instance (no asset whitelist, no per-model hole
coordinates).  The handle holes are parsed once from the composed USD (see
scissors_handle_geometry) and follow the rigid links with one pose read per
tick.

Grip rule (user-confirmed): the THUMB plus ONE other finger of the SAME hand
must pass through the TWO DIFFERENT holes (either assignment).  Either hand
may do it, but a left-right mix never passes.

ScissorsHandleInserted — latches after ``hold_ticks`` consecutive passing
ticks; independent score for "fingers correctly inserted".

ScissorsHandleActuated — within one continuously-held grip window, tracks the
scissors joint travel (max-min); passes when travel >= min_travel_ratio *
joint range for ``hold_ticks`` consecutive ticks.  A slipped finger RESETS
the window, so gravity-opened or after-the-fact openings never pass.  This is
direction-free, so no per-model angle interval is needed (benchmark_scissors_
001 has a negative joint range, the others positive).

RLinf note: these checkers need composed USD mesh topology via
``api_core.get_articulation_geometry``.  The benchmark APICore provides it;
the RLinf ApiCoreShim does NOT (poses/AABBs/joints only), so under RLinf the
checkers fail closed with a single warning.  Porting later means carrying the
parsed hole regions in SceneSchema and forwarding them through ApiCoreShim —
do not duplicate the USD parsing on the RLinf side.
"""

from __future__ import annotations

import math

import numpy as np

from geniesim_benchmark.plugins.ader.action.common_actions import (
    ActionBase,
    ActionEvent,
    EvaluateAction,
)
from geniesim_benchmark.plugins.ader.action.custom.scissors_handle_geometry import (
    HoleRegion,
    ScissorsGeometry,
    hole_contains_segment,
    resolve_handle_pose,
)
from geniesim_benchmark.plugins.logger import Logger

logger = Logger()

DEFAULT_HOLD_TICKS = 3
DEFAULT_TRAVEL_RATIO = 0.15

# Wuji five-finger dexterous hand link layout (see assets/robot/
# G2_crsB_wuji/robot_fix.usda): hand_{l|r}_{thumb|index|middle|ring|pinky}_
# link{2,3,4}.  A finger is a POLYLINE link2->link3->link4: a deep insertion
# can leave link3 and link4 both past the hole plane, so any one segment
# threading the ring proves insertion (link1 sits at the palm and is
# deliberately excluded — a palm-touch would falsely thread the ring).
_WUJI_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
_WUJI_NON_THUMB = ("index", "middle", "ring", "pinky")
_WUJI_BASE = "/genie/hand_{side}_{finger}_link{joint}"


def _wuji_finger_paths(side: str, finger: str) -> list[str]:
    s = "l" if side == "left" else "r"
    return [
        _WUJI_BASE.format(side=s, finger=finger, joint=j) for j in (2, 3, 4)
    ]


def _is_wuji(env) -> bool:
    robot_cfg = getattr(env, "robot_cfg", None)
    if robot_cfg is None:
        robot_cfg = getattr(env, "init_task_config", {}).get("robot_cfg", "")
    return "wuji" in str(robot_cfg).lower()


def evaluate_grip(
    api_core,
    geometry: ScissorsGeometry,
    clearance: float = 0.004,
) -> dict:
    """Same-hand thumb+finger through the two DIFFERENT holes, either hand.

    A finger counts as inserted when ANY segment of its link2->link3->link4
    polyline threads the hole (CROSS straddle or DEEP tip-in-slab).

    Returns {"ok": bool, "match": str} — ``match`` describes the passing pair
    ("" when not ok).
    """
    link_paths = list(geometry.links.keys())
    holes = []
    for link_path in link_paths:
        lg = geometry.links[link_path]
        if lg.hole is None:
            continue
        pose = resolve_handle_pose(api_core, link_path)
        holes.append(lg.hole.world_from_local(pose))
    if len(holes) != 2:
        return {"ok": False, "match": ""}

    def _finger_hits(side: str, finger: str) -> set[int]:
        """Hole indices (0/1) threaded by this finger's polyline."""
        poses = [
            np.asarray(api_core.get_obj_world_pose(p)[0], dtype=float)
            for p in _wuji_finger_paths(side, finger)
        ]
        hits = set()
        for hi, hole in enumerate(holes):
            for a, b in zip(poses, poses[1:]):
                if hole_contains_segment(hole, a, b, clearance=clearance):
                    hits.add(hi)
                    break
        return hits

    for side in ("left", "right"):
        thumb_hits = _finger_hits(side, "thumb")
        for finger in _WUJI_NON_THUMB:
            finger_hits = _finger_hits(side, finger)
            # Thumb and one other finger must occupy the two DIFFERENT holes
            # (either assignment); two fingers sharing a hole never passes.
            if ((0 in thumb_hits and 1 in finger_hits)
                    or (1 in thumb_hits and 0 in finger_hits)):
                return {"ok": True, "match": f"{side}:{finger}"}
    return {"ok": False, "match": ""}


class ScissorsHandleBase(EvaluateAction):
    """Shared plumbing: parameter handling + lazy geometry resolution."""

    def __init__(self, env, obj_name, hold_ticks=DEFAULT_HOLD_TICKS):
        super().__init__(env)
        self._obj_name = str(obj_name)
        self.hold_ticks = max(1, int(hold_ticks))
        self._geometry: ScissorsGeometry | None = None
        self._geometry_failed = False
        self._warned = False
        self._pass_ticks = 0
        self._done_flag = False
        self._last_grip = {"ok": False, "match": ""}

    # -- geometry ----------------------------------------------------------

    def _resolve_geometry(self) -> ScissorsGeometry | None:
        if self._geometry is not None or self._geometry_failed:
            return self._geometry
        getter = getattr(self.api_core, "get_articulation_geometry", None)
        if getter is None:
            # RLinf ApiCoreShim lands here: fail closed, warn once.
            if not self._warned:
                self._warned = True
                logger.warning(
                    f"[{type(self).__name__}] api_core has no get_articulation_geometry; "
                    "benchmark-only checker fails closed (see RLinf note in module docstring)"
                )
            self._geometry_failed = True
            return None
        try:
            geom = getter(self._analyze_obj_name(self._obj_name))
        except Exception as e:
            if not self._warned:
                self._warned = True
                logger.warning(
                    f"[{type(self).__name__}] geometry resolution failed for "
                    f"{self._obj_name!r}: {e}"
                )
            self._geometry_failed = True
            return None
        if geom is None:  # api_core without the capability (RLinf shim)
            if not self._warned:
                self._warned = True
                logger.warning(
                    f"[{type(self).__name__}] api_core returned no geometry for "
                    f"{self._obj_name!r}; checker fails closed (see RLinf note)"
                )
            self._geometry_failed = True
            return None
        if not geom.ok:
            reasons = "; ".join(
                f"{lg.link_path}: {lg.reject_reason}" for lg in geom.links.values()
            )
            logger.warning(
                f"[{type(self).__name__}] {self._obj_name!r} not usable as a "
                f"two-ring tool ({reasons}); checker fails closed"
            )
            self._geometry_failed = True
            return None
        self._geometry = geom
        return geom

    def _current_grip(self) -> dict:
        geom = self._resolve_geometry()
        if geom is None:
            return {"ok": False, "match": ""}
        try:
            grip = evaluate_grip(self.api_core, geom)
        except Exception as e:
            if not self._warned:
                self._warned = True
                logger.warning(f"[{type(self).__name__}] grip evaluation failed: {e}")
            return {"ok": False, "match": ""}
        self._warned = False  # transient errors may recover; allow one re-warn
        return grip

    # -- scoring ------------------------------------------------------------

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info(
            f"Action [{type(self).__name__}] obj={self._obj_name} "
            f"done={self._done_flag} evt={event.value}"
        )
        if event == ActionEvent.FINISHED:
            self.progress_info["SCORE"] = 1 if self._done_flag else 0


class ScissorsHandleInserted(ScissorsHandleBase):
    """Score the correct two-finger insertion, independently of the cutting."""

    def __init__(self, env, obj_name, hold_ticks=DEFAULT_HOLD_TICKS):
        super().__init__(env, obj_name, hold_ticks)

    def update(self, delta_time: float) -> float:
        if self._done_flag or not self.is_running():
            return super().update(delta_time)

        if not _is_wuji(self.env):
            if not self._warned:
                self._warned = True
                logger.warning(
                    "[ScissorsHandleInserted] non-Wuji robot: no finger segments "
                    "to thread the rings; checker fails closed"
                )
            return super().update(delta_time)

        self._last_grip = self._current_grip()
        if self._last_grip["ok"]:
            self._pass_ticks += 1
        else:
            self._pass_ticks = 0

        if self._pass_ticks >= self.hold_ticks:
            self._done_flag = True
            logger.info(
                "[ScissorsHandleInserted] PASSED: "
                f"{self._last_grip['match']} threads both rings "
                f"({self._pass_ticks}/{self.hold_ticks} ticks)"
            )
        return super().update(delta_time)


class ScissorsHandleActuated(ScissorsHandleBase):
    """Score actuation: joint travel accumulated under a continuously-held grip."""

    def __init__(
        self,
        env,
        obj_name,
        hold_ticks=2,
        min_travel_ratio=DEFAULT_TRAVEL_RATIO,
    ):
        super().__init__(env, obj_name, hold_ticks)
        self.min_travel_ratio = float(min_travel_ratio)
        if not 0.0 < self.min_travel_ratio < 1.0:
            raise ValueError(
                f"min_travel_ratio must be in (0,1), got {self.min_travel_ratio}"
            )
        # Grip session: starts when the grip first holds, resets on slip.
        self._session_min = None
        self._session_max = None
        self._travel_ok_ticks = 0

    def _joint_position(self, geom: ScissorsGeometry) -> float | None:
        info = self.api_core.get_obj_joint(self._analyze_obj_name(self._obj_name)) or {}
        names = list(info.get("joint_names") or [])
        positions = list(info.get("joint_positions") or [])
        if not positions:
            return None
        idx = None
        if geom.joint_name:
            for i, name in enumerate(names):
                if str(name) == geom.joint_name or str(name).rsplit("/", 1)[-1] == geom.joint_name:
                    idx = i
                    break
        if idx is None:
            # Fall back to the single-joint articulation case, else joint 0.
            idx = 0 if len(positions) == 1 or len(names) <= 1 else None
        if idx is None or idx >= len(positions):
            return None
        value = positions[idx]
        return None if value is None else float(value)

    def update(self, delta_time: float) -> float:
        if self._done_flag or not self.is_running():
            return super().update(delta_time)

        geom = self._resolve_geometry()
        if geom is None or not _is_wuji(self.env):
            return super().update(delta_time)

        self._last_grip = self._current_grip()
        if not self._last_grip["ok"]:
            # Finger slipped / scissors dropped: the window resets, so travel
            # accumulated before a slip can never satisfy the check later.
            self._session_min = None
            self._session_max = None
            self._travel_ok_ticks = 0
            return super().update(delta_time)

        position = self._joint_position(geom)
        if position is None:
            return super().update(delta_time)

        if self._session_min is None:
            self._session_min = self._session_max = position
        else:
            self._session_min = min(self._session_min, position)
            self._session_max = max(self._session_max, position)

        travel = self._session_max - self._session_min
        target = self.min_travel_ratio * abs(math.radians(geom.joint_upper) -
                                             math.radians(geom.joint_lower))
        if travel >= target:
            self._travel_ok_ticks += 1
        else:
            self._travel_ok_ticks = 0  # still holding, but not enough travel yet

        if self._travel_ok_ticks >= self.hold_ticks:
            self._done_flag = True
            logger.info(
                f"[ScissorsHandleActuated] PASSED: grip={self._last_grip['match']} "
                f"travel={travel:.3f}rad >= {target:.3f}rad "
                f"({self._travel_ok_ticks}/{self.hold_ticks} ticks)"
            )
        return super().update(delta_time)
