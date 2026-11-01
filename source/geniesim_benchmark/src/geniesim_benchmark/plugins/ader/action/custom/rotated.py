# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from geniesim_benchmark.plugins.ader.action.common_actions import (
    EvaluateAction,
    ActionBase,
    ActionEvent,
)
import numpy as np
from geniesim_benchmark.plugins.logger import Logger

logger = Logger()


def _rotation_matrix_to_angle_rad(R: np.ndarray) -> float:
    """Extract rotation angle (radians) from a 3x3 rotation matrix.

    Same math as stable_grasp._rotation_matrix_to_angle_rad, kept local so
    this checker never has to import (or modify) StableGrasp.
    """
    trace = np.trace(R)
    cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
    return float(np.arccos(cos_angle))


def _rotation_matrix_to_axis(R: np.ndarray) -> np.ndarray:
    """Rotation axis (unit vector) of a 3x3 rotation matrix.

    The axis is the eigenvector of R with eigenvalue 1. For near-identity
    rotations the formula degenerates; callers treat tiny angles as "no
    rotation" before asking for the axis.
    """
    # v = (R32-R23, R13-R31, R21-R12); ||v|| = 2*sin(angle)
    v = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]
    )
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0])
    return v / norm


class Rotated(EvaluateAction):
    """Evaluate whether a free rigid body has rotated away from its initial pose.

    Complements the existing orientation/joint checkers:

    * ``Upright``    — absolute pose vs the world vertical (degrees);
    * ``PushPull``   — articulated joint position (rad), e.g. hinged doors;
    * ``Rotated``    — free-body orientation change vs the FIRST-TICK pose.

    Config formats (thresholds are config-supplied, never hardcoded):

        "Rotated": "obj|angle_deg"              # total rotation angle
        "Rotated": "obj|angle_deg|ax,ay,az"     # signed component about the
                                                # given world axis only

    ``obj`` may be a single prim name, a comma-separated candidate list
    (passes when ANY candidate rotates past the threshold; the matched
    candidate is published to env placeholder_str1 so downstream checkers
    follow the SAME object — PickUpOnGripper pattern), or a placeholder
    ("{@placeholder_str1}") resolved at runtime from an earlier checker.
    In multi-candidate mode each candidate gets its own initial orientation
    record, so untouched neighbours never fire.

    With an axis, rotation about other axes is ignored: a book spun in place
    around the vertical does not count as "tilted out" when the axis pins the
    shelf-row direction. Without an axis, any rotation beyond the threshold
    passes (generic; use where the motion axis is not known).

    Debounce: 2 consecutive pass ticks (house style — InBBox /
    RelativePositionChecker), then latch. Angle unit is degrees in config,
    radians internally, matching Upright's convention.
    """

    def __init__(self, env, obj_name, angle_threshold_deg=15.0, axis=None):
        super().__init__(env)
        self._holder_name, self._obj_name = self.placeholder_sparser(obj_name)
        self._obj_candidates = None
        if not self._holder_name:
            self._obj_candidates = self.parse_obj_input(obj_name)
        self._done_flag = False
        self._pass_frame = 0
        self._matched_obj = None

        self.angle_threshold_deg = float(angle_threshold_deg)
        self.angle_threshold_rad = np.radians(self.angle_threshold_deg)

        # Optional signed world axis "ax,ay,az" (rotation about this axis
        # counts with its sign; other axes are projected out).
        self.axis = None
        if axis is not None:
            v = np.array([float(a) for a in axis], dtype=np.float64)
            norm = np.linalg.norm(v)
            if norm > 1e-9:
                self.axis = v / norm
            else:
                logger.warning(f"[Rotated] degenerate axis {axis!r}; using total angle")

        # Initial orientation per object, recorded on the first update tick
        # that sees the object (mirrors LiftUp's initial_z bookkeeping).
        self._initial_R_map: dict = {}
        # Legacy single-object view on the same store (keeps the property
        # `_initial_R` semantics for direct construction in tests).
        self._initial_R: np.ndarray | None = None

    @property
    def obj_name(self) -> str:
        if self._holder_name:
            return getattr(self, self._obj_name)
        return self._obj_name

    @property
    def obj_name_list(self) -> list:
        """Candidate list: placeholder value if set, else the parsed list."""
        if self._holder_name:
            resolved = getattr(self, self._obj_name)
            # Fall back to the previously matched object so a later-emptied
            # placeholder still checks the object this checker locked onto.
            if resolved:
                return [resolved]
            if self._matched_obj:
                return [self._matched_obj]
            return []
        return self._obj_candidates or [self._obj_name]

    def _check_rotated_obj(self, obj_name, obj_pose) -> bool:
        current_R = np.array(obj_pose[:3, :3], dtype=np.float64)

        if obj_name not in self._initial_R_map:
            self._initial_R_map[obj_name] = current_R.copy()
            if not self._holder_name and len(self.obj_name_list) == 1:
                self._initial_R = self._initial_R_map[obj_name]
            logger.info(
                f"[Rotated] Initial orientation recorded for {obj_name}; "
                f"threshold {self.angle_threshold_deg:.2f} deg"
            )
            return False

        delta_R = current_R @ self._initial_R_map[obj_name].T
        angle_rad = _rotation_matrix_to_angle_rad(delta_R)

        if self.axis is not None:
            # Signed rotation component about the configured axis: the
            # rotation vector is angle * axis(delta_R); project it.
            delta_axis = _rotation_matrix_to_axis(delta_R)
            signed_rad = angle_rad * float(np.dot(delta_axis, self.axis))
            passed = signed_rad >= self.angle_threshold_rad
            measure = f"about-axis {signed_rad:.3f} rad"
        else:
            passed = angle_rad >= self.angle_threshold_rad
            measure = f"total {angle_rad:.3f} rad"

        if passed:
            logger.info(
                f"[Rotated] {obj_name} rotated: {measure} "
                f">= {self.angle_threshold_rad:.3f} rad "
                f"({self.angle_threshold_deg:.2f} deg)"
            )
        return passed

    def check_rotated(self, obj_pose) -> bool:
        """Legacy single-object entry (direct construction / tests)."""
        return self._check_rotated_obj(self.obj_name, obj_pose)

    def update(self, delta_time: float) -> float:
        if self._done_flag:
            return super().update(delta_time)

        candidates = self.obj_name_list
        if not candidates:
            # Placeholder not yet populated: nothing to check this tick.
            return super().update(delta_time)

        # Unified path (a single configured object is the one-candidate
        # case): ANY candidate past the threshold; the 2 consecutive ticks
        # must agree on the SAME candidate — behaviour identical to the
        # legacy single-object implementation for one-candidate configs.
        matched = self.find_matching_object(
            candidates,
            lambda name: self._check_rotated_obj(name, self.get_obj_pose(name)),
        )
        if matched is not None:
            if matched == self._matched_obj:
                self._pass_frame += 1
            else:
                self._matched_obj = matched
                self._pass_frame = 1
            if self._pass_frame >= 2:
                self._done_flag = True
                # Publish ONLY in true multi-candidate mode: single-object
                # configs must not clobber placeholders owned by other
                # checkers (Follow/PickUpOnGripper). For take_the_book the
                # downstream InBBox (also multi-candidate) republishes the
                # EXTRACTED book, making it the binding authority — this
                # transient tilt-stage value is always overwritten before
                # the terminal composite can pass.
                if self._obj_candidates is not None and len(self._obj_candidates) > 1:
                    self.env.update_place_holder("placeholder_str1", matched)
                    logger.info(f"[Rotated] Matched object locked: {matched}")
        else:
            self._pass_frame = 0

        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info(f"Action [Rotated] obj: {self.obj_name}, evt: {event.value}")

        if event == ActionEvent.STARTED:
            pass
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            pass
        elif event == ActionEvent.FINISHED:
            if self._done_flag:
                self.progress_info["SCORE"] = 1
                logger.info(f"[Rotated] Task completed successfully for {self.obj_name}")
            else:
                self.progress_info["SCORE"] = 0
                logger.warning(f"[Rotated] Task failed for {self.obj_name}")
