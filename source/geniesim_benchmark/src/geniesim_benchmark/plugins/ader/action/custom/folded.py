# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Staged fold checkers for the cloth-fold task.

Judges the double-layer t-shirt's fold progress from the FEM cloth's simulation
nodal positions, mirroring the three grasp-cycles choreographed in the reference
demo (cloth_fold_fem_tshirt001_gripper_standalone.py):

  1. FoldCycle1 - both arms carry the hem corners (idx 2573/17766) onto the
     collar/hem anchors (idx 1349/17310).
  2. FoldCycle2 - both arms carry the collar corners (idx 1652/18532) onto the
     same anchors (idx 1349/17310).
  3. FoldCycle3 - single arm tucks the hem anchor (idx 17310) onto the collar
     anchor (idx 1349), the final adjustment.

Each stage checks exactly what the reference demo's own check_round() checks
for that cycle: has the vertex carried in that cycle actually arrived at its
target anchor vertex. This is a different (and more literal) criterion than
comparing the two cycles' grab points to each other.

Stage ordering is enforced by the Act tree, not by this file: problems.json
runs the three cycles as an ActionList of ActionSetWaitAny(cycle, StepOut)
stages, so each cycle only starts ticking once the previous one passed, and a
cycle that misses its own step budget ends the episode instead of burning the
remaining steps. The criteria are pure vertex-distance checks on the current
cloth state, so a policy that folds cycle2 before cycle1 still scores both:
cycle2's check passes immediately once its stage starts.

Thresholds (ROUND_DONE_M in the reference demo) are a first guess only -- they
were calibrated under a different physics setup (no jaw collision, pure
velocity-servo pin) than this benchmark's real FEM cloth-pin mechanism, and
should be recalibrated from real benchmark runs using the periodic calibration
log below.
"""

import numpy as np

from geniesim_benchmark.plugins.ader.action.common_actions import (
    EvaluateAction,
    ActionBase,
    ActionEvent,
)
from geniesim_benchmark.plugins.logger import Logger

logger = Logger()

# Require the fold to hold for >1 consecutive check ticks (debounce, like Stack/InBBox).
PASS_FRAMES = 2


def _pair_dist(P, i, j):
    """Return the distance (m) between vertices i and j, or None if out of range."""
    if i >= len(P) or j >= len(P):
        return None
    return float(np.linalg.norm(P[i] - P[j]))


def _check_pairs(P, pairs):
    """Shared AND-across-pairs check: each pairs[name] = (i, j, threshold_m)."""
    ok = True
    parts = []
    for nm, (i, j, thr) in pairs.items():
        d = _pair_dist(P, i, j)
        if d is None:
            return False, f"index {max(i, j)} out of range ({len(P)})"
        g = d <= thr
        ok = ok and g
        parts.append(f"{nm}={d*100:.1f}{'v' if g else 'x'}<={thr*100:.0f}")
    return ok, "  ".join(parts)


class _ClothFoldStageCheck(EvaluateAction):
    """Shared skeleton for cloth-fold stage checkers: debounced stage check -> SCORE."""

    stage_name = "Stage"

    def __init__(self, env, cloth_name="auto"):
        super().__init__(env)
        self.cloth_name = cloth_name  # reserved; get_cloth_nodal_positions auto-finds the cloth
        self._done_flag = False
        self._pass_frame = 0
        self._log_n = 0

    def _check_stage(self, P):
        """Return (done: bool, detail: str). Subclasses implement the actual criteria."""
        raise NotImplementedError

    def _check_fold(self):
        pos = self.api_core.get_cloth_nodal_positions()
        if pos is None or len(pos) == 0:
            return False, "no cloth nodal positions"
        return self._check_stage(np.asarray(pos))

    def update(self, delta_time: float) -> float:
        if self._done_flag:
            return super().update(delta_time)

        done, detail = self._check_fold()
        if done:
            self._pass_frame += 1
            if self._pass_frame > PASS_FRAMES - 1:
                self._done_flag = True
                logger.info(f"[{self.stage_name}] DONE  {detail}")
        else:
            self._pass_frame = 0
        # Periodic calibration log (helps confirm indices/scale, esp. the flat baseline).
        self._log_n += 1
        if self._log_n % 20 == 1:
            logger.info(f"[{self.stage_name}] {detail}")

        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info(f"Action [{self.stage_name}] evt: {event.value}")
        if event == ActionEvent.FINISHED:
            self.progress_info["SCORE"] = 1 if self._done_flag else 0


# Cycle 1: both arms carry the hem corners onto the collar/hem anchors.
# (name, grab_idx, anchor_idx, threshold_m) -- ROUND_DONE_M["cycle1"] from the reference demo,
# calibrated under a different physics setup; recalibrate against real benchmark runs.
CYCLE1_PAIRS = {"A": (2573, 1349, 0.12), "B": (17766, 17310, 0.08)}


class FoldCycle1(_ClothFoldStageCheck):
    stage_name = "FoldCycle1"

    def _check_stage(self, P):
        return _check_pairs(P, CYCLE1_PAIRS)


# Cycle 2: both arms carry the collar corners onto the same anchors.
# ROUND_DONE_M["cycle2"] from the reference demo -- same recalibration caveat as cycle1.
CYCLE2_PAIRS = {"A": (1652, 1349, 0.14), "B": (18532, 17310, 0.10)}


class FoldCycle2(_ClothFoldStageCheck):
    stage_name = "FoldCycle2"

    def _check_stage(self, P):
        return _check_pairs(P, CYCLE2_PAIRS)


# Cycle 3: single arm (B) tucks the hem anchor onto the collar anchor, the final adjustment.
# ROUND_DONE_M["cycle3"] from the reference demo (== its FOLD_DONE_M["top_bottom"]).
# Deliberately no footprint/thickness anti-crumple gate here -- ported as a literal 1:1
# match of the reference demo's own cycle3 criterion, not the stricter final check_fold.
CYCLE3_PAIRS = {"B": (17310, 1349, 0.20)}


class FoldCycle3(_ClothFoldStageCheck):
    stage_name = "FoldCycle3"

    def _check_stage(self, P):
        return _check_pairs(P, CYCLE3_PAIRS)
