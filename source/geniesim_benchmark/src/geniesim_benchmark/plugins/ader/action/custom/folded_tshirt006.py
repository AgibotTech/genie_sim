# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Staged fold checkers for the benchmark_tshirt_006 single-layer garment task.

Judges benchmark_tshirt_006's fold progress from the FEM cloth's simulation
nodal positions, mirroring the three grasp-cycles choreographed in the
reference demo (cloth_fold_fem_tshirt006_gripper_standalone.py):

  1. FoldCycle1Tshirt006 - both arms carry one side's corners (idx 4276/16889)
     onto the opposite-side anchors (idx 7369/18392).
  2. FoldCycle2Tshirt006 - both arms carry the other side's corners
     (idx 8941/19027) onto the same anchors (idx 7369/18392).
  3. FoldCycle3Tshirt006 - single arm (B) tucks the hem anchor (idx 18392)
     onto the other anchor (idx 7369), the final adjustment.

Stage ordering is enforced by the Act tree (an ActionList of
ActionSetWaitAny(cycle, StepOut) stages), not by this file -- see folded.py.

Thresholds (ROUND_DONE_M in the reference demo) were calibrated directly on
this garment (3 headless full runs, cross-checked against --screenshot
captures confirmed by eye to be a compact fold) -- see the constants and
comments in cloth_fold_fem_tshirt006_gripper_standalone.py.
"""

from geniesim_benchmark.plugins.ader.action.custom.folded import _ClothFoldStageCheck, _check_pairs

# Cycle 1: both arms carry one side's corners onto the opposite-side anchors.
# (name, grab_idx, anchor_idx, threshold_m) -- ROUND_DONE_M["cycle1"] from the reference demo.
CYCLE1_PAIRS = {"A": (4276, 7369, 0.08), "B": (16889, 18392, 0.06)}


class FoldCycle1Tshirt006(_ClothFoldStageCheck):
    stage_name = "FoldCycle1Tshirt006"

    def _check_stage(self, P):
        return _check_pairs(P, CYCLE1_PAIRS)


# Cycle 2: both arms carry the other side's corners onto the same anchors.
# ROUND_DONE_M["cycle2"] from the reference demo.
CYCLE2_PAIRS = {"A": (8941, 7369, 0.13), "B": (19027, 18392, 0.09)}


class FoldCycle2Tshirt006(_ClothFoldStageCheck):
    stage_name = "FoldCycle2Tshirt006"

    def _check_stage(self, P):
        return _check_pairs(P, CYCLE2_PAIRS)


# Cycle 3: single arm (B) tucks the hem anchor onto the other anchor, the final adjustment.
# ROUND_DONE_M["cycle3"] from the reference demo (== its FOLD_DONE_M["top_bottom"]).
CYCLE3_PAIRS = {"B": (18392, 7369, 0.16)}


class FoldCycle3Tshirt006(_ClothFoldStageCheck):
    stage_name = "FoldCycle3Tshirt006"

    def _check_stage(self, P):
        return _check_pairs(P, CYCLE3_PAIRS)
