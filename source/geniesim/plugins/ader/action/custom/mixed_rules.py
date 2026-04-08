# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from geniesim.plugins.ader.action.common_actions import (
    EvaluateAction,
    ActionBase,
    ActionEvent,
)
import numpy as np
from geniesim.plugins.logger import Logger

logger = Logger()


class MixedRules(EvaluateAction):
    def __init__(self, env, rules, check_interval=1):
        super().__init__(env)
        self._done_flag = False
        self._update_counter = 0
        self._check_interval = int(check_interval)
        self._best_score = 0.0
        self._sub_checkers = []
        self._sub_results = []

        self._build_sub_checkers(env, rules)

    def _build_sub_checkers(self, env, rules):
        factory = {
            "Inside": self._create_inside,
            "PushPull": self._create_pushpull,
            "Upright": self._create_upright,
            "RelativePosition": self._create_relative_position,
            "Ontop": self._create_ontop,
            "Onfloor": self._create_onfloor,
            "Cover": self._create_cover,
            "LiftUp": self._create_liftup,
            "InBBox": self._create_inbbox,
            "Stack": self._create_stack,
            "StableGrasp": self._create_stable_grasp,
        }

        for rule in rules:
            checker_name = rule["name"]
            params_str = rule["params"]
            creator = factory.get(checker_name)
            if creator is None:
                logger.warning(f"[MixedRules] Unknown checker: {checker_name}, skipping")
                continue
            try:
                checker = creator(env, params_str)
                checker._skip_progress_report = True
                checker.state = checker.State.RUNNING
                self._sub_checkers.append({"name": checker_name, "checker": checker})
            except Exception as e:
                logger.error(f"[MixedRules] Failed to create checker {checker_name}: {e}")

        self._sub_results = [False] * len(self._sub_checkers)
        logger.info(f"[MixedRules] Total sub-checkers: {len(self._sub_checkers)}")

    @staticmethod
    def _create_inside(env, params_str):
        from geniesim.plugins.ader.action.custom.inside import Inside

        params = params_str.split("|")
        return Inside(env, params[0], params[1], params[2])

    @staticmethod
    def _create_pushpull(env, params_str):
        from geniesim.plugins.ader.action.custom.push_pull import PushPull

        params = params_str.split("|")
        return PushPull(env, params[0], params[1], params[2], int(params[3]) if len(params) > 3 else 0)

    @staticmethod
    def _create_upright(env, params_str):
        from geniesim.plugins.ader.action.custom.upright import Upright

        params = params_str.split("|")
        allow_flipped = params[2].lower() == "true" if len(params) > 2 else False
        return Upright(env, params[0], float(params[1]), allow_flipped)

    @staticmethod
    def _create_relative_position(env, params_str):
        from geniesim.plugins.ader.action.custom.relative_position_checker import RelativePositionChecker

        params = params_str.split("|")
        return RelativePositionChecker(env, params[0], params[1], params[2])

    @staticmethod
    def _create_ontop(env, params_str):
        from geniesim.plugins.ader.action.custom.ontop import Ontop

        params = params_str.split("|")
        return Ontop(env, params[0], params[1])

    @staticmethod
    def _create_onfloor(env, params_str):
        from geniesim.plugins.ader.action.custom.onfloor import Onfloor

        params = params_str.split("|")
        return Onfloor(env, obj_name=params[0], height=params[1])

    @staticmethod
    def _create_cover(env, params_str):
        from geniesim.plugins.ader.action.custom.cover import Cover

        params = params_str.split("|")
        return Cover(env, params[0], params[1])

    @staticmethod
    def _create_liftup(env, params_str):
        from geniesim.plugins.ader.action.custom.liftup import LiftUp

        params = params_str.split("|")
        return LiftUp(env, params[0], float(params[1]))

    @staticmethod
    def _create_inbbox(env, params_str):
        from geniesim.plugins.ader.action.custom.inbbox import InBBox

        params = params_str.split("|")
        center_values = params[1].split(",")
        size_values = params[2].split(",")
        bbox_center = [float(v) for v in center_values]
        bbox_size = [float(v) for v in size_values]
        return InBBox(env, params[0], bbox_center, bbox_size)

    @staticmethod
    def _create_stack(env, params_str):
        from geniesim.plugins.ader.action.custom.stack import Stack

        params = params_str.split("|")
        return Stack(env, params[0], params[1])

    @staticmethod
    def _create_stable_grasp(env, params_str):
        from geniesim.plugins.ader.action.custom.stable_grasp import StableGrasp

        params = params_str.split("|")
        kwargs = {}
        if len(params) >= 3:
            kwargs["distance_threshold"] = float(params[2])
        if len(params) >= 4:
            kwargs["pose_diff_pos_threshold"] = float(params[3])
        if len(params) >= 5:
            kwargs["pose_diff_rot_threshold_rad"] = float(params[4])
        return StableGrasp(env, params[0], params[1], **kwargs)

    def _evaluate_sub_checker(self, entry):
        checker = entry["checker"]
        name = entry["name"]
        try:
            if hasattr(checker, "_done_flag") and checker._done_flag:
                return True

            checker.update(1.0)

            if hasattr(checker, "_done_flag"):
                return checker._done_flag
            return checker.is_finished()
        except Exception as e:
            logger.warning(f"[MixedRules] Error evaluating {name}: {e}")
            return False

    def update(self, delta_time: float) -> float:
        if not self.is_running():
            return 0.0

        self._update_counter += 1
        if self._update_counter % self._check_interval != 0:
            return super().update(delta_time)

        total = len(self._sub_checkers)
        if total == 0:
            self._done_flag = True
            return super().update(delta_time)

        for i, entry in enumerate(self._sub_checkers):
            if not self._sub_results[i]:
                if self._evaluate_sub_checker(entry):
                    self._sub_results[i] = True

        passed_count = sum(self._sub_results)
        current_score = passed_count / total

        if current_score > self._best_score:
            self._best_score = current_score

        status_parts = [
            f"{'✓' if self._sub_results[i] else '✗'} {entry['name']}" for i, entry in enumerate(self._sub_checkers)
        ]
        logger.info(
            f"[MixedRules] score={self._best_score:.3f} " f"({passed_count}/{total}) [{', '.join(status_parts)}]"
        )

        if self._best_score >= 1.0:
            self._done_flag = True

        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        self.progress_info["SCORE"] = self._best_score
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"
        details = []
        for i, entry in enumerate(self._sub_checkers):
            details.append(
                {
                    "name": entry["name"],
                    "passed": self._sub_results[i],
                }
            )
        self.progress_info["DETAILS"] = details

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        if event == ActionEvent.FINISHED:
            self.progress_info["SCORE"] = self._best_score
