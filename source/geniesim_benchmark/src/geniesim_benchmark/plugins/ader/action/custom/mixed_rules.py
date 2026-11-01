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


class MixedRules(EvaluateAction):
    def __init__(self, env, rules, check_interval=1, mode="or", strict=False, carry_distance=None):
        super().__init__(env)
        self._done_flag = False
        self._update_counter = 0
        self._check_interval = int(check_interval)
        self._best_score = 0.0
        self._sub_checkers = []
        self._sub_results = []
        self._mode = mode.lower() if mode else "or"
        if not isinstance(strict, bool):
            raise ValueError("strict must be a boolean")
        self._strict = strict
        if strict and (
            self._mode not in ("and", "or")
            or type(check_interval) is not int
            or self._check_interval < 1
            or not isinstance(rules, list)
            or not rules
        ):
            raise ValueError("Strict MixedRules requires rules, a positive interval, and and/or mode")

        self._build_sub_checkers(env, rules)
        self._carry_score = None
        if carry_distance is not None:
            from .carry_distance import CarryDistanceScore

            if not isinstance(carry_distance, dict) or not strict or check_interval != 1:
                raise ValueError("Carry distance requires strict rules and check_interval=1")
            self._carry_score = CarryDistanceScore(self, **carry_distance)

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
            "InsidePrecise": self._create_inside_precise,
            "CoverPrecise": self._create_cover_precise,
            "ThroughOpening": self._create_through_opening,
            "PointInVolume": self._create_point_in_volume,
            "ShaftInserted": self._create_shaft_inserted,
        }

        for rule in rules:
            checker_name = rule["name"]
            params_str = rule["params"]
            creator = factory.get(checker_name)
            if creator is None:
                if self._strict:
                    raise ValueError(f"Unknown checker: {checker_name}")
                logger.warning(f"[MixedRules] Unknown checker: {checker_name}, skipping")
                continue
            try:
                checker = creator(env, params_str)
                checker._skip_progress_report = True
                own = getattr(checker, "set_composite_managed", None)
                if callable(own):
                    own(True)
                checker.state = checker.State.RUNNING
                self._sub_checkers.append({"name": checker_name, "checker": checker})
            except Exception as e:
                if self._strict:
                    raise ValueError(f"Invalid checker {checker_name}: {e}") from e
                logger.error(f"[MixedRules] Failed to create checker {checker_name}: {e}")

        self._sub_results = [False] * len(self._sub_checkers)
        logger.info(f"[MixedRules] Total sub-checkers: {len(self._sub_checkers)}")

    @staticmethod
    def _create_shaft_inserted(env, params):
        from .shaft_inserted import ShaftInserted

        return ShaftInserted(env, **params)

    @staticmethod
    def _create_point_in_volume(env, params):
        from .point_in_volume import PointInVolume

        return PointInVolume(env, **params)

    @staticmethod
    def _create_through_opening(env, params):
        from .through_opening import ThroughOpening

        return ThroughOpening(env, **params)

    @staticmethod
    def _create_inside_precise(env, params):
        from .precise import InsidePrecise

        return InsidePrecise(env, **params)

    @staticmethod
    def _create_cover_precise(env, params):
        from .precise import CoverPrecise

        return CoverPrecise(env, **params)

    @staticmethod
    def _create_inside(env, params_str):
        from geniesim_benchmark.plugins.ader.action.custom.inside import Inside

        params = params_str.split("|")
        return Inside(env, params[0], params[1], params[2])

    @staticmethod
    def _create_pushpull(env, params_str):
        from geniesim_benchmark.plugins.ader.action.custom.push_pull import PushPull

        params = params_str.split("|")
        return PushPull(env, params[0], params[1], params[2], int(params[3]) if len(params) > 3 else 0)

    @staticmethod
    def _create_upright(env, params_str):
        from geniesim_benchmark.plugins.ader.action.custom.upright import Upright

        params = params_str.split("|")
        allow_flipped = params[2].lower() == "true" if len(params) > 2 else False
        return Upright(env, params[0], float(params[1]), allow_flipped)

    @staticmethod
    def _create_relative_position(env, params_str):
        from geniesim_benchmark.plugins.ader.action.custom.relative_position_checker import RelativePositionChecker

        params = params_str.split("|")
        threshold = params[3] if len(params) > 3 else None
        return RelativePositionChecker(env, params[0], params[1], params[2], threshold)

    @staticmethod
    def _create_ontop(env, params_str):
        from geniesim_benchmark.plugins.ader.action.custom.ontop import Ontop

        params = params_str.split("|")
        return Ontop(env, params[0], params[1])

    @staticmethod
    def _create_onfloor(env, params_str):
        from geniesim_benchmark.plugins.ader.action.custom.onfloor import Onfloor

        params = params_str.split("|")
        return Onfloor(env, obj_name=params[0], height=params[1])

    @staticmethod
    def _create_cover(env, params_str):
        from geniesim_benchmark.plugins.ader.action.custom.cover import Cover

        params = params_str.split("|")
        return Cover(env, params[0], params[1])

    @staticmethod
    def _create_liftup(env, params_str):
        from geniesim_benchmark.plugins.ader.action.custom.liftup import LiftUp

        params = params_str.split("|")
        return LiftUp(env, params[0], float(params[1]))

    @staticmethod
    def _create_inbbox(env, params_str):
        from geniesim_benchmark.plugins.ader.action.custom.inbbox import InBBox

        params = params_str.split("|")
        center_values = params[1].split(",")
        size_values = params[2].split(",")
        bbox_center = [float(v) for v in center_values]
        bbox_size = [float(v) for v in size_values]
        return InBBox(env, params[0], bbox_center, bbox_size)

    @staticmethod
    def _create_stack(env, params_str):
        from geniesim_benchmark.plugins.ader.action.custom.stack import Stack

        params = params_str.split("|")
        return Stack(env, params[0], params[1])

    @staticmethod
    def _create_stable_grasp(env, params_str):
        from geniesim_benchmark.plugins.ader.action.custom.stable_grasp import StableGrasp

        params = params_str.split("|")
        kwargs = {}
        if len(params) >= 3:
            kwargs["distance_threshold"] = float(params[2])
        if len(params) >= 4:
            kwargs["pose_diff_pos_threshold"] = float(params[3])
        if len(params) >= 5:
            kwargs["pose_diff_rot_threshold_rad"] = float(params[4])
        if len(params) >= 6:
            kwargs["min_movement_threshold"] = float(params[5])
        return StableGrasp(env, params[0], params[1], **kwargs)

    def _evaluate_sub_checker(self, entry, force_update=False):
        checker = entry["checker"]
        name = entry["name"]
        try:
            if force_update:
                if hasattr(checker, "_done_flag"):
                    checker._done_flag = False
                if hasattr(checker, "state"):
                    checker.state = checker.State.RUNNING
            elif hasattr(checker, "_done_flag") and checker._done_flag:
                return True

            checker.update(1.0)

            if hasattr(checker, "_done_flag"):
                return checker._done_flag
            return checker.is_finished()
        except Exception as e:
            logger.warning(f"[MixedRules] Error evaluating {name}: {e}")
            return False

    def _replay_placeholders(self):
        """Propagate env placeholder values onto sub-checkers.

        Sub-checkers built by the factory are NOT registered into
        task_progress (record_act_obj is never called for them), so
        AderEnv.update_place_holder's setattr loop never reaches them — a
        ``{@placeholder_str1}`` reference inside a MixedRules rule would
        never resolve. Mirror the replay llm_task.py applies to lazily-
        built subtask checkers: push env._placeholders down each tick.
        No-op for configs whose env has no placeholders.
        """
        cached = getattr(self.env, "_placeholders", None)
        if not cached:
            return
        for entry in self._sub_checkers:
            checker = entry["checker"]
            for key, value in cached.items():
                if getattr(checker, key, None) != value:
                    setattr(checker, key, value)

    def update(self, delta_time: float) -> float:
        if not self.is_running():
            return 0.0
        if self._carry_score is not None:
            return self._carry_score.update(delta_time)

        self._update_counter += 1
        if self._update_counter % self._check_interval != 0:
            return super().update(delta_time)

        self._replay_placeholders()

        total = len(self._sub_checkers)
        if total == 0:
            self._done_flag = True
            return super().update(delta_time)

        if self._mode == "and":
            for i, entry in enumerate(self._sub_checkers):
                self._sub_results[i] = self._evaluate_sub_checker(entry, force_update=True)
        else:
            for i, entry in enumerate(self._sub_checkers):
                if not self._sub_results[i]:
                    if self._evaluate_sub_checker(entry):
                        self._sub_results[i] = True

        passed_count = sum(self._sub_results)

        if self._mode == "and":
            current_score = 1.0 if passed_count == total else 0.0
        else:
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
        if self._carry_score is not None:
            self.progress_info["GOAL_COMPLETE"] = self._done_flag
            self.progress_info["CARRY_DISTANCE"] = dict(self._carry_score.last_measurement)
            self.progress_info["STATUS"] = (
                "SUCCESS" if self._done_flag else "FAILED" if self._carry_score.sealed else "PENDING"
            )
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"
        details = []
        for i, entry in enumerate(self._sub_checkers):
            detail = {"name": entry["name"], "passed": self._sub_results[i]}
            measurement = getattr(entry["checker"], "last_measurement", None)
            if measurement is not None:
                detail["measurement"] = measurement
            details.append(detail)
        self.progress_info["DETAILS"] = details

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        if event in (ActionEvent.FINISHED, ActionEvent.CANCELED):
            for entry in self._sub_checkers:
                close = getattr(entry["checker"], "close", None)
                if callable(close):
                    close()
        if event == ActionEvent.FINISHED:
            self.progress_info["SCORE"] = self._best_score
