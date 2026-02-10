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
from geniesim.evaluator.generators.auto_score import auto_score
from geniesim.evaluator.templates.VLM_TEMPLATE import VLM_TEMPLATE

logger = Logger()


class VLM(EvaluateAction):
    def __init__(self, env, task_id, instruction):
        super().__init__(env)
        self._done_flag = False
        self.env = env
        self._checked = False
        self.task_id = task_id
        self._update_counter = 0
        self._check_interval = 5
        self.description = VLM_TEMPLATE.get(self.task_id, instruction)
        # Store accumulated image observations for evaluation.
        self._image_history = []
        # Store the last evaluation score
        self._last_score = 0.0
        self._use_history = False
        self._debug = False

    def update(self, delta_time: float) -> float:
        if not self.is_running():
            return 0.0
        self._update_counter += 1
        if self._update_counter % self._check_interval == 0:
            try:
                if not self._use_history:
                    self._image_history = []
                image_dict = self.get_observation_image()
                self._image_history.append(image_dict)

                score, reasoning = auto_score(
                    self.description, self._image_history, target_size=(640, 480), save_debug_images=self._debug
                )
                self._last_score = score
                # Consider task done if score is 1.0 (all scoring points satisfied)
                if score >= 1.0:
                    self._done_flag = True
                    logger.info(f"VLM checker: Description  is fully satisfied (score: {score:.3f})")
                else:
                    logger.info(f"VLM checker: Description partially satisfied (score: {score:.3f})")

            except Exception as e:
                logger.error(f"VLM checker error: {e}")

        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def reset_image_history(self):
        """Reset the stored image history, typically when starting a new task."""
        # Clear previously saved images.
        self._image_history = []
        self._last_score = 0.0
        logger.info("VLM: Image history has been reset")

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info(f"Action [VLM] {self.description} evt: {event.value}")
        if event == ActionEvent.STARTED:
            pass
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            pass
        elif event == ActionEvent.FINISHED:
            # Use the last evaluation score
            self.progress_info["SCORE"] = self._last_score
