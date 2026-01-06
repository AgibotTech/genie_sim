# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import copy
import time
from typing import Any, Dict, List, Optional

from client.planner.action.grasp import PickStage
from client.planner.action.insert import InsertStage
from client.planner.action.place import PlaceStage
from client.planner.action.rotate import RotateStage
from client.planner.action.stage import Stage
from common.base_utils.logger import logger


class StageFactory:
    """Stage factory class, creates corresponding Stage derived classes based on configuration"""

    @staticmethod
    def create_stage(stage_config: Dict[str, Any], objects: Dict[str, Any]) -> Stage:
        action = stage_config["action"]

        stage_classes = {
            "pick": PickStage,
            "place": PlaceStage,
            "insert": InsertStage,
            "rotate": RotateStage,
        }

        if action in stage_classes:
            return stage_classes[action](stage_config, objects)
        else:
            logger.warning(f"Unknown action type: {action}, using base Stage class")

            # For unknown action types, create a generic Stage instance
            class GenericStage(Stage):
                def _execute_impl(self):
                    logger.info(f"Execute generic action: {self.action}")
                    time.sleep(0.1)  # Simulate execution time

            return GenericStage(stage_config, objects)


class ActionScript:
    """Class managing the entire action script"""

    def __init__(self):
        self.stages: List[Stage] = []
        self.current_stage_index = 0
        self.status = "initialized"  # initialized, running, paused, completed, failed
        self.start_time = None
        self.end_time = None
        self.initialized = False
        # Use global logger instance

    def initialize(self, task_definition: Dict[str, Any], objects: Dict[str, Any]):
        """Initialize action script based on task definition"""
        logger.info("Starting ActionScript initialization")
        self.reset()

        # Use factory to create all stages
        stages_config = copy.deepcopy(task_definition.get("stages", []))
        for i, stage_config in enumerate(stages_config):
            stage = StageFactory.create_stage(stage_config, objects)
            self.append_stage(stage)

        self.initialized = True
        logger.info(f"ActionScript initialization complete, {len(self.stages)} stages total")

    def __iter__(self):
        return self

    def __next__(self) -> Optional[Stage]:
        if self.current_stage_index < len(self.stages):
            stage = self.stages[self.current_stage_index]
            self.current_stage_index += 1
            return stage
        raise StopIteration

    def append_stage(self, stage: Stage):
        """Add a new stage"""
        if self.stages:
            stage.set_previous_stage(self.stages[-1])
            self.stages[-1].set_next_stage(stage)
        self.stages.append(stage)

    def insert_stage(self, index: int, stage: Stage):
        """Insert a stage at specified position"""
        if index < 0 or index > len(self.stages):
            raise IndexError("Index out of bounds")

        if self.stages:
            if index > 0:
                stage.set_previous_stage(self.stages[index - 1])
                self.stages[index - 1].set_next_stage(stage)
            if index < len(self.stages):
                stage.set_next_stage(self.stages[index])
                self.stages[index].set_previous_stage(stage)

        self.stages.insert(index, stage)

    def get_stage(self, stage_id: int) -> Optional[Stage]:
        """Get stage by ID"""
        if 0 <= stage_id < len(self.stages):
            return self.stages[stage_id]
        return None

    def get_current_stage(self) -> Optional[Stage]:
        """Get current stage"""
        return self.get_stage(self.current_stage_index)

    def run(self):
        """Run each stage sequentially"""
        self.status = "running"
        self.start_time = time.time()
        logger.info("Starting ActionScript execution")

        try:
            for i, stage in enumerate(self.stages):
                self.current_stage_index = i
                current_stage = self.get_current_stage()

                logger.info(f"Preparing to execute stage {i}: {current_stage.action}")

                # Check if can execute
                if not current_stage.can_execute():
                    logger.warning(f"Stage {i} prerequisites not met, skipping execution")
                    continue

                # Execute current stage
                current_stage.execute()

                # Check execution result
                if current_stage.status == "failed":
                    self.status = "failed"
                    logger.error(f"Stage {i} execution failed, stopping entire script")
                    break

            # Check final status
            if all(stage.status == "completed" for stage in self.stages):
                self.status = "completed"
                logger.info("ActionScript execution complete")
            else:
                self.status = "failed"
                logger.warning("ActionScript execution failed")

        except Exception as e:
            self.status = "failed"
            logger.error(f"Error occurred during ActionScript execution: {str(e)}")
            raise

        finally:
            self.end_time = time.time()

    def reset(self):
        """Reset all stage states"""
        self.current_stage_index = 0
        self.status = "initialized"
        self.start_time = None
        self.end_time = None

        self.stages.clear()

        logger.info("ActionScript has been reset")

    def get_execution_time(self) -> float:
        """Get total execution time"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    def __str__(self):
        return f"ActionScript: {len(self.stages)} stages, status: {self.status}"

    def __repr__(self):
        return f"ActionScript(stages={len(self.stages)}, status={self.status})"
