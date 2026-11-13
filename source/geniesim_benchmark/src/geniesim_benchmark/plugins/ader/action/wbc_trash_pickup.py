# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Observe WBC trash pickups during the preceding approach stage."""

from collections.abc import Iterator

from geniesim_benchmark.plugins.logger import Logger
from geniesim_benchmark.plugins.output_system.policy_budget import PolicyBudgetError

from .common_actions import ActionBase, ActionEvent, ActionList, ActionSetWaitAll, ActionSetWaitAny
from .custom.pickup_on_gripper import PickUpOnGripper
from .custom.robot_near_object import RobotNearObject
from .policy_dispatch_budget import PolicyDispatchBudget

logger = Logger()


class _ObservePickupDuringApproach(ActionBase):
    """Keep the native stage and its budget, sampling its upcoming pickup first."""

    def __init__(self, stage: ActionBase, near_checks: list[RobotNearObject], pickup: PickUpOnGripper):
        super().__init__(stage.env)
        self._actions = [stage]
        self.near_checks = near_checks
        self.pickup = pickup
        self._last_sample_step: int | None = None
        self._skip_started_update = False
        self._skip_progress_report = True

    @property
    def stage(self) -> ActionBase:
        return self._actions[0]

    def _observe_pickup(self) -> None:
        step = self.env.current_step
        if step == self._last_sample_step:
            return
        self._last_sample_step = step
        gripper_poses = {}
        for gripper in self.pickup.gripper_ids:
            try:
                gripper_poses[gripper] = self.pickup.get_world_pose_matrix(self.pickup.gripper_link_path(gripper))
            except Exception as exc:
                logger.warning(f"[WbcTrashPickup] Could not read {gripper} at step {step}: {exc}")
        if not gripper_poses:
            return

        def has_lifted(name: str) -> bool:
            if name in self.env._picked_objects:
                return False
            first_sample = name not in self.pickup._initial_z_map
            lifted = self.pickup._check_single_object_pickup(name, gripper_poses)
            if first_sample:
                logger.info(
                    f"[WbcTrashPickup] Baseline {name} {self.pickup.gripper_id} "
                    f"step={step} z={self.pickup._initial_z_map[name]:.4f}"
                )
            return lifted

        matched = self.pickup.find_matching_object(self.pickup.obj_name_list, has_lifted)
        if matched is None:
            return

        for near in self.near_checks:
            if near.is_end_of_life():
                continue
            if near.obj_name == matched or near.obj_name in self.env._picked_objects:
                near._done_flag = True
                near.progress_info["NEAR_SOURCE"] = "pickup"
        logger.info(f"[WbcTrashPickup] Early pickup evidence {matched} {self.pickup._matched_gripper} step={step}")

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        if event == ActionEvent.STARTED:
            self._observe_pickup()
            self.stage.start()
            self._skip_started_update = True
        elif event == ActionEvent.PAUSED:
            self.stage.pause()
        elif event == ActionEvent.RESUMED:
            self.stage.resume()
        elif event == ActionEvent.CANCELED:
            self.stage.stop()

    def update(self, delta_time: float) -> float:
        if not self.is_running():
            return 0.0
        if self._skip_started_update:
            self._skip_started_update = False
        elif self.stage.is_running():
            self._observe_pickup()
            delta_time = self.stage.update(delta_time)
        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self.stage.is_finished()


def _leaves(stage: ActionBase) -> Iterator[ActionBase]:
    if isinstance(stage, PolicyDispatchBudget):
        yield from _leaves(stage.stage)
    elif isinstance(stage, (ActionSetWaitAny, ActionSetWaitAll)):
        for child in stage._actions:
            yield from _leaves(child)
    else:
        yield stage


def configure_wbc_trash_pickup(action: ActionBase) -> list[ActionBase]:
    """Attach the observers and return them for score-tree registration."""
    if not isinstance(action, ActionList) or len(action._actions) != 7:
        raise PolicyBudgetError("WbcTrashPickup requires the seven-stage WBC trash layout")

    observers = []
    initial_z = {}
    for index in (0, 2):
        stage = action._actions[index]
        near_checks = [leaf for leaf in _leaves(stage) if isinstance(leaf, RobotNearObject)]
        pickups = [leaf for leaf in _leaves(action._actions[index + 1]) if isinstance(leaf, PickUpOnGripper)]
        if len(near_checks) != 2 or len(pickups) != 1 or pickups[0].gripper_id != "any_gripper":
            raise PolicyBudgetError("WbcTrashPickup requires two near checks followed by an any-hand pickup")
        pickup = pickups[0]
        if {near.obj_name for near in near_checks} != set(pickup.parse_obj_input(pickup.obj_name)):
            raise PolicyBudgetError("WbcTrashPickup approach and pickup must reference the same objects")
        pickup._initial_z_map = initial_z
        observer = _ObservePickupDuringApproach(stage, near_checks, pickup)
        action._actions[index] = observer
        observers.append(observer)
    return observers
