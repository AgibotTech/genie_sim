# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from typing import Dict, Tuple, Optional
from .common_actions import ActionBase


class ActionManager:
    def __init__(self):
        self._act_map: Dict[str, Tuple[str, ActionBase]] = {}
        self._update_buffer: list = []

    def start(self, slot: str, name: str, action: ActionBase) -> None:
        existing = self._act_map.get(slot)
        if existing:
            if existing[0] == name:
                return
            self.stop(slot)

        action.start()
        if action.is_finished():
            return

        self._act_map[slot] = (name, action)

    def pause(self, slot: str) -> None:
        if entry := self._act_map.get(slot):
            entry[1].pause()

    def pause_all(self) -> None:
        for entry in self._act_map.values():
            entry[1].pause()

    def resume(self, slot: str) -> None:
        if entry := self._act_map.get(slot):
            entry[1].resume()

    def resume_all(self) -> None:
        for entry in self._act_map.values():
            entry[1].resume()

    def stop(self, slot: str) -> None:
        if entry := self._act_map.pop(slot, None):
            entry[1].stop()

    def stop_all(self) -> None:
        actions = list(self._act_map.values())
        self._act_map.clear()
        for entry in actions:
            entry[1].stop()

    def update(self, delta_time: float) -> None:
        completed_slots = []

        # Phase 1: Update all actions
        for slot, (_, action) in list(self._act_map.items()):
            action.update(delta_time)
            if action.is_finished():
                completed_slots.append(slot)

        # Phase 2: Cleaning up completed actions
        for slot in completed_slots:
            if slot in self._act_map:
                del self._act_map[slot]

    def get_action_name(self, slot: str) -> Optional[str]:
        if entry := self._act_map.get(slot):
            return entry[0]
        return None

    def exist_action(self, slot: str, action_name: str = "") -> bool:
        entry = self._act_map.get(slot)
        if action_name == "":
            return entry is not None
        if entry is None:
            return False
        return entry[0] == action_name
