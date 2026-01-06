# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import asyncio
from abc import ABC, abstractmethod
from enum import IntEnum


class CheckerStatus(IntEnum):
    UNKNOWN = 0
    RUNNING = 1
    PASS = 1
    FAIL = 2
    ERROR = 3


class CheckerBase(ABC):
    def __init__(self, name, command_controller, **kwargs):
        # command_controller is a info integration
        self.name = name
        self.status = CheckerStatus.UNKNOWN
        self.command_controller = command_controller

    def check(self) -> CheckerStatus:
        if self.status == CheckerStatus.UNKNOWN:
            self.check_impl()
        return self.status

    @abstractmethod
    def check_impl(self) -> bool:
        pass


class SyncChecker(CheckerBase):
    def check(self) -> CheckerStatus:
        passed = self.check_impl()
        self.status = CheckerStatus.PASS if passed else CheckerStatus.FAIL
        return self.status


class AsyncChecker(CheckerBase):
    def __init__(self, check_time, check_interval=0.1, **kwargs):
        super().__init__(**kwargs)
        self.check_time = check_time
        self.check_interval = check_interval

    def check(self) -> CheckerStatus:
        if self.status == CheckerStatus.UNKNOWN:
            self.status = CheckerStatus.RUNNING
            asyncio.ensure_future(self.check_async())
        return self.status

    async def check_async(self):
        acc_time = 0
        while acc_time < self.check_time:
            passed = self.check_impl()
            if not passed:
                self.status = CheckerStatus.FAIL
                return
            await asyncio.sleep(self.check_interval)
            acc_time += self.check_interval
        self.status = CheckerStatus.PASS


class ValueChecker(SyncChecker):
    def __init__(self, name, value, rule, **kwargs):
        super().__init__(name, **kwargs)
        self.value = value
        self.rule = rule

    def check_impl(self):
        cur_value = self.get_value()
        if self.rule == "equalTo":
            return cur_value == self.value
        elif self.rule == "greaterThan":
            return cur_value > self.value
        elif self.rule == "lessThan":
            return cur_value < self.value
        return False

    @abstractmethod
    def get_value(self):
        pass
