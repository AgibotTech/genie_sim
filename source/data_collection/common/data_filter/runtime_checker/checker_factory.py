# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from typing import Any, Dict, List

from common.base_utils.logger import logger


class CheckerRegistry:
    def __init__(self):
        self._checkers: Dict[str, Any] = {}

    def register(self, name: str):
        def decorator(checker_class):
            self._checkers[name] = checker_class
            logger.info(f"Registered checker{name}")
            return checker_class

        return decorator

    def create_checker(self, checker_name: str, **kwargs):
        if checker_name not in self._checkers:
            raise ValueError(
                f"Checker '{checker_name}' not found. Available checkers: {list(self._checkers.keys())}"
            )
        checker_class = self._checkers[checker_name]
        return checker_class(**kwargs)

    def list_checkers(self) -> List[str]:
        return list(self._checkers.keys())


registry = CheckerRegistry()

register_checker = registry.register
create_checker = registry.create_checker
list_checkers = registry.list_checkers
