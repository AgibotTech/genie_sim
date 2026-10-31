# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Read-only local observation of persisted traces, outside the action loop."""

from typing import Protocol


class Observer(Protocol):
    @property
    def url(self) -> str: ...

    def close(self) -> None: ...
