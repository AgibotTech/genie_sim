# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Optional per-request artifact hook; no dependency on storage or service code."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_writer: ContextVar[Callable[[str, Any], None] | None] = ContextVar("artifact_writer", default=None)


@contextmanager
def capture_artifacts(writer: Callable[[str, Any], None] | None) -> Iterator[None]:
    token = _writer.set(writer)
    try:
        yield
    finally:
        _writer.reset(token)


def record_artifact(kind: str, value: Any) -> None:
    writer = _writer.get()
    if writer is not None:
        writer(kind, value)
