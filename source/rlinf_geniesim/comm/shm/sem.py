# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pair of named POSIX semaphores for SHM request/response wakeup.

Needed only by the shm backend: shared memory carries no notification of its
own. Semantics are lockstep, one request in flight at a time.

``posix_ipc`` is imported lazily inside the methods, so importing this module
alone does not require the wheel and the comm package stays importable on a
host without it.
"""
from __future__ import annotations
from typing import Optional


class SemPair:
    def __init__(self, req_name: str, resp_name: str, *, create: bool) -> None:
        import posix_ipc

        flags = posix_ipc.O_CREAT if create else 0
        # initial_value=0: both semaphores start with nothing pending.
        self._req = posix_ipc.Semaphore(req_name, flags=flags, initial_value=0)
        self._resp = posix_ipc.Semaphore(resp_name, flags=flags, initial_value=0)
        self._created = create

    # server side
    def wait_request(self, timeout: Optional[float] = None) -> None:
        self._req.acquire(timeout)

    def post_response(self) -> None:
        self._resp.release()

    # client side
    def post_request(self) -> None:
        self._req.release()

    def wait_response(self, timeout: Optional[float] = None) -> None:
        self._resp.acquire(timeout)

    def close(self, unlink: bool = False) -> None:
        import posix_ipc

        # Unlink before close: posix_ipc cannot unlink a closed handle. Only the
        # server removes names; open peer handles remain valid until they close.
        if unlink and self._created:
            for s in (self._req, self._resp):
                try:
                    s.unlink()
                except posix_ipc.ExistentialError:
                    pass
        for s in (self._req, self._resp):
            try:
                s.close()
            except Exception:  # noqa: BLE001
                pass


__all__ = ["SemPair"]
