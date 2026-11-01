# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Retry helpers for the inference client path.

Designed for the multi-env scenario where many simulators hit the same
inference server: exponential backoff with jitter avoids thundering herd when
all clients fail simultaneously and then retry in lockstep.
"""

import os
import random
import sys
import time
from typing import Any, Callable, Optional, Tuple, Type

_BASE_DELAY_SEC = 1.0
_MAX_DELAY_SEC = 30.0
_JITTER_SEC = 2.0


# Errors that indicate a transient connectivity / server-overload condition.
# These should be retried. Anything outside this set (RuntimeError on a
# malformed payload, KeyError, ValueError, ...) is treated as fatal because
# retrying won't help and the caller should surface the bug.
_TRANSIENT_BASE_TYPES: Tuple[Type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
    EOFError,
)


def _resolve_transient_types() -> Tuple[Type[BaseException], ...]:
    extra: Tuple[Type[BaseException], ...] = ()
    try:
        import websockets.exceptions as _ws_exc

        extra = (_ws_exc.WebSocketException,)
    except Exception:
        pass
    return _TRANSIENT_BASE_TYPES + extra


TRANSIENT_EXC_TYPES: Tuple[Type[BaseException], ...] = _resolve_transient_types()


class InferenceUnavailableError(RuntimeError):
    """Fatal "inference server unreachable" condition, after the configured
    number of consecutive failed attempts.

    Caveat: the current retry loop (`retry_until_ready`) does not raise this — on
    exhaustion it terminates the process directly via ``sys.exit(1)`` and lets an
    external supervisor (shell / k8s / scheduler) decide whether to restart. This
    type is kept for callers that prefer to catch a typed exception instead. The
    intent either way is to NOT skip the current episode — a dead server would
    otherwise silently burn through the remaining episodes as failures.
    """


def is_transient_error(exc: BaseException) -> bool:
    return isinstance(exc, TRANSIENT_EXC_TYPES)


def backoff_delay(
    attempt: int,
    base: float = _BASE_DELAY_SEC,
    cap: float = _MAX_DELAY_SEC,
    jitter: float = _JITTER_SEC,
) -> float:
    """Return seconds to sleep before retry ``attempt`` (1-indexed).

    Uses capped exponential backoff plus uniform jitter:
        min(base * 2**(attempt-1), cap) + U(0, jitter)
    """
    attempt = max(1, int(attempt))
    exp = min(base * (2 ** (attempt - 1)), cap)
    return exp + random.uniform(0.0, jitter)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def get_max_consecutive_failures(default: int = 30) -> int:
    """Max consecutive infer() failures before the retry loop gives up (sys.exit(1)).

    With the default backoff (cap=30s + 2s jitter) 30 attempts is roughly a
    13-minute outage tolerance — long enough to survive a typical inference
    server restart or a multi-minute load spike, short enough that a truly
    dead server doesn't hang the run forever.
    """
    return _env_int("GENIESIM_INFER_MAX_FAILURES", default)


def get_backoff_cap(default: float = _MAX_DELAY_SEC) -> float:
    return _env_float("GENIESIM_INFER_BACKOFF_CAP_SEC", default)


def run_with_inference_retry(
    infer_fn: Callable[[], bool],
    *,
    log: Any,
    max_consecutive_failures: Optional[int] = None,
    backoff_cap: Optional[float] = None,
    label: str = "inference",
) -> None:
    """Drive an infer callable until it succeeds.

    Contract for ``infer_fn``:
      - returns True on success → loop returns
      - returns False on transient failure → retried after backoff
      - raises a transient exception (see ``TRANSIENT_EXC_TYPES``) → retried
      - raises any other exception → propagated immediately (fatal)

    After ``max_consecutive_failures`` consecutive failed attempts, logs an
    error and terminates the process via ``sys.exit(1)``. The counter resets
    on the next call to this helper; within one call it only grows.
    """
    max_failures = max_consecutive_failures if max_consecutive_failures is not None else get_max_consecutive_failures()
    cap = backoff_cap if backoff_cap is not None else get_backoff_cap()

    attempt = 0
    while True:
        last_err: Optional[str] = None
        try:
            ok = infer_fn()
        except Exception as e:
            if not is_transient_error(e):
                raise
            ok = False
            last_err = f"{type(e).__name__}: {e}"

        if ok:
            return

        attempt += 1
        if attempt >= max_failures:
            log.error(
                f"{label} failed {attempt} consecutive times "
                f"(last: {last_err or 'returned False'}); inference server "
                f"unreachable, exiting."
            )
            sys.exit(1)

        delay = backoff_delay(attempt, cap=cap)
        log.info(
            f"Retrying {label} in {delay:.1f}s "
            f"(attempt {attempt}/{max_failures}, last: {last_err or 'returned False'})"
        )
        time.sleep(delay)
