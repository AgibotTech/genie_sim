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
_RECV_TIMEOUT_SEC = 120.0


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

    Callers that own episode cleanup can opt in to this exception from
    ``run_with_inference_retry``. They can report the final transport error
    before aborting the evaluation.
    """


class InferenceServiceError(RuntimeError):
    """The service replied with an error or an incompatible response schema."""


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


def get_recv_timeout(default: float = _RECV_TIMEOUT_SEC) -> float:
    """Deadline for a single inference response, in seconds.

    Without a deadline a lost request or a lost reply hangs the simulator
    forever: the WebSocket itself stays healthy (the gateway keeps answering
    keepalive pings), so nothing ever raises and the episode blocks until the
    pod watchdog kills it. The deadline is what converts that silent hang into
    a transient TimeoutError the retry loop below can act on.

    Picking the value: it must be comfortably ABOVE the slowest legitimate
    inference, or healthy-but-slow models get killed and re-asked in a loop.
    That also makes it the floor on recovery latency, so it is a backstop,
    not a fast path.
    """
    return _env_float("GENIESIM_INFER_RECV_TIMEOUT_SEC", default)


def run_with_inference_retry(
    infer_fn: Callable[[], bool],
    *,
    log: Any,
    max_consecutive_failures: Optional[int] = None,
    backoff_cap: Optional[float] = None,
    label: str = "inference",
    raise_on_exhaustion: bool = False,
) -> None:
    """Drive an infer callable until it succeeds.

    Contract for ``infer_fn``:
      - returns True on success → loop returns
      - returns False on transient failure → retried after backoff
      - raises a transient exception (see ``TRANSIENT_EXC_TYPES``) → retried
      - raises any other exception → propagated immediately (fatal)

    After ``max_consecutive_failures`` consecutive failed attempts, logs an
    error and terminates the process via ``sys.exit(1)``. Callers can pass
    ``raise_on_exhaustion=True`` to receive ``InferenceUnavailableError``.
    The failure counter resets for each call.
    """
    max_failures = max_consecutive_failures if max_consecutive_failures is not None else get_max_consecutive_failures()
    cap = backoff_cap if backoff_cap is not None else get_backoff_cap()

    attempt = 0
    while True:
        last_err: Optional[str] = None
        cause: Optional[Exception] = None
        try:
            ok = infer_fn()
        except Exception as e:
            if not is_transient_error(e):
                raise
            ok = False
            cause = e
            last_err = f"{type(e).__name__}: {e}"

        if ok:
            return

        attempt += 1
        if attempt >= max_failures:
            message = (
                f"{label} failed {attempt} consecutive times "
                f"(last: {last_err or 'returned False'}); inference server "
                f"unreachable, exiting."
            )
            log.error(message)
            if raise_on_exhaustion:
                raise InferenceUnavailableError(message) from cause
            sys.exit(1)

        delay = backoff_delay(attempt, cap=cap)
        log.info(
            f"Retrying {label} in {delay:.1f}s "
            f"(attempt {attempt}/{max_failures}, last: {last_err or 'returned False'})"
        )
        time.sleep(delay)
