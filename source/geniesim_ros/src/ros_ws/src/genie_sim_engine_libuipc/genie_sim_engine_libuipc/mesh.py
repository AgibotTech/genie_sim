# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Isolated convex decomposition for explicitly requested CPU colliders."""

from __future__ import annotations

_COACD_WORKER_SRC = r"""
import pickle
import sys

import coacd

try:
    coacd.set_log_level("error")
except Exception:
    pass

v, f, th, maxh = pickle.load(sys.stdin.buffer)
parts = coacd.run_coacd(coacd.Mesh(v, f), threshold=th, max_convex_hull=maxh)
pickle.dump(parts, sys.stdout.buffer)
"""


def _run_coacd_isolated(v, f, threshold: float, max_convex_hull: int, retries: int = 3):
    """Run ``coacd.run_coacd`` in a throwaway child process, retrying on crash.

    Rolling-specific hardening, not in the ported reference commit: CoACD's
    OpenMP-parallel search has an upstream, currently-unfixed thread-safety
    bug (SarahWeiii/CoACD PR #104 — a data race in a thread-unsafe helper
    touched during parallel execution) that intermittently SIGSEGVs on
    multi-core hosts. Verified empirically on this container (32 cores):
    roughly 1-in-8 to 1-in-18 ``run_coacd`` calls crash the whole process —
    which, in-process, would take down engine boot on ~1/3 of cold builds
    (4 pads/build). Forcing ``OMP_NUM_THREADS=1`` avoids the race but was
    measured to blow up per-pad decomposition time from under a second to
    minutes (single-threaded MCTS search), an unacceptable build-time cost.
    Isolating each call in a subprocess costs nothing on the (fast, fully
    parallel) happy path and turns a process-killing crash into a plain
    non-zero exit code the parent can retry — a fresh subprocess is an
    independent roll of the same low-probability dice, so 3 attempts drive
    the effective failure probability low enough to be negligible, and the
    existing "keep single hull" fallback (see caller) still catches the
    residual case where all retries are exhausted.
    """
    import pickle  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    payload = pickle.dumps((v, f, float(threshold), int(max_convex_hull)))
    last_err: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _COACD_WORKER_SRC],
                input=payload,
                capture_output=True,
                timeout=180,
            )
        except subprocess.TimeoutExpired as exc:
            last_err = exc
            continue
        if proc.returncode != 0:
            last_err = RuntimeError(
                f"coacd worker exited {proc.returncode} "
                f"(attempt {attempt + 1}/{retries}): "
                f"{proc.stderr[-500:]!r}"
            )
            continue
        return pickle.loads(proc.stdout)
    raise last_err or RuntimeError("coacd worker failed with no captured error")
