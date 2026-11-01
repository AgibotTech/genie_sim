# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Atomic file commands and acknowledgements for render reload and randomization.

Each channel has its own command/ack filenames and monotonic generation counter.
Reload sends a manifest path; randomization sends sampled scene changes and
returns application readback. Readers reject unrelated generations and tolerate
missing or malformed files. Writers use a same-directory temporary file and
os.replace so readers never see a partial write."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

CMD_FILENAME = "reload_cmd.json"
ACK_FILENAME = "reload_ack.json"
# The generalization-layer render-side command channel — a separate file,
# separate generation, see the module docstring.
RANDOMIZE_CMD_FILENAME = "randomize_cmd.json"
RANDOMIZE_ACK_FILENAME = "randomize_ack.json"


def _atomic_write_json(path: Path, data: dict) -> Path:
    """Atomically replace JSON using a temporary file on the same filesystem.

    Readers see a complete old or new file; write failures remove the temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, str(path))
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def write_cmd(manifest_dir, generation: int, manifest_path: str) -> Path:
    """Write ``<manifest_dir>/reload_cmd.json`` — the bridge's request to
    reload the render node onto a new manifest at ``generation``."""
    return _atomic_write_json(
        Path(manifest_dir) / CMD_FILENAME, {"generation": int(generation), "manifest": str(manifest_path)}
    )


def read_cmd_if_new(manifest_dir, last_generation: int) -> Optional[dict]:
    """Return the parsed cmd dict iff the file exists, parses cleanly, AND
    its generation is strictly greater than ``last_generation`` — else
    ``None``. NEVER raises: an absent file, a malformed/partial/garbage JSON
    body, or a same-or-stale generation are all just "nothing new" to the
    render node's per-frame poll."""
    path = Path(manifest_dir) / CMD_FILENAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        gen = int(data["generation"])
        manifest_path = str(data["manifest"])
    except Exception:  # noqa: BLE001 — malformed/partial/garbage -> "nothing new"
        return None
    if gen <= int(last_generation):
        return None
    return {"generation": gen, "manifest": manifest_path}


def peek_generation(manifest_dir) -> int:
    """Read the current reload generation, returning zero for unreadable files.

    A node starts with its argv manifest and treats any existing command generation
    as consumed, preventing a stale command from replacing that initial stage."""
    path = Path(manifest_dir) / CMD_FILENAME
    if not path.exists():
        return 0
    try:
        return int(json.loads(path.read_text())["generation"])
    except Exception:  # noqa: BLE001 — malformed/partial/garbage -> nothing to seed
        return 0


def clear_stale_control_files(manifest_dir) -> None:
    """Best-effort removal of both command/ack channels before spawning a node.

    A reused control directory must not carry commands or acknowledgements from
    the previous process into the new generation sequence."""
    for name in (CMD_FILENAME, ACK_FILENAME, RANDOMIZE_CMD_FILENAME, RANDOMIZE_ACK_FILENAME):
        try:
            (Path(manifest_dir) / name).unlink(missing_ok=True)
        except OSError:
            pass


def write_ack(manifest_dir, generation: int, ok: bool, error: Optional[str], dims: Optional[list]) -> Path:
    """Write ``<manifest_dir>/reload_ack.json`` — the render node's reply to
    a reload cmd, always ``{"generation", "ok", "error", "dims"}`` even on
    failure (``ok=False`` + ``error``) — the bridge decides what to do next."""
    return _atomic_write_json(
        Path(manifest_dir) / ACK_FILENAME,
        {
            "generation": int(generation),
            "ok": bool(ok),
            "error": None if error is None else str(error),
            "dims": None if dims is None else list(dims),
        },
    )


def poll_ack(
    manifest_dir,
    generation: int,
    timeout_s: float,
    proc_alive: Optional[Callable[[], bool]] = None,
    poll_interval_s: float = 0.25,
    *,
    time_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], Any] = time.sleep,
) -> dict:
    """Block until ``reload_ack.json`` carries ``generation`` (matches this
    call's request), then return it AS-IS — ``ok=False`` is still returned,
    not raised; the caller (the bridge) decides whether that's fatal.

    Raises:
        RuntimeError: ``proc_alive`` is given and returns ``False`` — the
            render process died, so no ack will ever come; raised
            immediately, without waiting out the rest of ``timeout_s``.
        TimeoutError: no matching-generation ack within ``timeout_s``.

    ``time_fn``/``sleep_fn`` are injectable (default: real wall clock) so
    unit tests can drive timeout/proc-dead paths without real sleeping.
    """
    path = Path(manifest_dir) / ACK_FILENAME
    deadline = time_fn() + float(timeout_s)
    while True:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                gen = int(data["generation"])
            except Exception:  # noqa: BLE001 — torn/partial read -> keep polling
                data, gen = None, None
            if isinstance(data, dict) and gen == int(generation):
                return data
        if proc_alive is not None and not proc_alive():
            raise RuntimeError(
                f"poll_ack: render process died waiting for ack " f"generation={generation} in {manifest_dir}"
            )
        if time_fn() >= deadline:
            raise TimeoutError(
                f"poll_ack: timed out after {timeout_s}s waiting for ack " f"generation={generation} in {manifest_dir}"
            )
        sleep_fn(poll_interval_s)


# ---------------------------------------------------------------------------
# Randomization uses its own generation sequence and RandomizationPlan.render
# payload. Acknowledgements omit dimensions because these changes preserve
# observation shape.
# ---------------------------------------------------------------------------


def write_randomize_cmd(manifest_dir, generation: int, render: dict) -> Path:
    """Write ``<manifest_dir>/randomize_cmd.json`` — the bridge's per-reset
    render-side generalization command at ``generation`` (``render`` may
    carry sibling keys such as ``"lights"``/``"material"``/``"camera_pose"``)."""
    return _atomic_write_json(
        Path(manifest_dir) / RANDOMIZE_CMD_FILENAME, {"generation": int(generation), "render": dict(render)}
    )


def read_randomize_cmd_if_new(manifest_dir, last_generation: int) -> Optional[dict]:
    """Return the parsed cmd dict iff ``randomize_cmd.json`` exists, parses
    cleanly, AND its generation is strictly greater than ``last_generation``
    — else ``None``. NEVER raises, same contract as ``read_cmd_if_new``: an
    absent file, a malformed/partial/garbage JSON body, or a same-or-stale
    generation are all just "nothing new" to the render node's per-frame
    poll. Writing a ``reload_cmd.json`` (the OTHER channel) never makes this
    return non-``None`` — it only ever looks at ``randomize_cmd.json``."""
    path = Path(manifest_dir) / RANDOMIZE_CMD_FILENAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        gen = int(data["generation"])
        render = dict(data["render"])
    except Exception:  # noqa: BLE001 — malformed/partial/garbage -> "nothing new"
        return None
    if gen <= int(last_generation):
        return None
    return {"generation": gen, "render": render}


def peek_randomize_generation(manifest_dir) -> int:
    """Return the ``generation`` currently written in
    ``<manifest_dir>/randomize_cmd.json``, or ``0`` if the file is absent,
    unreadable, or malformed — mirrors ``peek_generation``'s "never raise"
    contract and its reason for existing: a freshly launched node seeds its
    ``last_randomize_generation`` from this so a randomize cmd left over from
    a PRIOR (bridge, node) pair sharing this control dir is treated as
    ALREADY CONSUMED, never as a live command to apply at boot."""
    path = Path(manifest_dir) / RANDOMIZE_CMD_FILENAME
    if not path.exists():
        return 0
    try:
        return int(json.loads(path.read_text())["generation"])
    except Exception:  # noqa: BLE001 — malformed/partial/garbage -> nothing to seed
        return 0


def write_randomize_ack(
    manifest_dir, generation: int, ok: bool, error: Optional[str], applied: Optional[dict] = None
) -> Path:
    """Write ``<manifest_dir>/randomize_ack.json`` — the render node's reply
    to a randomize cmd, always ``{"generation", "ok", "error"}`` even on
    failure (``ok=False`` + ``error``) — the bridge decides what to do next
    (getting nothing back, or getting ``ok=False``, means raising loudly —
    it is never allowed to degrade to "no generalization this reset, oh
    well"; see ``isaac_tiled.IsaacTiledBackend.on_reset``).

    Optionally carries ``applied`` — the render node side's **readback**
    result (the count of prims it actually rebound or relit after its
    per-entry WARN-and-skip pass, a much stronger success signal than
    "this cmd's dispatch function did not raise"). The shape today is
    ``{"lights": {"applied": int, "requested": int}, "material":
    {"<env_id str>": {"applied": int, "requested": int,
    "first_missing_mesh": str | None}}}`` — either sub-key can be missing,
    depending on which dimensions this particular cmd actually carried (a
    lights-only cmd naturally has no ``"material"``).

    When ``applied`` is ``None`` (the default) or an empty dict, **this key
    is not written into the JSON at all** — backward compatibility is a
    hard requirement: existing call sites (lights commands, and
    hand-written acks in tests) do not need to change a single character;
    ``ack.get("applied")`` gets back ``None``, not a ``KeyError``, and the
    reader (``poll_randomize_ack`` forwards the whole JSON as-is) never
    needs to care whether this field exists."""
    payload = {"generation": int(generation), "ok": bool(ok), "error": None if error is None else str(error)}
    if applied:
        payload["applied"] = applied
    return _atomic_write_json(Path(manifest_dir) / RANDOMIZE_ACK_FILENAME, payload)


def poll_randomize_ack(
    manifest_dir,
    generation: int,
    timeout_s: float,
    proc_alive: Optional[Callable[[], bool]] = None,
    poll_interval_s: float = 0.25,
    *,
    time_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], Any] = time.sleep,
) -> dict:
    """Block until ``randomize_ack.json`` carries ``generation`` (matches
    this call's request), then return it AS-IS — ``ok=False`` is still
    returned, not raised; the caller (the bridge's ``on_reset``) decides
    whether that's fatal (it always is — the caller must raise loudly on
    ``ok=False``, never silently downgrade to "no generalization this
    reset"). "AS-IS" also means that the optional ``"applied"`` readback
    field is carried out unchanged — this function does not interpret it,
    only forwards it; ``on_reset`` reads it itself via
    ``ack.get("applied")``.

    Raises:
        RuntimeError: ``proc_alive`` is given and returns ``False`` — the
            render process died, so no ack will ever come; raised
            immediately, without waiting out the rest of ``timeout_s``.
        TimeoutError: no matching-generation ack within ``timeout_s``.

    Reading ``reload_ack.json`` (the OTHER channel) never satisfies this —
    it only ever looks at ``randomize_ack.json``. Mirrors ``poll_ack`` field
    for field except there is no ``dims`` to compare (a lights command never
    changes observation shape)."""
    path = Path(manifest_dir) / RANDOMIZE_ACK_FILENAME
    deadline = time_fn() + float(timeout_s)
    while True:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                gen = int(data["generation"])
            except Exception:  # noqa: BLE001 — torn/partial read -> keep polling
                data, gen = None, None
            if isinstance(data, dict) and gen == int(generation):
                return data
        if proc_alive is not None and not proc_alive():
            raise RuntimeError(
                f"poll_randomize_ack: render process died waiting for "
                f"randomize ack generation={generation} in {manifest_dir}"
            )
        if time_fn() >= deadline:
            raise TimeoutError(
                f"poll_randomize_ack: timed out after {timeout_s}s waiting "
                f"for randomize ack generation={generation} in {manifest_dir}"
            )
        sleep_fn(poll_interval_s)


__all__ = [
    "write_cmd",
    "read_cmd_if_new",
    "write_ack",
    "poll_ack",
    "peek_generation",
    "clear_stale_control_files",
    "CMD_FILENAME",
    "ACK_FILENAME",
    "write_randomize_cmd",
    "read_randomize_cmd_if_new",
    "peek_randomize_generation",
    "write_randomize_ack",
    "poll_randomize_ack",
    "RANDOMIZE_CMD_FILENAME",
    "RANDOMIZE_ACK_FILENAME",
]
