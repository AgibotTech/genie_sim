# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Discover usable background rooms within the configured size limit.

Task background requirements are resolved separately by task_background. This
module chooses among rooms; callers must account for tasks whose room contains
physical supports or manipulated objects.

Reject failed USD opens, composition diagnostics and empty bounds with warnings.
Filter oversized bounds quietly. Capture C++ USD diagnostics through stderr's
file descriptor. Cache measurements by file path and modification time."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable, MutableMapping, NamedTuple, Optional, Sequence, Union


class RoomMeasurement(NamedTuple):
    """The result of one ``measure_room`` call. When ``ok=False``,
    ``dx``/``dy``/``dz`` are placeholder ``0.0`` and do not represent any
    real size -- ``reason`` is what actually matters."""

    ok: bool
    dx: float
    dy: float
    dz: float
    reason: str


def _capture_stderr_during(fn: Callable):
    """Capture C++ USD diagnostics written directly to stderr's file descriptor."""
    fd = 2
    saved = os.dup(fd)
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        os.dup2(tmp.fileno(), fd)
        result = None
        exc: Optional[Exception] = None
        try:
            result = fn()
        except Exception as e:  # noqa: BLE001 -- reported by the caller
            exc = e
        finally:
            os.dup2(saved, fd)
            os.close(saved)
        tmp.seek(0)
        captured = tmp.read().decode("utf-8", "replace")
    return result, exc, captured


def measure_room(usda_path: str) -> RoomMeasurement:
    """Open a background USD and measure its world bounding box.

    Return a failed measurement for open/composition errors or empty geometry.
    Size-limit evaluation is separate so the measurement can be reused."""
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    def _open():
        return Usd.Stage.Open(str(usda_path))

    stage, exc, captured = _capture_stderr_during(_open)
    if exc is not None:
        return RoomMeasurement(False, 0.0, 0.0, 0.0, f"Usd.Stage.Open raised: {exc!r}")
    if not stage:
        return RoomMeasurement(False, 0.0, 0.0, 0.0, "Usd.Stage.Open returned None/falsy")
    if captured.strip():
        first_line = captured.strip().splitlines()[0]
        return RoomMeasurement(
            False,
            0.0,
            0.0,
            0.0,
            f"opened but a composition diagnostic was printed to stderr "
            f"during Open (likely a broken internal reference/payload "
            f"path): {first_line}",
        )
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    rng = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    if rng.IsEmpty():
        return RoomMeasurement(
            False,
            0.0,
            0.0,
            0.0,
            "opened cleanly but the world bounding box is empty -- no " "renderable geometry resolved",
        )
    lo, hi = rng.GetMin(), rng.GetMax()
    return RoomMeasurement(True, float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(hi[2] - lo[2]), "ok")


def passes_size_gate(m: RoomMeasurement, max_footprint_m: float) -> "tuple[bool, str]":
    """Require all three bounding-box extents to be below max_footprint_m.

    Propagate an unusable measurement's reason without applying the size gate."""
    if not m.ok:
        return False, m.reason
    if m.dx >= max_footprint_m or m.dy >= max_footprint_m or m.dz >= max_footprint_m:
        return False, (
            f"footprint {m.dx:.2f}x{m.dy:.2f}x{m.dz:.2f}m has an axis >= " f"max_footprint_m={max_footprint_m:.2f}m"
        )
    return True, "ok"


_DEFAULT_CACHE: "dict[str, tuple[Optional[float], RoomMeasurement]]" = {}


def _cached_measure(usda_path: Path, cache: MutableMapping) -> RoomMeasurement:
    key = str(usda_path)
    try:
        mtime: Optional[float] = usda_path.stat().st_mtime
    except OSError:
        mtime = None
    hit = cache.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    m = measure_room(str(usda_path))
    cache[key] = (mtime, m)
    return m


def discover_background_rooms(
    background_root: "Union[str, Path]",
    *,
    max_footprint_m: float,
    rooms: Optional[Sequence[str]] = None,
    cache: Optional[MutableMapping] = None,
) -> "list[str]":
    """Return sorted room keys whose assets open cleanly and pass the size gate.

    background_root is the asset's background directory. Explicit room keys still
    undergo validation; when omitted, discover background.usda files recursively.
    Use the supplied measurement cache or the process-wide cache. Sorted output is
    part of the seeded sampler's reproducibility contract."""
    root = Path(background_root)
    if cache is None:
        cache = _DEFAULT_CACHE

    if rooms:
        candidates = [(str(r), root / str(r) / "background.usda") for r in rooms]
    else:
        candidates = sorted((p.parent.relative_to(root).as_posix(), p) for p in root.rglob("background.usda"))

    passing: list[str] = []
    for room, usda_path in candidates:
        if not usda_path.is_file():
            print(
                f"[randomize.background] WARN: room {room!r} was "
                f"configured/discovered but {usda_path} does not exist -- "
                f"excluded",
                flush=True,
            )
            continue
        m = _cached_measure(usda_path, cache)
        ok, reason = passes_size_gate(m, max_footprint_m)
        if ok:
            passing.append(room)
        elif not m.ok:
            # Can't open / can't use at all -- an asset or environment
            # problem, must be seen (module docstring, exclusion reasons 1-3).
            print(f"[randomize.background] WARN: room {room!r} " f"({usda_path}) excluded -- {reason}", flush=True)
        # else: quietly filtered by size (exclusion reason 4) -- the normal,
        # expected shape of this candidate set, not a problem to report.

    if not passing:
        print(
            f"[randomize.background] WARN: {len(candidates)} candidate(s) "
            f"under {root} but NONE passed the size gate "
            f"(max_footprint_m={max_footprint_m}) -- background "
            f"randomization would have no effect",
            flush=True,
        )
    return sorted(passing)


def resolve_room_usda(background_root: "Union[str, Path]", room: str) -> str:
    """Room key (the kind ``discover_background_rooms`` returns, e.g.
    ``"room/room_1"``) -> absolute ``background.usda`` path. Lets the
    mounting side (the manifest's ``background_usda_per_env``) turn a key
    into the real file path ``build_gridcloner_stage`` needs."""
    return str(Path(background_root) / room / "background.usda")


__all__ = [
    "RoomMeasurement",
    "measure_room",
    "passes_size_gate",
    "discover_background_rooms",
    "resolve_room_usda",
]
