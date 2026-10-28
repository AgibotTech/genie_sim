# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Shared parse helpers for :mod:`genie_sim_engine_core.schema`.

Kept private (leading underscore) — the public surface is the dataclass
API in :mod:`.scene` / :mod:`.manifest`. These helpers exist so both
modules can share coercion logic without leaking their vocabulary into
each other.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Tuple

from .errors import SchemaError


def _require_mapping(value: Any, *, where: str, path: object = None) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SchemaError(
            f"{where}: expected a mapping, got {type(value).__name__}",
            path=path,
        )
    return value


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    """Non-throwing coercion: returns ``value`` if it's a mapping else ``{}``.

    Scene YAML is authored by humans; missing optional sub-blocks are
    normal. Use this at optional-section boundaries. Reserve
    :func:`_require_mapping` for places where a wrong type is a real
    typo (e.g. root ``robot:`` is a list).
    """
    return value if isinstance(value, Mapping) else {}


def _list_or_empty(value: Any) -> List[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _get_str(source: Mapping[str, Any], key: str, default: str = "") -> str:
    raw = source.get(key, default)
    if raw is None:
        return default
    return str(raw)


def _get_bool(source: Mapping[str, Any], key: str, default: bool) -> bool:
    raw = source.get(key, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        # Match the tolerant parser common/scene_config._parse_bool has used.
        return raw.strip().lower() not in {"false", "0", "no", "off", ""}
    return bool(raw)


def _get_int(source: Mapping[str, Any], key: str, default: int) -> int:
    raw = source.get(key, default)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError) as err:
        raise SchemaError(f"{key!r}: expected int, got {raw!r}") from err


def _get_float(source: Mapping[str, Any], key: str, default: float) -> float:
    raw = source.get(key, default)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError) as err:
        raise SchemaError(f"{key!r}: expected float, got {raw!r}") from err


def _get_opt_path(source: Mapping[str, Any], key: str) -> Optional[str]:
    raw = source.get(key)
    if raw is None:
        return None
    s = str(raw).strip()
    return s or None


def _xyz(source: Mapping[str, Any], key: str = "xyz") -> Optional[Tuple[float, float, float]]:
    raw = source.get(key)
    if raw is None:
        return None
    try:
        x, y, z = (float(raw[0]), float(raw[1]), float(raw[2]))
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    return (x, y, z)


def _wxyz(source: Mapping[str, Any], key: str = "wxyz") -> Optional[Tuple[float, float, float, float]]:
    raw = source.get(key)
    if raw is None:
        return None
    try:
        w, x, y, z = (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    return (w, x, y, z)


def resolve_under(base: Path, value: Optional[str]) -> Optional[Path]:
    """Resolve ``value`` against ``base`` if it's relative, absolute-passthrough otherwise.

    Manifest fields are stored **relative to ``base_path``** on disk (unchanged
    convention from the pre-schema flat manifest — every writer and every C++
    reader has always assumed this). This helper is the single place that
    walks that convention.

    Returns ``None`` when ``value`` is ``None`` or an empty string; the empty
    string sentinel is common in the legacy flat manifest for "no such asset"
    (e.g. mujoco scenes have no ``robot_visual_usda``).
    """
    if not value:
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    return Path(os.path.normpath(base / p))


def relativize_under(base: Path, value: Optional[Path]) -> str:
    """Convert an absolute path back to a ``base``-relative string for JSON.

    Used by manifest writers. Falls back to the absolute path when
    ``value`` is not under ``base`` (cross-drive on Windows, or a
    caller who legitimately stored an out-of-tree path).
    """
    if value is None:
        return ""
    p = Path(value)
    if not p.is_absolute():
        return str(p)
    try:
        return os.path.relpath(p, base)
    except ValueError:
        return str(p)


def check_unknown_keys(
    source: Mapping[str, Any],
    *,
    known: Iterable[str],
    where: str,
    strict: bool = False,
) -> List[str]:
    """Return keys in ``source`` that aren't in ``known``.

    ``strict=True`` raises. ``strict=False`` (default) returns the list
    so a caller can decide to warn or ignore. Human-authored YAML is
    forgiving by default; assemble outputs (``manifest.json``) are
    strict because a stray key means schema drift, not a typo.
    """
    unknown = [k for k in source if k not in set(known)]
    if unknown and strict:
        raise SchemaError(f"{where}: unknown key(s) {sorted(unknown)!r}; " f"expected one of {sorted(known)!r}")
    return unknown


__all__ = [
    "check_unknown_keys",
    "relativize_under",
    "resolve_under",
]
