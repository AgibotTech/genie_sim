# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Single source of truth for tier-1 / tier-2 peer classification.

The canonical declaration lives in **`source/geniesim/pyproject.toml`**:

- ``[project].dependencies``               → **tier 1** (always installed)
- ``[project.optional-dependencies]``      → **tier 2** (opt-in via extras)

Every consumer in the geniesim stack (``bootstrap``, ``status``, ``tool
deps-dag``, …) should call into here rather than maintaining its own list,
so that adding or re-tiering a peer is a **one-line edit** in the umbrella's
pyproject.

Three read paths (in order):

1. **Source-tree path** — walk up from this module and the current directory to find
   ``source/geniesim/pyproject.toml``. This is the authoritative read
   in a dev checkout: edits to the umbrella's pyproject take effect
   immediately, no ``pip install`` needed to refresh.

2. **Installed-metadata fallback** — ``importlib.metadata.distribution("geniesim").requires``.
   Used in production / deployed containers where ``source/`` isn't on
   disk. Works for both editable (PEP 660) and wheel installs.
   *Warning:* metadata is recorded at install time, so this can lag
   behind a freshly-edited source pyproject — which is why source is
   tried first.

3. **Bundled fallback** — the CLI wheel carries a copy of the umbrella's
   pyproject, so diagnostics and recovery also work before it is installed.
   The source file is a symlink to the canonical declaration, not a second list.

The standalone script ``docker/collect_deps.py`` re-implements the same
parse against the same pyproject (it runs at Docker build time, before
``geniesim_cli`` is pip-installed, so it can't import this module).
Keep the two parses in sync via the shared canonical input.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path


def _tomllib():
    """Return the TOML reader module. ``tomllib`` is stdlib on Python
    3.11+; fall back to the ``tomli`` backport on 3.10."""
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[import-not-found, no-redef]
    return tomllib


def _strip_specifier(req: str) -> str:
    """``geniesim_cli>=3.2.0`` → ``geniesim_cli``. Lightweight; only
    handles the operators that appear in our pyproject files."""
    for op in (">=", "<=", "==", "~=", ">", "<", "!=", "[", " ", ";"):
        idx = req.find(op)
        if idx != -1:
            req = req[:idx]
    return req.strip()


def _is_geniesim_peer(name: str) -> bool:
    return name.startswith("geniesim")


# --------------------------------------------------------------------------
# Path 1 — installed metadata (importlib.metadata)
# --------------------------------------------------------------------------


def _read_from_installed_metadata() -> tuple[list[str], dict[str, list[str]]] | None:
    """Query ``geniesim``'s recorded dependencies via importlib.metadata.

    Returns ``(tier1, tier2)`` or ``None`` if the umbrella isn't installed.

    ``dist.requires`` returns PEP 508 strings; entries with a
    ``; extra == 'name'`` marker are extras, everything else is required.
    """
    try:
        import importlib.metadata as md
    except ImportError:  # pragma: no cover — stdlib always present on supported Pythons
        return None
    try:
        dist = md.distribution("geniesim")
    except md.PackageNotFoundError:
        return None

    requires = dist.requires or []
    tier1_list: list[str] = []
    tier2_map: dict[str, list[str]] = {}

    for raw in requires:
        if "; extra ==" in raw or "; extra==" in raw:
            spec, marker = re.split(r";\s*extra\s*==", raw, maxsplit=1)
            extra_name = marker.strip().strip("'\"")
            name = _strip_specifier(spec)
            if _is_geniesim_peer(name):
                tier2_map.setdefault(extra_name, []).append(name)
        else:
            name = _strip_specifier(raw)
            if _is_geniesim_peer(name):
                tier1_list.append(name)

    return tier1_list, tier2_map


# --------------------------------------------------------------------------
# Path 2 — source-tree fallback (umbrella's pyproject.toml)
# --------------------------------------------------------------------------


def _source_pyproject() -> Path | None:
    """Locate the umbrella in the CLI's checkout or the current checkout."""
    here = Path(__file__).resolve()
    cwd = Path.cwd()
    for parent in dict.fromkeys((*here.parents, cwd, *cwd.parents)):
        candidate = parent / "source" / "geniesim" / "pyproject.toml"
        if candidate.is_file():
            return candidate
    return None


def _read_project(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    with path.open("rb") as fh:
        project = _tomllib().load(fh).get("project", {}) or {}
    tier1_list = [
        _strip_specifier(r)
        for r in (project.get("dependencies") or [])
        if _is_geniesim_peer(_strip_specifier(r))
    ]
    tier2_map: dict[str, list[str]] = {}
    for name, deps in (project.get("optional-dependencies") or {}).items():
        peers = [_strip_specifier(d) for d in deps if _is_geniesim_peer(_strip_specifier(d))]
        if peers:
            tier2_map[name] = peers
    return tier1_list, tier2_map


def _read_from_source_tree() -> tuple[list[str], dict[str, list[str]]] | None:
    path = _source_pyproject()
    return _read_project(path) if path is not None else None


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _tiers() -> tuple[list[str], dict[str, list[str]]]:
    """Return ``(tier1, tier2)``. Source tree wins over installed
    metadata so a freshly-edited umbrella pyproject takes effect
    without ``pip install``. The bundled declaration supports standalone installs."""
    result = _read_from_source_tree() or _read_from_installed_metadata()
    if result is None:
        result = _read_project(Path(__file__).with_name("_umbrella.toml"))
    return result


def tier1() -> list[str]:
    """Required ``geniesim_*`` peers (the umbrella's
    ``[project].dependencies``). Order preserved from pyproject."""
    return list(_tiers()[0])


def tier2() -> dict[str, list[str]]:
    """Optional ``geniesim_*`` peers keyed by extra name (e.g.
    ``{"teleop": ["geniesim_teleop"], "all": ["geniesim_teleop", …]}``).

    Aggregator extras (``all``, ``full``) are returned as-is; callers
    that want only the component extras should filter."""
    return {k: list(v) for k, v in _tiers()[1].items()}


def component_extras() -> dict[str, list[str]]:
    """Same as :func:`tier2` but filters out aggregator extras
    (``all``, ``full``). Use this when listing user-facing tier-2 peers
    that each correspond to one installable component."""
    aggregators = {"all", "full"}
    return {k: v for k, v in tier2().items() if k not in aggregators}


def all_peers() -> list[str]:
    """Union of tier-1 + every tier-2 peer (no duplicates, deterministic
    order: tier-1 first, then alphabetical tier-2)."""
    out = tier1()
    seen = set(out)
    for extra_name in sorted(tier2()):
        for pkg in tier2()[extra_name]:
            if pkg not in seen:
                seen.add(pkg)
                out.append(pkg)
    return out
