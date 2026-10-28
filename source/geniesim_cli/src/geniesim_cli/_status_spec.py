# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Derived status spec — single source of truth, no hand-maintained checklists.

The historical pattern was a hand-curated ``_STATUS_DISTRIBUTIONS`` tuple in
``commands/status.py`` that listed every distribution's deps, submodules, and
extras inline. That table drifted every time a peer changed: a renamed
submodule flipped ``geniesim status`` to ⚠️ until someone remembered to update
the tuple. This module deletes the tuple and derives the same shape from:

  * ``importlib.metadata.distributions()`` — what's actually installed
  * Each peer's ``pyproject.toml`` (source tree first, install-root fallback)
    — ``[project].dependencies`` + ``[project.optional-dependencies]``
  * ``pkgutil.iter_modules`` on each peer's ``__path__`` — actual subpackages

The only hand-curated state is presentation-only or genuinely irreducible:

  * ``_EMOJI`` — display label per distribution
  * ``_PIP_TO_IMPORT`` — the ~3 packages whose import name differs from pip
    name (everything else infers ``pip_name.replace("-", "_")``)
  * ``_AGGREGATOR_EXTRAS`` — extras like ``all`` / ``full`` that bundle other
    extras; filtered from the per-distribution extras section
  * ``_DIST_BLOCKLIST`` — locator shims (``geniesim-data-collection``) that
    we don't want to appear in the status table

Public API: :func:`status_distributions` — returns the same tuple-of-dicts
shape the old static ``DISTRIBUTIONS`` had, so callers (``status``,
``bootstrap``, ``doctor``) don't need to change.
"""

from __future__ import annotations

import importlib.util
import pkgutil
import re
from functools import lru_cache
from pathlib import Path

# --------------------------------------------------------------------------
# Hand-curated tables (presentation + irreducible mappings)
# --------------------------------------------------------------------------

# Display label per known distribution. Unknown peers fall back to "📦"
# so new peers show up immediately (without needing this table updated).
_EMOJI: dict[str, str] = {
    "geniesim": "📦",
    "geniesim_cli": "🧞",
    "geniesim_benchmark": "🧪",
    "geniesim_generator": "🎨",
    "geniesim_scene_agent": "🎨",
    "geniesim_ros": "📡",
    "geniesim_teleop": "🎮",
    "geniesim_assets": "🎨",
    "geniesim_world": "🌐",
}

# Pip dist name → top-level import name. Only listed when they truly
# differ — the default rule is ``import_name = pip_name.replace("-", "_")``
# which already handles ``langchain-chroma`` → ``langchain_chroma`` etc.
_PIP_TO_IMPORT: dict[str, str] = {
    "usd-core": "pxr",
    "Pillow": "PIL",
    "pillow": "PIL",
    "opencv-python": "cv2",
    "opencv-python-headless": "cv2",
    "PyYAML": "yaml",
}

# Extras that aggregate other extras (eg. ``all``, ``full``). Filtered
# from the displayed extras list because they don't correspond to a
# single capability the user installs deliberately.
_AGGREGATOR_EXTRAS = frozenset({"all", "full"})

# Distributions that look like geniesim peers by name but should NOT appear
# in the status table (locator shims, build helpers).
_DIST_BLOCKLIST = frozenset(
    {
        # ``data_collection`` legacy module installed under the
        # ``geniesim-data-collection`` PyPI name as a locator shim.
        "geniesim-data-collection",
        "geniesim_data_collection",
    }
)


# --------------------------------------------------------------------------
# pyproject.toml reader (shared between source-tree and install-root paths)
# --------------------------------------------------------------------------


def _tomllib():
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[import-not-found, no-redef]
    return tomllib


def _import_name(pip_name: str) -> str:
    if pip_name in _PIP_TO_IMPORT:
        return _PIP_TO_IMPORT[pip_name]
    return pip_name.replace("-", "_")


def _strip_specifier(req: str) -> str:
    """``geniesim_cli>=3.2.0`` → ``geniesim_cli``. Handles every operator
    we use in this repo's pyprojects."""
    for op in (">=", "<=", "==", "~=", ">", "<", "!=", "[", " ", ";"):
        idx = req.find(op)
        if idx != -1:
            req = req[:idx]
    return req.strip()


def _is_geniesim_peer(name: str) -> bool:
    return name.startswith("geniesim")


def _read_pyproject(path: Path) -> dict | None:
    try:
        with path.open("rb") as fh:
            return _tomllib().load(fh)
    except (OSError, ValueError):
        return None


def _locate_pyproject(dist_name: str, top_name: str) -> Path | None:
    """Find a pyproject.toml for ``dist_name``.

    Resolution order:
      1. ``<repo>/source/<dist>/pyproject.toml`` — editable installs from
         the umbrella checkout.
      2. ``<repo>/source/<top>/pyproject.toml`` — same, dash-vs-underscore
         tolerance (the directory name often uses underscores even when
         the PyPI name has dashes).
      3. ``<installed_pkg_dir>/pyproject.toml`` — flat installs that ship
         their packaging metadata next to the code, with no ``src/`` shim.
      4. ``<installed_pkg_dir>/../pyproject.toml`` — standard layout where
         pyproject sits one level above ``src/<pkg>``.
    """
    candidates: list[Path] = []

    here = Path(__file__).resolve()
    for parent in here.parents:
        source_dir = parent / "source"
        if source_dir.is_dir():
            for name in (dist_name, top_name):
                candidates.append(source_dir / name / "pyproject.toml")
            break  # only walk the first source/ ancestor

    try:
        spec = importlib.util.find_spec(top_name)
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.submodule_search_locations:
        pkg_dir = Path(list(spec.submodule_search_locations)[0]).resolve()
        candidates.append(pkg_dir / "pyproject.toml")
        candidates.append(pkg_dir.parent / "pyproject.toml")
        candidates.append(pkg_dir.parent.parent / "pyproject.toml")

    for c in candidates:
        if c.is_file():
            return c
    return None


# --------------------------------------------------------------------------
# Submodule discovery (replaces the hand-maintained submodule lists)
# --------------------------------------------------------------------------


def _discover_subpackages(top_name: str) -> tuple[str, ...]:
    """Walk the installed package and return its first-level subpackages.

    Returns ``("<top>.<sub>", ...)`` ready to feed into ``probe_module``.
    Returns an empty tuple when the package isn't importable or is a
    single-module distribution (no ``__path__``). Single-file modules
    are skipped — historically the status table listed only subpackages,
    and a noisy import-time module list isn't useful for health-checking.
    """
    try:
        spec = importlib.util.find_spec(top_name)
    except (ImportError, ValueError):
        return ()
    if spec is None or not spec.submodule_search_locations:
        return ()

    subs: list[str] = []
    for module_info in pkgutil.iter_modules(list(spec.submodule_search_locations)):
        if not module_info.ispkg:
            continue
        # Skip private subpackages (``_*``) and underscore-prefixed test trees.
        if module_info.name.startswith("_"):
            continue
        # Skip ``tests`` / ``test`` subpackages — health-irrelevant.
        if module_info.name in ("tests", "test"):
            continue
        subs.append(f"{top_name}.{module_info.name}")
    return tuple(sorted(subs))


# --------------------------------------------------------------------------
# Distribution discovery
# --------------------------------------------------------------------------


def _list_geniesim_distributions() -> list[tuple[str, str]]:
    """The canonical distribution set, as declared by the umbrella.

    Returns a list of ``(dist_name, tier)`` pairs, where ``tier`` is:

      * ``"required"`` — tier-1 peer (must be installed for a healthy stack)
      * ``"optional"`` — tier-2 peer (declared as an umbrella extra; absence
        is a ⏭️ skip, not a ❌ failure)

    The single source of truth is ``source/geniesim/pyproject.toml`` —
    its ``[project].dependencies`` lists every tier-1 peer, its
    ``[project.optional-dependencies]`` lists every tier-2 peer.
    ``_tiers.py`` already reads that for ``bootstrap`` / ``status`` /
    ``tool deps-dag``; we reuse it so the four consumers can't drift.

    The canonical order is:

      * ``geniesim_cli``       — root of the dep graph (required)
      * ``geniesim``           — the umbrella itself (required)
      * tier-1 peers in pyproject declaration order (required)
      * tier-2 peers in pyproject declaration order (optional)

    Anything else that happens to be installed with a ``geniesim*`` name
    is treated as a ghost (orphan editable installs from previous repos,
    abandoned forks, etc.) and silently dropped — those used to bloat the
    status report.
    """
    try:
        from geniesim_cli._tiers import component_extras, tier1
    except Exception:
        return []

    ordered: list[tuple[str, str]] = [("geniesim_cli", "required"), ("geniesim", "required")]
    seen: set[str] = {name for name, _tier in ordered}

    for peer in tier1():
        if peer not in seen and peer not in _DIST_BLOCKLIST:
            ordered.append((peer, "required"))
            seen.add(peer)
    for _label, peers in component_extras().items():
        for peer in peers:
            if peer not in seen and peer not in _DIST_BLOCKLIST:
                ordered.append((peer, "optional"))
                seen.add(peer)
    return ordered


# --------------------------------------------------------------------------
# Spec builders
# --------------------------------------------------------------------------


def _build_spec_from_pyproject(dist_name: str, pyproj: dict) -> dict:
    project = pyproj.get("project", {}) or {}
    declared_name = project.get("name") or dist_name
    top = _import_name(declared_name)

    deps_raw = project.get("dependencies") or []
    deps = tuple(
        (_strip_specifier(r), _import_name(_strip_specifier(r)))
        for r in deps_raw
        if _is_geniesim_peer(_strip_specifier(r))
    )

    extras: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
    for label, pkgs in (project.get("optional-dependencies") or {}).items():
        if label in _AGGREGATOR_EXTRAS:
            continue
        pip_names = tuple(_strip_specifier(p) for p in pkgs)
        import_names = tuple(_import_name(n) for n in pip_names)
        extras.append((label, pip_names, import_names))

    is_umbrella = declared_name == "geniesim"
    submodules: tuple[str, ...] = () if is_umbrella else _discover_subpackages(top)

    return {
        "dist": declared_name,
        "top": top,
        "emoji": _EMOJI.get(declared_name, "📦"),
        "deps": deps,
        "submodules": submodules,
        "extras": tuple(extras),
    }


def _build_spec_from_installed_metadata(dist_name: str) -> dict:
    """Best-effort spec for a peer with no readable pyproject.

    Reads ``importlib.metadata.distribution(name).requires`` — PEP 508
    strings, with ``; extra == 'name'`` markers identifying extras. Sibling
    peer deps are extracted from the unmarked block; everything else is
    routed into the appropriate extra.
    """
    top = _import_name(dist_name)
    deps: list[tuple[str, str]] = []
    extras_map: dict[str, list[str]] = {}

    try:
        import importlib.metadata as md

        dist = md.distribution(dist_name)
        requires = dist.requires or []
    except Exception:
        requires = []

    for raw in requires:
        if "; extra ==" in raw or "; extra==" in raw:
            spec_part, marker = re.split(r";\s*extra\s*==", raw, maxsplit=1)
            extra_name = marker.strip().strip("'\"")
            if extra_name in _AGGREGATOR_EXTRAS:
                continue
            extras_map.setdefault(extra_name, []).append(_strip_specifier(spec_part))
        else:
            pkg = _strip_specifier(raw)
            if _is_geniesim_peer(pkg):
                deps.append((pkg, _import_name(pkg)))

    extras: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
    for label, pkgs in extras_map.items():
        pip_names = tuple(pkgs)
        import_names = tuple(_import_name(p) for p in pip_names)
        extras.append((label, pip_names, import_names))

    is_umbrella = dist_name == "geniesim"
    submodules: tuple[str, ...] = () if is_umbrella else _discover_subpackages(top)

    return {
        "dist": dist_name,
        "top": top,
        "emoji": _EMOJI.get(dist_name, "📦"),
        "deps": tuple(deps),
        "submodules": submodules,
        "extras": tuple(extras),
    }


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

# Display priority is determined by ``_list_geniesim_distributions`` — it
# yields ``geniesim_cli`` first, then ``geniesim`` (the umbrella), then
# tier-1 peers in the order the umbrella declares them, then tier-2 peers.


@lru_cache(maxsize=1)
def status_distributions() -> tuple[dict, ...]:
    """Return the canonical status spec, derived from pyproject + installed packages.

    Output shape matches the historical ``_STATUS_DISTRIBUTIONS`` tuple
    (same dict keys) so callers in ``status``, ``bootstrap``, and
    ``doctor`` don't need to change. Each spec carries a ``"tier"`` field
    (``"required"`` | ``"optional"``) — callers that care about the
    overall health verdict skip ``"optional"`` peers when they're absent.
    """
    specs: list[dict] = []
    for name, tier in _list_geniesim_distributions():
        top_guess = _import_name(name)
        pyproj_path = _locate_pyproject(name, top_guess)
        pyproj = _read_pyproject(pyproj_path) if pyproj_path else None
        if pyproj is not None:
            spec = _build_spec_from_pyproject(name, pyproj)
        else:
            spec = _build_spec_from_installed_metadata(name)
        spec["tier"] = tier
        specs.append(spec)
    return tuple(specs)
