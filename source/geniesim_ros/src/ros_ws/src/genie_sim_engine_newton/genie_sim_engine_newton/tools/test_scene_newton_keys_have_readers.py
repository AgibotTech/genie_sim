#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Every ``newton.*`` key a shipped scene sets must be read by something.

``newton:`` is an opaque passthrough dict in the scene schema
(``schema/scene.py``: ``newton: Dict[str, Any]``).  That is deliberate — it
lets a scene carry backend-specific settings without the core schema growing
a field per solver — but it has one sharp edge: a key whose reader is absent
is **carried and ignored**.  No schema error, no warning, no log line.  The
scene runs, and it runs without the thing it asked for.

This has bitten three times on this branch alone:

* ``newton.coupling`` — a scene set ``enabled: true``; nothing read it, so
  the run was uncoupled rather than failing.
* ``newton.mujoco_enable_multiccd`` — set to ``true`` by the retired
  Swiftpicker chef configuration, read by nothing;
  the scene had single-point convex contacts while believing it had
  manifolds.
* ``solver.impratio`` — not a scene key, but the same shape: declared, and
  applied by only one of the two adapters that claim to support it.

Each was found by hand, months apart.  This is the cheap version of that
search, and it runs in milliseconds.

The check is deliberately crude: a top-level ``newton.<key>`` must appear as
a literal string somewhere under the engine packages.  That proves a reader
plausibly exists, not that it is correct — a passing test here says only
"someone typed this name on the reading side too".  It is a smoke alarm, not
an inspection.  Nested keys are not walked: the top level is where the
silent-degrade cases have actually landed, and every level down multiplies
the false-positive rate against a grep this blunt.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent.parent.parent  # .../src
_SCENES = _SRC / "genie_sim_bringup" / "config"
_ENGINE_PKGS = (
    _SRC / "genie_sim_engine_newton" / "genie_sim_engine_newton",
    _SRC / "genie_sim_engine_core" / "genie_sim_engine_core",
)


def _scene_files():
    if not _SCENES.is_dir():
        return []
    return sorted(_SCENES.glob("scene_*.yaml"))


def _newton_keys(path: Path):
    try:
        doc = yaml.safe_load(path.read_text()) or {}
    except Exception:  # a malformed scene is another test's problem
        return []
    section = doc.get("newton")
    return sorted(section) if isinstance(section, dict) else []


def _engine_sources():
    out = []
    for pkg in _ENGINE_PKGS:
        if pkg.is_dir():
            out += [p for p in pkg.rglob("*.py") if not p.name.startswith("test_")]
    return out


_SOURCES = None


def _reader_exists(key: str) -> bool:
    global _SOURCES
    if _SOURCES is None:
        _SOURCES = [p.read_text(errors="ignore") for p in _engine_sources()]
    return any(key in text for text in _SOURCES)


_CASES = [(p.name, k) for p in _scene_files() for k in _newton_keys(p)]


@pytest.mark.skipif(not _scene_files(), reason="genie_sim_bringup/config not present")
@pytest.mark.parametrize("scene,key", _CASES, ids=[f"{s}::{k}" for s, k in _CASES])
def test_every_newton_key_a_scene_sets_is_read_somewhere(scene, key):
    assert _reader_exists(key), (
        f"{scene} sets newton.{key!r} and no engine source mentions that name. "
        f"The scene schema carries unknown keys under `newton:` silently, so this "
        f"scene is running WITHOUT whatever it asked for — it is not failing, and "
        f"it is not warning. Either add the reader or delete the key."
    )


def test_the_check_found_some_keys_to_check():
    """Guards the guard: a path typo would make every case above vanish and
    the file would pass by testing nothing."""
    assert _CASES, "no newton.* keys discovered — the scene glob is probably wrong"
    assert _engine_sources(), "no engine sources discovered — the package paths are wrong"


def test_a_name_no_scene_uses_is_correctly_reported_missing():
    """The assertion is only worth anything if it can fail."""
    assert not _reader_exists("mujoco_enable_flux_capacitor")
