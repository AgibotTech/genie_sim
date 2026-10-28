#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``physics_knobs.md``'s "read by" column must match who actually reads.

Setting a config entry the active solver adapter never reads is silent: no
warning, no error, the yaml simply has no effect.  The document is the only
place that says which entries are like that, so it is the only thing standing
between a user and an afternoon of wondering why `impratio: 30` changed
nothing.

Why this can rot: the entries split across two layers.

  * shared engine layer (``engine/newton/setup/*``) — every mjwarp adapter
    reaches it, stock included;
  * ``adapters/mujoco_warp_multi_env.py`` — the isolated adapter only.

That split was created on 2026-08-17 when the multi-env work moved out of
``adapters/mujoco_warp.py``, and it moved entries from the first group into the
second WITHOUT the document noticing.  Move one back, or add a new knob to the
isolated adapter, and the table goes stale in exactly the same silent way.

The check is deliberately coarse — "is the entry's name mentioned in the
isolated adapter, in the shared layer, or both" — because that is the question
the column answers.  It cannot catch a knob that is read but ignored; nothing
static can.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

_HERE = pathlib.Path(__file__).resolve()
_INNER_PKG = _HERE.parents[1]  # genie_sim_engine_newton/
_ENGINE = _INNER_PKG / "engine" / "newton"
_ADAPTERS = _ENGINE / "adapters"
_DOC = _INNER_PKG.parents[1] / "genie_sim_engine_core" / "docs" / "physics_knobs.md"

_MULTI_ENV_ONLY = "*multi-env only*"
_ALL_MJWARP = "*all mjwarp*"


def _doc_rows() -> dict[str, str]:
    """``{config entry: read-by cell}`` from the section A table."""
    if not _DOC.is_file():
        pytest.fail(f"physics_knobs.md not found at {_DOC}", pytrace=False)
    rows: dict[str, str] = {}
    for line in _DOC.read_text().splitlines():
        m = re.match(r"\|\s*`([a-z_][a-z_0-9.]*)`\s*\|\s*(\*[^|]*\*)\s*\|", line)
        if m:
            rows[m.group(1)] = m.group(2).strip()
    return rows


def _leaf(entry: str) -> str:
    """The attribute name the code actually reads.

    The doc writes dotted paths (``gripper_contact.pad_decomposition.enabled``);
    the code reads the leaf off a dataclass (``pad_decomposition``).  ``enabled``
    is a yaml sub-key, not an attribute, so it is dropped.
    """
    parts = entry.split(".")
    if parts[-1] in ("enabled",):
        parts = parts[:-1]
    return parts[-1]


def _mentions(paths, name: str) -> bool:
    return any(name in p.read_text() for p in paths)


def _code_mentions(paths, name: str) -> bool:
    """``_mentions``, but blind to comments and docstrings.

    The coarse version answers "is the name written in this file", which is
    the right question for the *shared layer vs isolated adapter* checks
    above: a knob's name appearing anywhere in a file is evidence enough and
    a false positive there is the safe direction.

    It is the wrong question for deciding whether a layer READS an entry.
    Two files discuss knobs they do not touch: ``setup/solver.py`` names
    ``mjc_iterations`` when explaining why it routes construction through the
    adapter, and ``common/mjcf_postprocess.py`` lists ``impratio`` among the
    options it leaves alone. Both would otherwise read as shared-layer reads,
    and the entries would drop out of the warning's key list as "already
    covered elsewhere".

    String literals other than docstrings are KEPT, deliberately: every one
    of these reads is ``getattr(section, "leaf", default)`` -- the adapters
    duck-type ``physics_params`` rather than importing engine_core -- so the
    attribute name only ever appears as a string.
    """
    for p in paths:
        if not p.is_file():
            continue
        text = p.read_text()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            # Unparseable: fall back to the coarse answer rather than
            # silently reporting "not read".
            if name in text:
                return True
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = getattr(node, "body", None)
            if not body:
                continue
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                first.value.value = ""
        if name in ast.unparse(tree):
            return True
    return False


def test_the_doc_table_parses():
    """Guard the guard: a table restructure would make every check below pass
    vacuously."""
    rows = _doc_rows()
    assert len(rows) >= 6, (
        f"parsed only {len(rows)} rows out of physics_knobs.md's config table "
        f"({sorted(rows)}) — the table shape changed and this test now proves "
        f"nothing. Fix the parser before trusting it."
    )
    assert set(rows.values()) <= {
        _MULTI_ENV_ONLY,
        _ALL_MJWARP,
    }, f"unexpected 'read by' values: {sorted(set(rows.values()))}"


def test_multi_env_only_entries_really_are_multi_env_only():
    """★ An entry marked *multi-env only* must be absent from the shared
    engine layer AND from the stock adapter — otherwise the doc
    under-promises and a stock-adapter user skips a knob that would have
    worked.

    The stock adapter was added to this check on 2026-10-18, when the
    gripper and arm force caps were ported into it.  Without it the check
    only looked at ``setup/*.py``, so porting a knob into
    ``adapters/mujoco_warp.py`` and forgetting the column read as a pass —
    the exact rot this file exists to catch, in the one direction it was
    blind to."""
    shared = list(_ENGINE.rglob("setup/*.py")) + [_ENGINE / "mjc_contact.py"]
    stock = [_ADAPTERS / "mujoco_warp.py"]
    wrong = [e for e, who in _doc_rows().items() if who == _MULTI_ENV_ONLY and _code_mentions(shared + stock, _leaf(e))]
    assert not wrong, (
        f"marked *multi-env only* but also read where the stock adapter "
        f"reaches: {wrong}. Stock-adapter users are being told a working knob "
        f"is inert."
    )


def test_all_mjwarp_entries_are_reachable_without_the_isolated_adapter():
    """★★ The costly direction. An entry marked *all mjwarp* must be readable
    from the shared layer; if it only lives in the isolated adapter, every
    stock-adapter user setting it gets silence."""
    shared = list(_ENGINE.rglob("setup/*.py")) + [_ENGINE / "mjc_contact.py"]
    stock = [_ADAPTERS / "mujoco_warp.py"]
    wrong = [e for e, who in _doc_rows().items() if who == _ALL_MJWARP and not _mentions(shared + stock, _leaf(e))]
    assert not wrong, (
        f"marked *all mjwarp* but read nowhere the stock adapter reaches: "
        f"{wrong}. On `mujoco-warp` these are inert, and nothing says so."
    )


def test_the_isolated_adapter_has_not_quietly_taken_over_a_shared_knob():
    """The failure mode that created this test: isolating an adapter moved
    knobs out of the shared layer, and the doc kept advertising them to
    everyone. Flag any *all mjwarp* entry that the stock adapter cannot see."""
    multi = _ADAPTERS / "mujoco_warp_multi_env.py"
    shared = list(_ENGINE.rglob("setup/*.py")) + [_ENGINE / "mjc_contact.py"]
    stock = [_ADAPTERS / "mujoco_warp.py"]
    for entry, who in _doc_rows().items():
        if who != _ALL_MJWARP:
            continue
        name = _leaf(entry)
        if _mentions([multi], name) and not _mentions(shared + stock, name):
            pytest.fail(
                f"{entry} is read ONLY in mujoco_warp_multi_env.py but the doc "
                f"says *all mjwarp*. Either move the read back into the shared "
                f"layer or change the column to *multi-env only*."
            )


def test_the_inert_params_warning_covers_every_adapter_only_entry():
    """★★ ``warn_inert_params`` is the only thing that speaks up when the
    active solver is not a mjwarp one, and it works off a hand-written tuple
    in ``adapters/base.py``.  A new knob read only by the adapters and not
    added there is silent again — the same rot as a stale column, one layer
    down.

    "Adapter-only" is decided the same coarse way as every other check here:
    the leaf is mentioned in an adapter and nowhere the shared layer or the
    MJCF post-processor reaches.  An entry read outside the adapters applies
    under Featherstone too and must NOT be in the tuple, so this asserts set
    equality, not containment — over-reporting would name a knob that does
    work."""
    base_src = (_ADAPTERS / "base.py").read_text()
    registered = set(re.findall(r'\(\s*"[a-z_]+",\s*"[a-z_0-9]+",\s*"([a-z_0-9.]+)"\s*\)', base_src))
    assert registered, (
        "parsed no entries out of _ADAPTER_ONLY_PARAM_KEYS in adapters/base.py "
        "— its shape changed and this test now proves nothing"
    )

    outside = (
        list(_ENGINE.rglob("setup/*.py"))
        + [_ENGINE / "mjc_contact.py"]
        + [_INNER_PKG / "common" / "mjcf_postprocess.py"]
    )
    adapters = [_ADAPTERS / "mujoco_warp.py", _ADAPTERS / "mujoco_warp_multi_env.py"]
    expected = {
        e
        for e in _doc_rows()
        # ``newton.*`` rows are scene-yaml entries, not physics_params ones --
        # the table says so in the row itself. ``warn_inert_params`` reads a
        # params object, so it cannot see them and must not claim to.
        if not e.startswith("newton.") and _code_mentions(adapters, _leaf(e)) and not _code_mentions(outside, _leaf(e))
    }
    assert registered == expected, (
        f"_ADAPTER_ONLY_PARAM_KEYS in adapters/base.py is out of step with the "
        f"document: missing {sorted(expected - registered)}, extra "
        f"{sorted(registered - expected)}. Missing entries go unwarned on "
        f"featherstone/avbd; extra ones warn about a knob that does work."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
