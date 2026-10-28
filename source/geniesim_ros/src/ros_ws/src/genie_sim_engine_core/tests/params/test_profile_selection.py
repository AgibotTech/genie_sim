# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Profile resolution accepts filenames, relative paths and absolute paths.

Named profiles must resolve to readable files, and failed explicit requests
must be reported rather than silently selecting built-in defaults.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

from genie_sim_engine_core.common.session import _resolve_physics_params_file

_CONFIG = pathlib.Path(__file__).resolve().parents[2] / "config"
_LAUNCHERS = _CONFIG.parents[1] / "genie_sim_bringup" / "config"


@pytest.fixture(autouse=True)
def _share_dir(monkeypatch):
    """Point ament at the source tree.

    These tests run from source, where ``genie_sim_engine_core`` is not an
    installed ament package, so the real lookup raises.  The resolver's own
    ament-missing branch is exercised separately by
    ``test_ament_failure_still_reports_a_requested_profile``; everything else
    wants a working share dir so it is testing resolution, not the environment.
    """
    import ament_index_python.packages as _amp

    monkeypatch.setattr(_amp, "get_package_share_directory", lambda pkg: str(_CONFIG.parent))


class _Log:
    def __init__(self):
        self.info_lines: list = []
        self.warn_lines: list = []
        self.error_lines: list = []

    def info(self, m):
        self.info_lines.append(str(m))

    def warn(self, m):
        self.warn_lines.append(str(m))

    def error(self, m):
        self.error_lines.append(str(m))


# ---------------------------------------------------------------- resolver


def test_empty_lands_on_the_base_profile():
    log = _Log()
    out = _resolve_physics_params_file("", log)
    assert out.endswith("physics_params.yaml")
    assert any("base profile" in line for line in log.info_lines)


def test_bare_filename_resolves_against_the_package_config_dir():
    log = _Log()
    out = _resolve_physics_params_file("physics_params_rlinf.yaml", log)
    assert pathlib.Path(out).name == "physics_params_rlinf.yaml"
    assert pathlib.Path(out).is_file(), "a shipped profile name did not resolve to a real file"


def test_absolute_path_is_taken_as_given(tmp_path):
    p = tmp_path / "out_of_tree.yaml"
    p.write_text("solver:\n  mjc_iterations: 7\n")
    assert _resolve_physics_params_file(str(p), _Log()) == str(p)


def test_a_missing_profile_is_an_error_not_a_shrug():
    """The consequence of a typo is silent — ``load_physics_params`` falls back
    to built-in defaults, so the run uses physics nobody asked for.  The name
    that does not exist must be loud at the point it is resolved."""
    log = _Log()
    out = _resolve_physics_params_file("physics_params_typo.yaml", log)
    assert out == ""
    assert log.error_lines, "a nonexistent profile resolved without an error"
    assert "physics_params_typo.yaml" in log.error_lines[0]


def test_resolution_is_always_logged():
    """Silence is what let a dead selector survive: with nothing in the log, a
    fallback and a successful load looked identical."""
    for requested in ("", "physics_params.yaml", "physics_params_rlinf.yaml"):
        log = _Log()
        _resolve_physics_params_file(requested, log)
        assert log.info_lines, f"resolving {requested!r} logged nothing"


def test_ament_failure_still_reports_a_requested_profile(monkeypatch):
    """If the share dir cannot be resolved at all, a launcher that asked for a
    named profile must not quietly get built-in defaults."""
    import ament_index_python.packages as _amp

    def _boom(pkg):
        raise RuntimeError("no ament")

    monkeypatch.setattr(_amp, "get_package_share_directory", _boom)
    log = _Log()
    out = _resolve_physics_params_file("physics_params_rlinf.yaml", log)
    assert out == ""
    assert log.warn_lines, "ament failure was not reported"
    assert log.error_lines, "a requested profile silently vanished"


# ---------------------------------------------------------------- launchers


def _engine_launchers():
    out = []
    for p in sorted(_LAUNCHERS.glob("launcher_*.yaml")):
        doc = yaml.safe_load(p.read_text()) or {}
        if "genie_sim_engine" in doc:
            out.append(p)
    return out


def test_the_sweep_found_launchers_to_check():
    """Guards the glob: a moved config directory would otherwise make every
    test below pass by checking nothing."""
    found = _engine_launchers()
    assert len(found) >= 5, f"only {len(found)} engine launchers found under {_LAUNCHERS}"


@pytest.mark.parametrize("path", _engine_launchers(), ids=lambda p: p.name)
def test_every_engine_launcher_names_a_profile(path):
    params = (yaml.safe_load(path.read_text()) or {})["genie_sim_engine"]["ros__parameters"]
    assert "physics_params_file" in params, (
        f"{path.name} does not set physics_params_file. It will silently load the base "
        f"profile — which may even be what you want, but say so, because an unset key is "
        f"indistinguishable from the bug this file exists to prevent."
    )


@pytest.mark.parametrize("path", _engine_launchers(), ids=lambda p: p.name)
def test_every_named_profile_exists(path):
    params = (yaml.safe_load(path.read_text()) or {})["genie_sim_engine"]["ros__parameters"]
    name = params.get("physics_params_file", "")
    assert name, f"{path.name}: empty physics_params_file"
    resolved = pathlib.Path(name) if pathlib.Path(name).is_absolute() else _CONFIG / name
    assert resolved.is_file(), f"{path.name} names {name!r}, which does not exist at {resolved}"


def test_the_rlinf_profile_has_a_consumer():
    """The point of the exercise.  If this fails, the profile is back to being
    config nothing can reach — delete it or wire it, but do not ship it dead."""
    named = set()
    for p in _engine_launchers():
        params = (yaml.safe_load(p.read_text()) or {})["genie_sim_engine"]["ros__parameters"]
        named.add(params.get("physics_params_file", ""))
    assert (
        "physics_params_rlinf.yaml" in named
    ), "no launcher selects physics_params_rlinf.yaml — it ships with zero readers again"
