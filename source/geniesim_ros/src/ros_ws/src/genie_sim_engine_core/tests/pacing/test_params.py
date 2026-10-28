# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""The launcher-YAML surface: step_mode / step_idle_timeout / step_lockstep_ticks.

Three separate hops, each with its own way to fail silently:

  1. ``EngineNodeParams.from_dict`` — coercion.  A YAML value is always a
     string here, so ``step_lockstep_ticks: 8`` must survive as an int and
     ``Lockstep`` must normalise to a registry key.
  2. ``EngineSession`` — forwarding.  A field that is parsed but never
     passed on is the classic three-file-change bug this package has been
     bitten by before (``mujoco_njmax``); the constructor-signature check
     below is what catches it without booting a backend.
  3. ``pacing.make_pacer`` — validation.  Unknown modes RAISE.

The default must stay exactly rolling's behaviour: realtime, gate off,
pacer default burst.  Anything else is a silent behaviour change for
every existing launcher yaml in the tree.
"""

import inspect

import pytest

from genie_sim_engine_core.common.pacing import DEFAULT_STEP_MODE, make_pacer
from genie_sim_engine_core.common.params import EngineNodeParams


class Bare:
    """A backend exposing no command-activity signal.

    ``activity is None`` is how ``PhysicsEngine`` spells that, and it is a
    declared member rather than an absent one — the pacer asks a question it
    is guaranteed an answer to, instead of probing for private names that a
    typo would silently turn into "no signal".
    """

    activity = None


# ----------------------------------------------------------------------
# 1. coercion
# ----------------------------------------------------------------------


def test_unset_is_rollings_behaviour():
    ep = EngineNodeParams.from_dict({})
    assert ep.step_mode == ""  # -> DEFAULT_STEP_MODE
    assert ep.step_idle_timeout == 0.0  # gate off
    assert ep.step_lockstep_ticks == 0  # pacer default
    assert make_pacer(ep.step_mode or DEFAULT_STEP_MODE, Bare(), 0.01, 1).mode == "realtime"


def test_mode_is_normalised_for_the_registry():
    """YAML capitalisation must not change which pacer you get."""
    for raw in ("lockstep", "  LockStep ", "LOCKSTEP"):
        assert EngineNodeParams.from_dict({"step_mode": raw}).step_mode == "lockstep"


def test_numeric_knobs_survive_as_numbers():
    ep = EngineNodeParams.from_dict({"step_idle_timeout": "2.5", "step_lockstep_ticks": "8"})
    assert ep.step_idle_timeout == 2.5 and isinstance(ep.step_idle_timeout, float)
    assert ep.step_lockstep_ticks == 8 and isinstance(ep.step_lockstep_ticks, int)


@pytest.mark.parametrize("bad", ["-3", "banana", "", None])
def test_bad_numbers_fall_back_to_off_not_to_a_crash(bad):
    """A malformed number disables the feature; it never crashes bring-up,
    and it never becomes negative (a negative timeout would pause forever)."""
    ep = EngineNodeParams.from_dict({"step_idle_timeout": bad, "step_lockstep_ticks": bad})
    assert ep.step_idle_timeout == 0.0
    assert ep.step_lockstep_ticks == 0


# ----------------------------------------------------------------------
# 2. forwarding
# ----------------------------------------------------------------------


def test_the_loop_names_every_field_the_params_declare():
    """The ``mujoco_njmax`` failure mode: parsed, documented, never named.

    Checked by reflection rather than by booting a session, so it runs
    without ROS, a GPU, or the compiled core .so.
    """
    from genie_sim_engine_core.common.loop import EngineRunLoop

    accepted = set(inspect.signature(EngineRunLoop.__init__).parameters)
    for field in ("step_mode", "step_idle_timeout", "step_lockstep_ticks"):
        assert field in accepted, (
            f"EngineNodeParams declares {field!r} but EngineRunLoop.__init__ "
            f"does not name it — the value would be parsed and dropped."
        )


def test_the_session_forwards_all_three():
    """Source-level check on the one construction site.  Cheap, and it fails
    loudly the day someone adds a fourth knob and forwards only three."""
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[2] / "genie_sim_engine_core" / "common" / "session.py"
    text = src.read_text()
    for field in ("step_mode=", "step_idle_timeout=", "step_lockstep_ticks="):
        assert field in text, f"session.py never forwards {field}"


# ----------------------------------------------------------------------
# 3. validation
# ----------------------------------------------------------------------


def test_a_typo_stops_the_run():
    """Deliberately stricter than the tri-state contact_publish_* strings:
    a wrong step_mode redefines what one training step MEANS."""
    ep = EngineNodeParams.from_dict({"step_mode": "lockstepp"})
    with pytest.raises(ValueError) as exc:
        make_pacer(ep.step_mode, Bare(), 0.01, 1)
    assert "expected one of" in str(exc.value)


def test_every_documented_mode_is_registered():
    for mode in ("realtime", "lockstep", "freerun"):
        assert make_pacer(mode, Bare(), 0.01, 1).mode == mode
