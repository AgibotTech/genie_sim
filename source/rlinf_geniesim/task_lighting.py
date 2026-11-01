# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Default light rigs for tasks without a background room.

Tasks with rooms use room-authored lights. The explicit table selects legacy or
data_collection lighting so task observation distributions remain stable.
The data_collection rig follows data_collection/replay/kinematic_replay.py;
its low intensities require the specified exposure values."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LightRig:
    """Dome fill and distant key-light settings.

    A None rotation authors no xformOp. Disabling dome primary-ray visibility changes
    the visible background without changing illumination. The default visibility
    setting leaves that optional USD attribute unauthored."""

    dome_intensity: float
    dome_exposure: float
    sun_intensity: float
    sun_exposure: float
    sun_rotate: tuple
    dome_rotate: Optional[tuple] = None
    dome_kelvin: Optional[float] = None
    sun_kelvin: Optional[float] = None
    dome_visible_in_primary_ray: bool = True


LIGHT_RIGS: dict[str, LightRig] = {
    "legacy": LightRig(
        dome_intensity=200.0,
        dome_exposure=0.0,
        sun_intensity=600.0,
        sun_exposure=0.0,
        sun_rotate=(-55.0, 20.0, 0.0),
    ),
    # Every value here is copied one-for-one from ensure_default_lights,
    # including ``visibleInPrimaryRay=False``.
    #
    # That flag's effect needs to be spelled out, since it doesn't affect
    # lighting, only the background: once the dome is hidden, wherever a
    # camera ray misses everything is **pure black**, instead of the
    # dome's flat gray. The head camera on tabletop tasks mostly can't see
    # that area (the floor blocks it), but the two wrist cameras have very
    # steep viewing angles and can see past the edge of the table --
    # measured, hiding the dome puts 28.9%-33.5% of wrist camera pixels at
    # pure black, versus 0.2% when not hidden.
    #
    # A black background is the effect this task **wants**, not "matching
    # the data-collection setup": over on the data-collection side, the
    # wrist cameras see the whole room (sofa,
    # rug, walls), so pure black is only 0.9% there -- the reason they can
    # get away with setting it to False is precisely that there's always a
    # real room backing it up, so the void never shows through. We have no
    # room here, so hiding the dome amounts to choosing a black
    # background.
    #
    # A known cost, left for later: with multiple envs, the black fraction
    # is **not uniform**. GridCloner lays envs out at spacing 15 while the
    # floor only extends to +/-15 m, so with 4 envs the two outer ones fall
    # outside the floor, with 47.4% / 44.4% of the head camera pure black
    # versus 0.1% for the two inner ones; wrist camera 44.6% vs 28.9%. In
    # other words, the observed background is tied to the env index.
    # Enlarging the floor to match the grid extent (``build_gridcloner_
    # stage`` knows num_envs and spacing) would make all envs consistent --
    # the sky would still be black, but all four envs would be equally
    # black.
    "data_collection": LightRig(
        dome_intensity=1.0,
        dome_exposure=9.0,
        dome_kelvin=6150.0,
        dome_rotate=(-90.0, -30.0, 0.0),
        dome_visible_in_primary_ray=False,
        sun_intensity=1.0,
        sun_exposure=10.0,
        sun_kelvin=7250.0,
        sun_rotate=(-35.0, 45.0, 0.0),
    ),
}

DEFAULT_RIG = "legacy"

# Value = which light rig this sub-task uses. Comment = why it doesn't use the default rig.
TASK_LIGHT_RIG: dict[str, str] = {
    # Sub-task name -> rig name in ``LIGHT_RIGS``. Empty: every task runs on
    # ``DEFAULT_RIG``. A task whose test set replays recorded episodes can be
    # listed here with ``"data_collection"`` so RL sees the lights the data
    # was collected under; doing so changes its observation distribution,
    # so trained policies and eval numbers stop being comparable (see the
    # module docstring).
}


def resolve_light_rig(
    sub_task_name: str,
    default: str = DEFAULT_RIG,
) -> tuple[LightRig, str, str]:
    """Return (rig, rig_name, source), using the explicit task table before default.

    Raise KeyError for an unknown rig name rather than substituting another rig."""
    name, source = (
        (TASK_LIGHT_RIG[sub_task_name], "table") if sub_task_name in TASK_LIGHT_RIG else (str(default), "default")
    )
    if name not in LIGHT_RIGS:
        raise KeyError(
            f"unknown light rig {name!r} (task {sub_task_name!r}, source {source}). " f"available: {sorted(LIGHT_RIGS)}"
        )
    return LIGHT_RIGS[name], name, source


def perturb_light_rig(rig: LightRig, perturbation: dict) -> LightRig:
    """Return a new rig with sampled intensity scales and sun rotation offsets.

    perturbation accepts dome_intensity_scale, sun_intensity_scale and
    sun_rotate_delta_deg. Exposure, dome rotation, temperature and visibility remain
    unchanged. Empty or neutral perturbations preserve the rig field-for-field.
    The perturbation follows data_collection/replay/fuzzer/set.py's light contract."""
    dome_scale = float(perturbation.get("dome_intensity_scale", 1.0))
    sun_scale = float(perturbation.get("sun_intensity_scale", 1.0))
    dx, dy, dz = (float(v) for v in (perturbation.get("sun_rotate_delta_deg") or (0.0, 0.0, 0.0)))
    ox, oy, oz = rig.sun_rotate
    return dataclasses.replace(
        rig,
        dome_intensity=rig.dome_intensity * dome_scale,
        sun_intensity=rig.sun_intensity * sun_scale,
        sun_rotate=(ox + dx, oy + dy, oz + dz),
    )


__all__ = ["LightRig", "LIGHT_RIGS", "TASK_LIGHT_RIG", "DEFAULT_RIG", "resolve_light_rig", "perturb_light_rig"]
