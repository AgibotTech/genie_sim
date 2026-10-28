#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Lifecycle helpers for ``genie_sim_engine``.

``bootstrap_simulation_app`` MUST be called before any ``omni.*`` /
``isaacsim.*`` import; it constructs the ``SimulationApp`` (headless or
GUI), strips ``--gui`` from ``sys.argv``, and returns ``(app, headless)``.

``configure_carb_settings`` may import omni / isaacsim freely because it is
only invoked from the running node, well after bootstrap.
"""

from __future__ import annotations

import os
import sys
from typing import Tuple
from genie_sim_engine_newton.warmup import warmup_newton_kernels
from .launch_config import prepare_launch_config

_VALID_ENGINES = ("isaac_physx", "isaac_newton", "newton")

# ---------------------------------------------------------------------------
# Render-mode map: ``--render-mode`` choices → (hydra_engine, rtx_submode).
#
# A single CLI knob picks the RTX submode. We don't expose the Pixar Storm
# delegate because it isn't bundled in ``isaacsim.exp.base.python.kit``
# (the minimal kit our newton-standalone path uses). When we tried Storm,
# ``vp.set_hd_engine("Storm")`` spammed
# ``UsdContext::createViewport — unable to find suitable engine`` warnings
# at ~15 Hz and silently fell back to whatever RTX submode was active. If
# you really need Storm, you'd have to switch to the full Isaac Sim kit
# experience and pay the heavier startup — not worth it just for a debug
# viewport.
#
# Both layers (kit args at app create time + ``configure_viewport_for_debug``
# at runtime) target the same RTX submode so a late kit-defaults push
# doesn't override the launch arg.
#
# Rough cost at the fr3+cloth scene on a 5090:
#   raster     ~16 ms    RaytracedLighting: raster + RT shadows + RT denoise.
#   pathtrace  ~49 ms    RealTimePathTracing (kit default).
#   offline    100-300ms PathTracing (no denoiser, multi-sample).
# ---------------------------------------------------------------------------
RENDER_MODE_MAP: dict[str, tuple[str, str]] = {
    "raster": ("rtx", "RaytracedLighting"),
    "pathtrace": ("rtx", "RealTimePathTracing"),
    "offline": ("rtx", "PathTracing"),
}
VALID_RENDER_MODES = tuple(RENDER_MODE_MAP.keys())
DEFAULT_RENDER_MODE = "raster"


def _resolve_render_mode(render_mode: str) -> tuple[str, str, str]:
    """Return ``(canonical_name, hydra_engine, rtx_submode)`` for a mode.

    Falls back to ``DEFAULT_RENDER_MODE`` on unrecognised input rather than
    raising, so a malformed launch arg doesn't crash bringup.
    """
    name = (render_mode or "").strip().lower() or DEFAULT_RENDER_MODE
    if name not in RENDER_MODE_MAP:
        print(
            f"[genie_sim_engine] render_mode={render_mode!r} not recognised; "
            f"valid: {VALID_RENDER_MODES}. Falling back to {DEFAULT_RENDER_MODE!r}.",
            flush=True,
        )
        name = DEFAULT_RENDER_MODE
    engine, submode = RENDER_MODE_MAP[name]
    return name, engine, submode


def _validate_engine_id(physics_engine: str) -> str:
    """Lower-case + validate against the canonical engine set."""
    raw = physics_engine.strip().lower()
    if raw not in _VALID_ENGINES:
        raise ValueError(f"physics_engine={physics_engine!r} is not a valid engine id. " f"Valid: {_VALID_ENGINES}.")
    return raw


def _configure_isaac_physx(headless: bool, physics_solver: str, render_mode: str, cfg: dict) -> str:
    print("[genie_sim_engine] physics engine: isaac_physx", flush=True)
    return ""


# ---------------------------------------------------------------------------
# Isaac install-layout resolution (GENIESIM4_MIGRATION.md §4 R13).
#
# Three places in this package reach into Isaac's *filesystem layout*
# rather than its API, each with its own hardcoded literal naming the
# dist-packages root, the Python minor and a per-extension directory.
# An API break is a traceback; a layout break is whatever the
# surrounding code does with a missing path -- and 6.2 moved the URDF
# importer into an independently-built ``isaacsim-asset`` wheel, which
# is exactly when an ``exts/<ext>/pip_prebundle`` directory relocates.
#
# ``resolve_isaac_paths`` resolves all three from the installed
# ``isaacsim`` package and records WHICH branch produced each one, so
# the startup banner can print resolved paths instead of restating the
# literals.  It never raises: an unresolvable path comes back with
# ``exists=False`` and is reported as such.
#
# NOTE: ``assemble/_robot_impl.py`` still carries its own copy of the
# prebundle literal (it runs in the converter process, not here).  The
# banner prints what this function resolves, so a divergence between
# the two becomes visible rather than silent.
# ---------------------------------------------------------------------------
_ISAAC_PKG_LITERAL = "/usr/local/lib/python3.12/dist-packages/isaacsim"
_KIT_EXPERIENCE_LEAF = ("apps", "isaacsim.exp.full.newton.kit")
_URDF_PREBUNDLE_LEAF = ("exts", "isaacsim.asset.importer.urdf", "pip_prebundle")

# Filled by ``_configure_isaac_newton`` so the banner reports the
# experience the SimulationApp was actually launched with, and the
# solver requested by the launcher -- not what the
# resolution would pick if re-run later.
_LAUNCHED: dict[str, str] = {}


def _isaac_package_root() -> tuple[str, str]:
    """Return ``(path, how)`` for the installed ``isaacsim`` package dir.

    ``how`` is ``"importlib"`` when the package resolved properly,
    ``"literal"`` when only the hardcoded dist-packages path exists, or
    ``"unresolved"``.
    """
    try:
        import importlib.util as _iu  # local import keeps cold-path overhead off

        spec = _iu.find_spec("isaacsim")
        if spec is not None and spec.submodule_search_locations:
            return (list(spec.submodule_search_locations)[0], "importlib")
    except Exception:  # noqa: BLE001 -- a broken isaacsim must not break the banner
        pass
    if os.path.isdir(_ISAAC_PKG_LITERAL):
        return (_ISAAC_PKG_LITERAL, "literal")
    return ("", "unresolved")


def resolve_isaac_paths() -> dict[str, dict]:
    """Resolve the three Isaac filesystem paths this package depends on.

    Returns ``{name: {"path": str, "how": str, "exists": bool}}`` for
    ``package_root``, ``kit_experience`` and ``urdf_prebundle``.

    Resolution order per path: an operator env override, then the
    hardcoded literal (a documented fast path -- it is correct in the
    shipped image and costs no import), then the ``isaacsim`` package
    root.  ``how`` names the branch that won, which is the part worth
    printing: ``"literal"`` means the layout assumption still holds,
    ``"importlib"`` means it moved and we followed it, and
    ``"unresolved"`` means the caller is about to fail.
    """
    root, root_how = _isaac_package_root()
    out: dict[str, dict] = {
        "package_root": {"path": root, "how": root_how, "exists": bool(root) and os.path.isdir(root)},
    }

    for name, leaf, env_var, is_file in (
        ("kit_experience", _KIT_EXPERIENCE_LEAF, "GENIESIM_ISAACSIM_KIT", True),
        ("urdf_prebundle", _URDF_PREBUNDLE_LEAF, "GENIESIM_ISAACSIM_URDF_PREBUNDLE", False),
    ):
        check = os.path.isfile if is_file else os.path.isdir
        path, how = "", "unresolved"
        env_val = os.environ.get(env_var, "")
        if env_val and check(env_val):
            path, how = env_val, "env"
        else:
            literal = os.path.join(_ISAAC_PKG_LITERAL, *leaf)
            if check(literal):
                path, how = literal, "literal"
            elif root:
                candidate = os.path.join(root, *leaf)
                path, how = candidate, "importlib" if check(candidate) else "unresolved"
        out[name] = {"path": path, "how": how, "exists": bool(path) and check(path)}
    return out


def _configure_isaac_newton(headless: bool, physics_solver: str, render_mode: str, cfg: dict) -> str:
    solver = physics_solver.strip().lower()
    # The .kit experience, the prebundle and the package root all resolve
    # through ``resolve_isaac_paths`` (env override -> hardcoded literal ->
    # ``isaacsim`` package root).  Keeping the resolution in one function
    # is what lets the startup banner print the path this app was launched
    # with rather than restating the literal -- see R13.
    paths = resolve_isaac_paths()
    experience = paths["kit_experience"]["path"]
    if not experience:
        # Unresolvable: fall back to the literal so SimulationApp raises
        # against a concrete path instead of an empty string, which it
        # would silently read as "use the default experience".
        experience = os.path.join(_ISAAC_PKG_LITERAL, *_KIT_EXPERIENCE_LEAF)
        print(
            f"[genie_sim_engine] WARNING: could not resolve the isaacsim newton .kit experience; "
            f"falling back to {experience!r} (set GENIESIM_ISAACSIM_KIT to override).",
            flush=True,
        )
    # Solver selection is authored through the PhysicsScene API before reset;
    # 6.2 does not consume the former /exts/.../solver_type Kit setting.
    _LAUNCHED["kit_experience"] = experience
    _LAUNCHED["kit_experience_how"] = paths["kit_experience"]["how"]
    _LAUNCHED["requested_solver_type"] = solver
    print(f"[genie_sim_engine] physics engine: isaac_newton (solver={solver})", flush=True)
    return experience


def _configure_newton(headless: bool, physics_solver: str, render_mode: str, cfg: dict) -> str | None:
    # Newton-standalone is Kit-free.  This function unconditionally
    # returns the Kit-free sentinel so ``bootstrap_simulation_app``
    # early-returns ``(None, True)`` for callers that still reach it
    # (the supported entry point is ``genie_sim_engine_newton_node.py``,
    # which never calls it at all).
    warmup_newton_kernels("vbd")
    print("[genie_sim_engine] physics engine: newton (Kit-free; SimulationApp suppressed)", flush=True)
    return None


_ENGINE_CONFIGURE = {
    "isaac_physx": _configure_isaac_physx,
    "isaac_newton": _configure_isaac_newton,
    "newton": _configure_newton,
}

_ENGINE_POST_WARMUP = {
    "isaac_newton": lambda solver: warmup_newton_kernels(solver),
    "newton": lambda _: warmup_newton_kernels("vbd"),
}


def bootstrap_simulation_app(
    physics_engine: str = "isaac_physx",
    physics_solver: str = "mujoco-warp",
    render_mode: str = DEFAULT_RENDER_MODE,
) -> Tuple[object, bool]:
    """Create the ``SimulationApp`` and return ``(app, headless)``.

    Honors ``ISAACSIM_HEADLESS`` (set to a falsy value to force GUI) and
    consumes a ``--gui`` flag from ``sys.argv``.

    Args:
        physics_engine: ``"isaac_physx"`` (default) or ``"isaac_newton"``.
            Both go through Isaac Sim's wrappers; the choice picks
            ``omni.physx`` (PhysX 5) vs ``isaacsim.physics.newton``
            (Newton solver, rigid-body bridge only — no cloth / soft).
        physics_solver: Solver for ``isaac_newton`` — ``"mujoco-warp"``
            (default, MuJoCo-Warp), ``"xpbd"``, ``"featherstone"``, or
            ``"semiImplicit"``. Ignored for ``isaac_physx``.
    """
    import isaacsim  # noqa: F401
    from isaacsim import SimulationApp

    physics_engine = _validate_engine_id(physics_engine)

    headless = True
    if os.environ.get("ISAACSIM_HEADLESS", "").strip().lower() in ("0", "false", "no", "off"):
        headless = False
    if "--gui" in sys.argv:
        headless = False
        sys.argv.remove("--gui")

    launch_config: dict = {"headless": headless}

    experience = _ENGINE_CONFIGURE[physics_engine](headless, physics_solver, render_mode, launch_config)
    if experience is None:
        return None, True  # newton + headless: Kit-free early return

    app = SimulationApp(prepare_launch_config(launch_config), experience=experience)

    if physics_engine in _ENGINE_POST_WARMUP:
        _ENGINE_POST_WARMUP[physics_engine](physics_solver.strip().lower())

    return app, headless


def configure_viewport_for_debug(
    render_mode: str = DEFAULT_RENDER_MODE,
    updates_enabled: bool | None = None,
) -> None:
    """Configure the editor viewport for a debug-only use case.

    Call this AFTER the scene has loaded (i.e. after the engine has opened
    the stage and rendered a few frames). Otherwise the viewport never gets
    to render the scene and the GUI stays black.

    Args:
        render_mode: One of ``RENDER_MODE_MAP`` keys (``storm`` / ``raster``
            / ``pathtrace`` / ``offline``). Resolved to a ``(hydra_engine,
            rtx_submode)`` pair, then applied via BOTH ``carb.settings`` and
            ``vp.set_hd_engine`` / ``vp.render_mode``. Pinning at multiple
            layers is required because kit's RTX defaults will override a
            launch-arg-only ``/rtx/rendermode`` flip during composer setup.
        updates_enabled: If False, the viewport stops re-rendering and stays
            on the last frame (cheap: ~2ms / simulation_app.update()). If True,
            the viewport keeps re-rendering live. If None, read the
            ``ISAACSIM_VIEWPORT_UPDATES`` env var (default: **True**, i.e.
            live debug view).
    """
    rm_name, hydra_engine, rtx_submode = _resolve_render_mode(render_mode)

    try:
        import carb.settings
        import omni.kit.viewport.utility as vpu
    except Exception as exc:
        print(f"[genie_sim_engine] viewport configure: import failed ({exc})", flush=True)
        return

    vp = vpu.get_active_viewport()
    if vp is None:
        print("[genie_sim_engine] no active viewport; skipping config", flush=True)
        return

    # Diagnostic: what does /rtx/rendermode hold right now?
    try:
        s = carb.settings.get_settings()
        print(
            f"[genie_sim_engine] viewport configure: render_mode={rm_name!r} → "
            f"engine={hydra_engine}, submode={rtx_submode}  "
            f"| live /rtx/rendermode = {s.get('/rtx/rendermode')!r}",
            flush=True,
        )
    except Exception:
        pass

    # 0. Pin carb settings (live + defaults + transient) BEFORE flipping the
    # viewport. The kit's RTX defaults override the launch-arg
    # /rtx/rendermode when the renderer is first attached; flipping the
    # transient setting too guarantees the composer picks up the choice on
    # the next update.
    try:
        s.set("/rtx/rendermode", rtx_submode)
        s.set("/rtx-defaults/rendermode", rtx_submode)
        s.set("/rtx-transient/rendermode", rtx_submode)
    except Exception as exc:
        print(
            f"[genie_sim_engine] viewport: carb /rtx*/rendermode set failed: {exc}",
            flush=True,
        )

    # 1. Pin the Hydra engine on the live viewport. Always ``"rtx"`` now —
    # see RENDER_MODE_MAP for why Storm isn't supported.
    try:
        before_engine = getattr(vp, "hydra_engine", None)
        vp.set_hd_engine(hydra_engine)
        print(
            f"[genie_sim_engine] viewport hydra_engine: {before_engine} → {hydra_engine}",
            flush=True,
        )
    except Exception as exc:
        print(f"[genie_sim_engine] viewport.set_hd_engine failed: {exc}", flush=True)

    # 2. Set the submode on the live viewport. ``vp.render_mode = X`` is
    # what triggers the viewport to actually flip (carb settings alone
    # don't force a re-attach).
    try:
        before = vp.render_mode
        vp.render_mode = rtx_submode
        print(
            f"[genie_sim_engine] viewport render mode: {before} → {vp.render_mode}",
            flush=True,
        )
    except Exception as exc:
        print(f"[genie_sim_engine] viewport.render_mode setter failed: {exc}", flush=True)

    try:
        print(f"[genie_sim_engine] viewport hydra_engine (read-back) = {vp.hydra_engine}", flush=True)
    except Exception:
        pass

    # 2. Decide updates_enabled. Default is LIVE (True) — user wants the
    # viewport to refresh continuously by default for debug. Set
    # ISAACSIM_VIEWPORT_UPDATES=0 (or false/no/off) to freeze it instead,
    # which makes simulation_app.update() ~2ms instead of ~33ms.
    if updates_enabled is None:
        env = os.environ.get("ISAACSIM_VIEWPORT_UPDATES", "").strip().lower()
        if env in ("0", "false", "no", "off"):
            updates_enabled = False
        else:
            updates_enabled = True  # default: live

    try:
        vp.updates_enabled = bool(updates_enabled)
        if updates_enabled:
            print(
                "[genie_sim_engine] viewport updates_enabled = True  (live debug view; "
                "set ISAACSIM_VIEWPORT_UPDATES=0 to freeze for ~33ms→~2ms render savings)",
                flush=True,
            )
        else:
            print(
                "[genie_sim_engine] viewport updates_enabled = False  " "(GUI frozen at last frame; cheap render)",
                flush=True,
            )
    except Exception as exc:
        print(f"[genie_sim_engine] viewport.updates_enabled setter failed: {exc}", flush=True)


def configure_carb_settings(headless: bool, logger) -> None:
    """Apply the carb runtime settings used by the physics loop."""
    import carb

    settings = carb.settings.get_settings()
    try:
        if headless:
            settings.set_bool("/app/runLoops/main/rateLimitEnabled", False)
            settings.set_bool("/app/runLoopsGlobal/rateLimitEnabled", False)
            settings.set_int("/app/runLoops/main/rateLimitFrequency", 1000)
            settings.set_bool("/app/runLoops/present/rateLimitEnabled", False)
            settings.set_bool("/persistent/app/runLoops/main/rateLimitEnabled", False)
            settings.set_bool("/app/asyncRendering", True)
            settings.set_bool("/app/asyncRenderingLowLatency", True)
            settings.set_bool("/app/vsync", False)
        else:
            settings.set_bool("/app/runLoops/main/rateLimitEnabled", True)
            settings.set_int("/app/runLoops/main/rateLimitFrequency", 60)
            settings.set_bool("/app/asyncRendering", False)
            settings.set_bool("/app/vsync", False)
        settings.set_float("/physics/minFrameRate", 0.0)
        settings.set_int("/persistent/simulation/minFrameRate", 0)
        settings.set_bool("/rtx/ecoMode/enabled", False)
        settings.set_bool("/persistent/app/usd/muteUsdDiagnostics", False)
    except Exception as exc:
        logger.warn(f"failed to set carb settings: {exc}")
