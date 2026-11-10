# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Manage SimulationApp startup, backend initialization, settings, and extensions."""

from __future__ import annotations

from typing import Any, Iterable, Literal, Mapping, Optional

from .backend import BackendConfig, init_backend

RtxMode = Literal["realtime", "pathtrace", "off"]


_RTX_MODE_TO_RENDERER = {
    "realtime": "RayTracedLighting",
    "pathtrace": "RealTimePathTracing",
    "off": "RayTracedLighting",
}


# Use Any to avoid importing isaacsim before startup. The handle exposes
# SimulationApp methods such as is_running, close, and update.
AppHandle = Any


_app: Optional[AppHandle] = None


def init_app(
    *,
    backend_config: BackendConfig,
    headless: bool = True,
    rtx_mode: RtxMode = "pathtrace",
    extensions: Iterable[str] = (),
    async_rendering: bool = False,
    extra_args: Optional[Iterable[str]] = None,
    hide_ui: bool = False,
    rtx_settings: Optional[Mapping[str, Any]] = None,
) -> AppHandle:
    """Start SimulationApp, apply settings, enable extensions, and initialize the backend.

    Args:
        backend_config: Physics/render devices and tensor output configuration.
        headless: Run without the application window.
        rtx_mode: RTX preset: realtime, pathtrace, or off.
        extensions: Extensions to enable after application startup.
        async_rendering: Replicator async rendering; disabled for synchronized capture.
        extra_args: Additional SimulationApp arguments, e.g. livestream options.
        hide_ui: Hide the UI.
        rtx_settings: Additional carb path/value settings, applied after the fixed
            physics and async-rendering settings.

    Returns:
        SimulationApp handle; use is_running() in the main loop.
    Repeated startup raises RuntimeError."""
    global _app
    if _app is not None:
        raise RuntimeError(
            "init_app has already been called. Only one SimulationApp may be started per process."
        )

    # 1. Initialize the backend before settings and World construction.
    init_backend(backend_config)

    # 2. Start SimulationApp before importing Isaac Sim submodules.
    from isaacsim import SimulationApp  # noqa: PLC0415 (intentional lazy import)

    app_kwargs: dict = {
        "headless": headless,
        "renderer": _RTX_MODE_TO_RENDERER[rtx_mode],
        "hide_ui": hide_ui,
    }
    if extra_args is not None:
        app_kwargs["extra_args"] = list(extra_args)

    _app = SimulationApp(prepare_launch_config(app_kwargs))

    # 3. Apply fixed carb settings followed by optional caller overrides.
    _apply_carb_settings(_app, async_rendering=async_rendering, rtx_settings=rtx_settings)

    # 4. Enable extensions after SimulationApp starts.
    if extensions:
        from isaacsim.core.utils import extensions as _ext  # noqa: PLC0415

        for ext_name in extensions:
            _ext.enable_extension(ext_name)

    # 5. Import and patch torch compatibility after the app and extensions load.
    if backend_config.resolve_tensors_as() == "torch":
        _apply_torch_backend_compat_patches()

    return _app


def prepare_launch_config(config: dict) -> dict:
    """Keep explicit Replicator captures working with Isaac Sim 6.2.

    The 6.2 multi-tick renderer stalls after stage replacement and fails to
    advance to Replicator's scheduled frame. The same scene with legacy tick
    scheduling returns RGB and metre-valued depth, including mixed-height tiled
    captures. Set this before Kit starts so its initial viewport uses the same
    scheduling contract as later render products. Legacy 5.1/4.5 stays unchanged.
    This is an SDK compatibility requirement, not a physics or pacing knob.
    """
    from importlib import metadata
    import os
    from pathlib import Path
    import sys

    result = dict(config)
    try:
        root = os.environ.get("ISAAC_PATH")
        if root and Path(sys.executable).resolve().is_relative_to(Path(root).resolve() / "kit/python"):
            # Bundled Python has no wheel metadata and may inherit another SDK's paths.
            installed_version = (Path(root) / "VERSION").read_text().strip()
        else:
            installed_version = metadata.version("isaacsim")
        version = tuple(int(part) for part in installed_version.split(".")[:2])
    except (metadata.PackageNotFoundError, OSError, ValueError):
        return result
    if version == (6, 2):
        args = list(result.get("extra_args", []))
        args.append("--/rtx/hydra/supportMultiTickRate=false")
        result["extra_args"] = args
    return result


def _apply_torch_backend_compat_patches() -> None:
    """Bridge CUDA tensor outputs to numpy-only consumers and read live Fabric poses.
    Lula motion generation and SciPy require numpy inputs. ArticulationSubset
    outputs and single-prim poses are converted before reaching those consumers.
    Physical prims use Fabric to avoid stale USD transforms on the GPU pipeline;
    nonphysical markers use USD because Fabric does not track their updates."""
    from isaacsim.core.api.articulations.articulation_subset import (  # noqa: PLC0415
        ArticulationSubset,
    )
    # Try both wrapper module locations used by Isaac Sim releases.
    # Warn if neither exists; GPU pose reads may then remain stale.
    SinglePrimWrapper = None
    try:
        from isaacsim.core.prims.impl._impl.single_prim_wrapper import (  # noqa: PLC0415
            _SinglePrimWrapper as SinglePrimWrapper,
        )
    except ImportError:
        try:
            from isaacsim.core.prims.impl.single_prim_wrapper import (  # noqa: PLC0415
                _SinglePrimWrapper as SinglePrimWrapper,
            )
        except ImportError as exc:
            import warnings
            warnings.warn(
                f"_SinglePrimWrapper not found in isaacsim.core.prims.impl{{._impl}}.single_prim_wrapper "
                f"({exc}); GPU pipeline pose read patch skipped. May affect GPU-mode physics pose freshness."
            )
    from pxr import UsdPhysics  # noqa: PLC0415

    from .backend import to_numpy as _to_numpy  # noqa: PLC0415

    _orig_gjp = ArticulationSubset.get_joint_positions
    _orig_gjv = ArticulationSubset.get_joint_velocities
    _orig_glp = SinglePrimWrapper.get_local_pose if SinglePrimWrapper is not None else None

    def _patched_get_joint_positions(self):
        return _to_numpy(_orig_gjp(self))

    def _patched_get_joint_velocities(self):
        return _to_numpy(_orig_gjv(self))

    def _is_physics_tracked(prim_view) -> bool:
        """Check whether any view prim has rigid-body or articulation-root APIs.
        Physical prims use Fabric for current poses. Nonphysical markers must use USD
        because Fabric pose updates can fail silently or return an initial snapshot."""
        try:
            prims = prim_view._prims
        except AttributeError:
            return False
        return any(
            p.HasAPI(UsdPhysics.RigidBodyAPI) or p.HasAPI(UsdPhysics.ArticulationRootAPI)
            for p in prims
        )

    def _patched_get_world_pose(self):
        # Read physical prims from Fabric and nonphysical markers from USD.
        usd = not _is_physics_tracked(self._prim_view)
        positions, orientations = self._prim_view.get_world_poses(usd=usd)
        return _to_numpy(positions[0]), _to_numpy(orientations[0])

    def _patched_get_local_pose(self):
        # Local pose reads use USD parent transforms with no usd selector.
        # Convert CUDA tensor outputs to numpy.
        pos, orient = _orig_glp(self)
        return _to_numpy(pos), _to_numpy(orient)

    def _patched_set_world_pose(self, position=None, orientation=None):
        # Write physical prims through Fabric and nonphysical markers through USD.
        if position is not None:
            position = self._backend_utils.convert(position, device=self._device)
            position = self._backend_utils.expand_dims(position, 0)
        if orientation is not None:
            orientation = self._backend_utils.convert(orientation, device=self._device)
            orientation = self._backend_utils.expand_dims(orientation, 0)
        usd = not _is_physics_tracked(self._prim_view)
        self._prim_view.set_world_poses(
            positions=position, orientations=orientation, usd=usd
        )

    ArticulationSubset.get_joint_positions = _patched_get_joint_positions
    ArticulationSubset.get_joint_velocities = _patched_get_joint_velocities
    if SinglePrimWrapper is not None:
        SinglePrimWrapper.get_world_pose = _patched_get_world_pose
        SinglePrimWrapper.get_local_pose = _patched_get_local_pose
        SinglePrimWrapper.set_world_pose = _patched_set_world_pose


def _apply_carb_settings(
    app: AppHandle,
    *,
    async_rendering: bool,
    rtx_settings: Optional[Mapping[str, Any]] = None,
) -> None:
    """Apply fixed physics/readback and async-rendering settings, then overrides.
    Disable ujitsoCollisionCooking, set both async-rendering keys, and enable USD
    readback for poses, velocities, and particles so USD readers see current state."""
    settings = app._carb_settings  # Private settings handle exposed by SimulationApp.
    settings.set("/physics/cooking/ujitsoCollisionCooking", False)
    settings.set("/omni/replicator/asyncRendering", async_rendering)
    settings.set("/app/asyncRendering", async_rendering)
    # Enable readback for USD readers such as cuRobo XformCache on the GPU pipeline.
    settings.set("/physics/suppressReadback", False)
    # Explicitly enable pose, velocity, and particle writeback channels.
    # Setting names vary between Isaac Sim versions.
    settings.set("/physics/updateToUsd", True)
    settings.set("/physics/updateVelocitiesToUsd", True)
    settings.set("/physics/updateParticlesToUsd", True)
    if rtx_settings:
        for path, value in rtx_settings.items():
            settings.set(path, value)


def get_app() -> AppHandle:
    if _app is None:
        raise RuntimeError("SimulationApp is not running. Call sim_backend.runtime.init_app(...) first.")
    return _app


def is_app_started() -> bool:
    return _app is not None


def shutdown_app() -> None:
    """Close SimulationApp for tests or cleanup; do nothing if it is already closed."""
    global _app
    if _app is None:
        return
    _app.close()
    _app = None
