# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Scene-plugin loader for newton-standalone engine.

A scene plugin is a Python file referenced from the scene yaml under
``newton.scene_plugin`` (path resolved against the yaml's directory).  It
provides imperative hooks that the engine calls at well-defined moments
in the build / step / render pipeline:

  * ``on_stage(stage, ctx)``         — before ``ModelBuilder.add_usd``. Use
                                       an anonymous USD session-layer edit for
                                       per-scene schema filtering; never save
                                       the source asset.
  * ``on_build(builder, ctx)``       — after ``add_usd`` + ``add_ground_plane``
                                       + cloth injection, before ``finalize``.
                                       Use to ``add_body`` / ``add_soft_grid`` /
                                       ``add_shape_*``.
  * ``on_model_ready(model, ctx)``   — after ``builder.finalize``, before the
                                       solver is instantiated.  Use to set
                                       ``model.particle_*`` / ``soft_contact_*``.
  * ``on_post_step(state, sim_time, dt, ctx)``  — each frame, after the
                                       physics substep loop.  Used for
                                       autonomous controllers that drive
                                       kinematic bodies (e.g. the chow-mein
                                       wok controller).
  * ``record_snapshot_enabled()``   — optional dynamic consumer query. False
                                       disables particle host copies and snapshot
                                       callbacks. Without this query, a callable
                                       ``on_record_snapshot`` opts in.
  * ``on_record_snapshot(particle_q, body_q, body_paths, sim_time, ctx)`` —
                                       after the async host mirrors refresh.
                                       Particle and rigid arrays are the same
                                       one-tick-delayed physics snapshot.
  * ``on_render(viewer, state, sim_time, ctx)`` — each render frame inside
                                       the engine's render hook.  Use to
                                       call ``viewer.log_mesh`` for custom
                                       overlays (e.g. smooth tubes built
                                       on the GPU from particle positions).
  * ``get_deformable_render_bindings()`` — optional post-build query returning
                                       authored Mesh paths and fixed particle
                                       reconstruction maps for inline OVRtx.

The plugin module may either expose a ``ScenePlugin`` class (the engine
constructs ``ScenePlugin()`` once) or expose top-level functions named
``on_stage`` / ``on_build`` / ``on_model_ready`` / ``on_post_step`` /
``record_snapshot_enabled`` / ``on_record_snapshot`` / ``on_render`` /
``get_deformable_render_bindings``.

``ctx`` is a ``SimpleNamespace`` with these attributes:

  * ``logger``        — the engine's ``SimpleLogger``
  * ``scene_cfg``     — full parsed scene yaml as a dict
  * ``scene_yaml``    — absolute path to the scene yaml (str)
  * ``device``        — ``"cuda:0"`` (the device the model is built on)
  * ``robot_prefix``  — the robot's USD prefix (e.g. ``"genie"``)
  * ``physics_hz``    — float
  * ``sim_substeps``  — int
"""

from __future__ import annotations

import importlib.util
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

_PLUGIN_HOOKS = (
    "on_stage",
    "on_build",
    "on_model_ready",
    "on_post_step",
    "record_snapshot_enabled",
    "on_record_snapshot",
    "on_render",
    "get_deformable_render_bindings",
)


def _resolve_plugin_path(scene_cfg: dict, scene_yaml: str) -> Optional[Path]:
    """Pull ``newton.scene_plugin`` out of the scene yaml and resolve it.

    Resolution order: absolute path → relative to scene yaml dir → fail.
    Returns ``None`` if the field is unset.
    """
    newton_cfg = scene_cfg.get("newton") or {}
    raw = newton_cfg.get("scene_plugin")
    if not raw:
        return None
    if not isinstance(raw, str):
        raise RuntimeError(f"newton.scene_plugin must be a string path, got {type(raw).__name__}")
    p = Path(raw)
    if not p.is_absolute() and scene_yaml:
        p = Path(scene_yaml).parent / p
    p = p.resolve()
    if not p.is_file():
        raise FileNotFoundError(f"newton.scene_plugin resolved to {p} which does not exist")
    return p


def _load_plugin(plugin_path: Path, logger: Any) -> Any:
    """Import a scene plugin from a file path.

    Returns either a ``ScenePlugin`` instance (if the module exposes that
    class) or the module itself (when hooks are top-level functions).
    """
    mod_name = f"_genie_scene_plugin_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(mod_name, str(plugin_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build import spec for {plugin_path}")
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec_module — Python 3.12+ dataclasses
    # do ``sys.modules[cls.__module__].__dict__`` during ``@dataclass``
    # processing, and that fails with AttributeError on NoneType if the
    # module isn't registered yet.
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    plugin_cls = getattr(module, "ScenePlugin", None)
    if plugin_cls is not None:
        plugin = plugin_cls()
    else:
        plugin = module
    found = [h for h in _PLUGIN_HOOKS if callable(getattr(plugin, h, None))]
    logger.info(f"[newton-standalone] loaded scene plugin {plugin_path.name}: " f"hooks={found or 'none'}")
    return plugin


class _PluginMixin:
    def _load_scene_plugin(self) -> None:
        """Load ``newton.scene_plugin`` (if present) onto ``self._scene_plugin``.

        Idempotent — repeat calls are no-ops.  Failures raise; a missing
        plugin is *not* an error (the field is optional).
        """
        if getattr(self, "_scene_plugin", None) is not None:
            return
        scene_yaml_path = ""
        # The scene_yaml path is not stored on the engine directly; the
        # session resolves it from the manifest.  Fall back to the same
        # heuristic as ``newton_solvers_path`` (sibling of the manifest).
        # Precedence: explicit attribute → manifest sibling.
        scene_yaml_path = getattr(self, "_scene_yaml_path", "") or ""
        if not scene_yaml_path and self._newton_solvers_path:
            # newton_solvers.json sits next to the scene yaml's manifest;
            # the engine doesn't get a direct path, but when the plugin
            # path in the yaml is relative we need *some* anchor.  Use
            # the cwd as a last-resort fallback (CI / dev runs from the
            # repo root).
            scene_yaml_path = os.getcwd()
        try:
            plugin_path = _resolve_plugin_path(self._scene_cfg, scene_yaml_path)
        except Exception as exc:
            self._logger.warn(f"[newton-standalone] scene plugin resolve failed: {exc}")
            self._scene_plugin = None
            return
        if plugin_path is None:
            self._scene_plugin = None
            return
        self._scene_plugin = _load_plugin(plugin_path, self._logger)
        self._scene_plugin_path = str(plugin_path)

    def _record_snapshot_enabled(self) -> bool:
        """Only an active snapshot consumer needs particle host mirrors."""
        plugin = getattr(self, "_scene_plugin", None)
        if not callable(getattr(plugin, "on_record_snapshot", None)):
            return False
        if callable(getattr(plugin, "record_snapshot_enabled", None)):
            return bool(self._call_plugin("record_snapshot_enabled"))
        # Existing plugins with only the snapshot callback remain opted in.
        return True

    def _plugin_ctx(self) -> SimpleNamespace:
        """Build the ``ctx`` passed to every plugin hook."""
        return SimpleNamespace(
            logger=self._logger,
            scene_cfg=self._scene_cfg,
            scene_yaml=getattr(self, "_scene_yaml_path", "") or "",
            device="cuda:0",
            robot_prefix=self._robot_prefix_str,
            physics_hz=float(self._physics_hz),
            sim_substeps=int(self._sim_substeps),
        )

    def _call_plugin(self, hook: str, *args) -> Any:
        """Dispatch a single hook on ``self._scene_plugin``.

        Missing hooks (or no plugin) are silent no-ops; exceptions
        from a hook are logged and re-raised so a buggy plugin can't
        silently desync the engine.
        """
        plugin = getattr(self, "_scene_plugin", None)
        if plugin is None:
            return None
        fn = getattr(plugin, hook, None)
        if not callable(fn):
            return None
        try:
            return fn(*args)
        except Exception:
            self._logger.warn(f"[newton-standalone] scene plugin hook {hook} raised; re-raising")
            raise
