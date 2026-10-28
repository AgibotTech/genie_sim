# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Newton-standalone visualizers (RViz markers + OVRtx viewport).

Two independent paths share this subpackage:

  * :mod:`markers` — sibling rclpy publishers (one shared node) that
    surface engine state to RViz: ``DeformableMarkerPublisher`` for
    cloth / FEM tet surfaces, ``DeformablePointCloudPublisher`` for raw
    ``particle_q`` as a PointCloud2, ``ObjectMarkerPublisher`` for
    FREE-joint rigid bodies.  Gated by ``newton.debug.pub_*`` scene-yaml
    flags.

  * :mod:`ovrtx` — in-process OVRtx (path-traced) viewport with its
    own GPU stream, camera management, and per-frame compositor.
    Companion :mod:`ovrtx_camera` holds ``CameraCfg`` and manifest
    loading; :mod:`ovrtx_kernels` holds the Warp kernels the
    compositor invokes per frame.

Both paths import ``rclpy`` / OVRtx lazily so processes that don't
enable either knob never pay the cost.

The re-exports below are resolved lazily too, via :pep:`562`.  They used
to be plain module-level imports, which broke that promise in a way that
only showed up under pytest: ``ovrtx`` imports the compiled
``genie_sim_engine_py`` at module scope, and ROS's ``launch_testing``
collector ``pyimport()``s every module it walks
(``launch_testing/pytest/hooks.py:193``).  So ANY pytest run whose
rootdir contained this package aborted collection outright with
``ModuleNotFoundError: No module named 'genie_sim_engine_py'`` — not a
failed test, a dead run — on every host without the compiled core built.

Deferring costs nothing: every in-tree consumer already imports from the
submodule inside a function (``setup/debug_pubs.py``, ``stats.py``,
``scripts/genie_sim_engine_newton_node.py``), so these names exist for
the package's public contract, not for any caller on the hot path.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # import-time only for type checkers, never at runtime
    from genie_sim_engine_newton.engine.newton.visualizers.markers import (
        DeformableMarkerPublisher,
        DeformablePointCloudPublisher,
        ObjectMarkerPublisher,
        shutdown as shutdown_marker_node,
    )
    from genie_sim_engine_newton.engine.newton.visualizers.ovrtx import InlineOvrtxVisualizer, get_render_stats

__all__ = [
    "DeformableMarkerPublisher",
    "DeformablePointCloudPublisher",
    "ObjectMarkerPublisher",
    "shutdown_marker_node",
    "InlineOvrtxVisualizer",
    "get_render_stats",
]

# name -> (submodule, attribute in that submodule)
_LAZY: dict = {
    "DeformableMarkerPublisher": ("markers", "DeformableMarkerPublisher"),
    "DeformablePointCloudPublisher": ("markers", "DeformablePointCloudPublisher"),
    "ObjectMarkerPublisher": ("markers", "ObjectMarkerPublisher"),
    "shutdown_marker_node": ("markers", "shutdown"),
    "InlineOvrtxVisualizer": ("ovrtx", "InlineOvrtxVisualizer"),
    "get_render_stats": ("ovrtx", "get_render_stats"),
}


def __getattr__(name: str) -> Any:
    """Resolve a re-export on first access (:pep:`562`).

    Raises ``AttributeError`` for unknown names so ``hasattr`` and
    ``from ... import *`` behave normally.  The import error from a
    genuinely missing dependency still propagates on real use — it is
    only module *collection* that no longer triggers it.
    """
    try:
        where, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    from importlib import import_module

    value = getattr(import_module(f"{__name__}.{where}"), attr)
    globals()[name] = value  # cache; __getattr__ is not consulted again
    return value


def __dir__() -> list:
    return sorted(__all__)
