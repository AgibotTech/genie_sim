# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Build entry point + warmup + graph capture + runtime USD dump.

Holds ``_build`` (the phase table-of-contents), ``_warmup``,
``_capture_graph``, ``_dump_runtime_usd``, plus ``_warmup_renders`` /
``_configure_viewport`` no-ops kept for the ``PhysicsEngine``
interface.
"""

from __future__ import annotations

import json
import math
import os
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import warp as wp


@contextmanager
def _dof_target_layout():
    """Keep the standalone target ABI when Newton 1.6.1 defaults to coordinates."""
    import newton

    previous = newton.use_coord_layout_targets
    newton.use_coord_layout_targets = False
    try:
        yield
    finally:
        newton.use_coord_layout_targets = previous


class _RuntimeMixin:
    def _warmup_renders(self) -> None:
        """No-op.  Newton-standalone is Kit-free; nothing to render-tick."""
        return

    def _configure_viewport(self) -> None:
        """No-op.  Newton-standalone is Kit-free; no editor viewport to configure."""
        return

    def _build(self) -> None:
        """Engine build pipeline.

        Split into phase methods so this entry point reads like a table
        of contents and each phase's locals + side-effects are visible
        at the seam.  Each ``_phase_*`` method runs exactly once per
        engine and never recurses; the order below is the only valid
        order (later phases consume ``self`` state set by earlier ones,
        plus the four cross-phase locals returned by phase 2).
        """
        self._phase_open_stage_and_overrides()
        cloth_entries, solver_pref, plugin_has_build, needs_particle_solver = self._phase_resolve_scene()
        # ModelBuilder and finalize snapshot this choice. Keep the original
        # DOF indexing used by Newton 1.4, without changing other models' defaults.
        with _dof_target_layout():
            self._phase_build_model(cloth_entries, solver_pref, plugin_has_build, needs_particle_solver)
        self._phase_normalize_model()
        self._phase_states_and_robot_solver()
        self._phase_cloth_solver_and_readback(cloth_entries, solver_pref, plugin_has_build)
        self._phase_debug_publishers()
        self._phase_finalize_init_state()
        self._phase_publish_static_facts()

    def _warmup(self) -> None:
        """Run one uncaptured frame so JIT-compiled kernels are warm.

        CUDA graph capture (in ``_capture_graph``) requires the kernels to
        already be loaded — capture fails if it triggers a JIT compile.
        """
        if self._model is None:
            return
        self._logger.info("[newton-standalone] warming up kernels…")
        try:
            if self._cloth_solver is not None and hasattr(self._cloth_solver, "rebuild_bvh"):
                self._cloth_solver.rebuild_bvh(self._state_0)
            self._simulate_substeps(dt=1.0 / self._physics_hz, captured=False)
            self._warmup_renders()
            self._logger.info("[newton-standalone] kernel warmup complete.")
        except Exception as exc:
            self._logger.warn(f"[newton-standalone] warmup failed: {exc}")

    def _dump_runtime_usd(self) -> None:
        """Snapshot the live composed USD stage to ``self._runtime_usd_dump_path``.

        The dump is the stage exactly as ``add_usd`` saw it — robot.usda
        + the newton_scene.usda sublayer (cloth prim, static box, …) +
        any runtime authoring we did before this point. Open the file in
        usdview / Isaac Sim to verify:

          * which physics colliders Newton would have parsed
          * what xforms got composed onto cloth / box prims
          * whether the robot prefix path matches the body labels Newton
            built joint / body indices against

        Newton-standalone is Kit-free, so there is no live Fabric /
        USDRT layer that this snapshot would miss — what the asset
        pipeline gave Newton is the same thing Newton runs on.

        We export the ROOT LAYER ONLY (not the flattened stage) so the
        dump stays a small shell that references the cached payload
        files.  ``Stage.Export()`` flattens the composed stage and pulls
        every Mesh/inertia/collider inline — for the G2 + cloth scene
        that's a ~47 MB file with 99 % of its contents already
        identically present in ``payloads/geometries.usd``.  Exporting
        the root layer alone produces a few-KB file that opens
        identically in usdview/Isaac Sim because USD chases the
        references at load time.

        Empty path → no-op (e.g. headless CI). Non-fatal on failure —
        logs and continues so an export error doesn't break the run.
        """
        path = (self._runtime_usd_dump_path or "").strip()
        if not path:
            return
        if self._stage is None:
            self._logger.warn(
                f"[newton-standalone] runtime USD dump requested at {path!r} " f"but self._stage is None — skipping."
            )
            return
        try:
            from pathlib import Path as _Path

            _Path(path).parent.mkdir(parents=True, exist_ok=True)
            # Root layer only — keeps references to the cached payloads
            # intact instead of flattening multi-MB mesh data inline.
            self._stage.GetRootLayer().Export(path)
            # Make asset references relative to the dump's parent dir.
            # The export step preserves the source layer's references
            # verbatim, and the source stage was opened with absolute
            # paths (Newton's ``add_usd`` resolves to absolute), so the
            # exported file carries lines like ``subLayers = [@/abs/path/
            # newton_scene.usda@]`` and ``prepend references = @/abs/
            # path/robot.usda@``.  Rewriting them to ``@./newton_scene
            # .usda@`` / ``@./robot.usda@`` makes the dump portable
            # (move ``/geniesim_assets/`` somewhere else and the dump
            # still loads), and matches the convention the
            # ``robot.usda`` shell already uses for its
            # ``@./payloads/*.usda@`` references.  Best-effort — if
            # the rewrite errors, the dump file already exists with
            # absolute paths; we log and move on.
            #
            # ``source_dir`` is the dir of the ORIGINAL scene USD layer:
            # ``Stage.GetRootLayer().Export()`` copies the source's
            # content (including bare references like
            # ``@supermarket_shelf_002/Aligned.usda@``) verbatim to the
            # new dump location, where the bare paths no longer
            # resolve.  Passing the source dir lets the relativizer
            # rewrite those bare paths to their original target
            # (anchored as ``@../GDK/supermarket_shelf_002/Aligned.usda@``)
            # rather than falling back to the ambiguous basename scan
            # — which fails when many assets share a leaf filename
            # like ``Aligned.usda``.
            try:
                from genie_sim_engine_newton.common.usd_path_helpers import (  # noqa: PLC0415
                    make_layer_asset_paths_relative,
                )

                source_dir = (
                    os.path.dirname(os.path.abspath(self._scene_usda)) if getattr(self, "_scene_usda", "") else None
                )
                make_layer_asset_paths_relative(
                    path,
                    source_dir=source_dir,
                    logger=self._logger,
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.warn(
                    f"[newton-standalone] runtime USD dump path-relativizer "
                    f"failed ({exc!r}); dump kept absolute paths."
                )
            try:
                size = _Path(path).stat().st_size
                self._logger.info(f"[newton-standalone] runtime USD dumped: {path}  ({size} bytes)")
            except Exception:
                self._logger.info(f"[newton-standalone] runtime USD dumped: {path}")
        except Exception as exc:
            self._logger.warn(f"[newton-standalone] runtime USD dump failed ({path}): {exc}")

    def _apply_mjc_contact_overrides(self) -> None:
        """Read-only post-finalize diagnostic.

        Per-class MJW contact compliance + friction injection happens
        BEFORE ``builder.finalize()`` so the writes reach both the MJCF
        compile path AND the runtime contact kernel (see ``engine/newton/
        mjc_contact.py``'s module docstring for the asymmetry that drives
        the placement).

        This method is a thin redirect to the module's read-only
        ``readback_post_finalize`` for any external call site.
        """
        from genie_sim_engine_newton.engine.newton.mjc_contact import readback_post_finalize  # noqa: PLC0415

        if self._model is None:
            return
        readback_post_finalize(
            self._model,
            scene_cfg=self._scene_cfg,
            robot_prefix=str(getattr(self, "_robot_prefix_str", "") or ""),
            adapter_name=getattr(self._adapter, "name", ""),
            logger=self._logger,
        )

    def _capture_graph(self) -> None:
        """Capture all 10 substeps into a CUDA graph for fast replay.

        After this, ``step()`` becomes one ``wp.capture_launch`` call instead
        of a 10-iteration Python loop. The captured ``sub_dt`` is fixed
        (a graph constant), so the run loop must always use the same dt.
        """
        if self._model is None:
            return
        self._cuda_graph = None
        try:
            self._sim_dt_captured = (1.0 / self._physics_hz) / self._sim_substeps
            if self._cloth_solver is not None and hasattr(self._cloth_solver, "rebuild_bvh"):
                self._cloth_solver.rebuild_bvh(self._state_0)
            with wp.ScopedCapture() as cap:
                self._simulate_substeps(captured=True)
            self._cuda_graph = cap.graph
            self._logger.info(f"[newton-standalone] CUDA graph captured (sub_dt={self._sim_dt_captured*1000:.3f}ms)")
        except Exception as exc:
            self._logger.warn(f"[newton-standalone] CUDA graph capture failed: {exc}; " f"falling back to Python loop")
            self._cuda_graph = None
