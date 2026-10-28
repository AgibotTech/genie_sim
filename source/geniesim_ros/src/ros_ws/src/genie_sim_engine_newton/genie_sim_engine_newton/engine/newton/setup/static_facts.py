# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Build-phase 9: publish the static engine facts, once, latched.

A "static fact" is something a consumer in another process needs, that is
decided by the build and never changes afterwards.  Today that is the
per-body local AABB table on ``/rlinf/body_aabb_local``; the joint->prim
map is the obvious next one if the wrapper ever asks for it out of process.

Why this is its own phase rather than a tail on phase 8
-------------------------------------------------------
It used to live at the end of ``_phase_finalize_init_state``, for a reason
that was about data availability rather than about init pose: the table is
computed by the mjwarp adapter at ``build_solver`` time and copied onto the
engine by ``topology.py::_build_body_map``, which runs at the *top* of phase
8.  Phase 7 (``debug_pubs``) is the only other phase that publishes anything,
and there the table is still ``None`` — so the publish would have been a
silent no-op.

Phase 8 was never the semantically right home.  The AABB is a **local**,
body-frame quantity: it does not depend on the init pose, on ``eval_fk``, or
on the reset snapshot.  Its only real dependency is ``_build_body_map``.
Sitting at the end of phase 8 made a module named ``init_pose`` end by doing
ROS I/O, and made the ordering constraint look like an init-pose constraint.

``debug_pubs`` is not the right home either, in the other direction: every
publisher there is gated behind a ``newton.debug.pub_*`` flag and defaults
off, whereas this table is load-bearing for rlinf's eval checkers.  Filing it
under debug would invite somebody to gate it.

So: its own phase, after phase 8, holding the publishes that are neither
debug nor per-tick.
"""

from __future__ import annotations

from typing import Any


class _StaticFactsMixin:
    def _phase_publish_static_facts(self) -> None:
        """Phase 9: one-shot latched publishes of build-decided facts."""
        self._publish_body_aabb_local()
        from genie_sim_engine_newton.engine.newton.static_props import compute_static_prop_aabbs
        from genie_sim_engine_newton.engine.newton.benchmark_publisher import publish_static_prop_aabb

        requested = ((self._scene_cfg or {}).get("newton") or {}).get("eval_static_props") or []
        names, boxes = compute_static_prop_aabbs(self._stage, requested, logger=self._logger)
        publish_static_prop_aabb(names, boxes, logger=self._logger)

    def _publish_body_aabb_local(self) -> None:
        """Publish ``/rlinf/body_aabb_local`` once, latched.

        Names are the PER-ENV bare body paths — one env's worth, in the same
        order as the AABB rows (both derive from ``self.body_paths`` / the
        Newton per-env body index).

        NOT a debug knob — rlinf's eval checkers (Inside / Ontop / Cover /
        GripperPassing / RelativePosition) compose this table with their own
        memoized world poses in ``get_body_aabbs()``.  Adapters that expose no
        table (everything except the mjwarp multi-env one today) make it a
        no-op.

        The publish lives on the backend rather than in
        ``genie_sim_engine_core``: core is the base package every backend
        build-depends on, so it must not import from a backend.
        """
        aabb_local: Any = getattr(self, "body_aabb_local", None)
        if aabb_local is None:
            return
        try:
            from genie_sim_engine_newton.engine.newton.benchmark_publisher import (  # noqa: PLC0415
                publish_body_aabb_local,
            )

            paths = list(self.body_paths or [])
            ne = int(getattr(self, "_num_envs", 1) or 1)
            if ne > 1 and paths and len(paths) % ne == 0:
                paths = paths[: len(paths) // ne]
            publish_body_aabb_local(
                [str(p).lstrip("/") for p in paths],
                aabb_local,
                logger=self._logger,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[newton-standalone] body_aabb_local publish failed: {exc!r}")
