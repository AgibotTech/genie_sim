# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Build-phase 7: optional rclpy debug publishers.

Each publisher (deformable / object) is gated by its own
``newton.debug.pub_*`` flag.  When all are off, rclpy is never
imported and the engine has zero per-tick debug cost.
"""

from __future__ import annotations

import json
import math
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import warp as wp


class _DebugPubsMixin:
    def _phase_debug_publishers(self) -> None:
        """Phase 7: optional rclpy debug publishers (deformable + objects).

        Each publisher is gated by its own ``newton.debug.pub_*`` flag
        in the scene yaml; when all are off, ``rclpy`` is never imported
        and per-tick cost is zero.
        """
        # ----------------------------------------------------------------
        # Debug-only: sibling rclpy publishers for engine state.  Hosted
        # by ``engine/newton/debug_visualizer.py`` (one shared rclpy node
        # for all debug pubs; QoS = sensor_data so RViz auto-connects).
        # Each publisher is gated by its own knob in ``newton.debug:``;
        # all default off — when none are set, debug_visualizer is never
        # imported and rclpy stays out of the engine process.
        # ----------------------------------------------------------------
        _debug_cfg = (self._scene_cfg.get("newton") or {}).get("debug") or {}
        # Deformable surface (cloth / FEM tet) as TRIANGLE_LIST Marker.
        # Requires model.tri_count > 0 — VBD/XPBD cloth produces tris
        # from add_cloth_grid / add_cloth_mesh, FEM tet bars from
        # add_soft_grid also produce surface tris, so this covers both
        # the shirt and chef demos.
        if bool(_debug_cfg.get("pub_deformable_marker", False)) and self._model.tri_count > 0:
            try:
                from genie_sim_engine_newton.engine.newton.visualizers.markers import (
                    DeformableMarkerPublisher,
                )  # noqa: PLC0415

                _topic = str(_debug_cfg.get("deformable_marker_topic", "deformable_marker"))
                _frame = str(_debug_cfg.get("deformable_marker_frame_id", "map"))
                self._deformable_pub = DeformableMarkerPublisher(model=self._model, topic=_topic, frame_id=_frame)
                self._logger.info(
                    f"[newton-standalone] debug.pub_deformable_marker: ON "
                    f"(topic={_topic!r}, frame_id={_frame!r}, "
                    f"tri_count={self._model.tri_count})"
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.warn(
                    f"[newton-standalone] debug.pub_deformable_marker requested but "
                    f"DeformableMarkerPublisher init failed: {exc!r}; continuing without it"
                )
                self._deformable_pub = None

        # Raw particle_q as sensor_msgs/PointCloud2 — independent of the
        # TRIANGLE_LIST marker above so an operator can pick either, both,
        # or neither.  Gate is ``model.particle_count > 0`` (not
        # ``tri_count``) so particle-only soft bodies that emit no surface
        # tris still publish.
        if (
            bool(_debug_cfg.get("pub_deformable_pointcloud", False))
            and int(getattr(self._model, "particle_count", 0) or 0) > 0
        ):
            try:
                from genie_sim_engine_newton.engine.newton.visualizers.markers import (
                    DeformablePointCloudPublisher,
                )  # noqa: PLC0415

                _topic = str(_debug_cfg.get("deformable_pointcloud_topic", "deformable_pointcloud"))
                _frame = str(_debug_cfg.get("deformable_pointcloud_frame_id", "map"))
                self._deformable_pc_pub = DeformablePointCloudPublisher(
                    model=self._model, topic=_topic, frame_id=_frame
                )
                self._logger.info(
                    f"[newton-standalone] debug.pub_deformable_pointcloud: ON "
                    f"(topic={_topic!r}, frame_id={_frame!r}, "
                    f"particle_count={int(getattr(self._model, 'particle_count', 0) or 0)})"
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.warn(
                    f"[newton-standalone] debug.pub_deformable_pointcloud requested but "
                    f"DeformablePointCloudPublisher init failed: {exc!r}; continuing without it"
                )
                self._deformable_pc_pub = None

        # MarkerArray pub for free-joint rigid objects (hangers, dropped
        # objects, free wok).  Uses primitive markers only (CUBE / SPHERE
        # / CYLINDER) — meshes would force a per-frame asset reload in
        # RViz.
        if bool(_debug_cfg.get("pub_object_marker", False)):
            try:
                from genie_sim_engine_newton.engine.newton.visualizers.markers import (
                    ObjectMarkerPublisher,
                )  # noqa: PLC0415

                _topic = str(_debug_cfg.get("object_marker_topic", "object_marker"))
                _frame = str(_debug_cfg.get("object_marker_frame_id", "map"))
                _obj_dir = str(_debug_cfg.get("object_marker_obj_dir", ""))
                # robot_prefix filters the robot's own FREE-jointed base
                # out of the marker array (only relevant when
                # ``pin_base_to_world: false`` — e.g. the WBC scene).
                # Per-body collider meshes are baked to OBJ files on
                # disk at startup; per-tick wire traffic is just the
                # body pose.  ``obj_out_dir`` defaults to
                # /tmp/genie_sim_engine/markers/<pid> when empty.
                self._object_pub = ObjectMarkerPublisher(
                    model=self._model,
                    topic=_topic,
                    frame_id=_frame,
                    robot_prefix=str(getattr(self, "_robot_prefix_str", "") or ""),
                    obj_out_dir=_obj_dir,
                    logger=self._logger,
                )
                # Detailed body listing — for debugging "why is my marker
                # array empty / why isn't body N showing the right shape".
                # Each template's kind + per-body tri count (or primitive
                # type) tells you what the publisher latched onto.
                _body_lines = []
                for _tpl in self._object_pub._templates:
                    if _tpl.get("kind") == "mesh_resource":
                        _ntri = int(_tpl.get("n_tri", 0))
                        _res = str(_tpl.get("mesh_resource", ""))
                        _body_lines.append(
                            f"    body[{_tpl['body']}] label={_tpl['label']!r} " f"mesh_resource={_res} n_tri={_ntri}"
                        )
                    else:
                        _body_lines.append(
                            f"    body[{_tpl['body']}] label={_tpl['label']!r} " f"primitive type={_tpl.get('type')}"
                        )
                self._logger.info(
                    f"[newton-standalone] debug.pub_object_marker: ON "
                    f"(topic={_topic!r}, frame_id={_frame!r}, "
                    f"obj_out_dir={self._object_pub._obj_out_dir!r}, "
                    f"free_bodies={len(self._object_pub._free_bodies)})"
                    + ("\n" + "\n".join(_body_lines) if _body_lines else "")
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.warn(
                    f"[newton-standalone] debug.pub_object_marker requested but "
                    f"ObjectMarkerPublisher init failed: {exc!r}; continuing without it"
                )
                self._object_pub = None
