# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""InlineOvrtxVisualizer — newton-standalone in-process OVRtx render thread.

Selected at launch via ``physics_engine_visualizer:=ovrtx`` (alongside
``newton`` GL).  Runs OVRtx 0.4.x inside the physics
process on a dedicated Python thread.  Body transforms flow from Newton's
``state.body_q`` (GPU) to OVRtx's internal Fabric buffer via a single
zero-copy Warp kernel launch.  Camera images are extracted on the OVRtx
thread and published via :class:`gsi::RosBridge` (extended with
``publish_camera_image_*`` / ``publish_camera_info`` / ``has_image_subscribers``).

Design doc: :doc:`ovrtx_sync.md`.

Threading contract:
    * Physics thread records :class:`wp.Event` ``physics_step_event`` after
      each step (one ``wp.record_event`` call in
      :meth:`NewtonHeadlessEngine.tick_extras`).  No CPU-side wait.
    * OVRtx thread (this class) waits on ``physics_step_event`` via its
      own Warp stream (``ovrtx_stream``) before each frame's sync kernel.
      Physics never CPU-blocks; physics and OVRtx GPU work overlap.
    * ROS publishes from the OVRtx thread go through ``_core.publish_*``
      with ``py::gil_scoped_release`` — same pattern as the physics-thread
      publishes in :mod:`common.loop`.

OVRtx zero-copy hot path (verbatim from the OVRtx
``mapping-attributes`` skill and the reference test at
``tests/docs/python/test_attribute_bindings.py`` in the OVRtx repo):

    with binding.map(device=Device.CUDA) as mapping:
        tensor = wp.from_dlpack(mapping.tensor, dtype=wp.mat44d)
        wp.launch(kernel, dim=N, inputs=[tensor, ...], stream=stream)
        mapping.unmap(stream=stream.cuda_stream)

We do not rely on ``__exit__`` to call ``unmap(stream=...)`` — its default
``unmap()`` has no stream argument, so the kernel could still be queued on
``ovrtx_stream`` when the C unmap fires.  We call ``mapping.unmap(stream=...)``
explicitly inside the with block so OVRtx receives the stream handle and
serialises against the kernel via CUDA stream order.

OVRtx attribute conventions (see ``writing-transforms`` skill):
    * Attribute name  : "omni:xform"
    * Element type    : float64
    * Element shape   : (4, 4)
    * Memory layout   : USD row-vector (translation in last row)
    * Semantic        : OVRTX_SEMANTIC_XFORM_MAT4x4

Newton's ``wp.transform_to_matrix`` produces column-vector form, hence the
transpose in :func:`ovrtx_kernels.sync_body_q_to_ovrtx_mat44d`.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import warp as wp

import genie_sim_engine_py as _core
from genie_sim_engine_newton.engine.newton.visualizers.ovrtx_kernels import (
    reconstruct_tet_bound_points,
    sync_body_q_to_ovrtx_mat44d,
    sync_particle_q_slice_to_points,
)
from genie_sim_engine_newton.engine.newton.visualizers.deformable_bindings import (
    prepare_deformable_bindings,
)
from genie_sim_engine_newton.engine.newton.visualizers.camera_zoom import (
    clamp_zoom_factor,
    scaled_intrinsics,
)


def _log(msg: str, *args: Any) -> None:
    """Print to stderr so ROS launch and ros2 topic logs both surface it.

    Accepts ``%``-style formatting (``msg % args``) to match
    ``logger.info(fmt, *args)`` call sites.  Falls back to ``msg``
    unchanged if formatting fails (mismatched specifier count) so a
    logging bug never crashes the OVRtx thread.

    The rest of the engine uses ``common.session.SimpleLogger`` (plain
    prints to stderr).  This mirrors that style instead of stdlib
    logging so operators don't need to set logging levels to see the
    visualizer's setup / per-frame stats / error reasons.
    """
    if args:
        try:
            msg = msg % args
        except (TypeError, ValueError):
            pass
    print(f"[ovrtx-viz] {msg}", file=sys.stderr, flush=True)


def _logexc(msg: str, *args: Any) -> None:
    """Like :func:`_log` but appends the current exception traceback."""
    _log(msg, *args)
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Cross-thread render-rate publication (OVRtx thread → physics-thread stats)
# ---------------------------------------------------------------------------

# Latest 1-second render rate, frame time (ms), and stale-after wall time.
# Updated by :class:`InlineOvrtxVisualizer` once per second.  Read by the
# physics-loop stats logger (engine/newton/stats.py) so the 1 Hz log can
# carry a single ``ovrtx=NN.NHz`` line alongside the in-loop viewport
# counter.  All three values are plain floats — Python assignments to
# attributes on a module are atomic w.r.t. the GIL, which is enough for
# this read pattern (no torn read of any single field; we never need
# rate and frame_ms to be mutually consistent across a single read).
_render_hz: float = 0.0
_render_avg_ms: float = 0.0
_render_failed: int = 0
# ``_render_stale_after`` is wall-clock seconds (time.monotonic).  When
# the reader sees ``time.monotonic() > _render_stale_after`` it knows
# the OVRtx thread hasn't published a stat in over ~3 s, so the values
# above are stale and should not be displayed.  This avoids ghosting an
# old rate after the OVRtx thread has died or been stopped.
_render_stale_after: float = 0.0


def get_render_stats() -> Optional[tuple]:
    """Return ``(hz, avg_ms, failed)`` or ``None`` if no fresh data.

    Called once per 1 Hz log tick from the physics-thread stats logger
    (engine/newton/stats.py).  Returns ``None`` when the OVRtx thread is
    not running or has fallen behind — caller omits the line entirely
    rather than printing a stale value.
    """
    if _render_stale_after == 0.0:
        return None
    if time.monotonic() > _render_stale_after:
        return None
    return _render_hz, _render_avg_ms, _render_failed


# ---------------------------------------------------------------------------
# Camera config (mirrors the schema in genie_sim_render/render_node.cpp:225-253
# but parsed in-engine; no shared compiled code with genie_sim_render)
# ---------------------------------------------------------------------------


@dataclass
class CameraCfg:
    """One camera entry from manifest.json.

    Field semantics match render_node.cpp's CameraConfig (read-only borrow;
    we don't import its C++ struct).
    """

    path: str  # USD prim path inside the robot, e.g. "Head_Camera/Camera"
    topic: str  # ROS topic prefix; we publish <topic>/image_raw + /camera_info
    depth_topic: str = ""
    render_product_path: str = ""
    width: int = 1280
    height: int = 800
    fx: float = 610.0
    fy: float = 610.0
    cx: float = 640.0
    cy: float = 400.0
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    is_free_cam: bool = False

    # ----- cached intrinsics (built once in __post_init__) ---------------------
    # Pre-built ``sensor_msgs/CameraInfo`` arrays, materialised once at scene
    # load so the OVRtx render thread doesn't allocate four numpy arrays per
    # camera per frame in ``_publish_outputs``.  K/P/R are float64 numpy
    # arrays (the C++ shim memcpy's their .data straight into the message);
    # D is a plain list because the bridge takes ``std::vector<double>`` by
    # value and would copy a numpy array element-by-element anyway.
    K_np: np.ndarray = field(init=False, repr=False)
    P_np: np.ndarray = field(init=False, repr=False)
    R_np: np.ndarray = field(init=False, repr=False)
    D_list: List[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.K_np = np.asarray(
            [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0],
            dtype=np.float64,
        )
        self.P_np = np.asarray(
            [
                self.fx,
                0.0,
                self.cx,
                0.0,
                0.0,
                self.fy,
                self.cy,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
            dtype=np.float64,
        )
        self.R_np = np.asarray(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float64,
        )
        self.D_list = [self.k1, self.k2, self.p1, self.p2, self.k3]

    @property
    def frame_id(self) -> str:
        """ROS TF frame_id used in image headers; matches render_node.cpp."""
        return self.path

    @property
    def K(self) -> List[float]:
        """Row-major 3x3 intrinsics (sensor_msgs/CameraInfo K).

        Returns a plain Python list — used by callers outside the hot path
        (e.g. logging).  Hot-path callers should consume ``self.K_np``.
        """
        return self.K_np.tolist()

    @property
    def P(self) -> List[float]:
        """Row-major 3x4 projection (sensor_msgs/CameraInfo P)."""
        return self.P_np.tolist()

    @property
    def R(self) -> List[float]:
        """Identity rectification (sensor_msgs/CameraInfo R)."""
        return self.R_np.tolist()

    @property
    def D(self) -> List[float]:
        """plumb_bob distortion (sensor_msgs/CameraInfo D)."""
        return list(self.D_list)


def load_cameras_from_manifest(manifest_path: str) -> tuple:
    """Parse the ``cameras`` array + ``free_cam_prim_path`` from manifest.json.

    Delegates to :class:`genie_sim_engine_core.schema.Manifest` — same
    reader the C++ ``RenderNode`` and every other Python consumer uses,
    so a rename of any camera field in the shared schema is picked up
    here for free.

    Returns:
        ``(cameras, free_cam_prim_path)`` — ``free_cam_prim_path`` is the
        absolute USD prim path of the FreeCam camera (e.g.
        ``/RenderOVRTX/Cameras/FreeCam``), or empty when the scene has no
        free camera authored.
    """
    from genie_sim_engine_core.schema import Manifest, SchemaError

    p = Path(manifest_path).resolve()
    if not p.is_file():
        _log("manifest.json not found at %s", p)
        return [], ""
    try:
        manifest = Manifest.read(p)
    except SchemaError as exc:
        _log("manifest.json unreadable: %s", exc)
        return [], ""
    cams: List[CameraCfg] = []
    for i, cm in enumerate(manifest.cameras):
        cams.append(
            CameraCfg(
                path=cm.path or f"camera_{i}",
                topic=cm.topic or f"/camera_{i}",
                depth_topic=cm.depth_topic,
                render_product_path=cm.render_product_path or f"/RenderOVRTX/Cam_{i}",
                width=cm.width,
                height=cm.height,
                fx=cm.fx,
                fy=cm.fy,
                cx=cm.cx if cm.cx else cm.width / 2.0,
                cy=cm.cy if cm.cy else cm.height / 2.0,
                k1=cm.k1,
                k2=cm.k2,
                p1=cm.p1,
                p2=cm.p2,
                k3=cm.k3,
                is_free_cam=cm.is_free_cam,
            )
        )
    return cams, manifest.free_cam_prim_path


# ---------------------------------------------------------------------------
# InlineOvrtxVisualizer
# ---------------------------------------------------------------------------


@dataclass
class _ThreadStats:
    """Lightweight stats counters owned by the OVRtx thread."""

    frames: int = 0
    skipped_no_subs: int = 0
    failed: int = 0
    last_log_time: float = field(default_factory=time.monotonic)
    sum_frame_ms: float = 0.0


class InlineOvrtxVisualizer:
    """Newton-standalone in-process OVRtx render thread.

    Lifecycle:
        1. ``__init__``               — version guard, construct ovrtx.Renderer,
                                        load USD layers, parse cameras.
        2. ``bind_newton_bodies(model)`` — create the persistent
                                        ``omni:xform`` binding + index map.
        3. ``attach_physics_event(event)`` — receive the wp.Event the physics
                                        thread will record after each step.
        4. ``start()``                — spawn the render thread.
        5. (per frame on the render thread) ``_render_one_frame()``.
        6. ``stop()``                 — join thread, unbind, destroy renderer.
    """

    # OVRtx render-var names we extract.  Both come from
    # genie_sim_engine_isaacsim/assemble/_scene_impl.py — see render_node.cpp:644, 704.
    _RV_RGB = "LdrColor"
    _RV_DEPTH = "DistanceToImagePlaneSD"

    def __init__(
        self,
        scene_usda: str,
        robot_usda: str,
        render_layer_usda: str,
        manifest_path: str,
        robot_prefix: str,
        device: str = "cuda:0",
        ovrtx_root: Optional[str] = None,
        render_fps: float = 30.0,
        realtime_factor: float = 1.0,
        log_path: str = "/tmp/ovrtx_inline.log",
        log_level: str = "info",
        camera_pose_topic: str = "/genie_sim_engine/viewer/camera_pose",
        camera_zoom_topic: str = "",
    ) -> None:
        # Version guard — installed header vs installed wheel skew is
        # the most common setup bug (e.g. an older site-packages copy).
        import ovrtx

        if ovrtx.__version__.split(".")[:2] != ["0", "4"]:
            raise RuntimeError(f"InlineOvrtxVisualizer requires ovrtx 0.4.x, got {ovrtx.__version__}")

        self._device = device
        self._render_period = 1.0 / max(render_fps * realtime_factor, 1e-3)
        self._scene_usda = scene_usda
        self._robot_usda = robot_usda
        self._render_layer_usda = render_layer_usda
        self._robot_prefix = robot_prefix.lstrip("/")

        # Mount-path → cloth asset USD path, populated when newton_scene.usda
        # is composed below.  Must be initialised BEFORE the newton_scene
        # block runs because that block writes into it; later cloth-binding
        # bookkeeping reads it.  Box entries don't get an entry here.
        self._cloth_assets: Dict[str, str] = {}

        # Renderer construction — port of render_node.cpp:281-326 in Python.
        # Note: ``binary_package_root`` is forwarded via
        # ``ovrtx.register_schema_paths(binary_package_root=...)`` BEFORE
        # constructing the Renderer, not as a RendererConfig field.  See
        # ovrtx/_src/schema_paths.py.
        from ovrtx import Renderer, RendererConfig, register_schema_paths

        if ovrtx_root:
            try:
                register_schema_paths(binary_package_root=ovrtx_root)
            except Exception:
                _logexc(
                    "register_schema_paths(%s) failed — continuing " "with default search paths",
                    ovrtx_root,
                )

        # OVRtx 0.4 enables Sensor Processing Graphs by default.  The inline
        # visualizer publishes geometric ``DistanceToImagePlaneSD`` depth,
        # so keep SPG disabled just like the standalone C++ renderer does;
        # otherwise OVRtx applies the depth-sensor model and clamps the
        # usable range to roughly [0.3, 2.5] m.
        self._renderer: Optional[Renderer] = Renderer(
            RendererConfig(log_file_path=log_path, log_level=log_level, enable_spg=False)
        )
        _log("renderer constructed")

        # USD load + references
        _log("open_usd %s", scene_usda)
        self._renderer.open_usd(scene_usda)
        if robot_usda and os.path.isfile(robot_usda):
            robot_pfx = "/" + self._robot_prefix
            _log("add_usd_reference %s @ %s", robot_usda, robot_pfx)
            self._renderer.add_usd_reference(robot_usda, robot_pfx)
        if render_layer_usda and os.path.isfile(render_layer_usda):
            _log(
                "add_usd_reference %s @ /RenderOVRTX",
                render_layer_usda,
            )
            self._renderer.add_usd_reference(render_layer_usda, "/RenderOVRTX")

        # Newton extras layer: cloth/softbody USDs and procedural box
        # colliders authored by the isaac_scene stage at /World/<name>.
        # The Kit-using paths sublayer this into the physics stage via
        # kit/stage.py:_open_scene_with_references; we do the equivalent
        # here for the inline OVRtx visualizer.
        #
        # CANNOT use ``add_usd_reference(newton_scene.usda, "/World")`` —
        # blank.usda already authors a /World prim, and OVRtx rejects a
        # reference compose at an existing prim path with
        # "A prim already exists at path: /World".  Instead we reference
        # each /World/<name> prim INSIDE newton_scene.usda individually,
        # by emitting a tiny in-memory USDA that references the specific
        # prim and mounting that at "/World/<name>" — which doesn't yet
        # exist, so the compose succeeds.
        newton_scene_path = (
            os.path.join(os.path.dirname(render_layer_usda), "newton_scene.usda") if render_layer_usda else ""
        )
        if newton_scene_path and os.path.isfile(newton_scene_path):
            from pxr import Sdf, Usd

            ns_stage = Usd.Stage.Open(newton_scene_path)
            world = ns_stage.GetPrimAtPath("/World") if ns_stage else None

            # Walk newton_scene.usda's /World/<name> children.  For each,
            # determine the right per-prim mount strategy:
            #
            #   * Self-contained (e.g. UsdGeom.Cube):  reference
            #     newton_scene.usda's prim — compose pulls in the inline
            #     definition without needing additional asset paths.
            #
            #   * Reference-only (cloth Xform with `references = @...@`):
            #     extract the underlying asset path and reference THAT
            #     directly.  A reference-of-reference through an
            #     in-memory wrapper composes empty in OVRtx because
            #     the wrapper's prim isn't the underlying file's
            #     defaultPrim, so the inner reference doesn't propagate
            #     into the consumer's prim spec.  Going to the original
            #     asset bypasses that.
            child_specs: List[tuple] = []  # (name, mount_path, layer_to_reference)
            if world and world.IsValid():
                for child in world.GetChildren():
                    name = child.GetName()
                    mount = f"/World/{name}"
                    # Inspect the prim's authored references.  Cloth
                    # entries are pure Xforms with a single external
                    # reference and no other authored geometry; box
                    # entries are UsdGeom.Cube with their own attrs.
                    ref_asset = None
                    for spec in child.GetPrimStack():
                        refs = spec.referenceList
                        # prepended + appended + explicit — the
                        # cloth entry uses ``prepend references``
                        # so we check both.
                        for arc in list(refs.prependedItems) + list(refs.explicitItems) + list(refs.appendedItems):
                            if arc.assetPath:
                                ref_asset = arc.assetPath
                                break
                        if ref_asset:
                            break

                    # If the child has only an asset reference and no
                    # geometry of its own, point at the original
                    # asset.  Otherwise (Cube, etc.) point at the
                    # newton_scene.usda prim.
                    is_geometry = child.GetTypeName() in {"Cube", "Sphere", "Capsule", "Cylinder", "Cone", "Mesh"}
                    if ref_asset and not is_geometry:
                        # Resolve relative paths against newton_scene.usda's directory.
                        if not os.path.isabs(ref_asset):
                            ref_asset = os.path.normpath(os.path.join(os.path.dirname(newton_scene_path), ref_asset))
                        child_specs.append((name, mount, ref_asset))
                    else:
                        child_specs.append((name, mount, newton_scene_path))

            _log(
                "newton_scene.usda authors %d prim(s) under /World: %s",
                len(child_specs),
                [(n, os.path.basename(layer)) for n, _, layer in child_specs],
            )

            # Mount each child at /World/<name>.  Two cases:
            #
            #   * Self-contained prim inside newton_scene.usda (e.g. the
            #     ``UsdGeom.Cube`` for ``fold_box``): we need a per-prim
            #     selector ``@layer@</World/<name>>`` to pick the specific
            #     prim out of the layer.  ``add_usd_reference`` mounts the
            #     file's defaultPrim only — so we go through
            #     ``add_usd_reference_from_string`` with a tiny in-memory
            #     wrapper that uses the selector syntax.
            #
            #   * Already an external asset (cloth Xform whose only
            #     contribution is a reference to its mesh USD): the asset's
            #     defaultPrim IS what we want to mount.  Skip the in-memory
            #     wrapper and call ``add_usd_reference`` directly — same
            #     result, less work for OVRtx (no extra USDA parse).
            #
            # No try/except here: a failed mount means the scene-yaml
            # object won't render at all and the operator should see the
            # error immediately, not buried in a "(non-fatal)" log line.
            for name, mount, layer in child_specs:
                if layer == newton_scene_path:
                    snippet = (
                        f"#usda 1.0\n"
                        f"(\n"
                        f'    defaultPrim = "{name}"\n'
                        f")\n"
                        f'def "{name}" (\n'
                        f"    references = @{layer}@</World/{name}>\n"
                        f") {{}}\n"
                    )
                    self._renderer.add_usd_reference_from_string(snippet, mount)
                else:
                    # Direct asset reference — OVRtx mounts the asset's
                    # defaultPrim at ``mount``.  Remember the asset path
                    # so cloth_info can walk to the Mesh prim for points
                    # binding.
                    self._cloth_assets[mount] = layer
                    self._renderer.add_usd_reference(layer, mount)
            _log("newton_scene per-prim references mounted: %d", len(child_specs))
        elif newton_scene_path:
            _log(
                "no newton_scene.usda at %s (scene yaml has no box/cloth — OK)",
                newton_scene_path,
            )

        # Cameras
        self._cameras: List[CameraCfg]
        self._free_cam_prim_path: str
        self._cameras, self._free_cam_prim_path = load_cameras_from_manifest(manifest_path)
        self._free_cam_cfg = next((cam for cam in self._cameras if cam.is_free_cam), None)
        self._free_cam_base_fx = self._free_cam_cfg.fx if self._free_cam_cfg else 0.0
        self._free_cam_base_fy = self._free_cam_cfg.fy if self._free_cam_cfg else 0.0
        # Must match define_free_cam() in engine_core/common/render_layer.py.
        # This is an authored camera invariant, not a second tuning default.
        self._free_cam_base_focal_length = 18.0
        self._free_cam_zoom = 1.0
        # Build the render-product set we pass to renderer.step every frame.
        # Free cams are included unconditionally — the FreeCam is the
        # default viewport for users running with physics_engine_visualizer:=ovrtx
        # and is expected to render even with no /image_raw subscriber.
        self._render_products: set = {cam.render_product_path for cam in self._cameras}
        # Register publishers up-front so subscriber-count gating is meaningful
        # from the very first frame (matches render_node.cpp:setup_cameras).
        # Free cams get publishers too — RViz / other clients subscribe to
        # /genie_sim/free_camera_rgb the same way as any other camera.
        for cam in self._cameras:
            _core.create_camera_publisher(cam.topic, cam.width, cam.height, False)
            if cam.depth_topic:
                _core.create_camera_publisher(cam.depth_topic, cam.width, cam.height, True)

        # Free-cam pose subscription.  RViz's view_camera_pose_publisher_display
        # plugin defaults to /genie_sim_engine/viewer/camera_pose (see
        # genie_sim_rviz_plugins/view_camera_pose_publisher_display.cpp:32).
        # Pose is applied every frame — see _apply_free_cam_pose.
        self._free_cam_binding: Any = None
        self._camera_pose_topic = camera_pose_topic
        if self._free_cam_prim_path:
            try:
                _core.subscribe_camera_pose(camera_pose_topic)
                _log(
                    "free_cam pose subscription wired: %s -> %s",
                    camera_pose_topic,
                    self._free_cam_prim_path,
                )
            except Exception:
                _logexc(
                    "subscribe_camera_pose(%s) failed; free-cam " "will sit at its rest pose",
                    camera_pose_topic,
                )

        # Optical zoom is deliberately opt-in and separate from the existing
        # pose contract. Only the Newton inline OVRtx launcher enables this
        # subscription; the external g2sim renderer keeps pose-only behavior.
        self._camera_zoom_topic = camera_zoom_topic
        if self._free_cam_prim_path and self._free_cam_cfg and camera_zoom_topic:
            try:
                _core.subscribe_camera_zoom(camera_zoom_topic)
                _log("free_cam optical zoom subscription wired: %s", camera_zoom_topic)
            except Exception:
                _logexc(
                    "subscribe_camera_zoom(%s) failed; FreeCam keeps authored FOV",
                    camera_zoom_topic,
                )

        _log(
            "loaded %d camera(s); %d render product(s); " "free_cam_prim_path=%s",
            len(self._cameras),
            len(self._render_products),
            self._free_cam_prim_path or "(none)",
        )

        # Filled by bind_newton_bodies().
        self._binding: Any = None
        self._body_indices: Optional[wp.array] = None
        self._body_q_provider: Optional[Any] = None
        self._physics_event: Optional[wp.Event] = None

        # Cloth bookkeeping — populated by bind_newton_bodies() from the
        # engine's ``_cloth_particle_start/_end/_cloth_usd_prim_path``
        # fields (set by ``cloth.py:_inject_cloth``).  v1 supports a
        # single cloth slot; if the scene authors more than one, the
        # later ones overwrite the engine's own bookkeeping anyway.
        self._cloth_binding: Any = None
        self._cloth_points: Optional[wp.array] = None  # output buffer
        self._cloth_start: int = 0
        self._cloth_count: int = 0
        self._cloth_prim_path: str = ""
        self._particle_q_provider: Optional[Any] = None
        # ``self._cloth_assets`` is initialised at the top of __init__
        # (before the newton_scene.usda block runs).  Populated there with
        # mount-path → cloth-asset-USD entries; bind_newton_bodies reads
        # it to find the Mesh prim path inside the asset.

        # Multi-Mesh FEM reconstruction. Each output buffer has the authored
        # vertex count/order for one source USDA Mesh. The fixed four-particle
        # indices and barycentric weights are uploaded once at bind time.
        self._deformable_binding: Any = None
        self._deformable_points: List[wp.array] = []
        self._deformable_particle_indices: List[wp.array] = []
        self._deformable_weights: List[wp.array] = []
        self._deformable_prim_paths: List[str] = []

        # Dedicated CUDA stream so OVRtx GPU work doesn't serialise behind
        # physics on Warp's default stream (see docs/ovrtx_sync.md).
        self._ovrtx_stream: wp.Stream = wp.Stream(device=device)

        self._thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()
        self._ready = threading.Event()
        self._stats = _ThreadStats()
        # Latched across the lifetime of the visualizer; ``self._stats`` is
        # reset every 1 Hz stats log, so ``self._stats.frames`` can't be
        # used to gate first-frame diagnostics.
        self._first_frame_logged = False
        # First free_cam pose log fires once on the first
        # /viewer/camera_pose message so we have a known matrix to diff
        # against the cross-process renderer's on_free_cam_pose for the
        # same viewport.
        self._free_cam_first_pose_logged = False

    # ----- setup --------------------------------------------------------------

    def bind_newton_bodies(
        self,
        model: Any,
        body_q_provider: Any,
        cloth_info: Optional[dict] = None,
        particle_q_provider: Optional[Any] = None,
        deformable_bindings: Optional[List[dict]] = None,
    ) -> None:
        """Build the persistent OVRtx binding and the Newton body-index map.

        Args:
            model: object exposing ``body_paths`` either as a list/sequence
                attribute or as a zero-arg method returning the list.  In
                practice the entrypoint shims a small object whose
                ``body_paths()`` returns ``engine._body_paths`` so
                downstream depends on a stable schema.
            body_q_provider: Callable returning the live ``state.body_q``
                ``wp.array``.  We call this every frame because Newton swaps
                ``state_0`` ↔ ``state_1`` between substeps; binding to a
                snapshot would point at the wrong buffer after the first
                swap.  See engine/newton/engine.py:481.
            cloth_info: Optional dict with keys
                ``{"prim_path", "start", "end"}`` describing the single
                cloth tracked by the engine
                (``engine.newton.cloth._inject_cloth``).  When present we
                bind ``points`` on that prim and animate it from
                ``state.particle_q[start:end]`` every frame.
            particle_q_provider: Callable returning the live
                ``state.particle_q`` ``wp.array``.  Required when
                ``cloth_info`` or ``deformable_bindings`` is set.
            deformable_bindings: Authored-Mesh records carrying exact prim
                paths plus fixed ``tet_barycentric_v1`` particle mappings.
        """
        import ovrtx

        bp_attr = getattr(model, "body_paths", None)
        if callable(bp_attr):
            raw_paths = list(bp_attr() or [])
        elif bp_attr is not None:
            raw_paths = list(bp_attr)
        else:
            raw_paths = []
        if not raw_paths:
            raise RuntimeError("no body paths available; cannot bind omni:xform")

        # Newton's body labels can be either:
        #   * bare names like "upperarm" — we must prefix with the robot's
        #     OVRtx-side root (e.g. "/genie/upperarm") so OVRtx's prim
        #     resolver finds them.  add_usd_reference(robot_usda, "/genie")
        #     places every robot prim under "/genie/...".
        #   * absolute paths like "/World/ground" or "/genie/upperarm" —
        #     pass through unchanged.
        # Mirrors render_node.cpp:476-481 (the cross-process renderer's
        # tf_scene handler runs the same logic on every frame).
        prefix = "/" + self._robot_prefix.lstrip("/")
        body_paths: List[str] = []
        for name in raw_paths:
            if not name:
                continue
            if name.startswith("/"):
                body_paths.append(name)
            else:
                body_paths.append(f"{prefix}/{name}")

        n = len(body_paths)

        # Diagnostic: dump every transformed path so a misplaced robot is
        # tied to a specific name mismatch the operator can fix.  Writes
        # to /tmp/ovrtx_inline.bodies.txt — too verbose for the live log
        # but cheap to diff against the OVRtx stage when something looks
        # wrong.
        try:
            dump_path = "/tmp/ovrtx_inline.bodies.txt"
            with open(dump_path, "w") as fh:
                fh.write(f"# robot_prefix={prefix}\n")
                fh.write(f"# {n} body paths after prefix transform\n")
                fh.write("# format: index <tab> raw_name <tab> resolved_path\n")
                for i, (raw, resolved) in enumerate(zip(raw_paths, body_paths)):
                    fh.write(f"{i}\t{raw}\t{resolved}\n")
            _log("body paths dumped to %s (n=%d, prefix=%s)", dump_path, n, prefix)
        except Exception:
            pass

        # Verify resolution: read back ``omni:xform`` for every path we're
        # about to bind.  Paths that don't resolve are reported here — the
        # silent skip from ``PrimMode.EXISTING_ONLY`` later would otherwise
        # leave those bodies frozen at their authored rest pose, which is
        # exactly the "robot apart" symptom.
        try:
            probe = self._renderer.read_attribute("omni:xform", body_paths)
            _log("read_attribute round-trip OK (probe type=%s)", type(probe).__name__)
        except Exception as exc:
            _log("read_attribute round-trip FAILED: %r — some prim paths may not resolve", exc)

        # CRITICAL: Newton's body_q is per-body WORLD-space transforms, but
        # OVRtx normally composes a prim's omni:xform with its parent's
        # accumulated transform.  Newton's body labels for a hierarchical
        # robot USD are nested paths (e.g.
        # /genie/.../base_link/body_link1/body_link2), so writing
        # world-space matrices onto these would compound with parent
        # transforms and pull the robot apart.
        #
        # Fix: flag every bound prim with ``omni:resetXformStack=True`` so
        # OVRtx interprets the prim's omni:xform as the FINAL world
        # transform, ignoring parent composition.  Same semantic the
        # FreeCam already uses (``genie_sim_engine_core/common/render_layer.py``).
        # This is a one-shot
        # write at bind time; from this point on the per-frame Warp kernel
        # writes world-space transforms directly.
        try:
            reset_flags = np.ones(n, dtype=bool)
            self._renderer.write_attribute(body_paths, "omni:resetXformStack", reset_flags)
            _log("set omni:resetXformStack=True on %d body prims", n)
        except Exception:
            _logexc(
                "failed to set omni:resetXformStack — robot will likely render "
                "apart because per-prim world-space writes will compose with "
                "parent transforms"
            )

        sample = body_paths[: min(5, n)]
        _log(
            "binding omni:xform for %d bodies (robot_prefix=%s); first=%s",
            n,
            prefix,
            sample,
        )

        # Persistent binding pays prim resolution + descriptor cost once.
        #
        # IMPORTANT: keep ``dtype="float64", shape=(4, 4)`` here, NOT
        # ``semantic=Semantic.XFORM_MAT4x4``.  The CUDA-mapped path
        # (``binding.map(device=Device.CUDA)``) hands back a tensor whose
        # layout depends on whether the binding was constructed with the
        # raw dtype or the transform semantic.  For the kernel-writing hot
        # path we want a contiguous ``(N, 4, 4)`` float64 buffer that the
        # kernel can stride with ``wp.mat44d`` per slot — which is exactly
        # what the raw dtype/shape gives us.  Using
        # ``Semantic.XFORM_MAT4x4`` here triggers an OVRtx-side crash
        # (Carbonite minidump, mid-run, after ~1 minute) consistent with
        # heap corruption from a per-slot stride mismatch.  The cross-
        # process renderer uses ``ovrtx_set_xform_mat`` (the C semantic
        # helper) but only for CPU writes; the same pattern doesn't apply
        # 1:1 here because we go through the zero-copy GPU map.
        self._binding = self._renderer.bind_attribute(
            prim_paths=body_paths,
            attribute_name="omni:xform",
            dtype="float64",
            shape=(4, 4),
            prim_mode=ovrtx.PrimMode.EXISTING_ONLY,
            flags=ovrtx.BindingFlag.OPTIMIZE,
        )

        # Identity index map for the single-env newton-standalone case.
        # If we ever add cloned envs, this becomes the flat enumeration.
        self._body_indices = wp.array(np.arange(n, dtype=np.int32), dtype=wp.int32, device=self._device)
        self._body_q_provider = body_q_provider

        # Free-cam binding — one persistent binding for the FreeCam's
        # ``omni:xform`` attribute, written from a CPU mat44d every frame
        # (low-rate, no kernel needed).  Built here because the renderer is
        # ready and the manifest's free_cam_prim_path is known.
        #
        # We use ``semantic=Semantic.XFORM_MAT4x4`` rather than the
        # ``dtype + shape`` path so OVRtx uses the same semantic-conversion
        # code-path that ``ovrtx_set_xform_mat`` (the C helper used by
        # genie_sim_render's on_free_cam_pose, render_node.cpp:543) takes.
        # Both should be layout-equivalent in theory, but some historical
        # OVRtx builds differ in how they handle a "raw" matrix4d binding vs a
        # semantically-tagged one — the explicit semantic is safer.
        if self._free_cam_prim_path:
            try:
                self._free_cam_binding = self._renderer.bind_attribute(
                    prim_paths=[self._free_cam_prim_path],
                    attribute_name="omni:xform",
                    semantic=ovrtx.Semantic.XFORM_MAT4x4,
                    prim_mode=ovrtx.PrimMode.EXISTING_ONLY,
                    flags=ovrtx.BindingFlag.NONE,
                )
            except Exception:
                _logexc(
                    "free_cam bind_attribute(%s) failed; pose " "updates will be skipped",
                    self._free_cam_prim_path,
                )
                self._free_cam_binding = None

        # Cloth binding — when the engine has a single cloth slot
        # tracked, bind ``points`` on its prim and allocate a
        # per-cloth Warp output buffer.  The OVRtx ``points`` array
        # binding accepts a list-of-tensors (one per prim); for the
        # single-cloth case the list has length 1.
        #
        # Frame note: the isaac_scene stage's newton-cloth branch
        # (``genie_sim_engine_isaacsim/assemble/_scene_impl.py``) keeps the cloth
        # Xform at identity so we can write world-space ``particle_q``
        # straight into ``points``.  If you ever switch to a non-
        # identity authored xform, this needs an inverse-world
        # transform first.
        #
        # ``points`` lives on the underlying ``UsdGeom.Mesh``, which is a
        # CHILD of the cloth Xform — not the Xform itself.  cloth.py:_inject_cloth
        # walks ``Usd.PrimRange(prim)`` to find the first Mesh under
        # ``/World/<name>``; we do the equivalent here, but on the cloth
        # USD layer (which doesn't have access to the live physics stage).
        # The Mesh's path inside the layer is relative to the layer's
        # defaultPrim, which is what we mounted at ``/World/<name>``;
        # so we recompute the OVRtx-stage absolute path by taking the
        # Mesh's path-relative-to-defaultPrim and prepending
        # ``/World/<name>``.
        if cloth_info and particle_q_provider:
            prim = cloth_info.get("prim_path") or ""
            start = int(cloth_info.get("start", 0))
            end = int(cloth_info.get("end", 0))
            # ``cloth_info`` may carry an explicit asset_path; otherwise
            # look it up from the mount-path table populated in __init__
            # while composing newton_scene.usda.
            asset = cloth_info.get("asset_path") or self._cloth_assets.get(prim, "")
            count = max(0, end - start)
            mesh_mount = ""
            if prim and count > 0 and asset and os.path.isfile(asset):
                # No try/except: a Mesh-path resolution failure means
                # the cloth USD is malformed and we want to see the
                # traceback, not a "skipping" log line.
                from pxr import Usd, UsdGeom

                cloth_stage = Usd.Stage.Open(asset)
                default_prim = cloth_stage.GetDefaultPrim()
                if not default_prim or not default_prim.IsValid():
                    raise RuntimeError(f"cloth USD {asset} has no defaultPrim — cannot resolve " f"Mesh path")
                mesh_prim = None
                for q in Usd.PrimRange(default_prim):
                    if q.IsA(UsdGeom.Mesh):
                        mesh_prim = q
                        break
                if mesh_prim is None:
                    raise RuntimeError(
                        f"cloth USD {asset} has no UsdGeom.Mesh under " f"defaultPrim {default_prim.GetPath()}"
                    )
                # Path inside the cloth USD relative to its defaultPrim.
                # E.g. defaultPrim=/Root, mesh=/Root/shirt → relative=shirt
                rel = mesh_prim.GetPath().MakeRelativePath(default_prim.GetPath()).pathString
                # Mount path: /World/<name>/<rel>.  For T_Shirt_fold.usd
                # this is /World/tshirt/shirt.
                mesh_mount = f"{prim}/{rel}" if rel and rel != "." else prim

            if mesh_mount:
                # No try/except: a failed cloth bind means the per-frame
                # points write will go nowhere and the operator should
                # see why immediately.
                self._cloth_binding = self._renderer.bind_array_attribute(
                    prim_paths=[mesh_mount],
                    attribute_name="points",
                    dtype="float32",
                    shape=(3,),
                    prim_mode=ovrtx.PrimMode.EXISTING_ONLY,
                    flags=ovrtx.BindingFlag.OPTIMIZE,
                )
                self._cloth_points = wp.empty(count, dtype=wp.vec3f, device=self._device)
                self._cloth_start = start
                self._cloth_count = count
                self._cloth_prim_path = mesh_mount
                self._particle_q_provider = particle_q_provider
                _log(
                    "bound cloth points: %s [%d particles, start=%d]",
                    mesh_mount,
                    count,
                    start,
                )

        if deformable_bindings:
            if particle_q_provider is None:
                raise RuntimeError("deformable bindings require a particle_q provider")
            particle_q = particle_q_provider()
            if particle_q is None:
                raise RuntimeError("particle_q is unavailable while binding deformable meshes")
            prepared = prepare_deformable_bindings(deformable_bindings, particle_count=int(particle_q.shape[0]))
            self._deformable_prim_paths = [item.prim_path for item in prepared]

            # Reconstruction writes world-space points. Reset each Mesh's
            # transform stack and set identity so OVRtx consumes those points
            # without composing the original authored parent transforms again.
            count = len(prepared)
            self._renderer.write_attribute(
                self._deformable_prim_paths,
                "omni:resetXformStack",
                np.ones(count, dtype=bool),
            )
            self._renderer.write_attribute(
                self._deformable_prim_paths,
                "omni:xform",
                np.tile(np.eye(4, dtype=np.float64), (count, 1, 1)),
                semantic=ovrtx.Semantic.XFORM_MAT4x4,
            )
            self._deformable_binding = self._renderer.bind_array_attribute(
                prim_paths=self._deformable_prim_paths,
                attribute_name="points",
                dtype="float32",
                shape=(3,),
                prim_mode=ovrtx.PrimMode.EXISTING_ONLY,
                flags=ovrtx.BindingFlag.OPTIMIZE,
            )
            self._deformable_points = [
                wp.empty(
                    item.global_particle_indices.shape[0],
                    dtype=wp.vec3f,
                    device=self._device,
                )
                for item in prepared
            ]
            self._deformable_particle_indices = [
                wp.array(
                    item.global_particle_indices.reshape(-1),
                    dtype=wp.int32,
                    device=self._device,
                )
                for item in prepared
            ]
            self._deformable_weights = [
                wp.array(
                    item.barycentric_weights.reshape(-1),
                    dtype=wp.float32,
                    device=self._device,
                )
                for item in prepared
            ]
            self._particle_q_provider = particle_q_provider
            _log(
                "bound %d deformable Mesh points arrays (%d authored vertices)",
                count,
                sum(item.global_particle_indices.shape[0] for item in prepared),
            )

        self._ready.set()

    def attach_physics_event(self, event: wp.Event) -> None:
        """Receive the per-step CUDA event from the physics thread."""
        self._physics_event = event

    # ----- thread control ----------------------------------------------------

    def start(self) -> None:
        if not self._ready.is_set():
            raise RuntimeError("start() called before bind_newton_bodies()")
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="ovrtx-render", daemon=False)
        self._thread.start()
        _log("render thread started @ %.1f Hz target", 1.0 / self._render_period)

    def stop(self) -> None:
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        # Drop the cross-thread render-rate readout immediately so the
        # physics-thread stats logger doesn't ghost an old rate after
        # shutdown.
        global _render_stale_after
        _render_stale_after = 0.0
        if self._binding is not None:
            try:
                self._binding.unbind()
            except Exception:
                _logexc("binding.unbind failed")
            self._binding = None
        if self._free_cam_binding is not None:
            try:
                self._free_cam_binding.unbind()
            except Exception:
                _logexc("free_cam_binding.unbind failed")
            self._free_cam_binding = None
        if self._cloth_binding is not None:
            try:
                self._cloth_binding.unbind()
            except Exception:
                _logexc("cloth_binding.unbind failed")
            self._cloth_binding = None
        if self._deformable_binding is not None:
            try:
                self._deformable_binding.unbind()
            except Exception:
                _logexc("deformable_binding.unbind failed")
            self._deformable_binding = None
        # Drop the renderer ref; ovrtx.Renderer cleans up via __del__.
        self._renderer = None
        _log("stopped")

    # ----- render loop -------------------------------------------------------

    def _run(self) -> None:
        _log("render thread loop entered")
        next_tick = time.monotonic()
        first_failure_logged = False
        while not self._shutdown.is_set():
            now = time.monotonic()
            if now < next_tick:
                # Idle until the next slot.  Wait on shutdown so Ctrl-C can
                # break out cleanly without burning CPU.
                self._shutdown.wait(timeout=(next_tick - now))
                continue
            # Slot-aligned cadence: if we're more than one period late, snap
            # forward instead of catching up (latest-state-wins).
            next_tick += self._render_period
            if next_tick < now:
                next_tick = now + self._render_period

            t0 = time.monotonic()
            try:
                self._render_one_frame()
                self._stats.frames += 1
                self._stats.sum_frame_ms += (time.monotonic() - t0) * 1000.0
            except Exception:
                self._stats.failed += 1
                # Log the first failure unconditionally so a hung thread
                # leaves a trace; later failures are throttled to 1 Hz via
                # the regular stats log to avoid log spam in a bad-state
                # loop.
                if not first_failure_logged:
                    _logexc("first frame failed")
                    first_failure_logged = True

            # 1 Hz stats log so operators can see the thread is alive without
            # peeking at /clock.  Also publishes the latest rate to the
            # module-level globals so engine/newton/stats.py can fold a
            # one-line ``ovrtx=NN.NHz`` summary into the physics-thread 1 Hz
            # log.  Stale-after is +3 s so a transient frame stutter doesn't
            # blank the readout for the next physics-thread log.
            now2 = time.monotonic()
            if now2 - self._stats.last_log_time >= 1.0:
                n = max(self._stats.frames, 1)
                elapsed = now2 - self._stats.last_log_time
                hz = self._stats.frames / max(elapsed, 1e-6)
                avg_ms = self._stats.sum_frame_ms / n
                failed = self._stats.failed
                global _render_hz, _render_avg_ms, _render_failed
                global _render_stale_after
                _render_hz = hz
                _render_avg_ms = avg_ms
                _render_failed = failed
                _render_stale_after = now2 + 3.0
                _log(
                    "%.1f frames/s (skipped_no_subs=%d, failed=%d, " "avg=%.2f ms/frame)",
                    hz,
                    self._stats.skipped_no_subs,
                    failed,
                    avg_ms,
                )
                self._stats = _ThreadStats(last_log_time=now2)
                first_failure_logged = False

    def _any_subscribers(self) -> bool:
        """True if any camera RGB or depth topic has a live subscriber.

        Free cams are checked too — if a client is subscribed to
        /genie_sim/free_camera_rgb we want to deliver its frames just like
        any other camera.  The render-frame-skip decision in
        ``_render_one_frame`` is separate from this: when a FreeCam exists
        we render unconditionally because the FreeCam IS the viewport, even
        when no image consumer is attached.
        """
        for cam in self._cameras:
            if _core.has_image_subscribers(cam.topic):
                return True
            if cam.depth_topic and _core.has_image_subscribers(cam.depth_topic):
                return True
        return False

    def _apply_free_cam_pose(self) -> None:
        """Drain the latest /<node>/viewer/camera_pose into FreeCam's xform.

        The pose-to-matrix conversion lives in C++ (``RosBridge``'s
        subscription callback at realtime_ros_node.cpp) — same byte-for-byte
        formula as ``genie_sim_render::on_free_cam_pose`` (render_node.cpp:531-540).
        We just receive a ready-to-write ``(1, 4, 4)`` float64 mat44d and
        hand it to the persistent OVRtx binding.

        The RViz plugin (genie_sim_rviz_plugins) latches the most recent
        pose, so a fresh subscriber sees the current viewport immediately;
        between pose updates we leave FreeCam where it was.
        """
        if self._free_cam_binding is None:
            return
        m = _core.take_free_cam_xform()
        if m is None:
            return

        # Diagnostic: dump the matrix on the very first successful drain
        # so any mismatch with genie_sim_render's behaviour is easy to
        # spot in the launch log.
        if not self._free_cam_first_pose_logged:
            try:
                _log(
                    "first free_cam mat44d row-major (from C++):\n"
                    "  [%+.3f %+.3f %+.3f %+.3f]\n"
                    "  [%+.3f %+.3f %+.3f %+.3f]\n"
                    "  [%+.3f %+.3f %+.3f %+.3f]\n"
                    "  [%+.3f %+.3f %+.3f %+.3f]",
                    m[0, 0, 0],
                    m[0, 0, 1],
                    m[0, 0, 2],
                    m[0, 0, 3],
                    m[0, 1, 0],
                    m[0, 1, 1],
                    m[0, 1, 2],
                    m[0, 1, 3],
                    m[0, 2, 0],
                    m[0, 2, 1],
                    m[0, 2, 2],
                    m[0, 2, 3],
                    m[0, 3, 0],
                    m[0, 3, 1],
                    m[0, 3, 2],
                    m[0, 3, 3],
                )
            except Exception:
                _logexc("first free_cam pose: diagnostic dump failed")
            self._free_cam_first_pose_logged = True

        # No try/except — a failed write means the FreeCam isn't
        # tracking RViz; an operator should see the traceback rather
        # than a one-line "failed" log they have to grep for.
        self._free_cam_binding.write(m)

    def _apply_free_cam_zoom(self) -> None:
        """Apply the latest focal-length multiplier and matching CameraInfo."""
        if not self._camera_zoom_topic or self._free_cam_cfg is None:
            return
        raw_zoom = _core.take_free_cam_zoom()
        if raw_zoom is None:
            return
        try:
            zoom = clamp_zoom_factor(raw_zoom)
        except (TypeError, ValueError):
            _log("ignoring invalid free_cam zoom sample: %r", raw_zoom)
            return
        if abs(zoom - self._free_cam_zoom) <= 1.0e-6:
            return

        focal_length = self._free_cam_base_focal_length * zoom
        self._renderer.write_attribute(
            [self._free_cam_prim_path],
            "focalLength",
            np.asarray([focal_length], dtype=np.float32),
        )

        cam = self._free_cam_cfg
        cam.fx = self._free_cam_base_fx * zoom
        cam.fy = self._free_cam_base_fy * zoom
        cam.K_np, cam.P_np = scaled_intrinsics(
            base_fx=self._free_cam_base_fx,
            base_fy=self._free_cam_base_fy,
            cx=cam.cx,
            cy=cam.cy,
            zoom=zoom,
        )
        self._free_cam_zoom = zoom
        _log("free_cam optical zoom=%.3f focalLength=%.3f", zoom, focal_length)

    def _apply_cloth_points(self) -> None:
        """Copy ``state.particle_q[start:end]`` into the cloth mesh's points.

        Runs on the OVRtx render thread, on the same Warp stream as the
        body sync kernel.  ``binding.write(..., DataAccess.ASYNC, cuda_stream=...)``
        makes OVRtx wait on our stream before consuming, so the kernel's
        write is fully ordered against OVRtx's read.

        See :func:`ovrtx_kernels.sync_particle_q_slice_to_points` for the
        kernel itself and the frame-convention note (cloth prim authored
        at identity → world == local).
        """
        import ovrtx

        if (
            self._cloth_binding is None
            or self._cloth_points is None
            or self._particle_q_provider is None
            or self._cloth_count <= 0
        ):
            return

        particle_q = self._particle_q_provider()
        if particle_q is None:
            return
        # Newton's particle_q is wp.vec3f; if a build hands it back as
        # transformf or another type, skip rather than crash the thread.
        if getattr(particle_q, "dtype", None) is not wp.vec3f:
            return

        # No try/except — a write failure should propagate.  The render
        # thread's outer ``_run`` catches it once with a traceback and
        # rate-limits subsequent failures via the 1 Hz stats log.
        wp.launch(
            sync_particle_q_slice_to_points,
            dim=self._cloth_count,
            inputs=[self._cloth_points, particle_q, self._cloth_start],
            stream=self._ovrtx_stream,
        )
        # ``data_access=ASYNC`` is required for GPU buffers — OVRtx
        # rejects SYNC (the default) on a CUDA tensor with
        # ``OVRTX_DATA_ACCESS_SYNC is not supported for GPU buffers``.
        # The cuda_stream= forwards our Warp stream so OVRtx serialises
        # against the kernel write via stream order — same pattern the
        # cuda-interop skill documents for ``write_attribute_async``.
        self._cloth_binding.write(
            [self._cloth_points],
            data_access=ovrtx.DataAccess.ASYNC,
            cuda_stream=self._ovrtx_stream.cuda_stream,
        )

    def _apply_deformable_points(self) -> None:
        """Reconstruct all authored Mesh vertices from live FEM particles."""
        import ovrtx

        if self._deformable_binding is None or not self._deformable_points or self._particle_q_provider is None:
            return
        particle_q = self._particle_q_provider()
        if particle_q is None or getattr(particle_q, "dtype", None) is not wp.vec3f:
            return

        for points, indices, weights in zip(
            self._deformable_points,
            self._deformable_particle_indices,
            self._deformable_weights,
        ):
            wp.launch(
                reconstruct_tet_bound_points,
                dim=points.shape[0],
                inputs=[points, particle_q, indices, weights],
                stream=self._ovrtx_stream,
            )
        self._deformable_binding.write(
            self._deformable_points,
            data_access=ovrtx.DataAccess.ASYNC,
            cuda_stream=self._ovrtx_stream.cuda_stream,
        )

    def _render_one_frame(self) -> None:
        import ovrtx

        # 0. Subscriber-aware skip — port of render_node.cpp:313.
        # When a FreeCam exists we render unconditionally because the
        # FreeCam IS the viewport for this mode (always-on by default).
        # Without a FreeCam we still gate on subscriber count: a scene with
        # no camera consumers is pure-headless physics and shouldn't burn
        # GPU.
        has_free_cam = bool(self._free_cam_prim_path)
        if not has_free_cam and not self._any_subscribers():
            self._stats.skipped_no_subs += 1
            return

        # First-frame banner — JIT compilation + shader cache warmup means
        # the first render.step() can take many seconds (typically 30-90s
        # on first run; subsequent runs hit the on-disk shader cache and
        # start instantly).  We log a one-shot banner here and then let
        # the thread block silently in the C extensions; an in-process
        # Python heartbeat won't tick because OVRtx's MDL / shader compile
        # holds the GIL.  When the first frame completes we resume the
        # normal per-frame stats logging.
        is_first_frame = not self._first_frame_logged
        if is_first_frame:
            _log("=" * 64)
            _log("OVRTX first run on this machine: compiling shaders now.")
            _log("This typically takes 30-90 seconds. Render images will")
            _log("stay blank until compilation finishes; the cache will")
            _log("make subsequent runs instant.")
            _log("=" * 64)

        # 1. Cross-stream sync: wait on the latest physics commit.
        if self._physics_event is not None:
            self._ovrtx_stream.wait_event(self._physics_event)

        # 2. ZERO-COPY MAP — Warp kernel writes directly into OVRtx's
        # internal Fabric buffer.  Lifetime of mapping.tensor is the with
        # block; the kernel launch is queued on ovrtx_stream BEFORE the
        # explicit mapping.unmap(stream=...), so OVRtx serialises against
        # the kernel via stream order.
        body_q = self._body_q_provider()
        if body_q is None or self._body_indices is None or self._binding is None:
            return
        with self._binding.map(device=ovrtx.Device.CUDA) as mapping:
            ovrtx_xforms = wp.from_dlpack(mapping.tensor, dtype=wp.mat44d)
            wp.launch(
                sync_body_q_to_ovrtx_mat44d,
                dim=ovrtx_xforms.shape[0],
                inputs=[ovrtx_xforms, self._body_indices, body_q],
                stream=self._ovrtx_stream,
            )
            # Explicit stream-aware unmap — see module docstring.  Without
            # this OVRtx would synchronise with cudaStreamLegacy on __exit__,
            # which on RTX hosts can fail or stall the render path.
            mapping.unmap(stream=self._ovrtx_stream.cuda_stream)

        # 2b. Apply the latest free-cam pose (from RViz / pose publisher).
        # Cheap: a single mat44d write through the persistent binding.
        if has_free_cam:
            self._apply_free_cam_pose()
            self._apply_free_cam_zoom()

        # 2c. Cloth points sync — animate cloth meshes from particle_q.
        # Same stream as the body kernel; OVRtx serialises against it via
        # ``cuda_stream=`` on the array-binding write.
        if self._cloth_binding is not None:
            self._apply_cloth_points()

        # 2d. Scene-plugin FEM meshes use fixed tet-barycentric maps to
        # reconstruct every authored vertex in stable source order.
        if self._deformable_binding is not None:
            self._apply_deformable_points()

        # 3. Render — synchronous step (waits on the GPU before returning,
        # but only this thread is gated; physics is off doing its thing).
        # First call compiles OVRTX shaders + JITs render kernels and can
        # block for tens of seconds with no output.  See the first-frame
        # banner above; we deliberately stay silent here because the C
        # extension holds the GIL and any in-process heartbeat would not
        # be able to run during the compile anyway.
        outputs = self._renderer.step(
            render_products=self._render_products,
            delta_time=self._render_period,
        )

        # 4. Publish camera outputs from THIS thread.
        sim_time = _core.last_sim_time()
        self._publish_outputs(sim_time, outputs)

        if is_first_frame:
            _log("OVRTX first frame complete — shader cache warmed; subsequent renders run at the configured render_hz")
            self._first_frame_logged = True

    def _publish_outputs(self, sim_time: float, outputs: Any) -> None:
        import ovrtx

        for cam in self._cameras:
            product = outputs.get(cam.render_product_path) if hasattr(outputs, "get") else None
            if product is None:
                # Fall back to dict-style access (RenderProductSetOutputs
                # exposes __contains__ + __getitem__).
                if cam.render_product_path not in outputs:
                    continue
                product = outputs[cam.render_product_path]
            if not getattr(product, "frames", None):
                continue
            frame = product.frames[0]

            # ---- RGB ----
            if _core.has_image_subscribers(cam.topic):
                rv = frame.render_vars.get(self._RV_RGB)
                if rv is not None:
                    try:
                        with rv.map(device=ovrtx.Device.CPU) as mrv:
                            arr = np.from_dlpack(mrv)
                            # OVRtx hands back (H, W, 4) uint8 for LdrColor.
                            if arr.ndim == 3 and arr.shape[2] == 4 and arr.dtype == np.uint8:
                                _core.publish_camera_image_rgba8(sim_time, cam.topic, cam.frame_id, arr)
                                _core.publish_camera_info(
                                    sim_time,
                                    cam.topic,
                                    cam.frame_id,
                                    cam.height,
                                    cam.width,
                                    cam.K_np,
                                    cam.P_np,
                                    cam.R_np,
                                    cam.D_list,
                                    "plumb_bob",
                                )
                            else:
                                _log(
                                    "%s: unexpected LdrColor shape %s " "dtype %s",
                                    cam.topic,
                                    arr.shape,
                                    arr.dtype,
                                )
                    except Exception:
                        _logexc("%s: LdrColor publish failed", cam.topic)

            # ---- Depth (only when scene yaml set topic.depth) ----
            if cam.depth_topic and _core.has_image_subscribers(cam.depth_topic):
                drv = frame.render_vars.get(self._RV_DEPTH)
                if drv is not None:
                    try:
                        with drv.map(device=ovrtx.Device.CPU) as mrv:
                            arr = np.from_dlpack(mrv)
                            # OVRtx contract: (H, W) or (H, W, 1) float32.
                            if arr.ndim == 3 and arr.shape[2] == 1:
                                arr = arr[..., 0]
                            if arr.ndim == 2 and arr.dtype == np.float32:
                                _core.publish_camera_image_depth32f(sim_time, cam.depth_topic, cam.frame_id, arr)
                                _core.publish_camera_info(
                                    sim_time,
                                    cam.depth_topic,
                                    cam.frame_id,
                                    cam.height,
                                    cam.width,
                                    cam.K_np,
                                    cam.P_np,
                                    cam.R_np,
                                    cam.D_list,
                                    "plumb_bob",
                                )
                            else:
                                _log(
                                    "%s: unexpected depth shape %s " "dtype %s",
                                    cam.depth_topic,
                                    arr.shape,
                                    arr.dtype,
                                )
                    except Exception:
                        _logexc("%s: depth publish failed", cam.depth_topic)
