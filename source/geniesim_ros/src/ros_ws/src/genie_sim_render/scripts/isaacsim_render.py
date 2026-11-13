#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""
isaacsim_render.py

Isaac Sim replacement for ``genie_sim_render_node`` (ovrtx-based).

Mirrors the C++ node contract:
  * Reads the ``manifest.json`` produced by ``assemble_scene`` (same JSON
    schema as the C++ RenderNode):
      ``scene_usda`` / ``usd_path``, ``robot_usda``, ``render_layer_usda``,
      ``robot_prefix``, ``free_cam_prim_path``, ``base_path``,
      ``cameras[...]`` (each with ``path``, ``topic``, ``render_product_path``,
      intrinsics, optional ``depth_topic``, ``is_free_cam``).
  * Loads the scene + optional robot sublayer + optional render layer into
    the Isaac Sim USD stage.
  * Subscribes to ``/tf_render`` (v1, deprecated — this script is the sole
    remaining consumer; new render paths use the v2 split via ``/tf_scene`` +
    ``robot_state_publisher``) and applies every transform to the matching
    prim under ``/{robot_prefix}`` (mirrors ``RenderNode::on_tf_render``).
  * Subscribes to ``~/free_cam_pose`` and applies it to ``free_cam_prim_path``.
  * Publishes RGB images + CameraInfo per manifest camera through the Isaac
    Sim ROS2 bridge OmniGraph. Topics follow the same convention as the
    default ``RosImagePublisherPlugin``: ``<topic>/image_raw`` and
    ``<topic>/camera_info``.

Run with the system python3 that has ``isaacsim`` installed via pip:

    python3 isaacsim_render.py --ros-args \\
        -p stage_manifest:=/path/to/manifest.json \\
        -p render_fps:=30.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Launch SimulationApp FIRST. All omni.* imports must come after this.
#
# preload_warp_before_kit() must run before that: Isaac Sim's bundled
# omni.warp.core extension merges its own (older) warp/ directory onto the
# already-imported warp package's __path__ instead of replacing it, which
# otherwise breaks Newton's eager `import warp.fem` during extension
# startup. See its docstring in _isaac_render_common.py for the full chain.
# ---------------------------------------------------------------------------
from _isaac_render_common import preload_warp_before_kit  # noqa: E402  (sibling module)

preload_warp_before_kit()

import isaacsim  # noqa: F401  (registers extensions)
from isaacsim import SimulationApp
from genie_sim_engine_isaacsim.kit.launch_config import prepare_launch_config


def _env_truthy(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "off", "")


# Headless by default; override with ISAACSIM_HEADLESS=0 to show the viewport.
_DEFAULT_HEADLESS = _env_truthy("ISAACSIM_HEADLESS", True)
simulation_app = SimulationApp(prepare_launch_config({"headless": _DEFAULT_HEADLESS}))

# ---------------------------------------------------------------------------
# Omni / Isaac Sim imports (post-SimulationApp)
# ---------------------------------------------------------------------------
import omni  # noqa: E402
import omni.graph.core as og  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.core.utils.stage import (  # noqa: E402
    add_reference_to_stage,
    get_current_stage,
)
from pxr import Gf, Sdf, UsdGeom  # noqa: E402

# ROS2 bridge extension must be enabled before creating the camera graph
enable_extension("isaacsim.ros2.bridge")
enable_extension("omni.graph.core")

# ---------------------------------------------------------------------------
# ROS2
# ---------------------------------------------------------------------------
import rclpy  # noqa: E402
import rclpy.time  # noqa: E402  (rclpy.time is a submodule, not re-exported)
from rclpy.node import Node  # noqa: E402
from rclpy.qos import (  # noqa: E402
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from geometry_msgs.msg import PoseStamped  # noqa: E402
from std_msgs.msg import Float64MultiArray, MultiArrayDimension  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

from _isaac_render_common import (  # noqa: E402  (sibling module)
    DeformableChannel,
    disable_physics_subtree,
    load_scene_config,
    reference_scene_stage,
    select_physics_variant,
)

_SENSOR_QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
)


@dataclass
class CameraConfig:
    """Backwards-compat view of a manifest camera entry.

    Historical shape used by :class:`Manifest` below; the canonical
    typed source is
    :class:`genie_sim_engine_core.schema.CameraManifest`. This wrapper
    stays because the isaacsim_render node references
    ``CameraConfig(...)`` in a handful of places.
    """

    path: str
    topic: str
    render_product_path: str
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
    k4: float = 0.0
    model: str = ""
    min_range: float = 0.01
    max_range: float = 10000.0
    is_free_cam: bool = False
    depth_topic: str = ""
    frame_id: str = ""


@dataclass
class Manifest:
    """View of ``manifest.json`` for the isaacsim_render node.

    Delegates parsing to :class:`genie_sim_engine_core.schema.Manifest`
    (the shared cross-language ABI); this wrapper only preserves the
    local field surface downstream Python callers already reference.
    New readers should import
    :class:`~genie_sim_engine_core.schema.Manifest` directly.
    """

    scene_usda: str
    render_layer_usda: str
    robot_usda: str = ""
    #: Prebaked visual robot USD (``robot_source.prebaked`` + ``visual_only``).
    #: When set, it is attached for rendering *instead of* ``robot_usda`` (the
    #: staged URDF output whose materials the converter flattens to white); the
    #: prebaked asset carries the authored materials. Its prim tree may differ
    #: from ``robot_usda`` -- see ``_remap_body_paths_to_visual``.
    robot_visual_usda: str = ""
    robot_prefix: str = ""
    free_cam_prim_path: str = ""
    cameras: List[CameraConfig] = field(default_factory=list)
    #: ``robot.body_paths`` — short link name -> absolute USD prim path, used to
    #: drive the robot from live tf (the short names match /tf frame ids).
    robot_body_paths: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, manifest_path: Path) -> "Manifest":
        """Load a ``manifest.json`` produced by the shared assemble
        pipeline.

        Path resolution semantics match the C++ ``RenderNode::load_manifest``
        exactly: entries stored relative to ``base_path`` (or the
        manifest's own directory when ``base_path`` is absent) resolve
        to absolute. Errors go through the shared
        :class:`~genie_sim_engine_core.schema.SchemaError` /
        ``SchemaVersionError`` hierarchy so a stale stage dir gets a
        clear "re-run assemble" message.
        """
        from genie_sim_engine_core.schema import Manifest as _SharedManifest

        shared = _SharedManifest.read(Path(manifest_path).resolve())

        def _s(p) -> str:
            return str(p) if p else ""

        scene_usda = _s(shared.scene_usda)
        if not scene_usda:
            raise KeyError("Manifest missing both 'scene_usda' and 'usd_path'")
        robot_usda = _s(shared.robot_usda)
        robot_visual_usda = _s(shared.robot_visual_usda)
        render_layer_usda = _s(shared.render_layer_usda)

        if not Path(scene_usda).exists():
            raise FileNotFoundError(f"Scene USD not found: {scene_usda}")
        if robot_usda and not Path(robot_usda).exists():
            raise FileNotFoundError(f"Robot USD not found: {robot_usda}")
        if robot_visual_usda and not Path(robot_visual_usda).exists():
            raise FileNotFoundError(f"Robot visual USD not found: {robot_visual_usda}")
        if render_layer_usda and not Path(render_layer_usda).exists():
            raise FileNotFoundError(f"Render layer USDA not found: {render_layer_usda}")

        cams: List[CameraConfig] = []
        for i, cm in enumerate(shared.cameras):
            cams.append(
                CameraConfig(
                    path=cm.path,
                    topic=cm.topic,
                    render_product_path=cm.render_product_path or f"/RenderOVRTX/Cam_{i}",
                    width=cm.width,
                    height=cm.height,
                    fx=cm.fx,
                    fy=cm.fy,
                    cx=cm.cx,
                    cy=cm.cy,
                    k1=cm.k1,
                    k2=cm.k2,
                    p1=cm.p1,
                    p2=cm.p2,
                    k3=cm.k3,
                    k4=cm.k4,
                    model=cm.model,
                    min_range=cm.min_range,
                    max_range=cm.max_range,
                    is_free_cam=cm.is_free_cam,
                    depth_topic=cm.depth_topic,
                    frame_id=cm.frame_id,
                )
            )

        return cls(
            scene_usda=scene_usda,
            render_layer_usda=render_layer_usda,
            robot_usda=robot_usda,
            robot_visual_usda=robot_visual_usda,
            robot_prefix=shared.robot_prefix,
            free_cam_prim_path=shared.free_cam_prim_path,
            cameras=cams,
            robot_body_paths=dict((shared.robot or {}).get("body_paths") or {}),
        )


def _set_camera_intrinsics(prim, cam: CameraConfig, focal_length: float = 1.0) -> None:
    """Author the calibrated RTX lens and a consistent USD fallback frustum."""
    from genie_sim_engine_core.common.render_layer import author_opencv_fisheye, author_opencv_pinhole

    horizontal_aperture = cam.width * focal_length / cam.fx
    vertical_aperture = cam.height * focal_length / cam.fy
    prim.GetAttribute("focalLength").Set(focal_length)
    prim.GetAttribute("horizontalAperture").Set(horizontal_aperture)
    prim.GetAttribute("verticalAperture").Set(vertical_aperture)
    prim.GetAttribute("horizontalApertureOffset").Set((cam.cx - cam.width / 2) * focal_length / cam.fx)
    prim.GetAttribute("verticalApertureOffset").Set((cam.cy - cam.height / 2) * focal_length / cam.fy)
    author = author_opencv_fisheye if getattr(cam, "model", "").lower() == "opencvfisheye" else author_opencv_pinhole
    author(
        prim,
        {
            **{key: getattr(cam, key) for key in ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")},
            "k4": getattr(cam, "k4", 0.0) if getattr(cam, "model", "").lower() == "opencvfisheye" else 0.0,
        },
        cam.width,
        cam.height,
    )


def _apply_xform(prim, translation: Gf.Vec3d, rotation: Gf.Quatd, *, reset_xform_stack: bool = False) -> None:
    """Write translate + orient onto an existing xform prim.

    ``reset_xform_stack`` makes the prim ignore its ancestors' transforms — so a
    world-absolute pose can be written onto a NESTED prim without the ancestor
    chain composing on top of it. This is the USD equivalent of the C++ node's
    ``omni:resetXformStack=1``, and it is what lets robot link prims (authored
    nested as ``base_link/body_link1/...``) take root-relative tf poses.

    The xformOpOrder is only re-authored when it is not already in the wanted
    shape. ``SetXformOpOrder`` is a STRUCTURAL edit (it changes the prim's
    composition), and doing it unconditionally for ~88 link prims every frame
    re-notifies the Hydra/Fabric render delegate each tick, which is how a long
    run degenerates into the delegate spinning in its TBB scheduler without ever
    producing another frame. The op VALUES are still written every frame; only
    the structural change is made once.
    """
    if not prim or not prim.IsValid():
        return
    xformable = UsdGeom.Xformable(prim)
    t_attr = prim.GetAttribute("xformOp:translate")
    o_attr = prim.GetAttribute("xformOp:orient")
    if not t_attr or not t_attr.IsValid() or not o_attr or not o_attr.IsValid():
        xformable.ClearXformOpOrder()
        t_attr = prim.GetAttribute("xformOp:translate") or xformable.AddTranslateOp().GetAttr()
        o_attr = prim.GetAttribute("xformOp:orient") or xformable.AddOrientOp().GetAttr()
    t_attr.Set(translation)
    if o_attr.GetTypeName() == Sdf.ValueTypeNames.Quatf:
        o_attr.Set(Gf.Quatf(rotation))
    else:
        o_attr.Set(rotation)

    if reset_xform_stack:
        current = list(prim.GetAttribute("xformOpOrder").Get() or [])
        wanted = ["!resetXformStack!", "xformOp:translate", "xformOp:orient"]
        if current != wanted:
            xformable.SetXformOpOrder([UsdGeom.XformOp(t_attr), UsdGeom.XformOp(o_attr)], resetXformStack=True)


def _build_camera_omnigraph(
    graph_path: str, cameras: List[CameraConfig], camera_prim_paths: List[str], fps_skip: int
) -> None:
    """Create camera transport with explicit calibrated CameraInfo.

    ``camera_prim_paths`` must hold the resolved Camera prim path per camera (see
    ``IsaacSimRenderNode._camera_prim_path``) -- the manifest's ``cam.path`` is a
    short body name that does not resolve on the stage.
    """
    nodes = [("Tick", "omni.graph.action.OnPlaybackTick"), ("Time", "isaacsim.core.nodes.IsaacReadSimulationTime")]
    edges = []
    for i, _cam in enumerate(cameras):
        nodes += [
            (f"CreateRP{i}", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
            (f"CamHelper{i}", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            (f"CamInfo{i}", "isaacsim.ros2.bridge.ROS2PublishCameraInfo"),
        ]
        edges += [
            ("Tick.outputs:tick", f"CreateRP{i}.inputs:execIn"),
            (f"CreateRP{i}.outputs:execOut", f"CamHelper{i}.inputs:execIn"),
            (f"CreateRP{i}.outputs:execOut", f"CamInfo{i}.inputs:execIn"),
            (f"CreateRP{i}.outputs:renderProductPath", f"CamHelper{i}.inputs:renderProductPath"),
            ("Time.outputs:simulationTime", f"CamInfo{i}.inputs:timeStamp"),
        ]
        if _cam.depth_topic:
            nodes.append((f"DepthHelper{i}", "isaacsim.ros2.bridge.ROS2CameraHelper"))
            edges += [
                (f"CreateRP{i}.outputs:execOut", f"DepthHelper{i}.inputs:execIn"),
                (f"CreateRP{i}.outputs:renderProductPath", f"DepthHelper{i}.inputs:renderProductPath"),
            ]

    og.Controller.edit(
        {"graph_path": graph_path},
        {
            og.Controller.Keys.CREATE_NODES: nodes,
            og.Controller.Keys.CONNECT: edges,
        },
    )

    for i, cam in enumerate(cameras):
        og.Controller.set(f"{graph_path}/CreateRP{i}.inputs:cameraPrim", camera_prim_paths[i])
        og.Controller.set(f"{graph_path}/CreateRP{i}.inputs:width", int(cam.width))
        og.Controller.set(f"{graph_path}/CreateRP{i}.inputs:height", int(cam.height))
        og.Controller.set(f"{graph_path}/CamHelper{i}.inputs:topicName", cam.topic + "/image_raw")
        og.Controller.set(f"{graph_path}/CamHelper{i}.inputs:frameId", cam.frame_id or cam.path)
        og.Controller.set(f"{graph_path}/CamHelper{i}.inputs:frameSkipCount", int(fps_skip))
        og.Controller.set(f"{graph_path}/CamInfo{i}.inputs:topicName", cam.topic + "/camera_info")
        og.Controller.set(f"{graph_path}/CamInfo{i}.inputs:frameId", cam.frame_id or cam.path)
        og.Controller.set(f"{graph_path}/CamInfo{i}.inputs:queueSize", 1)
        # The SDK helper forces fy=fx even for an OpenCV lens. Pass the same
        # calibration authored above directly to the publisher instead.
        for key, value in {
            "width": int(cam.width),
            "height": int(cam.height),
            "projectionType": "equidistant" if getattr(cam, "model", "").lower() == "opencvfisheye" else "plumb_bob",
            "k": [cam.fx, 0.0, cam.cx, 0.0, cam.fy, cam.cy, 0.0, 0.0, 1.0],
            "r": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "p": [cam.fx, 0.0, cam.cx, 0.0, 0.0, cam.fy, cam.cy, 0.0, 0.0, 0.0, 1.0, 0.0],
            "physicalDistortionModel": (
                "equidistant" if getattr(cam, "model", "").lower() == "opencvfisheye" else "plumb_bob"
            ),
            "physicalDistortionCoefficients": (
                [cam.k1, cam.k2, cam.k3, cam.k4]
                if getattr(cam, "model", "").lower() == "opencvfisheye"
                else [cam.k1, cam.k2, cam.p1, cam.p2, cam.k3]
            ),
        }.items():
            og.Controller.set(f"{graph_path}/CamInfo{i}.inputs:{key}", value)
        if cam.depth_topic:
            og.Controller.set(f"{graph_path}/DepthHelper{i}.inputs:type", "depth")
            og.Controller.set(f"{graph_path}/DepthHelper{i}.inputs:topicName", cam.depth_topic + "/image_raw")
            og.Controller.set(f"{graph_path}/DepthHelper{i}.inputs:frameId", cam.frame_id or cam.path)
            og.Controller.set(f"{graph_path}/DepthHelper{i}.inputs:frameSkipCount", int(fps_skip))


class IsaacSimRenderNode(Node):
    """ROS2 node that drives Isaac Sim rendering to mirror genie_sim_render_node."""

    def __init__(self) -> None:
        super().__init__("render_isaacsim")

        # Parameters (names mirror the C++ RenderNode)
        self.declare_parameter("stage_manifest", "")
        self.declare_parameter("render_fps", 30.0)
        self.declare_parameter("prim_paths", "")
        # Opt-in live deformable-particle channel. Empty (the default) leaves
        # the feature off, so an ordinary scene is unaffected.
        self.declare_parameter("deformable_topic", "")
        self.declare_parameter("sync_observation_topic", "")
        self.declare_parameter("sync_camera_topics", "{}")
        self.declare_parameter("sync_render_on_demand", False)
        self._sync_topic = self.get_parameter("sync_observation_topic").value
        self._sync_render_on_demand = self.get_parameter("sync_render_on_demand").value
        self._sync_capture = None

        manifest_path = self.get_parameter("stage_manifest").get_parameter_value().string_value
        self._render_fps = float(self.get_parameter("render_fps").get_parameter_value().double_value or 30.0)
        self._deformable_topic = self.get_parameter("deformable_topic").get_parameter_value().string_value

        if not manifest_path:
            raise RuntimeError("Parameter 'stage_manifest' is required")
        self._manifest_path = Path(manifest_path).resolve()
        self._material_application = None
        self._manifest = Manifest.load(Path(manifest_path))
        self._snapshot_robot_source_paths = dict(self._manifest.robot_body_paths)

        self._timing_pub = self.create_publisher(Float64MultiArray, "~/render_timing", 10)
        # Robot link poses. ``/tf_render`` below is the **v1** channel and is
        # deprecated: this script is its last render-side consumer, kept for
        # no-RSP deployments only (see ``use_tf_render`` in
        # ``genie_sim_engine_core/common/params.py``).  New render nodes must
        # take the v2 split instead.  Under that protocol the engine fans robot
        # links onto /tf via robot_state_publisher and keeps /tf_scene for
        # scene bodies only; /tf_render is the pre-v2 single-topic dialect that
        # a deployment without robot_state_publisher opts into
        # (``use_tf_render``). Bringup starts a robot_state_publisher, so /tf is
        # the channel that actually carries the robot here — same as the C++
        # node, which drives links from a tf2 buffer rather than a raw topic.
        self._tf_sub = self.create_subscription(TFMessage, "/tf", self._on_tf, _SENSOR_QOS)
        # Deprecated v1 compatibility channel; new deployments use /tf and /tf_scene.
        self._tf_scene_sub = self.create_subscription(TFMessage, "/tf_scene", self._on_tf_scene, _SENSOR_QOS)
        self._tf_render_sub = self.create_subscription(TFMessage, "/tf_render", self._on_tf, _SENSOR_QOS)
        self._tf_buffer = None
        self._tf_listener = None
        if self._manifest.robot_body_paths:
            # tf2 gives relative lookups between the URDF's own frames (the link
            # tree is published parent->child, so base_footprint -> base_link ->
            # ... resolves without a world frame existing). Isaac Sim's bundled
            # tf2_ros takes the node explicitly, unlike upstream.
            from tf2_ros import Buffer, TransformListener

            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
        if self._manifest.free_cam_prim_path:
            self._free_cam_sub = self.create_subscription(
                PoseStamped, "~/free_cam_pose", self._on_free_cam_pose, _SENSOR_QOS
            )

        self._world: Optional[World] = None
        self._stage = None
        self._prim_cache: Dict[str, object] = {}
        self._cam_body_to_render_path: Dict[str, str] = {}
        self._frame_count = 0
        self._drop_count = 0
        self._last_frame_time = 0.0

        self._init_stage()
        self._setup_cameras()

        # Snapshot mode exclusively owns geometry; disable the live particle
        # subscription before running the original channel initialization.
        if self._sync_topic:
            self._deformable_topic = ""

        # Live deformable-particle channel. Subscribed AFTER the stage is up,
        # so the latched topology can resolve its prim_paths against a real
        # stage (and a topology that arrived during load is re-requested by
        # the transient-local QoS).
        self._deformable = DeformableChannel(self, self._stage, self._deformable_topic, logger=self.get_logger())

        self._setup_sync_capture()

        self.get_logger().info(
            f"Isaac Sim render node ready ({self._render_fps:.1f} fps, "
            f"{len(self._manifest.cameras)} cameras, manifest={manifest_path})"
        )

    # -- stage / cameras --------------------------------------------------

    def _setup_sync_capture(self):
        """Add snapshot capture after the original stage/channel initialization."""
        if not self._sync_topic:
            return
        try:
            from _sync_capture import SyncObservationRenderer

            requested = json.loads(self.get_parameter("sync_camera_topics").value)
            by_topic = {camera.topic: camera for camera in self._manifest.cameras}
            if (
                not isinstance(requested, dict)
                or not requested
                or any(
                    not isinstance(name, str) or not name or topic not in by_topic for name, topic in requested.items()
                )
                or len(set(requested.values())) != len(requested)
            ):
                raise ValueError("sync_camera_topics must select distinct configured manifest cameras")
            cameras = {
                name: (self._camera_prim_path(by_topic[topic]), (by_topic[topic].width, by_topic[topic].height))
                for name, topic in requested.items()
            }
            self._sync_capture = SyncObservationRenderer(
                self,
                self._sync_topic,
                cameras,
                self._apply_observation_snapshot,
                completed_capture=self._record_material_capture,
            )
        except BaseException:
            self.shutdown()
            self.destroy_node()
            raise

    def _select_robot_usd(self) -> str:
        """Keep sync rendering on a complete, explicitly named robot inventory.

        A prebaked visual may use an older hand naming convention or carry
        retired geometry. Do not guess aliases or drop missing physical bodies;
        use the assembled robot when the visual cannot represent every body.
        The ordinary live-render selection retains its existing behavior.
        """
        visual, physical = self._manifest.robot_visual_usda, self._manifest.robot_usda
        if not self._sync_topic or not visual:
            return visual or physical
        from pxr import Usd

        robot_root = "/" + self._manifest.robot_prefix if self._manifest.robot_prefix else "/Robot"
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(robot_root, "Xform").GetReferences().AddReference(visual)
        missing = []
        for body, original in self._manifest.robot_body_paths.items():
            candidates = (stage.GetPrimAtPath(robot_root + "/" + body), stage.GetPrimAtPath(original))
            if not any(prim and prim.IsA(UsdGeom.Xformable) and not prim.IsInstanceProxy() for prim in candidates):
                missing.append(body)
        if not missing:
            return visual
        if not physical:
            raise ValueError("sync visual has an incomplete robot inventory and no assembled robot_usda")
        shown = ", ".join(sorted(missing)[:8]) + (" ..." if len(missing) > 8 else "")
        self.get_logger().warning(
            f"Sync visual lacks {len(missing)} declared robot bodies ({shown}); "
            f"using complete assembled robot USD: {physical}"
        )
        return physical

    def _init_stage(self) -> None:
        self.get_logger().info(f"Opening scene USD: {self._manifest.scene_usda}")
        # reference_scene_stage, not open_stage -- see its docstring for the
        # Isaac Sim 5.1 + ros2.bridge segfault it avoids.
        reference_scene_stage(self._manifest.scene_usda)
        for _ in range(10):
            simulation_app.update()
        self._world = World(stage_units_in_meters=1.0)
        self._stage = get_current_stage()

        robot_root = ""
        # Render the prebaked visual asset when the scene ships one
        # (robot_source.prebaked + visual_only): robot_usda is the staged
        # URDF output, whose converter drops the authored materials and renders
        # flat white. robot_visual_usda carries the real materials.
        robot_usd_to_attach = self._manifest.robot_visual_usda or self._manifest.robot_usda
        if self._sync_topic:
            robot_usd_to_attach = self._select_robot_usd()
        self._robot_usd_to_attach = robot_usd_to_attach
        if robot_usd_to_attach:
            robot_root = "/" + self._manifest.robot_prefix if self._manifest.robot_prefix else "/Robot"
            self.get_logger().info(f"Attaching robot USD: {robot_usd_to_attach} at {robot_root}")
            add_reference_to_stage(robot_usd_to_attach, robot_root)
            if select_physics_variant(self._stage, robot_root, "none"):
                self.get_logger().info(f"Physics variant -> none on {robot_root} (render needs no solver)")

        # The render-layer USDA parallels the ovrtx "/RenderOVRTX" sublayer. It may
        # contain camera prim definitions; reference it under the same root so that
        # manifest paths starting with "/RenderOVRTX/..." resolve. When the
        # render_config is a plain scene JSON (no render_layer_usda), skip it —
        # camera prims are expected to live directly under the scene USD.
        if self._manifest.render_layer_usda:
            self.get_logger().info(f"Attaching render layer USD: {self._manifest.render_layer_usda}")
            add_reference_to_stage(self._manifest.render_layer_usda, "/RenderOVRTX")
        else:
            self.get_logger().info("No render_layer_usda configured; using scene USD camera prims directly")

        # Every link pose comes from tf, so the articulation/rigid-body solver has
        # nothing to contribute and would only push back against those writes.
        # This is replay's link-replay mode (disable_physics_on_stage with
        # include_robot=True).
        if robot_root:
            disable_physics_subtree(self._stage, robot_root, tag="render_isaacsim")

        if self._sync_topic:
            # No solver owns this stage. The physical backend keeps running in
            # its own process while this renderer waits for inference.
            for root in self._stage.GetPseudoRoot().GetChildren():
                disable_physics_subtree(self._stage, str(root.GetPath()), tag="sync_render")
        self._world.reset()
        if self._sync_topic:
            self._world.pause()

        from _material_generalization import MaterialEpisodes, apply_material_plan, load_episode_material_config

        material_cfg = load_episode_material_config(self._manifest_path, load_scene_config)
        self._material_episodes = MaterialEpisodes(material_cfg) if material_cfg is not None else None
        if material_cfg is not None:

            plan, receipt = material_cfg["episode_plan"], material_cfg["receipt_path"]
            apply_material_plan(self._stage, plan, receipt, manifest_path=self._manifest_path)
            self._material_application = (plan, receipt)

        # body_paths point into robot_usda's (staged URDF) prim tree; when we
        # attached the prebaked visual instead, its layout differs, so rebase
        # the link->prim map onto the visual asset before tf starts driving it.
        if self._manifest.robot_visual_usda and robot_root and self._manifest.robot_body_paths:
            self._remap_body_paths_to_visual(robot_root)

        # Build body-name -> /RenderOVRTX/Cameras/<body> mapping (matches C++ node).
        for cam in self._manifest.cameras:
            if cam.is_free_cam:
                continue
            slash = cam.path.find("/", 1) if cam.path.startswith("/") else cam.path.find("/")
            if slash != -1:
                body_name = cam.path[:slash].lstrip("/")
                self._cam_body_to_render_path[body_name] = f"/RenderOVRTX/Cameras/{body_name}"

    def _camera_prim_path(self, cam: CameraConfig) -> str:
        """Resolve a manifest camera entry to its actual Camera prim path.

        ``manifest.cameras[].path`` is a stage-relative path, not a prim path:
        the free camera's is just ``"FreeCam"``, and a link camera's is
        ``<link>/<Camera>``. The render layer mirrors that whole relative path
        under ``/RenderOVRTX/Cameras/`` (see ``author_render_layer``), so the
        Camera prim is ``/RenderOVRTX/Cameras/<link>/<Camera>`` -- the ``<link>``
        level alone is the parent Xform that ``_apply_robot_tf`` re-poses, and
        handing that to the OmniGraph's ``cameraPrim`` would bind a render
        product to an Xform instead of a Camera.

        Resolving through the RenderProduct's ``camera`` relationship first
        keeps this in step with whatever the assembler actually authored
        (the same binding ``resolve_camera_prims`` reads on the tiled path).
        """
        if cam.is_free_cam:
            if self._manifest.free_cam_prim_path:
                return self._manifest.free_cam_prim_path
        else:
            if self._stage is not None and cam.render_product_path:
                try:
                    from pxr import UsdRender

                    rp = self._stage.GetPrimAtPath(cam.render_product_path)
                    if rp and rp.IsValid():
                        rel = UsdRender.Product(rp).GetCameraRel()
                        targets = rel.GetTargets() if rel else []
                        if targets:
                            return str(targets[0])
                except Exception:  # noqa: BLE001 - fall through to the path form
                    pass
            if cam.path:
                return f"/RenderOVRTX/Cameras/{cam.path.lstrip('/')}"
        # Fall back to the render product's parent scope, then to the raw value.
        if cam.render_product_path.startswith("/"):
            parent = cam.render_product_path.rsplit("/", 1)[0]
            if parent:
                return f"{parent}/Cameras/{cam.path.lstrip('/')}"
        return cam.path

    def _setup_cameras(self) -> None:
        if not self._manifest.cameras:
            self.get_logger().warn("No cameras configured in manifest")
            return

        cam_prims: List[str] = []
        for i, cam in enumerate(self._manifest.cameras):
            prim_path = self._camera_prim_path(cam)
            cam_prims.append(prim_path)
            prim = self._stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                self.get_logger().warn(f"Camera prim {prim_path} not in stage; skipping intrinsics")
                continue
            try:
                _set_camera_intrinsics(prim, cam)
            except Exception as exc:  # pragma: no cover - best-effort logging
                self.get_logger().warn(f"Intrinsics setup failed for {prim_path}: {exc}")
            self.get_logger().info(f"Camera[{i}]: {prim_path} -> {cam.topic} ({cam.width}x{cam.height})")

        # Frame-skip keeps the ROS publish rate near render_fps given Isaac's
        # default 60Hz playback. 0 means publish every tick.
        fps_skip = max(0, int(round(60.0 / max(self._render_fps, 1e-3))) - 1)
        # Sync observations own their three Replicator products. Keep the
        # ordinary RGB/depth/CameraInfo graph only in the live camera mode.
        if self._sync_topic:
            return
        _build_camera_omnigraph("/World/CameraGraph", self._manifest.cameras, cam_prims, fps_skip)

    def _apply_observation_snapshot(self, snapshot):
        """Preflight the entire frame, then apply world poses and world points.

        The public xyz+wxyz poses already include the authored-root offset.
        Deformable points remain full-precision JSON until the USD Vec3f boundary;
        mesh IDs are identities, not array indices or task-specific ranges.
        """
        import numpy as np
        from pxr import Vt
        from _sync_capture import validate_snapshot

        snapshot = validate_snapshot(snapshot)
        poses, meshes, targets = [], [], set()
        for source_path, values in snapshot["bodies"].items():
            body = source_path.rsplit("/", 1)[-1]
            mapped = self._manifest.robot_body_paths.get(body)
            original = getattr(self, "_snapshot_robot_source_paths", {}).get(body)
            is_robot = mapped is not None and source_path in (mapped, original)
            path = mapped if is_robot else source_path
            paths = [path]
            mount = self._cam_body_to_render_path.get(body) if is_robot else None
            if mount:
                paths.append(mount)
            for target in paths:
                prim = self._get_prim(target)
                if prim is None or not prim.IsValid() or not prim.IsA(UsdGeom.Xformable) or prim.IsInstanceProxy():
                    raise ValueError(f"snapshot body or camera mount missing: {target}")
                if target in targets:
                    raise ValueError(f"duplicate snapshot body target: {target}")
                targets.add(target)
                poses.append((prim, Gf.Vec3d(*values[:3]), Gf.Quatd(*values[3:])))
        for entry in snapshot["deformables"]:
            path = entry["prim_path"]
            prim = self._stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Mesh) or path in targets:
                raise ValueError(f"snapshot deformable mesh missing or conflicts with body: {path}")
            if prim.IsInstanceProxy():
                raise ValueError(f"snapshot mesh is an immutable instance proxy: {path}")
            mesh = UsdGeom.Mesh(prim)
            old_points = mesh.GetPointsAttr().Get()
            points = np.asarray(entry["points"], dtype=np.float64)
            if old_points is None or len(old_points) != len(points):
                raise ValueError(f"snapshot mesh point count differs from authored topology: {path}")
            indices = mesh.GetFaceVertexIndicesAttr().Get()
            if indices and (min(indices) < 0 or max(indices) >= len(points)):
                raise ValueError(f"snapshot mesh topology indices are out of range: {path}")
            if np.max(np.abs(points)) > np.finfo(np.float32).max:
                raise ValueError(f"snapshot mesh points overflow USD Vec3f: {path}")
            local = np.ascontiguousarray(points, dtype=np.float32)
            meshes.append(
                (
                    mesh,
                    Vt.Vec3fArray.FromNumpy(local),
                    Vt.Vec3fArray([Gf.Vec3f(*map(float, local.min(axis=0))), Gf.Vec3f(*map(float, local.max(axis=0)))]),
                )
            )
        if self._material_episodes is not None:
            from _material_generalization import apply_material_plan

            plan, receipt, changed = self._material_episodes.select(snapshot)
            if changed:
                apply_material_plan(self._stage, plan, receipt, manifest_path=self._manifest_path)
                self._material_application = (plan, receipt)
        # No validation below this boundary. A bad final mesh never leaves a
        # partially reposed robot in the render stage.
        for prim, translation, rotation in poses:
            _apply_xform(prim, translation, rotation, reset_xform_stack=True)
        for mesh, points, extent in meshes:
            # The solver supplies world coordinates. Reset just this mesh's
            # stack to identity, retaining the authored hierarchy and topology.
            xform = UsdGeom.Xformable(mesh)
            if list(mesh.GetPrim().GetAttribute("xformOpOrder").Get() or []) != ["!resetXformStack!"]:
                xform.SetXformOpOrder([], resetXformStack=True)
            mesh.GetPointsAttr().Set(points)
            mesh.CreateExtentAttr().Set(extent)

    def _record_material_capture(self, response):
        if self._material_application is not None:
            from _material_generalization import record_material_capture

            plan, receipt = self._material_application
            record_material_capture(self._stage, plan, receipt, response, manifest_path=self._manifest_path)

    # -- subscriptions ----------------------------------------------------

    def _get_prim(self, prim_path: str):
        prim = self._prim_cache.get(prim_path)
        if prim is None:
            prim = self._stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                self._prim_cache[prim_path] = prim
            else:
                return None
        return prim

    def _on_tf(self, msg: TFMessage) -> None:
        """Apply robot link poses from tf.

        Handles both dialects so either publisher works:
          * v2 (bringup): robot links arrive on ``/tf`` via
            robot_state_publisher, and the manifest's ``robot.body_paths``
            maps short link name -> prim path. Because the message carries only
            a parent->child edge per link, the prim is posed by composing the
            chain from ``base_footprint``; a tf2 buffer does that lookup.
          * pre-v2 (``use_tf_render``): every body on ``/tf_render`` with
            absolute child_frame_ids, applied directly.
        """
        if getattr(self, "_sync_topic", ""):
            return
        if self._stage is None:
            return
        self._apply_tf_render(msg)
        if self._tf_buffer is not None:
            self._apply_robot_tf()

    def _on_tf_scene(self, msg: TFMessage) -> None:
        """The v2 scene channel carries world poses at absolute USD paths."""
        if getattr(self, "_sync_topic", ""):
            return
        for tf in msg.transforms:
            if not tf.child_frame_id.startswith("/"):
                continue
            prim = self._get_prim(tf.child_frame_id)
            if prim is None:
                continue
            t = tf.transform.translation
            r = tf.transform.rotation
            _apply_xform(
                prim,
                Gf.Vec3d(float(t.x), float(t.y), float(t.z)),
                Gf.Quatd(float(r.w), float(r.x), float(r.y), float(r.z)),
                reset_xform_stack=True,
            )

    def _apply_tf_render(self, msg: TFMessage) -> None:
        prefix = self._manifest.robot_prefix
        for tf in msg.transforms:
            body = tf.child_frame_id
            if body.startswith("/World") or body.startswith("/"):
                prim_path = body
            elif prefix:
                prim_path = f"/{prefix}/{body}"
            else:
                prim_path = f"/{body}"

            t = tf.transform.translation
            r = tf.transform.rotation
            translation = Gf.Vec3d(float(t.x), float(t.y), float(t.z))
            rotation = Gf.Quatd(float(r.w), float(r.x), float(r.y), float(r.z))

            prim = self._get_prim(prim_path)
            if prim is not None:
                _apply_xform(prim, translation, rotation)

            render_path = self._cam_body_to_render_path.get(body)
            if render_path:
                render_prim = self._get_prim(render_path)
                if render_prim is not None:
                    _apply_xform(render_prim, translation, rotation)

    def _remap_body_paths_to_visual(self, robot_root: str) -> None:
        """Rebase ``robot_body_paths`` onto the prebaked visual asset's prims.

        The manifest's ``body_paths`` are authored against ``robot_usda`` (the
        staged URDF output), but rendering attaches ``robot_visual_usda``, whose
        prim tree can differ. The prebaked G2 asset lays every link out flat as
        ``/<prefix>/<link>``, so point each link there when that prim exists.
        Links with no flat match keep their manifest path, so a partial mismatch
        degrades to skipping only those links in ``_apply_robot_tf`` rather than
        freezing the whole robot. ``_apply_xform`` uses ``resetXformStack`` so
        the flat layout poses correctly regardless of the original nesting.
        """
        if getattr(self, "_sync_topic", "") and self._robot_usd_to_attach != self._manifest.robot_visual_usda:
            return
        remapped: Dict[str, str] = {}
        missing: List[str] = []
        for body, orig in self._manifest.robot_body_paths.items():
            flat = f"{robot_root}/{body}"
            if self._get_prim(flat) is not None:
                remapped[body] = flat
            else:
                remapped[body] = orig
                missing.append(body)
        self._manifest.robot_body_paths = remapped
        matched = len(remapped) - len(missing)
        self.get_logger().info(
            f"Remapped {matched}/{len(remapped)} robot link prim paths to the "
            f"prebaked visual layout under {robot_root}"
        )
        if missing:
            shown = ", ".join(sorted(missing)[:8]) + (" ..." if len(missing) > 8 else "")
            self.get_logger().warning(
                f"robot_visual_usda has no flat prim for {len(missing)} link(s); "
                f"keeping manifest paths for: {shown}"
            )

    def _apply_robot_tf(self) -> None:
        """Pose each manifest robot link prim from its chain in the tf buffer.

        Mirrors the C++ node's ``apply_robot_tf``, including its reset of the
        xform stack, but rooted at the tree's actual root instead of ``map``:
        this deployment publishes no ``map`` (only the URDF chain from
        ``base_footprint``), so the lookup is taken relative to ``base_footprint``.

        ``!resetXformStack!`` matters here. ``lookup_transform(root, link)``
        returns the link's pose in the ROOT frame, absolute — but these prims
        are nested (``base_link/body_link1/body_link2/...``), so a USD
        xformOp on a nested prim composes with every ancestor's, and writing a
        root-absolute pose to each link would apply the ancestors twice (the
        arms and body visibly detach). Putting ``!resetXformStack!`` first in
        ``xformOpOrder`` makes the prim ignore ancestor transforms, so the
        authored world pose is used as-is — the USD equivalent of the C++
        ``omni:resetXformStack=1``.
        """
        if getattr(self, "_sync_topic", ""):
            return
        if self._tf_buffer is None or not self._manifest.robot_body_paths:
            return
        root = "base_footprint"
        try:
            self._tf_buffer.lookup_transform(root, root, rclpy.time.Time())
        except Exception:
            # Root frame not published yet (bring-up), or a different root.
            return
        for body, prim_path in self._manifest.robot_body_paths.items():
            prim = self._get_prim(prim_path)
            if prim is None:
                continue
            try:
                tf = self._tf_buffer.lookup_transform(root, body, rclpy.time.Time())
            except Exception:
                continue
            t = tf.transform.translation
            r = tf.transform.rotation
            translation = Gf.Vec3d(float(t.x), float(t.y), float(t.z))
            rotation = Gf.Quatd(float(r.w), float(r.x), float(r.y), float(r.z))
            _apply_xform(prim, translation, rotation, reset_xform_stack=True)
            render_path = self._cam_body_to_render_path.get(body)
            if render_path:
                render_prim = self._get_prim(render_path)
                if render_prim is not None:
                    _apply_xform(render_prim, translation, rotation, reset_xform_stack=True)

    def _on_free_cam_pose(self, msg: PoseStamped) -> None:
        if getattr(self, "_sync_topic", ""):
            return
        if self._stage is None or not self._manifest.free_cam_prim_path:
            return
        p = msg.pose.position
        o = msg.pose.orientation
        translation = Gf.Vec3d(float(p.x), float(p.y), float(p.z))
        rotation = Gf.Quatd(float(o.w), float(o.x), float(o.y), float(o.z))
        prim = self._get_prim(self._manifest.free_cam_prim_path)
        if prim is not None:
            _apply_xform(prim, translation, rotation)

    # -- main loop --------------------------------------------------------

    def _publish_timing(self, step_ms: float, total_ms: float, interval_ms: float) -> None:
        msg = Float64MultiArray()
        dim = MultiArrayDimension()
        dim.label = "timing"
        dim.size = 7
        dim.stride = 7
        msg.layout.dim.append(dim)
        # [step, wait, fetch, map, publish, total, interval]
        msg.data = [step_ms, 0.0, 0.0, 0.0, 0.0, total_ms, interval_ms]
        self._timing_pub.publish(msg)

    def spin(self) -> None:
        period = 1.0 / max(self._render_fps, 1e-3)
        next_tick = time.monotonic()
        while simulation_app.is_running() and rclpy.ok():
            if self._sync_capture is not None and self._sync_render_on_demand:
                # Idle Kit updates may submit viewport/product rendering and
                # contend with native IPC. Only an actual capture owns Kit
                # work, including its existing GPU fences and Writer drains.
                started = time.monotonic()
                rclpy.spin_once(self, timeout_sec=0.01)
                self._sync_capture.process()
                remaining = 0.01 - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
                continue
            rclpy.spin_once(self, timeout_sec=0.0)
            if self._sync_capture is not None:
                self._sync_capture.process()
                self._world.render()
                time.sleep(0.001)
                continue

            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(next_tick - now, period))
                continue

            interval_ms = 0.0 if self._last_frame_time == 0.0 else (now - self._last_frame_time) * 1000.0
            self._last_frame_time = now

            t0 = time.monotonic()
            try:
                self._world.step(render=True)
            except Exception as exc:  # pragma: no cover
                self._drop_count += 1
                self.get_logger().warn(f"world.step failed: {exc}")
                next_tick = now + period
                continue
            step_ms = (time.monotonic() - t0) * 1000.0
            total_ms = step_ms  # Isaac Sim handles publishing asynchronously via OmniGraph

            self._frame_count += 1
            self._publish_timing(step_ms, total_ms, interval_ms)

            next_tick += period
            if next_tick < now:
                next_tick = now + period

    def shutdown(self) -> None:
        capture, self._sync_capture = self._sync_capture, None
        if capture is not None:
            try:
                capture.close()
            except Exception as exc:
                self.get_logger().warn(f"sync capture cleanup failed: {exc}")
        try:
            if self._world is not None:
                self._world.stop()
        except Exception:  # pragma: no cover
            pass


def _strip_ros_args(argv: List[str]) -> List[str]:
    """Return argv without the ROS2 '--ros-args ...' section."""
    if "--ros-args" not in argv:
        return argv
    idx = argv.index("--ros-args")
    return argv[:idx]


def main() -> int:
    argv = list(sys.argv)
    parser = argparse.ArgumentParser(description="Isaac Sim renderer (mirrors genie_sim_render_node).")
    parser.add_argument(
        "--manifest",
        default=os.environ.get("STAGE_MANIFEST", ""),
        help=(
            "Path to the stage manifest JSON. If omitted, the ROS parameter "
            "'stage_manifest' must be set (e.g. via --ros-args -p stage_manifest:=...)."
        ),
    )
    cli_args, _ = parser.parse_known_args(_strip_ros_args(argv)[1:])

    rclpy.init(args=argv)

    extra_params: List[rclpy.parameter.Parameter] = []
    if cli_args.manifest:
        extra_params.append(rclpy.parameter.Parameter("stage_manifest", value=cli_args.manifest))

    node: Optional[IsaacSimRenderNode] = None
    try:
        node = IsaacSimRenderNode()
        if extra_params:
            node.set_parameters(extra_params)
        node.spin()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"[isaacsim_render] fatal: {exc}", file=sys.stderr)
        return 1
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
