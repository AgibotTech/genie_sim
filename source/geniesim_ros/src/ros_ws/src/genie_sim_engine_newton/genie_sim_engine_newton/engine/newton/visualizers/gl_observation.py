# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Render mounted RGB views with Newton 1.4 ViewerGL, including plugin meshes.

The public ``ViewerGL.get_frame(render_ui=False)`` API returns top-left RGB
uint8 pixels. All GL calls stay on the viewer/physics thread. ROS imports are
lazy; camera geometry and capture ordering can be tested without ROS or a GPU.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import re
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MountedView:
    name: str
    topic: str
    body: str
    eye: tuple[float, float, float]
    look_at: tuple[float, float, float]
    up: tuple[float, float, float]
    fov_degrees: float
    width: int | None = None
    height: int | None = None
    calibration: dict | None = None


@dataclass(frozen=True)
class GlObservationConfig:
    width: int
    height: int
    views: tuple[MountedView, ...]

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> GlObservationConfig | None:
        raw = params.get("newton_rgb_views_json", "")
        if not raw:
            return None
        rows = json.loads(str(raw))
        if not isinstance(rows, list) or not rows:
            raise ValueError("newton_rgb_views_json must be a nonempty JSON list")
        dimensions = []
        for key in ("newton_rgb_width", "newton_rgb_height"):
            value = params.get(key)
            # Both core._parse_args(-p ...) and --params-file scalar values
            # arrive as strings. Keep strict integer syntax at this boundary.
            if isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value):
                value = int(value)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be an explicit positive integer")
            dimensions.append(value)
        views = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("each RGB view must be an object")
            for key in ("name", "topic", "body"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"RGB view {key} must be nonempty")
            if not row["topic"].startswith("/") or not row["topic"].endswith("/image_raw"):
                raise ValueError("RGB view topic must be an absolute /image_raw topic")
            vectors = []
            for key in ("eye", "look_at", "up"):
                vector = np.asarray(row.get(key), dtype=float)
                if vector.shape != (3,) or not np.isfinite(vector).all():
                    raise ValueError(f"RGB view {key} must contain three finite metres")
                vectors.append(tuple(float(x) for x in vector))
            direction = np.subtract(vectors[1], vectors[0])
            if np.linalg.norm(direction) < 0.05:
                raise ValueError("RGB view eye and look_at must differ by at least 0.05 m")
            if np.linalg.norm(np.cross(direction, vectors[2])) < 1e-8:
                raise ValueError("RGB view up must be nonzero and not parallel to the view direction")
            fov = float(row["fov_degrees"])
            if not np.isfinite(fov) or not 1 < fov < 179:
                raise ValueError("RGB fov_degrees must be between 1 and 179")
            width, height = row.get("width"), row.get("height")
            if (width is None) != (height is None) or any(
                value is not None and (type(value) is not int or value <= 0) for value in (width, height)
            ):
                raise ValueError("RGB view dimensions must be paired positive integers")
            calibration = row.get("calibration")
            if calibration is not None:
                if not isinstance(calibration, dict) or calibration.get("model") not in (
                    "opencvPinhole",
                    "opencvFisheye",
                ):
                    raise ValueError("RGB view calibration requires a supported OpenCV model")
                values = [float(calibration[key]) for key in ("fx", "fy", "cx", "cy")]
                if not np.isfinite(values).all() or min(values[:2]) <= 0:
                    raise ValueError("RGB view calibration requires finite positive focal lengths")
            views.append(MountedView(row["name"], row["topic"], row["body"], *vectors, fov, width, height, calibration))
        for key in ("name", "topic"):
            values = [getattr(view, key) for view in views]
            if len(set(values)) != len(values):
                raise ValueError(f"RGB view {key}s must be unique")
        return cls(*dimensions, tuple(views))


def mounted_camera_pose(body_pose: np.ndarray, view: MountedView) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform mount-local eye/target by Newton's xyz + quaternion XYZW pose."""
    pose = np.asarray(body_pose, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("camera parent must have a finite Newton body pose")
    q = pose[3:]
    length = np.linalg.norm(q)
    if length < 1e-8:
        raise ValueError("camera parent quaternion is zero")
    q = q / length

    def rotate(local: Sequence[float]) -> np.ndarray:
        vector = np.asarray(local, dtype=float)
        uv = np.cross(q[:3], vector)
        return vector + 2 * (q[3] * uv + np.cross(q[:3], uv))

    return pose[:3] + rotate(view.eye), pose[:3] + rotate(view.look_at), rotate(view.up)


def body_indices(paths: Sequence[str], views: Sequence[MountedView]) -> tuple[int, ...]:
    indices = []
    for view in views:
        matches = [index for index, path in enumerate(paths) if path == view.body or path.endswith("/" + view.body)]
        if len(matches) != 1:
            raise ValueError(f"RGB camera parent {view.body!r} resolves to {len(matches)} bodies")
        indices.append(matches[0])
    return tuple(indices)


@contextmanager
def fixed_render_size(viewer: Any, width: int, height: int) -> Iterator[None]:
    """Render/read a real fixed-size FBO, then restore the native window target.

    Newton 1.4 has no public offscreen-size API: Camera dimensions only control
    projection, and get_frame(target_image=...) still reads the renderer's full
    FBO. Its RendererGL render/setup functions use _screen_width/_screen_height;
    resize() instead reads the native window size. Isolate that narrow 1.4
    implementation boundary here so WM events cannot change an in-flight RGB
    readback or its PBO allocation. The native window is never resized.
    """
    renderer = viewer.renderer
    original_resize = renderer.resize
    original_callbacks = renderer._resize_callbacks
    renderer.resize = lambda *_: None
    renderer._resize_callbacks = []
    try:
        renderer._screen_width, renderer._screen_height = width, height
        renderer._setup_frame_buffer()
        viewer._invalidate_pbo()
        viewer.camera.update_screen_size(width, height)
        yield
    finally:
        renderer.resize = original_resize
        renderer._resize_callbacks = original_callbacks
        if viewer.is_running():
            # A WM resize may have arrived while its callbacks were suspended.
            # Rebuild from the CURRENT native size, then replay callbacks once
            # so observer projection/UI and readback buffers agree again.
            native_size = renderer.window.get_size()
            renderer.resize(*native_size)
            viewer._invalidate_pbo()
            viewer.camera.update_screen_size(*renderer.window.get_framebuffer_size())
            for callback in original_callbacks:
                callback(*native_size)


def capture_views(
    viewer: Any,
    config: GlObservationConfig,
    poses: np.ndarray,
    indices: Sequence[int],
    render_scene: Callable[[], None],
) -> tuple[np.ndarray, ...]:
    """Render each actual view from the same stopped state, then restore the UI.

    ``render_scene`` must call begin_frame/log_state/plugin.on_render/end_frame.
    No pixels are returned on partial capture, and the UI camera is restored on
    any failure. The caller must not step physics during this function.
    """
    if any(view.calibration is not None or view.width is not None for view in config.views):
        raise ValueError("calibrated per-camera RGB views require the Isaac Sim snapshot renderer")
    original_camera = viewer.camera
    original_ui = viewer.show_ui
    original_gui = viewer.gui
    images = []
    try:
        with fixed_render_size(viewer, config.width, config.height):
            try:
                viewer.show_ui = False
                # ViewerGL 1.4 routes navigation and UI drawing through gui.
                viewer.gui = None
                for view, index in zip(config.views, indices, strict=True):
                    eye, target, up = mounted_camera_pose(poses[index], view)
                    front = target - eye
                    front /= np.linalg.norm(front)
                    up = up - front * np.dot(up, front)
                    up /= np.linalg.norm(up)

                    class MountedCamera(type(original_camera)):
                        # Preserve measured wrist roll instead of world-up.
                        def get_front(self):
                            return self._as_vec3(front)

                        def get_up(self):
                            return self._as_vec3(up)

                    viewer.camera = MountedCamera(
                        width=config.width, height=config.height, fov=view.fov_degrees, pos=tuple(eye), up_axis="Z"
                    )
                    viewer.camera.look_at(tuple(target))
                    render_scene()
                    if not viewer.is_running():
                        raise RuntimeError("Newton GL window closed during RGB capture")
                    image = np.asarray(viewer.get_frame(render_ui=False).numpy())
                    expected = (config.height, config.width, 3)
                    if image.shape != expected or image.dtype != np.uint8:
                        raise RuntimeError(
                            f"Newton GL RGB frame must be uint8 {expected}, got {image.dtype} {image.shape}"
                        )
                    images.append(np.ascontiguousarray(image).copy())
            finally:
                viewer.camera = original_camera
                viewer.show_ui = original_ui
                viewer.gui = original_gui
    finally:
        if viewer.is_running():
            render_scene()
    return tuple(images)


class GlObservationPublisher:
    """Own image publishers on the existing lifecycle node, never its context."""

    def __init__(self, node: Any, config: GlObservationConfig) -> None:
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image

        self._node = node
        self._message = Image
        self.config = config
        self._publishers = []
        qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        try:
            for view in config.views:
                self._publishers.append(node.create_publisher(Image, view.topic, qos))
        except BaseException:
            self.close()
            raise

    def publish(self, images: Sequence[np.ndarray]) -> None:
        if len(images) != len(self.config.views):
            raise ValueError("RGB capture must contain every configured view")
        # One stamp identifies a complete multi-camera capture of one frozen
        # state. It is wall clock, consistent with the engine's joint/body wire.
        stamp = self._node.get_clock().now().to_msg()
        messages = []
        for view, image in zip(self.config.views, images, strict=True):
            if (
                image.shape != (view.height or self.config.height, view.width or self.config.width, 3)
                or image.dtype != np.uint8
            ):
                raise ValueError("RGB capture has an invalid shape or dtype")
            message = self._message()
            message.header.stamp = stamp
            message.header.frame_id = view.name
            message.height, message.width = image.shape[:2]
            message.encoding = "rgb8"
            message.is_bigendian = False
            message.step = message.width * 3
            message.data = image.tobytes()
            messages.append(message)
        for publisher, message in zip(self._publishers, messages, strict=True):
            publisher.publish(message)

    def close(self) -> None:
        publishers, self._publishers = self._publishers, []
        failure = None
        for publisher in publishers:
            try:
                self._node.destroy_publisher(publisher)
            except Exception as exc:
                failure = failure or exc
        if failure is not None:
            raise failure
