# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Lazy Isaac Sim 6.2 sensor factories and collection protocol adapters."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Tuple

import numpy as np


class LidarAdapter:
    """Retain the RTX runtime that owns a lidar's render product."""

    def __init__(self, authoring: Any, *, runtime_factory: Callable[..., Any]) -> None:
        self._authoring = authoring
        self._runtime_factory = runtime_factory
        self._runtime = None

    @property
    def prim_path(self) -> str:
        return str(self._authoring.paths[0])

    @property
    def render_product_path(self) -> str:
        if self._runtime is None:
            raise RuntimeError("Lidar must be initialized before accessing its render product")
        return str(self._runtime.render_product.GetPath())

    def get_render_product_path(self) -> str:
        return self.render_product_path

    def initialize(self) -> None:
        if self._runtime is None:
            self._runtime = self._runtime_factory(self._authoring)


class CameraAdapter:
    """Preserve the collection camera protocol over Isaac Sim 6.2 sensors."""

    _AXES = {
        "usd": np.eye(3, dtype=np.float64),
        "world": np.array(
            [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        "ros": np.diag([1.0, -1.0, -1.0]),
    }

    def __init__(
        self,
        authoring: Any,
        *,
        resolution: Tuple[int, int],
        annotators: Iterable[str] | None = None,
        runtime_factory: Callable[..., Any] | None = None,
        frequency: float | None = None,
    ) -> None:
        self._authoring = authoring
        self._resolution = (int(resolution[0]), int(resolution[1]))
        if min(self._resolution) <= 0:
            raise ValueError("Camera resolution must be positive (width, height)")
        self._frequency = frequency
        self._annotators = list(dict.fromkeys(annotators or []))
        self._runtime_factory = runtime_factory
        self._runtime = None

    @property
    def prim_path(self) -> str:
        return str(self._authoring.paths[0])

    @property
    def render_product_path(self) -> str:
        if self._runtime is None:
            raise RuntimeError("Camera must be initialized before accessing its render product")
        return str(self._runtime.render_product.GetPath())

    @property
    def _render_product_path(self) -> str:
        """Compatibility alias for consumers predating render_product_path."""
        return self.render_product_path

    def initialize(self) -> None:
        if self._runtime is not None:
            return
        runtime_factory = self._runtime_factory
        if runtime_factory is None:
            from isaacsim.sensors.experimental.rtx import CameraSensor  # noqa: PLC0415

            runtime_factory = CameraSensor
        width, height = self._resolution
        self._runtime = runtime_factory(
            self._authoring,
            resolution=(height, width),
            annotators=list(self._annotators),
        )

    def get_render_product_path(self) -> str:
        return self.render_product_path

    def get_frequency(self) -> float:
        if self._frequency is not None:
            return float(self._frequency)
        value = self._authoring.prims[0].GetAttribute("omni:sensor:tickRate").Get()
        return float(value) if value is not None else -1.0

    def get_resolution(self) -> Tuple[int, int]:
        if self._runtime is not None and getattr(self._runtime, "resolution", None) is not None:
            height, width = self._runtime.resolution
            return int(width), int(height)
        return self._resolution

    @staticmethod
    def _as_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "numpy"):
            return np.asarray(value.numpy())
        if hasattr(value, "detach"):
            return np.asarray(value.detach().cpu().numpy())
        return np.asarray(value)

    @classmethod
    def _scalar(cls, value: Any) -> float:
        return float(cls._as_numpy(value).reshape(-1)[0])

    def get_focal_length(self) -> float:
        return self._scalar(self._authoring.camera.get_focal_lengths())

    def set_focal_length(self, value: float) -> None:
        self._authoring.camera.set_focal_lengths(np.asarray([[value]], dtype=np.float32))

    def get_horizontal_aperture(self) -> float:
        horizontal, _ = self._authoring.camera.get_apertures()
        return self._scalar(horizontal)

    def set_horizontal_aperture(self, value: float) -> None:
        self._authoring.camera.set_apertures(horizontal_apertures=np.asarray([[value]], dtype=np.float32))

    def get_vertical_aperture(self) -> float:
        _, vertical = self._authoring.camera.get_apertures()
        return self._scalar(vertical)

    def set_vertical_aperture(self, value: float) -> None:
        self._authoring.camera.set_apertures(vertical_apertures=np.asarray([[value]], dtype=np.float32))

    @classmethod
    def _axes_matrix(cls, camera_axes: str) -> np.ndarray:
        try:
            return cls._AXES[camera_axes]
        except KeyError as exc:
            raise ValueError(f"Unsupported camera_axes {camera_axes!r}; expected one of {tuple(cls._AXES)}") from exc

    @staticmethod
    def _quaternion_to_matrix(quaternion: Any) -> np.ndarray:
        q = np.asarray(quaternion, dtype=np.float64).reshape(4)
        norm = np.linalg.norm(q)
        if norm == 0:
            raise ValueError("orientation quaternion must be non-zero")
        w, x, y, z = q / norm
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _matrix_to_quaternion(matrix: Any) -> np.ndarray:
        m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
        trace = float(np.trace(m))
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2.0
            q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
        else:
            i = int(np.argmax(np.diag(m)))
            if i == 0:
                s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
                q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
            elif i == 1:
                s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
                q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
            else:
                s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
                q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
        q /= np.linalg.norm(q)
        return -q if q[0] < 0 else q

    @classmethod
    def _read_pose(cls, values: tuple[Any, Any], camera_axes: str) -> tuple[np.ndarray, np.ndarray]:
        position = cls._as_numpy(values[0]).reshape(-1, 3)[0].copy()
        usd_quaternion = cls._as_numpy(values[1]).reshape(-1, 4)[0]
        target_rotation = cls._quaternion_to_matrix(usd_quaternion) @ cls._axes_matrix(camera_axes)
        return position, cls._matrix_to_quaternion(target_rotation)

    @classmethod
    def _write_orientation(cls, orientation: Any, camera_axes: str) -> np.ndarray:
        target_rotation = cls._quaternion_to_matrix(orientation)
        usd_rotation = target_rotation @ cls._axes_matrix(camera_axes).T
        return cls._matrix_to_quaternion(usd_rotation).reshape(1, 4)

    def get_world_pose(self, camera_axes: str = "usd") -> tuple[np.ndarray, np.ndarray]:
        return self._read_pose(self._authoring.get_world_poses(), camera_axes)

    def _ensure_pose_edit_ops(self) -> None:
        # New cameras have no pose ops; authored cameras can use a matrix op.
        # Normalize only when editing a pose. The SDK preserves the world pose,
        # while read-only wrapping keeps the authored transform stack intact.
        required = {"xformOp:translate", "xformOp:orient"}
        if any(not required.issubset(prim.GetPropertyNames()) for prim in self._authoring.prims):
            self._authoring.reset_xform_op_properties()

    def set_world_pose(self, position: Any = None, orientation: Any = None, camera_axes: str = "usd") -> None:
        self._ensure_pose_edit_ops()
        self._authoring.set_world_poses(
            positions=None if position is None else np.asarray(position, dtype=np.float32).reshape(1, 3),
            orientations=(
                None if orientation is None else self._write_orientation(orientation, camera_axes).astype(np.float32)
            ),
        )

    def get_local_pose(self, camera_axes: str = "usd") -> tuple[np.ndarray, np.ndarray]:
        return self._read_pose(self._authoring.get_local_poses(), camera_axes)

    def set_local_pose(self, translation: Any = None, orientation: Any = None, camera_axes: str = "usd") -> None:
        self._ensure_pose_edit_ops()
        self._authoring.set_local_poses(
            translations=None if translation is None else np.asarray(translation, dtype=np.float32).reshape(1, 3),
            orientations=(
                None if orientation is None else self._write_orientation(orientation, camera_axes).astype(np.float32)
            ),
        )

    def add_distance_to_image_plane_to_frame(self) -> None:
        self._ensure_annotator("distance_to_image_plane")

    def _ensure_annotator(self, annotator: str) -> None:
        if annotator in self._annotators:
            return
        self._annotators.append(annotator)
        if self._runtime is not None:
            self._runtime.attach_annotators(annotator)

    def get_data(self, annotator: str):
        self._ensure_annotator(annotator)
        self.initialize()
        return self._runtime.get_data(annotator)


def make_camera(
    prim_path: str,
    *,
    resolution: Tuple[int, int],
    frequency: float | None = None,
    annotators: Iterable[str] | None = None,
    **kwargs,
) -> CameraAdapter:
    """Keep width/height order while the experimental runtime uses height/width."""
    from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera  # noqa: PLC0415

    if frequency is not None and "tick_rate" in kwargs:
        raise TypeError("Specify only one of frequency and tick_rate")
    tick_rate = kwargs.pop("tick_rate", frequency)
    # Wrapping an authored camera must preserve its transform operation stack.
    kwargs.setdefault("reset_xform_op_properties", False)
    authoring = RtxCamera(prim_path, tick_rate=tick_rate, **kwargs)
    return CameraAdapter(
        authoring,
        resolution=resolution,
        annotators=annotators,
        runtime_factory=CameraSensor,
        frequency=tick_rate,
    )


def make_lidar(prim_path: str, *, frequency: float | None = None, **kwargs) -> LidarAdapter:
    """Wrap an authored lidar or create one from an explicit SDK configuration."""
    from isaacsim.sensors.experimental.rtx import Lidar, LidarSensor  # noqa: PLC0415

    if frequency is not None and "tick_rate" in kwargs:
        raise TypeError("Specify only one of frequency and tick_rate")
    tick_rate = kwargs.pop("tick_rate", frequency)
    kwargs.setdefault("reset_xform_op_properties", False)
    factory = Lidar.create if "config" in kwargs or "usd_path" in kwargs else Lidar
    return LidarAdapter(factory(prim_path, tick_rate=tick_rate, **kwargs), runtime_factory=LidarSensor)


def make_imu(prim_path: str, **kwargs) -> Any:
    """Use the 6.2 runtime's positional path contract."""
    if kwargs:
        raise TypeError("Configure IMU attributes with IMU.create before make_imu")
    from isaacsim.sensors.experimental.physics import IMUSensor  # noqa: PLC0415

    return IMUSensor(prim_path)


def make_contact_sensor(prim_path: str, **kwargs) -> Any:
    """Use the 6.2 runtime's positional path contract."""
    if kwargs:
        raise TypeError("Configure contact attributes with Contact.create before make_contact_sensor")
    from isaacsim.sensors.experimental.physics import ContactSensor  # noqa: PLC0415

    return ContactSensor(prim_path)


def create_render_product(camera_prim_or_path, resolution: Tuple[int, int]) -> Any:
    """Create a Replicator render product for the camera and resolution."""
    import omni.replicator.core as rep  # noqa: PLC0415

    return rep.create.render_product(camera_prim_or_path, resolution)


def get_annotator(name: str) -> Any:
    """Get a Replicator annotator by name, e.g. rgb, depth, semantic, or instance."""
    import omni.replicator.core as rep  # noqa: PLC0415

    return rep.AnnotatorRegistry.get_annotator(name)


# ---------------------------------------------------------------------------
# RTX settings; startup presets are applied by runtime.init_app.
# ---------------------------------------------------------------------------


def apply_rtx_settings(preset) -> None:
    """Reserved interface; RTX presets are currently applied by runtime.init_app."""
    raise NotImplementedError(
        "The RTX preset is currently configured by sim_backend.runtime.init_app(rtx_mode=...). "
        "Extend this implementation if runtime switching is required."
    )


def set_async_rendering(enable: bool) -> None:
    """Change both Replicator and application async-rendering settings at runtime."""
    from sim_backend.runtime import get_app  # noqa: PLC0415

    app = get_app()
    app._carb_settings.set("/omni/replicator/asyncRendering", enable)
    app._carb_settings.set("/app/asyncRendering", enable)
