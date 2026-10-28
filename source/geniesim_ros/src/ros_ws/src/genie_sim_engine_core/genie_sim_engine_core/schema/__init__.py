# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``genie_sim_engine_core.schema`` — canonical typed views of the
``scene_*.yaml`` and ``manifest.json`` contracts.

Three schemas live here:

* :class:`SceneConfig` (``scene.py``) — human-authored, tolerant, YAML.
  Consumed by every launcher and every engine startup path.
* :class:`Manifest` (``manifest.py``) — machine-emitted, strict,
  versioned, JSON. The cross-language ABI: Python producers write it,
  Python and C++ readers parse it (C++ mirror in
  ``genie_sim_render/include/genie_sim_render/manifest_schema.hpp``).
* :class:`JointParamOverlay` (``joint_params.py``) — human-authored,
  strict, sparse YAML. A *sparse overlay* on the parameters the URDF
  already carries: the URDF as generated is the ``sim-defaults``
  baseline, and an overlay names only the fields some run actually
  identified. The overlay files themselves live in
  ``genie_sim_robot_model``; this package holds only the method, and
  takes paths rather than package names, so there is no dependency
  edge. See ``source/geniesim_ros/docs/JOINT_PARAMS.md``.

All three come with :meth:`from_dict` / :meth:`to_dict` / :meth:`read` /
:meth:`write` and raise :class:`SchemaError` on shape mismatches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .errors import SchemaError, SchemaVersionError
from .manifest import (
    MANIFEST_VERSION,
    CameraManifest,
    LidarBaseLinkT,
    LidarManifest,
    Manifest,
)
from .scene import (
    CAMERA_MODE_RANK,
    CameraEntry,
    CameraIntrinsic,
    CameraSensor,
    CameraTopic,
    ContactPublishSection,
    LidarEntry,
    LidarSensor,
    PinBaseMode,
    Pose,
    PrebakedSource,
    RobotSection,
    RobotSource,
    SceneConfig,
    UrdfSource,
    camera_mode_rank,
)

__all__ = [
    "CAMERA_MODE_RANK",
    "CONVERSION_TARGETS",
    "CameraEntry",
    "CameraIntrinsic",
    "CameraManifest",
    "CameraSensor",
    "CameraTopic",
    "ContactPublishSection",
    "ENGINES",
    "ENGINE_SCOPED_FIELDS",
    "FieldValue",
    "JOINT_PARAMS_SCHEMA_ID",
    "JointParamOverlay",
    "LidarBaseLinkT",
    "LidarEntry",
    "LidarManifest",
    "LidarSensor",
    "MANIFEST_VERSION",
    "Manifest",
    "OVERLAY_ACTUATOR_PLANES",
    "OverlayStack",
    "PHYSX_FIELD_MAP",
    "PLANE_BOUND_FIELDS",
    "PLANE_ISAACSIM_PHYSX",
    "PLANE_MUJOCO_PD",
    "PLANE_UNRECORDED",
    "PhysxConversion",
    "PhysxJointGains",
    "PinBaseMode",
    "Pose",
    "PrebakedSource",
    "Resolution",
    "RobotSection",
    "RobotSource",
    "Scalar",
    "SceneConfig",
    "SchemaError",
    "SchemaVersionError",
    "UrdfSource",
    "camera_mode_rank",
    "convert_to_physx",
    "load_joint_overlays",
    "validate_against_urdf",
]


# ``joint_params`` is exported **lazily** (PEP 562), unlike ``scene`` and
# ``manifest`` above. Two reasons, both concrete:
#
# * It is irrelevant to the hot path. Every launcher and every engine
#   startup imports this package for ``SceneConfig`` / ``Manifest``;
#   none of them touch an overlay unless one is selected.
# * Importing it here eagerly makes
#   ``python -m genie_sim_engine_core.schema.joint_params`` — the
#   documented CLI invocation (JOINT_PARAMS.md section 6) — emit a runpy
#   double-import RuntimeWarning, because the module is already in
#   ``sys.modules`` by the time runpy goes to execute it as ``__main__``.
#
# Resolution is cached into ``globals()`` on first access, so this costs
# one dict lookup once and nothing thereafter.
_LAZY_EXPORTS = {
    "CONVERSION_TARGETS": "joint_params",
    "ENGINES": "joint_params",
    "ENGINE_SCOPED_FIELDS": "joint_params",
    "FieldValue": "joint_params",
    "JOINT_PARAMS_SCHEMA_ID": "joint_params",
    "JointParamOverlay": "joint_params",
    "OVERLAY_ACTUATOR_PLANES": "joint_params",
    "OverlayStack": "joint_params",
    "PHYSX_FIELD_MAP": "joint_params",
    "PLANE_BOUND_FIELDS": "joint_params",
    "PLANE_ISAACSIM_PHYSX": "joint_params",
    "PLANE_MUJOCO_PD": "joint_params",
    "PLANE_UNRECORDED": "joint_params",
    "PhysxConversion": "joint_params",
    "PhysxJointGains": "joint_params",
    "Resolution": "joint_params",
    "Scalar": "joint_params",
    "convert_to_physx": "joint_params",
    "load_joint_overlays": "joint_params",
    "validate_against_urdf": "joint_params",
}

if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDEs only
    from .joint_params import (  # noqa: F401
        ENGINE_SCOPED_FIELDS,
        ENGINES,
        CONVERSION_TARGETS,
        JOINT_PARAMS_SCHEMA_ID,
        OVERLAY_ACTUATOR_PLANES,
        PHYSX_FIELD_MAP,
        PLANE_BOUND_FIELDS,
        PLANE_ISAACSIM_PHYSX,
        PLANE_MUJOCO_PD,
        PLANE_UNRECORDED,
        FieldValue,
        JointParamOverlay,
        OverlayStack,
        PhysxConversion,
        PhysxJointGains,
        Resolution,
        Scalar,
        convert_to_physx,
        load_joint_overlays,
        validate_against_urdf,
    )


def __getattr__(name: str):
    """PEP 562 lazy attribute access for :data:`_LAZY_EXPORTS`."""
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value  # __getattr__ is only consulted on a miss
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
