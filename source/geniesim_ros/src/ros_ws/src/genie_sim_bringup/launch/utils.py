# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Common helpers shared by ``minimal.launch.py`` and ``experimental.launch.py``.

This module centralizes the cross-launch utilities so that both launch files
stay focused on their distinct physics backend wiring while sharing the
identical scaffolding for argument parsing, config resolution and node
construction.

Anything here is launch-file scope only (executed by ros2 launch in the host
Python interpreter); it does NOT run inside the spawned nodes.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import yaml
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# ---------------------------------------------------------------------------
# Console colors
# ---------------------------------------------------------------------------

MSG_COLOR = "\033[34m"
ERR_COLOR = "\033[31m"
RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Robot-description package defaults
#
# A scene yaml can point at any ROS package that ships the same xacro entry
# point as the default via ``robot.robot_source.package`` and (optionally)
# ``robot.robot_source.urdf.xacro_entry``.  These constants mirror the
# canonical defaults used by ``assemble/_robot_impl.py``
# (DEFAULT_ROBOT_MODEL_PACKAGE / DEFAULT_ROBOT_XACRO_ENTRY in that file);
# keep them in sync so the
# assemble pipeline and the runtime ``robot_description`` builder resolve
# the same xacro tree.
# ---------------------------------------------------------------------------

DEFAULT_ROBOT_MODEL_PACKAGE = "genie_sim_robot_model"
DEFAULT_ROBOT_XACRO_ENTRY = "xacro/robot.xacro"


# ---------------------------------------------------------------------------
# Package share root — resolved lazily.
#
# ``PACKAGE_PATH_SELF`` is the share dir of ``genie_sim_bringup`` itself,
# exposed via PEP 562 ``__getattr__`` so it's only computed when something
# actually reads it.
# ---------------------------------------------------------------------------


def __getattr__(name: str):  # PEP 562 module-level lazy attribute resolver.
    if name == "PACKAGE_PATH_SELF":
        value = Path(get_package_share_directory("genie_sim_bringup"))
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value  # cache for subsequent accesses
    return value


def _pkg_self() -> Path:
    """In-module accessor for ``PACKAGE_PATH_SELF``.

    PEP 562 ``__getattr__`` only fires for *external* module-attribute
    access (``lu.PACKAGE_PATH_SELF``); unqualified references from inside
    this module bypass it and would raise ``NameError`` until something
    has already populated the global. Routing in-module callers through
    this helper guarantees the lazy resolution path is taken regardless
    of access order.
    """
    return __getattr__("PACKAGE_PATH_SELF")


# ---------------------------------------------------------------------------
# Config-file resolution
# ---------------------------------------------------------------------------

_source_config_dir: Path | None = None
_source_config_dir_searched: bool = False


def _find_source_config_dir() -> Path | None:
    """Walk up from the install share directory to locate the source config dir.

    In a standard colcon workspace::

        <ws>/install/<pkg>/share/<pkg>/   ← _pkg_self()
        <ws>/src/<pkg>/config/            ← source config (what we want)

    Returns ``None`` when running from a pure-install layout.
    """
    global _source_config_dir, _source_config_dir_searched
    if _source_config_dir_searched:
        return _source_config_dir
    _source_config_dir_searched = True
    share = _pkg_self()
    for ancestor in share.parents:
        candidate = ancestor / "src" / "genie_sim_bringup" / "config"
        if candidate.is_dir():
            _source_config_dir = candidate
            return candidate
    return None


def resolve_bringup_config_file(raw: str) -> str:
    """Resolve a path to an existing file under ``genie_sim_bringup/config/`` or CWD.

    Search order (per name variant, in order):
      1. The raw value as given (absolute or cwd-relative).
      2. ``<bringup_share>/config/<raw>``
      3. ``<bringup_share>/config/<basename(raw)>``
      4. ``<bringup_source>/config/<raw>``          (source workspace, if found)
      5. ``<bringup_source>/config/<basename(raw)>``

    If ``raw`` does not already end in ``.yaml``/``.yml``, the search is
    repeated with each of those extensions appended. This lets callers pass a
    bare stem (e.g. ``scene:=scene_flat_acone``) and have it resolve to
    ``<bringup_share>/config/scene_flat_acone.yaml``.

    Returns the resolved absolute path as ``str``, or ``""`` if no candidate exists.
    """
    raw = raw.strip()
    if not raw:
        return ""
    name_variants: List[str] = [raw]
    if not raw.lower().endswith((".yaml", ".yml")):
        name_variants.extend([raw + ".yaml", raw + ".yml"])
    source_config = _find_source_config_dir()
    for name in name_variants:
        name_path = Path(name).expanduser()
        candidates = [
            name_path if name_path.is_absolute() else (Path.cwd() / name_path).resolve(),
            (_pkg_self() / "config" / name).resolve(),
            (_pkg_self() / "config" / name_path.name).resolve(),
        ]
        if source_config is not None:
            candidates.append((source_config / name).resolve())
            candidates.append((source_config / name_path.name).resolve())
        for cand in candidates:
            if cand.is_file():
                return str(cand)
    return ""


class _IncludeLoader(yaml.SafeLoader):
    """SafeLoader that resolves the ``!include <path>`` tag.

    Path is either absolute or relative to the *including* file's
    directory (matches how humans read ``!include ../shared.yaml``).
    Included files parse under this same loader class, so nested
    ``!include`` chains work; a per-load ``_seen`` set guards cycles.
    Everything else stays SafeLoader — no arbitrary Python
    construction, no eval.

    ``compose_node`` is overridden so ``!include`` scalars are
    substituted with the target file's *node tree* during
    composition, not converted to a Python value at construction
    time. This is what makes ``<<: !include base.yaml`` work: PyYAML
    processes merge keys during construction, before scalar
    constructors run, so if the merge value is still a
    tagged-scalar node the merge fails with "expected a mapping".
    Substituting at compose time hands the merge an actual
    ``MappingNode`` and everything just works.
    """

    _seen: set = set()

    def compose_node(self, parent, index):
        node = super().compose_node(parent, index)
        if isinstance(node, yaml.ScalarNode) and node.tag == "!include":
            ref = str(node.value).strip()
            if not ref:
                raise yaml.constructor.ConstructorError(
                    None, None, "!include requires a non-empty path", node.start_mark
                )
            base_dir = os.path.dirname(os.path.abspath(self.name)) if getattr(self, "name", None) else os.getcwd()
            resolved = ref if os.path.isabs(ref) else os.path.join(base_dir, ref)
            resolved_abs = os.path.abspath(resolved)
            if not os.path.isfile(resolved_abs):
                raise yaml.constructor.ConstructorError(
                    None,
                    None,
                    f"!include target does not exist: {ref!r} (resolved to {resolved_abs!r})",
                    node.start_mark,
                )
            if resolved_abs in _IncludeLoader._seen:
                raise yaml.constructor.ConstructorError(
                    None, None, f"!include cycle detected at {resolved_abs!r}", node.start_mark
                )
            _IncludeLoader._seen.add(resolved_abs)
            try:
                with open(resolved_abs, encoding="utf-8") as f:
                    sub_loader = _IncludeLoader(f)
                    sub_loader.name = resolved_abs
                    try:
                        sub_node = sub_loader.get_single_node()
                    finally:
                        sub_loader.dispose()
                return sub_node if sub_node is not None else node
            finally:
                _IncludeLoader._seen.discard(resolved_abs)
        return node


def load_launcher_yaml(path: str) -> Tuple[dict, dict, str]:
    """Split a launcher YAML into ``(launcher_section, ros_params_dict, ros_params_path)``.

    The YAML at ``path`` may contain:
      * a top-level ``launcher:`` section — consumed by the launch file (physics
        engine selector, industrial bridge, renders list). NEVER forwarded to
        rclcpp.
      * per-node ``<node_name>: { ros__parameters: {...} }`` sections —
        forwarded verbatim to rclcpp.

    Cross-file composition uses the ``!include <path>`` tag (custom
    :class:`_IncludeLoader`) — path is absolute or relative to the
    including file's directory. Combine with YAML's built-in merge
    key to fold a base document into an overlay:

    .. code-block:: yaml

        # launcher_ovrtx_mujoco.yaml
        <<: !include launcher_mujoco.yaml
        # per-render node blocks the base doesn't have
        render_ovrtx:
          ros__parameters:
            plugin: [genie_sim_render/RosImagePublisherPlugin]

    ``<<:`` merges *shallowly*: nested keys under ``launcher:`` /
    ``genie_sim_engine.ros__parameters`` do NOT recurse. To change a
    nested value the overlay must re-emit that whole sub-mapping.

    The ``launcher:`` key is stripped from the second return value and the
    remaining dict is written to a temp YAML, whose path is returned as the
    third value (empty string if there are no per-node sections to forward).
    """
    if not path:
        return {}, {}, ""
    with open(path, encoding="utf-8") as f:
        full = yaml.load(f, Loader=_IncludeLoader) or {}
    launcher_section = full.pop("launcher", {}) if isinstance(full, dict) else {}
    if not isinstance(launcher_section, dict):
        launcher_section = {}
    if not full:
        return launcher_section, {}, ""
    fd, tmp_path = tempfile.mkstemp(prefix="launcher_params_", suffix=".yaml")
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        yaml.safe_dump(full, out, default_flow_style=False, sort_keys=False)
    return launcher_section, full, tmp_path


# ---------------------------------------------------------------------------
# Scene YAML robot.robot_source resolution
# ---------------------------------------------------------------------------
#
# Single signal: PRESENCE of the ``robot.robot_source.urdf`` key decides
# everything (the value may be an empty mapping ``{}`` once the only
# previously-required nested field, ``package``, has been hoisted to the
# flat ``robot_source.package`` position).
#   - key present  -> URDF route: the launch composer runs xacro -> URDF,
#                     RSP publishes it on /robot_description, assemble_robot
#                     subscribes to the same topic and converts URDF -> USD,
#                     and assemble_scene reads the staged USD. Gripper
#                     override is suppressed.
#   - key absent   -> Legacy USD route: pre-baked ``assets/robot/<robot_name>/robot.usda``
#                     is loaded as-is, RViz/RSP uses the legacy publisher.

_SCENE_ROBOT_PARAM_KEYS = ("robot_model", "arm", "body", "gripper")


def _pick_robot_source_str(robot_source: dict, key: str) -> str:
    """Read a string field from ``robot_source`` or nested ``robot_source.urdf``."""
    urdf = robot_source.get("urdf") if isinstance(robot_source.get("urdf"), dict) else {}
    for container in (robot_source, urdf):
        if key not in container:
            continue
        raw = container.get(key)
        if raw is None:
            continue
        s = str(raw).strip()
        if s:
            return s
    return ""


def resolve_robot_params_from_scene(
    robot_source: dict,
    *,
    robot_model: str,
    body: str,
    arm: str,
    gripper: str,
) -> dict[str, str]:
    """Merge launch CLI defaults with ``robot.robot_source`` (scene wins when non-empty).

    Thin BC shim over
    :meth:`genie_sim_engine_core.schema.scene.SceneConfig.resolve_robot_params`;
    new call sites should read the typed schema directly.
    """
    from genie_sim_engine_core.schema.scene import (
        RobotSection as _RS,
        RobotSource as _SchemaRobotSource,
        SceneConfig as _SchemaSceneConfig,
    )

    rs = _SchemaRobotSource.from_dict(robot_source if isinstance(robot_source, dict) else {})
    scene = _SchemaSceneConfig(robot=_RS(robot_source=rs))
    return scene.resolve_robot_params(
        robot_model=robot_model,
        body=body,
        arm=arm,
        gripper=gripper,
    )


def uses_urdf_assemble_pipeline(robot_source: dict) -> bool:
    """True when the scene declares a ``robot_source.urdf`` block (URDF->USD assemble path).

    The mere PRESENCE of the ``urdf`` key selects the URDF route. The value
    may be an empty mapping ``{}`` for scenes that don't need any nested
    URDF-route options beyond the flat ``package`` / ``robot_model`` / ``arm``
    fields already on ``robot_source``.

    Which ROS package provides the xacro is decided per-scene via
    ``robot_source.package`` (default :data:`DEFAULT_ROBOT_MODEL_PACKAGE`).
    """
    if not isinstance(robot_source, dict):
        return False
    return "urdf" in robot_source


def needs_assemble_robot(robot_source: dict) -> bool:
    """Whether the urdf_to_usd stage should convert URDF->USD for this scene.

    Returns True when the URDF assemble pipeline is used AND the staged
    URDF→USD product is the canonical physics input for ``assemble_scene``.

    A ``robot.robot_source.prebaked.usda`` normally makes that product
    unnecessary: the prebaked USDA is the canonical robot input for both
    physics and rendering, so the URDF→USD stage is skipped entirely (while
    xacro processing — ``uses_urdf_assemble_pipeline`` /
    ``build_robot_description`` — still runs to populate ``/robot_description``
    for RSP / MoveIt).

    The exception is ``prebaked.visual_only: true``: there the prebaked USDA
    drives rendering only, and physics still needs the staged URDF→USD
    product, so this stage must run. ``_scene_impl.py`` mirrors the same
    ``visual_only`` split when it picks ``robot_usda`` vs ``robot_visual``.
    """
    if not uses_urdf_assemble_pipeline(robot_source):
        return False
    prebaked = robot_source.get("prebaked") if isinstance(robot_source, dict) else None
    if isinstance(prebaked, dict) and str(prebaked.get("usda") or "").strip():
        # visual_only prebaked drives rendering only; physics still needs the
        # staged URDF→USD product, so the stage must run.
        return bool(prebaked.get("visual_only", False))
    return True


def build_urdf_xacro_mappings(robot_source: dict, resolved: dict[str, str]) -> dict[str, str]:
    """Xacro mappings for the URDF-assemble pipeline (default package: ``genie_sim_robot_model``)."""
    mappings: dict[str, str] = {"robot_model": resolved["robot_model"]}
    if resolved.get("arm"):
        mappings["arm"] = resolved["arm"]
    if resolved.get("body"):
        mappings["body"] = resolved["body"]
    gripper = resolved.get("gripper", "")
    if gripper and gripper.lower() != "none":
        mappings["gripper"] = gripper
    urdf = robot_source.get("urdf") if isinstance(robot_source.get("urdf"), dict) else {}
    for k, v in urdf.items():
        if k in {*_SCENE_ROBOT_PARAM_KEYS, "gripper", "mimic", "package", "xacro_entry"}:
            continue
        if isinstance(v, str) and v.strip():
            mappings.setdefault(k, v.strip())
    return mappings


# ---------------------------------------------------------------------------
# Shared launch-argument declarations
# ---------------------------------------------------------------------------


def common_declared_arguments() -> List[DeclareLaunchArgument]:
    """Return the launch-argument set shared by minimal and experimental launches.

    ``robot_model`` / ``body`` / ``arm`` / ``gripper`` defaults match the
    G2 lineage but are not enumerated — the scene yaml's
    ``robot.robot_source`` block is the authoritative source for those
    fields; the launch args exist as overrides only.
    """
    args: List[DeclareLaunchArgument] = []
    args.append(
        DeclareLaunchArgument(
            name="use_sim_time",
            default_value="false",
            choices=["true", "false"],
            description="Enable ROS sim time (/clock) for all nodes",
        )
    )
    args.append(
        DeclareLaunchArgument(
            "robot_model",
            default_value="G2",
            description="Robot model (override; scene yaml's robot.robot_source.robot_model wins)",
        )
    )
    args.append(
        DeclareLaunchArgument(
            "body",
            description="Robot body model (override; scene yaml's robot.robot_source.body wins)",
            default_value="t2",
        )
    )
    args.append(
        DeclareLaunchArgument(
            "arm",
            description="Arm model (override; scene yaml's robot.robot_source.arm wins)",
            default_value="crs",
        )
    )
    args.append(
        DeclareLaunchArgument(
            "gripper",
            description="Gripper model (override; scene yaml's robot.robot_source.gripper wins)",
            default_value="none",
        )
    )
    args.append(
        DeclareLaunchArgument(
            "rviz_config_file",
            description="RViz config file (absolute path) to use when launching rviz.",
            default_value="",
        )
    )
    args.append(
        DeclareLaunchArgument(
            name="debug",
            default_value="true",
            choices=["true", "false"],
            description="Debug urdf file use joint state publisher gui",
        )
    )
    args.append(
        DeclareLaunchArgument(
            name="teleop",
            default_value="false",
            choices=["true", "false"],
            description="with teleop",
        )
    )
    args.append(
        DeclareLaunchArgument(
            name="navigation",
            default_value="false",
            choices=["true", "false"],
            description="with navigation (omni cmd_twist via servo node)",
        )
    )
    args.append(
        DeclareLaunchArgument(
            name="navigation_mode",
            default_value="move_base",
            choices=["move_base", "clothoid"],
            description="navigation mode: move_base or clothoid",
        )
    )
    args.append(
        DeclareLaunchArgument(
            name="launcher_config",
            default_value="launcher_physx.yaml",
            description=(
                "ROS 2 parameter YAML keyed by node name. " "See share/genie_sim_bringup/config/launcher*.yaml."
            ),
        )
    )
    args.append(
        DeclareLaunchArgument(
            name="tf_prefix",
            default_value="",
            description="TF prefix for the robot",
        )
    )
    args.append(
        DeclareLaunchArgument(
            name="fake_slam",
            default_value="true",
            description="fake_slam",
        )
    )
    args.append(
        DeclareLaunchArgument(
            name="remap_tf",
            default_value="false",
            description="remap_tf",
        )
    )
    args.append(
        DeclareLaunchArgument(
            name="rviz",
            default_value="false",
            description="Show RViz",
        )
    )
    args.append(
        DeclareLaunchArgument(
            name="log_level",
            default_value="info",
            description="ROS 2 log level for all nodes (debug, info, warn, error, fatal)",
        )
    )
    args.append(
        DeclareLaunchArgument(
            name="interaction_tools",
            default_value="false",
            choices=["true", "false"],
            description="Enable interactive markers for free-joint scene objects",
        )
    )
    return args


# ---------------------------------------------------------------------------
# Substitution helpers (run inside ``parse_args(context)``)
# ---------------------------------------------------------------------------


def perform(context, key: str) -> str:
    """Shorthand for ``context.perform_substitution(LaunchConfiguration(key))``."""
    return context.perform_substitution(LaunchConfiguration(key))


def _cli_explicit(key: str) -> bool:
    """True if ``key:=…`` was typed on the ros2 launch command line.

    Used to distinguish "user explicitly set this on CLI" from "user took
    the DeclareLaunchArgument default". Lets a launcher-yaml's
    ``ros__parameters`` block carry the default while the CLI override
    still wins when present:

        physics_params: list = []
        if plugins_params_path:
            physics_params.append(plugins_params_path)              # yaml defaults
        cli_overrides = {
            key: value for key, value in {…}.items() if _cli_explicit(key)
        }
        physics_params.append({...always_set_keys, **cli_overrides}) # CLI wins

    ROS 2 applies later entries in the ``parameters=`` list ON TOP OF
    earlier ones, so this pattern gives yaml → DeclareLaunchArgument →
    CLI precedence (low → high).
    """
    import sys

    needle = f"{key}:="
    for arg in sys.argv:
        if arg.startswith(needle):
            return True
    return False


# ---------------------------------------------------------------------------
# Node factories (shared)
# ---------------------------------------------------------------------------


JOINT_STATES_TOPIC = "/joint_states"


def make_robot_state_publisher_node(
    *,
    robot_description: str,
    publish_frequency: float,
    tf_prefix: str,
    common_param: dict,
    remap_tf: bool,
    ros_log_args: Iterable[str],
) -> Node:
    remappings = [
        ("/joint_states", JOINT_STATES_TOPIC),
        ("/robot_description", "/robot_description"),
    ]
    if remap_tf:
        remappings.append(("/tf", "/tf_sim"))
    return Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description,
                "publish_frequency": publish_frequency,
                "frame_prefix": tf_prefix,
                "ignore_timestamp": True,
                **common_param,
            }
        ],
        remappings=remappings,
        arguments=list(ros_log_args),
    )


def resolve_rviz_config(rviz_config_file: str, robot_model: str) -> Path:
    """Resolve the RViz config path with the same fallback chain as both launches.

    Falls back to ``<bringup>/rviz/<robot_model>.rviz`` then ``view_robot.rviz``.
    """
    if rviz_config_file == "":
        candidate = _pkg_self() / f"rviz/{robot_model}.rviz"
    else:
        candidate = Path(rviz_config_file)
    if not candidate.exists() or candidate.suffix != ".rviz":
        print(f"{MSG_COLOR}Can not find rviz config file: {candidate}, use default config file{RESET}")
        candidate = _pkg_self() / "rviz/view_robot.rviz"
    return candidate


def make_rviz_node(*, rviz_config_file: Path, common_param: dict, ros_log_args: Iterable[str]) -> Node:
    return Node(
        package="rviz2",
        executable="rviz2",
        name="robot_state",
        output="both",
        arguments=["-d", str(rviz_config_file), *ros_log_args],
        parameters=[common_param],
    )


def make_chassis_controller_node(*, chassis: str, common_param: dict, ros_log_args: Iterable[str]) -> Node:
    """Standalone four-wheel-steering servo node.

    Loads the rclcpp::Node form of the servo from
    ``genie_sim_controllers`` — same ``ServoCore`` as the
    ``ChassisServoController`` controller-plugin form, just driven by
    a wall-timer instead of the controller_manager's update loop. Use
    this when the bringup graph doesn't include a controller_manager
    (teleop / navigation / dev rigs). For the ros2_control path, the
    controller plugin is spawned from ``genie_sim_moveit/launch/wbc.launch.py``
    instead — same parameter schema (the shipped yamls use ``/**`` so
    a single file drives both forms).

    ``chassis`` selects the tuning file: ``4wd.yaml`` (four driven +
    steered corners, full 4WD/4WS) or ``2wd.yaml`` (diagonal 2WD, two
    driven corners + two passive casters). Scene yamls declare the
    topology via ``robot.robot_source.chassis``; default is ``4wd``.
    """
    chassis_config_name = f"{chassis}.yaml"
    chassis_controller_config = os.path.join(
        get_package_share_directory("genie_sim_controllers"),
        "config",
        chassis_config_name,
    )
    return Node(
        package="genie_sim_controllers",
        executable="genie_sim_chassis_servo_node",
        name="servo_node",
        parameters=[chassis_controller_config, common_param],
        output="both",
        arguments=list(ros_log_args),
    )


def make_teleop_nodes(*, publish_frequency: float, common_param: dict, ros_log_args: Iterable[str]) -> List[Node]:
    return [
        Node(
            package="genie_sim_controllers",
            executable="teleop_joy.py",
            name="geniesim_rt_teleop",
            parameters=[common_param],
            remappings=[("/wheel_command", "/pnc/chassis_joint_cmd")],
            output="both",
            arguments=list(ros_log_args),
        ),
        Node(
            package="joy",
            executable="game_controller_node",
            name="geniesim_rt_joy",
            output="both",
            parameters=[{"autorepeat_rate": publish_frequency, **common_param}],
            arguments=list(ros_log_args),
        ),
    ]


def make_navigation_node(*, navigation_mode: str, common_param: dict, ros_log_args: Iterable[str]) -> Node:
    return Node(
        package="genie_sim_planning",
        executable="simple_navigation.py",
        name="geniesim_rt_navigation",
        parameters=[{"mode": navigation_mode, **common_param}],
        output="both",
        arguments=list(ros_log_args),
    )


def make_industrial_bridge_node(
    *, industrial_bridge: str, common_param: dict, remap_tf: bool, ros_log_args: Iterable[str]
) -> Node:
    return Node(
        package="genie_sim_bringup_industrial",
        executable=industrial_bridge,
        name=Path(industrial_bridge).stem,
        output="both",
        parameters=[{**common_param, "remap_tf": remap_tf}],
        arguments=list(ros_log_args),
    )


# ---------------------------------------------------------------------------
# Physics-engine selection (shared launcher-YAML schema)
# ---------------------------------------------------------------------------
#
# Schema:
#
#   launcher:
#     physics:
#       engine: genie_sim_engine_isaacsim   # selector key (default if absent)
#       engines:
#         genie_sim_engine_isaacsim:
#           package:    genie_sim_engine_isaacsim
#           executable: genie_sim_engine_isaacsim_node.py
#           name:       genie_sim_engine
#       industrial_bridge: ""              # optional bridge executable name
#
# ``resolve_physics_engine`` returns package/executable/name metadata for
# the selected physics engine. Engines may optionally define
# ``executable_gui``; callers should use it only for non-headless launches.
# the active engine. When the YAML omits ``engine`` (or the whole
# ``engines.<engine>`` block) we fall back to a sensible default for the
# launch file in question — the caller passes ``default_*`` kwargs.
#
# Why the indirection: the physics-backend ``Node(package=…,
# executable=…)`` is chosen entirely from the launcher YAML, so swapping
# backends — or adding one in a downstream distribution — is a YAML edit
# rather than a launch-file edit.

DEFAULT_PHYSICS_ENGINES: dict[str, dict[str, str]] = {
    "genie_sim_engine_isaacsim": {
        "package": "genie_sim_engine_isaacsim",
        "executable": "genie_sim_engine_isaacsim_node.py",
        "name": "genie_sim_engine",
    },
}


def resolve_physics_engine(
    launcher_section: dict,
    *,
    default_engine: str,
) -> Tuple[str, str, str, str, str]:
    """Resolve (engine, package, executable, executable_gui, name) from the launcher YAML.

    ``launcher_section`` is the dict returned by :func:`load_launcher_yaml`
    (its first element). The physics block is read at
    ``launcher_section["physics"]``; the active engine's per-engine block is
    read at ``launcher_section["physics"]["engines"][<engine>]``.

    Resolution order (per (package, executable, name) field):
      1. ``physics.engines.<engine>.<field>`` from the YAML if non-empty,
      2. :data:`DEFAULT_PHYSICS_ENGINES[<engine>][<field>]` if known,
      3. raise :class:`ValueError` (the YAML mentions an engine we have no
         defaults for and didn't override every required field).

    The ``engine`` selector itself defaults to ``default_engine`` when the
    YAML is silent. This means launches that ship a bare
    ``launcher_physx.yaml`` keep the same physics node they always had.
    """
    physics_section = launcher_section.get("physics") or {}
    if not isinstance(physics_section, dict):
        physics_section = {}
    engine = str(physics_section.get("engine") or "").strip() or default_engine

    engines_section = physics_section.get("engines") or {}
    if not isinstance(engines_section, dict):
        engines_section = {}
    engine_cfg = engines_section.get(engine) or {}
    if not isinstance(engine_cfg, dict):
        engine_cfg = {}

    defaults = DEFAULT_PHYSICS_ENGINES.get(engine, {})

    def _pick(field: str) -> str:
        from_yaml = str(engine_cfg.get(field) or "").strip()
        if from_yaml:
            return from_yaml
        return str(defaults.get(field, "")).strip()

    package = _pick("package")
    executable = _pick("executable")
    executable_gui = str(engine_cfg.get("executable_gui") or defaults.get("executable_gui", "") or "").strip()
    name = _pick("name")
    missing = [k for k, v in (("package", package), ("executable", executable), ("name", name)) if not v]
    if missing:
        raise ValueError(
            f"launcher.physics.engines.{engine!r} is missing required field(s) {missing}; "
            f"either fill them in the YAML or pick one of the built-in engines: "
            f"{sorted(DEFAULT_PHYSICS_ENGINES.keys())}"
        )
    return engine, package, executable, executable_gui, name


def resolve_physics_scene(launcher_section: dict, engine: str) -> str:
    """Return the scene file path for the given physics engine from the launcher YAML.

    Looks under ``launcher.physics.engines.<engine>.scene``. Returns empty string
    if the engine has no scene entry (caller decides the fallback).
    """
    if not isinstance(launcher_section, dict):
        return ""
    physics_section = launcher_section.get("physics") or {}
    if not isinstance(physics_section, dict):
        return ""
    engines_section = physics_section.get("engines") or {}
    if not isinstance(engines_section, dict):
        return ""
    engine_cfg = engines_section.get(engine) or {}
    if not isinstance(engine_cfg, dict):
        return ""
    return str(engine_cfg.get("scene") or "").strip()


def resolve_prebaked_pipeline(robot_source: dict) -> dict | None:
    """Return the prebaked-assemble pipeline binding from scene yaml, or None.

    Reads ``robot.robot_source.prebaked.pipeline``. Same shape as
    ``launcher.physics.engines.<key>``: ``package`` / ``executable`` /
    ``name``. Returns a dict with those three fields (all non-empty) when
    the block is set, or ``None`` when ``prebaked`` is absent / empty /
    has no ``pipeline`` sub-block — caller falls through to the standard
    URDF->USD assemble pipeline in that case.

    Raises :class:`ValueError` when ``prebaked.pipeline`` is set but
    missing required fields, so a half-configured scene yaml fails loud
    instead of silently falling back to the wrong path.
    """
    if not isinstance(robot_source, dict):
        return None
    prebaked = robot_source.get("prebaked")
    if not isinstance(prebaked, dict) or not prebaked:
        return None
    pipeline = prebaked.get("pipeline")
    if not isinstance(pipeline, dict) or not pipeline:
        return None
    out = {k: str(pipeline.get(k) or "").strip() for k in ("package", "executable", "name")}
    missing = [k for k, v in out.items() if not v]
    if missing:
        raise ValueError(
            f"robot.robot_source.prebaked.pipeline is missing required field(s) {missing}; "
            f"set package/executable/name in the scene yaml."
        )
    return out


def sanitized_usd_env() -> dict:
    """Scrub OVRTX-injected USD plugin paths from the environment.

    Importing ``ovrtx`` writes its renamed-USD plugin dir into
    ``PXR_PLUGINPATH_NAME``; if those paths leak into ``assemble_scene``
    (which uses stock OpenUSD) the result is ``TfType::AddAlias`` collisions
    on ``ParticleField*``.  We strip any path containing ``ovrtx`` from the
    inherited variable (and clear the renamed key for good measure) before
    spawning child processes.
    """
    sep = os.pathsep
    sanitized: dict = {}
    for key in ("PXR_PLUGINPATH_NAME",):
        raw = os.environ.get(key, "")
        if not raw:
            continue
        kept = [p for p in raw.split(sep) if p and "ovrtx" not in p.lower()]
        sanitized[key] = sep.join(kept)
    sanitized["OV_PXR_PLUGINPATH_2511"] = ""
    return sanitized


# ---------------------------------------------------------------------------
# Assemble pipeline
# ---------------------------------------------------------------------------


def make_assemble_pipeline(
    *,
    scene_resolved: str,
    stage_dir: str,
    assets_folder: str,
    always_regenerate_robot_usd: bool,
    runtime_nodes: list,
    assemble_robot_artefacts: list[str] | None = None,
    physics_engine: str = "isaac_physx",
    urdf_text: str | None = None,
    pre_runtime_hook: dict | None = None,
    launcher_config: str = "",
    assemble_style: str = "",
    assemble_pipeline: dict | None = None,
) -> Tuple[Any, Any | None, Any | None, Any]:
    """Build the assemble pipeline as a single core-CLI invocation.

    Backend stages belong to a pipeline registered at
    :mod:`genie_sim_engine_core.assemble`. This helper emits a single
    ``ExecuteProcess`` that dispatches through
    ``python -m genie_sim_engine_core.assemble``; the pipeline runner
    inside runs each stage, honours per-stage ``skip_reason`` (e.g.
    ``UrdfToUsdStage`` self-skips when ``robot.robot_source.urdf`` is
    absent), and writes a single ``manifest.json``.

    Cache-hit for a configured robot stage is preserved via the CLI's
    ``--skip-stage <assemble_pipeline.robot_stage>`` flag: this helper computes the same
    ``need_assemble_robot`` decision as before (missing artefacts /
    URDF-hash mismatch / ``always_regenerate`` override) and, on a
    cache hit, passes the skip flag so the pipeline runner bypasses
    that stage.

    Returns
    -------
    ``(assemble_proc, None, None, gate_runtime)``

    The 2nd and 3rd return values are always ``None`` in the migrated
    form; they remain in the signature so callers that unpack the
    tuple (``a_scene, a_robot, gate_scene, gate_runtime = ...``) keep
    working during the migration window.

    Caller pattern (unchanged shape)::

        a_scene, a_robot, gate_scene, gate_runtime = \\
            lu.make_assemble_pipeline(...)

        if a_robot is not None:            # never true after commit 5
            return [a_robot, gate_scene, gate_runtime]
        return [a_scene, gate_runtime]     # this is the always-taken path
    """
    from launch.actions import ExecuteProcess, RegisterEventHandler  # noqa: F401
    from launch.event_handlers import OnProcessExit

    # Launcher-swap detection: read the previous manifest (if any) and
    # compare its ``launcher_config`` stamp against this launch's
    # value. On mismatch, force UrdfToUsdStage to re-run so the stage
    # dir doesn't carry launcher-conditional artefacts from a
    # different launcher (v2 protocol: robot.body_paths, backend-
    # specific USD conversions, etc.). Missing / empty stamp on the
    # old manifest is treated as mismatch — legacy bakes trigger one
    # regen and thereafter cache correctly.
    launcher_config_mismatch = False
    prior_launcher = ""
    manifest_path = os.path.join(stage_dir, "manifest.json")
    if launcher_config and os.path.isfile(manifest_path):
        try:
            with open(manifest_path) as fh:
                prior = json.load(fh)
            prior_launcher = str(prior.get("launcher_config") or "")
        except (OSError, json.JSONDecodeError):
            prior_launcher = ""
        launcher_config_mismatch = prior_launcher != launcher_config
        if launcher_config_mismatch:
            if prior_launcher:
                print(
                    f"{MSG_COLOR}[assemble_pipeline] launcher_config changed "
                    f"({prior_launcher!r} -> {launcher_config!r}); "
                    f"rebuilding stage dir{RESET}"
                )
            else:
                print(
                    f"{MSG_COLOR}[assemble_pipeline] existing manifest has no "
                    f"launcher_config stamp; rebuilding stage dir under "
                    f"{launcher_config!r}{RESET}"
                )

    # Cache-hit decision (unchanged) — controls whether the
    # UrdfToUsdStage runs or gets --skip-stage'd.
    need_assemble_robot = False
    if assemble_robot_artefacts is not None:
        missing = [p for p in assemble_robot_artefacts if not os.path.isfile(p)]
        urdf_hash_mismatch = False
        if urdf_text:
            current_hash = hashlib.sha256(urdf_text.encode("utf-8")).hexdigest()
            stamp_path = os.path.join(stage_dir, "urdf.sha256")
            stored_hash = ""
            if os.path.isfile(stamp_path):
                try:
                    with open(stamp_path) as fh:
                        stored_hash = fh.read().strip()
                except OSError:
                    stored_hash = ""
            urdf_hash_mismatch = stored_hash != current_hash
            if urdf_hash_mismatch:
                if stored_hash:
                    print(
                        f"{MSG_COLOR}[assemble_pipeline] urdf hash changed "
                        f"({stored_hash[:12]} -> {current_hash[:12]}); rebuilding robot USD{RESET}"
                    )
                else:
                    print(
                        f"{MSG_COLOR}[assemble_pipeline] no urdf hash stamp at "
                        f"{stamp_path}; building robot USD{RESET}"
                    )
            elif not missing and not launcher_config_mismatch:
                print(
                    f"{MSG_COLOR}[assemble_pipeline] urdf hash matches "
                    f"({current_hash[:12]}); skipping robot USD rebuild{RESET}"
                )

        if always_regenerate_robot_usd:
            need_assemble_robot = True
        elif missing:
            need_assemble_robot = True
        elif urdf_hash_mismatch:
            need_assemble_robot = True
        elif launcher_config_mismatch:
            need_assemble_robot = True
        elif len(assemble_robot_artefacts) == 0:
            need_assemble_robot = True
        else:
            need_assemble_robot = False

    extra_env = sanitized_usd_env()

    if assemble_pipeline is not None and not isinstance(assemble_pipeline, dict):
        raise ValueError("assemble_pipeline must be a mapping")
    if assemble_pipeline and assemble_style != "registered_pipeline":
        raise ValueError("assemble_pipeline is only valid with assemble_style='registered_pipeline'")

    # ``assemble_style`` is a data-driven dispatch flag threaded from the
    # launcher YAML (``launcher.physics.engines.<engine>.assemble_style``).
    # ``"standalone_hook"`` — the scene's ``prebaked.pipeline`` hook is
    # the entire assemble (skip the Kit pipeline).  Any backend that
    # ships a prebaked scene + its own assemble script declares this,
    # so bringup stays engine-name-agnostic.  Launcher YAMLs declare this
    # flag directly; no engine-name mapping lives here.
    if assemble_style == "standalone_hook":
        if not pre_runtime_hook:
            raise ValueError(
                f"physics_engine={physics_engine!r} declares "
                f"``assemble_style: standalone_hook`` but the scene has no "
                f"``robot.robot_source.prebaked.pipeline`` — the hook is "
                f"required to produce the stage_dir artefacts (bringup will "
                f"NOT invoke the Kit pipeline for standalone-hook engines)."
            )
        from launch.substitutions import PathJoinSubstitution
        from launch_ros.substitutions import FindPackagePrefix

        hook_pkg = pre_runtime_hook["package"]
        hook_exe = pre_runtime_hook["executable"]
        hook_name = pre_runtime_hook["name"]
        hook_path = PathJoinSubstitution([FindPackagePrefix(hook_pkg), "lib", hook_pkg, hook_exe])
        hook_proc = ExecuteProcess(
            cmd=[
                "python3",
                hook_path,
                "--scene",
                scene_resolved,
                "--output-dir",
                stage_dir,
                "--base-path",
                assets_folder,
                *(["--launcher-config", launcher_config] if launcher_config else []),
            ],
            output="screen",
            name=hook_name,
            additional_env=extra_env,
        )
        gate_runtime = RegisterEventHandler(
            OnProcessExit(
                target_action=hook_proc,
                on_exit=lambda event, context: (
                    list(runtime_nodes)
                    if event.returncode == 0
                    else [Shutdown(reason=f"{hook_name} failed — aborting launch")]
                ),
            )
        )
        return hook_proc, None, None, gate_runtime

    # Pipeline name — one per physics-engine family. Downstream backends
    # declare ``assemble_style: registered_pipeline`` plus the registered
    # pipeline name and module imports in launcher YAML, keeping this shared
    # bringup independent from their package names. Legacy built-in engines
    # continue to use the mapping below.
    #
    # ``physics_engine`` here is the launcher-yaml selector — same set
    # of tokens the runtime-node executable is chosen by (``newton``
    # picks the Kit-free newton-standalone runtime, ``isaac_newton`` /
    # ``isaac_physx`` pick the Kit path). Do NOT confuse with
    # ``physics_solver`` (mjvbd / mjxpbd / mjwarp / fsvbd / avbd / … —
    # a within-newton knob the runtime reads from ROS params;
    # irrelevant to which assemble pipeline runs).
    # Accept both the short-token form (`isaac_newton`, `isaac_physx`,
    # `newton`) and the package-name form (`genie_sim_engine_isaacsim`,
    # `genie_sim_engine_newton`) that ships in the launcher YAMLs' physics.engine.
    _ISAAC_PIPELINE_ENGINES = frozenset(
        {
            "isaac_newton",  # Kit + Newton path
            "isaac_physx",  # Kit + PhysX path
            "newton",  # newton-standalone (Kit-free) — same asset needs
            "genie_sim_engine_isaacsim",  # package-name form (Kit path)
            "genie_sim_engine_newton",  # package-name form (newton-standalone)
        }
    )
    if assemble_style == "registered_pipeline":
        pipeline_cfg = assemble_pipeline or {}
        pipeline_name = str(pipeline_cfg.get("name") or "").strip()
        raw_modules = pipeline_cfg.get("modules") or []
        raw_robot_stage = pipeline_cfg.get("robot_stage")
        if raw_robot_stage is not None and not isinstance(raw_robot_stage, str):
            raise ValueError("assemble_pipeline.robot_stage must be a string")
        robot_stage = (raw_robot_stage or "").strip()
        if not pipeline_name:
            raise ValueError(
                f"physics_engine={physics_engine!r} declares "
                "``assemble_style: registered_pipeline`` but "
                "``assemble_pipeline.name`` is empty"
            )
        if not isinstance(raw_modules, list):
            raise ValueError("assemble_pipeline.modules must be a list of Python module names")
        pipeline_modules = [str(module).strip() for module in raw_modules if str(module).strip()]
        if not pipeline_modules:
            raise ValueError(
                f"physics_engine={physics_engine!r} declares "
                "``assemble_style: registered_pipeline`` but "
                "``assemble_pipeline.modules`` is empty"
            )
    elif assemble_style:
        raise ValueError(
            f"physics_engine={physics_engine!r} declares unknown "
            f"assemble_style={assemble_style!r}; expected "
            "'standalone_hook' or 'registered_pipeline'"
        )
    elif physics_engine in _ISAAC_PIPELINE_ENGINES:
        pipeline_name = "isaac_newton"
        pipeline_modules = ["genie_sim_engine_newton.assemble", "genie_sim_engine_isaacsim.assemble"]
        robot_stage = "urdf_to_usd"
    else:
        # Unknown engine — refuse rather than silently pick a pipeline.
        # Add it here (and register a matching @Pipeline.register
        # factory in the backend) when a new Kit-family engine lands.
        #
        # ``standalone_hook`` engines (declared via
        # ``assemble_style: standalone_hook`` in the launcher YAML)
        # return above without reaching this branch.
        raise ValueError(
            f"[assemble_pipeline] physics_engine={physics_engine!r} has no "
            f"registered pipeline mapping. Known: "
            f"{sorted(_ISAAC_PIPELINE_ENGINES)}"
            f". Downstream backends must set ``assemble_style`` to "
            f"'standalone_hook' or 'registered_pipeline' in their launcher YAML."
        )

    cmd = [
        "python3",
        "-m",
        "genie_sim_engine_core.assemble",
        "--scene",
        scene_resolved,
        "--output-dir",
        stage_dir,
        "--base-path",
        assets_folder,
        "--physics-engine",
        pipeline_name,
    ]
    for pipeline_module in pipeline_modules:
        cmd += ["--pipeline-module", pipeline_module]
    if launcher_config:
        # Stamp launcher_config onto manifest so a subsequent launch
        # under a different launcher YAML detects the swap and bypasses
        # the URDF→USD cache. Empty ⇒ omit; the manifest stamp then
        # stays empty and the next launch will treat that as mismatch.
        cmd += ["--launcher-config", launcher_config]
    if assemble_robot_artefacts is not None and not need_assemble_robot and robot_stage:
        # Cache hit — skip the configured robot stage explicitly. When
        # assemble_robot_artefacts is None the caller signals "no robot
        # pipeline needed at all" and the stage's own skip_reason handles it.
        cmd += ["--skip-stage", robot_stage]

    assemble_proc = ExecuteProcess(
        cmd=cmd,
        output="screen",
        name="assemble_pipeline",
        additional_env=extra_env,
    )

    # Optional post-assemble hook — a per-scene stage declared via
    # ``robot.robot_source.prebaked.pipeline``.  Runs after the Kit
    # pipeline for engines that need extra per-scene artefacts on top
    # of the standard ``robot.usda`` + ``render_layer.usda`` + manifest.
    # Kept as a separate ExecuteProcess so existing hook contracts stay
    # unchanged.
    hook_proc: ExecuteProcess | None = None
    gate_after_hook: RegisterEventHandler | None = None
    if pre_runtime_hook:
        from launch.substitutions import PathJoinSubstitution
        from launch_ros.substitutions import FindPackagePrefix

        hook_pkg = pre_runtime_hook["package"]
        hook_exe = pre_runtime_hook["executable"]
        hook_name = pre_runtime_hook["name"]
        hook_path = PathJoinSubstitution([FindPackagePrefix(hook_pkg), "lib", hook_pkg, hook_exe])
        hook_proc = ExecuteProcess(
            cmd=[
                "python3",
                hook_path,
                "--scene",
                scene_resolved,
                "--output-dir",
                stage_dir,
                "--base-path",
                assets_folder,
                *(["--launcher-config", launcher_config] if launcher_config else []),
            ],
            output="screen",
            name=hook_name,
            additional_env=extra_env,
        )
        gate_after_hook = RegisterEventHandler(
            OnProcessExit(
                target_action=hook_proc,
                on_exit=lambda event, context: (
                    list(runtime_nodes)
                    if event.returncode == 0
                    else [Shutdown(reason=f"{hook_name} failed — aborting launch")]
                ),
            )
        )

    if hook_proc is not None:
        gate_runtime = RegisterEventHandler(
            OnProcessExit(
                target_action=assemble_proc,
                on_exit=lambda event, context: (
                    [hook_proc, gate_after_hook]
                    if event.returncode == 0
                    else [Shutdown(reason="assemble pipeline failed — aborting launch")]
                ),
            )
        )
    else:
        gate_runtime = RegisterEventHandler(
            OnProcessExit(
                target_action=assemble_proc,
                on_exit=lambda event, context: (
                    list(runtime_nodes)
                    if event.returncode == 0
                    else [Shutdown(reason="assemble pipeline failed — aborting launch")]
                ),
            )
        )

    # Second and third slots stay None so unchanged callers keep their
    # ``if a_robot is not None`` branch dead-but-safe during migration.
    return assemble_proc, None, None, gate_runtime


# ---------------------------------------------------------------------------
# Render-backend selection (shared CLI → name map)
# ---------------------------------------------------------------------------

CLI_TO_RENDER_NAME = {
    "rmagine": "render_rmagine",
    "ovrtx": "render_ovrtx",
    "isaacsim": "render_isaacsim",
}


def resolve_active_renders(renders_from_yaml: Iterable[str]) -> set:
    """Return the set of active render-node names declared by the launcher YAML."""
    return set(renders_from_yaml)


# ---------------------------------------------------------------------------
# Higher-level helpers extracted for composer-style launch files
# ---------------------------------------------------------------------------


def resolve_scene_yaml_robot_params(
    context,
    *,
    required: bool,
    robot_model: str,
    body: str,
    arm: str,
    gripper: str,
) -> dict:
    """Load the ``scene`` YAML and resolve the robot_model/body/arm/gripper block.

    Returns a dict with keys:
      * ``scene_resolved``: absolute path to the YAML (``""`` when absent and
        ``required`` is False).
      * ``scene_yaml``: full parsed mapping (``{}`` when absent).
      * ``robot_section``: ``scene_yaml['robot']`` as a dict (``{}`` when absent).
      * ``robot_source``: ``robot_section['robot_source']`` as a dict (``{}``).
      * ``init_joint_pos``: ``robot_section['init_joint_pos']`` (``{}``).
      * ``viewer_camera``: ``scene_yaml['viewer_camera']`` (``{}``) — optional
        ``{pos, lookat}`` (and/or ``pitch``/``yaw``) for the GL viewer / FreeCam.
      * ``resolved``: mapping returned by :func:`resolve_robot_params_from_scene`
        (always populated — falls back to launch CLI values).

    On hard error (file unreadable, ``required`` violated, wrong extension),
    prints a colored diagnostic and calls ``sys.exit(1)``.
    """
    scene_arg = perform(context, "scene").strip()
    scene_resolved = resolve_bringup_config_file(scene_arg) if scene_arg else ""
    if required and not scene_resolved:
        if scene_arg:
            print(
                f"{ERR_COLOR}scene YAML not found: {scene_arg} (check the path/name or rebuild genie_sim_bringup){RESET}"
            )
        else:
            print(f"{ERR_COLOR}scene is required (YAML scene file){RESET}")
        sys.exit(1)
    scene_yaml: dict = {}
    if scene_resolved:
        if not scene_resolved.endswith((".yaml", ".yml")):
            print(f"{ERR_COLOR}scene must be a YAML file (.yaml/.yml), got: {scene_resolved}{RESET}")
            sys.exit(1)
        try:
            with open(scene_resolved) as _f:
                scene_yaml = yaml.safe_load(_f) or {}
        except Exception as exc:
            print(f"{ERR_COLOR}failed to load scene YAML {scene_resolved}: {exc}{RESET}")
            sys.exit(1)
    robot_section = scene_yaml.get("robot") if isinstance(scene_yaml.get("robot"), dict) else {}
    robot_source = robot_section.get("robot_source") if isinstance(robot_section.get("robot_source"), dict) else {}
    init_joint_pos = (
        robot_section.get("init_joint_pos") if isinstance(robot_section.get("init_joint_pos"), dict) else {}
    )
    viewer_camera = scene_yaml.get("viewer_camera") if isinstance(scene_yaml.get("viewer_camera"), dict) else {}
    resolved = resolve_robot_params_from_scene(
        robot_source,
        robot_model=robot_model,
        body=body,
        arm=arm,
        gripper=gripper,
    )
    return {
        "scene_resolved": scene_resolved,
        "scene_yaml": scene_yaml,
        "robot_section": robot_section,
        "robot_source": robot_source,
        "init_joint_pos": init_joint_pos,
        "viewer_camera": viewer_camera,
        "resolved": resolved,
    }


def resolve_assets_folder() -> str:
    """Return the assets folder path.

    Search order:
      1. ``./assets`` relative to cwd.
      2. ``/geniesim_assets`` provisioned by runtime.
      3. ``geniesim_assets.ASSETS_PATH`` from the legacy module.

    Hard-exits if none is available.
    """
    local_assets = os.path.abspath("./assets")
    if os.path.isdir(local_assets):
        print(f"{MSG_COLOR}assets folder: {local_assets} (local){RESET}")
        return local_assets

    global_assets = "/geniesim_assets"
    if os.path.isdir(global_assets):
        print(f"{MSG_COLOR}assets folder: {global_assets} (industrial){RESET}")
        return global_assets

    try:
        import geniesim_assets as _ga  # type: ignore
    except ImportError as exc:
        print(
            f"{ERR_COLOR}No usable assets folder found.\n"
            f"  Checked: {local_assets}, {global_assets}, geniesim_assets.ASSETS_PATH.\n"
            f"  ImportError: {exc}\n"
            f"  Fix: provide ./assets, provision /geniesim_assets, or install geniesim_assets.{RESET}"
        )
        sys.exit(1)
    assets_path = getattr(_ga, "ASSETS_PATH", None)
    if assets_path is None or not isinstance(assets_path, (str, os.PathLike)):
        print(
            f"{ERR_COLOR}geniesim_assets imported but does not expose a "
            f"valid ``ASSETS_PATH`` (expected str or os.PathLike, "
            f"got: {type(assets_path).__name__} = {assets_path!r}).{RESET}"
        )
        sys.exit(1)
    assets_path = os.fspath(assets_path)
    if not assets_path:
        print(f"{ERR_COLOR}geniesim_assets.ASSETS_PATH is empty after normalization.{RESET}")
        sys.exit(1)
    if not os.path.isdir(assets_path):
        print(
            f"{ERR_COLOR}geniesim_assets.ASSETS_PATH points at {assets_path!r} "
            f"but no such directory exists on disk.{RESET}"
        )
        sys.exit(1)
    print(f"{MSG_COLOR}assets folder: {assets_path} (geniesim_assets){RESET}")
    return assets_path


def stage_yaml_snapshot(stage_dir: str, scene_path: str) -> str:
    """Copy ``scene_path`` to ``<stage_dir>/scene.yaml`` for post-mortem debugging.

    The snapshot is overwritten on every launch (including cache hits) so the
    operator can ``diff`` the yaml that actually drove this run against the
    yaml that baked any cached USDs. Best-effort: a copy failure is logged
    but does not abort the launch.
    """
    os.makedirs(stage_dir, exist_ok=True)
    try:
        import shutil

        staged_yaml = os.path.join(stage_dir, "scene.yaml")
        shutil.copyfile(scene_path, staged_yaml)
        print(f"{MSG_COLOR}staged scene yaml: {staged_yaml}{RESET}")
        return staged_yaml
    except OSError as exc:
        print(f"{MSG_COLOR}WARNING: failed to stage scene yaml ({exc}); continuing{RESET}")
        return ""


def discover_mujoco_plugin_dir() -> str:
    """Locate ``mujoco_plugin/`` under ``mujoco_vendor``'s share dir and export it.

    Sets ``MUJOCO_PLUGIN_DIR`` in ``os.environ`` so the MuJoCo physics node
    can find optional plugin shared objects. Silent no-op if ``mujoco_vendor``
    is not installed or the plugin directory is missing — MuJoCo runs fine
    without plugins.

    Returns the discovered absolute path (or ``""`` when not found).
    """
    try:
        share = Path(get_package_share_directory("mujoco_vendor"))
    except PackageNotFoundError:
        return ""
    candidate = share / "mujoco_plugin"
    if not candidate.exists():
        return ""
    os.environ["MUJOCO_PLUGIN_DIR"] = str(candidate)
    print(f"{MSG_COLOR}MUJOCO_PLUGIN_DIR: {candidate}{RESET}")
    return str(candidate)


def pad_urdf_joint_limits(urdf_text: str, pad_revolute_rad: float, pad_prismatic_m: float) -> str:
    """Thin re-export of :func:`genie_sim_robot_model.urdf_utils.pad_urdf_joint_limits`.

    Both ``genie_sim_bringup`` (simulator-side ``robot_state_publisher``)
    and ``genie_sim_moveit`` (MoveIt's URDF) need to apply the same pad,
    and putting the implementation in ``genie_sim_robot_model`` (a shared
    dependency of both) avoids a bringup<->moveit cycle.  We re-export
    here so existing call sites under ``launch/`` don't need to change
    imports.
    """
    from genie_sim_robot_model.urdf_utils import pad_urdf_joint_limits as _impl  # noqa: PLC0415

    return _impl(urdf_text, pad_revolute_rad, pad_prismatic_m)


def build_robot_description(
    *,
    robot_source: dict,
    resolved: dict[str, str],
) -> str:
    """Return the robot_description XML string for the active scene.

    Reads ``robot.robot_source.package`` (default :data:`DEFAULT_ROBOT_MODEL_PACKAGE`)
    and ``robot.robot_source.urdf.xacro_entry`` (default
    :data:`DEFAULT_ROBOT_XACRO_ENTRY`) to locate the xacro entry, then
    processes it with the mappings derived from ``robot_source`` / ``resolved``
    via :func:`build_urdf_xacro_mappings`.  Returns ``""`` on any failure
    (caller should treat as a fatal launch error).

    The same ``package`` / ``xacro_entry`` contract is honored by
    the urdf_to_usd stage at offline-cache build time, so the runtime
    ``robot_description`` here resolves to the same xacro tree the
    cached ``robot.usda`` was produced from.
    """
    try:
        import xacro  # type: ignore
    except ImportError as exc:
        print(f"{ERR_COLOR}xacro Python module not importable: {exc}{RESET}")
        return ""

    pkg = (robot_source.get("package") or "").strip() or DEFAULT_ROBOT_MODEL_PACKAGE
    urdf_block = robot_source.get("urdf") if isinstance(robot_source.get("urdf"), dict) else {}
    xacro_entry = (urdf_block.get("xacro_entry") or "").strip() or DEFAULT_ROBOT_XACRO_ENTRY
    try:
        pkg_share = Path(get_package_share_directory(pkg))
    except Exception as exc:
        print(f"{ERR_COLOR}Failed to locate robot description package {pkg!r}: {exc}{RESET}")
        return ""
    robot_xacro_file = pkg_share / xacro_entry
    if not robot_xacro_file.exists():
        print(f"{ERR_COLOR}xacro entry not found: {robot_xacro_file}{RESET}")
        return ""
    xacro_args = build_urdf_xacro_mappings(robot_source, resolved)
    print(f"{MSG_COLOR}xacro_args (description): {xacro_args} (package={pkg}, entry={xacro_entry}){RESET}")
    if "robot_model" not in xacro_args:
        print(f"{ERR_COLOR}no robot_model in scene.robot_source (or launch arg){RESET}")
        return ""
    try:
        rd = xacro.process_file(str(robot_xacro_file), mappings=xacro_args).toprettyxml(indent="  ")
    except Exception as exc:
        print(f"{ERR_COLOR}xacro processing failed for {robot_xacro_file} with {xacro_args}: {exc}{RESET}")
        return ""
    if not rd or "<link" not in rd:
        print(
            f"{ERR_COLOR}robot_description empty or has no <link> elements — "
            f"check scene.robot_source robot_model/arm/body/gripper. "
            f"xacro_args={xacro_args}{RESET}"
        )
        return ""

    # Widen URDF joint <limit> by a small pad so MoveIt's CheckStartStateBounds
    # tolerates the sub-mrad drift that MuJoCo's soft joint limit lets through
    # under heavy contact load.  Pads are intentionally tiny — this is a
    # tolerance widening, not a relax — and configurable per scene via
    # ``robot.robot_source.urdf.joint_limit_pad_{rev_rad,prismatic_m}``.
    pad_rev = float(urdf_block.get("joint_limit_pad_rev_rad", 0.01))  # 0.57°
    pad_pri = float(urdf_block.get("joint_limit_pad_prismatic_m", 0.001))  # 1 mm
    if pad_rev > 0.0 or pad_pri > 0.0:
        widened = pad_urdf_joint_limits(rd, pad_rev, pad_pri)
        if widened is not rd:
            print(
                f"{MSG_COLOR}widened URDF joint <limit> by "
                f"±{pad_rev:g} rad (revolute) / ±{pad_pri:g} m (prismatic) "
                f"so MoveIt tolerates MuJoCo soft-limit drift{RESET}"
            )
            rd = widened
    return rd


def _isaacsim_render_env() -> tuple[str, dict]:
    """Interpreter + environment for ``isaacsim_render.py``.

    Honor the CLI's ``$GENIESIM_PY_CMD`` convention. When Isaac is installed
    in the launch interpreter, use that interpreter and preserve its ROS
    underlay. Otherwise use bundled ``omni_python`` and remove incompatible
    system-Python native packages as described below.

    The launch process has already sourced the system ROS tree, which puts
    ``/opt/ros/<distro>/lib/python3.<system>`` on ``PYTHONPATH`` and its ``lib``
    on ``LD_LIBRARY_PATH``, and a robotics image typically also has
    ``/opt/openrobots/lib/python3.<system>/site-packages`` (Pinocchio & co.) on
    ``PYTHONPATH``. All of those are built for the system Python, so inside
    Isaac's own interpreter they make ``import rclpy`` resolve to the wrong ABI
    (``ModuleNotFoundError: No module named 'rclpy._rclpy_pybind11'``) and make
    ``import torch`` abort in ``torch._C`` (``SystemError: ... bad call flags``).
    Strip every entry tagged with the system interpreter's ``pythonX.Y`` (not
    just ``/opt/ros``) and let the node use Isaac's bundled bridge instead — the
    same thing the ``omni_python`` wrapper does, repeated here so the node works
    even when that wrapper is an older revision that lacks the stripping.

    Two further Isaac Sim 5.1 packaging defects are compensated for here, since
    neither is fixable from the node itself:

    * ``/isaac-sim/setup_ros_env.sh`` only appends the ROS2-bridge lib dir to
      ``LD_LIBRARY_PATH`` inside an ``if [ -z "$ROS_DISTRO" ]`` guard, but the
      ``omni_python`` wrapper exports ``ROS_DISTRO`` before sourcing it — so the
      guard never fires and ``rclpy`` fails to load
      (``libament_index_cpp.so: cannot open shared object file``).
    * Isaac Sim's bundled numpy carries a dangling ``libgfortran`` symlink,
      which breaks every extension that imports numpy unless it has been
      repaired in the image.

    Returns ``(interpreter, extra_env)``.
    """

    import importlib.metadata
    import shlex

    interpreter = os.environ.get("GENIESIM_PY_CMD") or ""
    if not interpreter:
        try:
            importlib.metadata.distribution("isaacsim")
        except importlib.metadata.PackageNotFoundError:
            interpreter = "omni_python" if shutil.which("omni_python") else (sys.executable or "python3")
        else:
            interpreter = sys.executable or "python3"

    executable = shutil.which(shlex.split(interpreter)[0])
    if executable and os.path.realpath(executable) == os.path.realpath(sys.executable):
        return interpreter, {}

    # The launch process runs under the system Python (e.g. 3.10 in the Humble
    # tree). Any PYTHONPATH / LD_LIBRARY_PATH entry that carries that version's
    # ``pythonX.Y`` marker holds extensions built against the system ABI; inside
    # Isaac Sim's own interpreter (3.11 on 5.1, 3.12 on 6.2) they load the wrong
    # ABI. It is NOT enough to strip ``/opt/ros`` — sibling robotics trees like
    # ``/opt/openrobots/lib/python3.10/site-packages`` are just as poisonous:
    # they inject a system-Python numpy/torch that makes ``torch._C`` abort with
    # ``SystemError: ... bad call flags``. Strip by the system interpreter's own
    # ``pythonX.Y`` tag so every such tree is dropped, whatever its prefix.
    _sys_pytag = f"python{sys.version_info.major}.{sys.version_info.minor}"

    # ...but exempt this repo's own colcon overlay. ``ament_cmake`` installs a
    # package's python modules under ``<prefix>/lib/pythonX.Y/site-packages``,
    # which carries the system ``pythonX.Y`` tag too — so on a base image whose
    # system Python (which built the workspace) differs from Isaac's bundled
    # interpreter (here: 3.12 workspace vs 3.11 ``omni_python``), the blanket
    # strip above deletes the entry that holds ``genie_sim_engine_core`` and the
    # render node dies with ``No module named 'genie_sim_engine_core'``. The
    # overlay holds only geniesim's own pure-python packages — never a
    # system-ABI numpy/torch/rclpy — so keeping it is safe under Isaac's
    # interpreter. ``COLCON_PREFIX_PATH`` lists exactly these overlay prefixes
    # (the ROS underlay at ``/opt/ros`` is not among them and stays stripped).
    _ws_prefixes = tuple(p for p in os.environ.get("COLCON_PREFIX_PATH", "").split(":") if p)

    def _under_workspace(path: str) -> bool:
        return any(path == pre or path.startswith(pre + os.sep) for pre in _ws_prefixes)

    def _strip_system_python(value: str) -> str:
        keep = []
        for p in value.split(":"):
            if not p:
                continue
            if _under_workspace(p):
                keep.append(p)
                continue
            if p.startswith("/opt/ros/") or _sys_pytag in p:
                continue
            keep.append(p)
        return ":".join(keep)

    extra_env: dict = {}
    # The colcon overlay tree (genie_sim_engine_core, the sibling render module)
    # is on PYTHONPATH; ``_strip_system_python`` preserves it (see above) while
    # dropping the system-Python entries that would break Isaac's interpreter.
    extra_env["PYTHONPATH"] = _strip_system_python(os.environ.get("PYTHONPATH", ""))

    ld_library_path = _strip_system_python(os.environ.get("LD_LIBRARY_PATH", ""))
    isaac_root = Path("/isaac-sim")
    if isaac_root.is_dir():
        distro = os.environ.get("ROS_DISTRO", "jazzy")
        bridge_lib = isaac_root / "exts" / "isaacsim.ros2.bridge" / distro / "lib"
        if bridge_lib.is_dir() and str(bridge_lib) not in ld_library_path.split(":"):
            ld_library_path = f"{ld_library_path}:{bridge_lib}" if ld_library_path else str(bridge_lib)
        extra_env.setdefault("ROS_DISTRO", distro)
    extra_env["LD_LIBRARY_PATH"] = ld_library_path

    return interpreter, extra_env


def make_render_isaacsim_node(
    *,
    stage_manifest: str,
    physics_node_name: str,
    plugins_params_path: str,
    common_param: dict,
    ros_log_args: Iterable[str],
    period: float = 3.0,
) -> "TimerAction":  # noqa: F821 — runtime import below
    """Construct the deferred ``render_isaacsim`` node tied to the stage manifest.

    Spawns ``isaacsim_render.py`` from the ``genie_sim_render`` package.
    Reads the same ``stage_manifest`` produced by ``assemble_scene`` as the
    C++ ovrtx node — the two render backends are drop-in swappable via the
    ``renders:`` list in the launcher YAML.
    """
    from launch.actions import TimerAction

    interpreter, render_env = _isaacsim_render_env()

    isaacsim_parameters: list = []
    if plugins_params_path:
        isaacsim_parameters.append(plugins_params_path)
    isaacsim_parameters.append(
        {
            "stage_manifest": stage_manifest,
            "render_fps": 30.0,
            **common_param,
        }
    )
    print(f"{MSG_COLOR}render_isaacsim stage_manifest: {stage_manifest}{RESET}")
    return TimerAction(
        period=period,
        actions=[
            Node(
                package="genie_sim_render",
                executable="isaacsim_render.py",
                name="render_isaacsim",
                output="both",
                parameters=isaacsim_parameters,
                remappings=[("~/free_cam_pose", f"/{physics_node_name}/viewer/camera_pose")],
                arguments=list(ros_log_args),
                prefix=interpreter if interpreter != "isaacsim_render.py" else None,
                additional_env=render_env,
            )
        ],
    )


def make_render_ovrtx_node(
    *,
    stage_manifest: str,
    physics_node_name: str,
    plugins_params_path: str,
    common_param: dict,
    ros_log_args: Iterable[str],
    period: float = 3.0,
) -> "TimerAction":  # noqa: F821 — runtime import below
    """Construct the deferred ``render_ovrtx`` node tied to the stage manifest.

    The OVRTX render node reads the same ``stage_manifest`` produced by
    ``assemble_scene``, ensuring a single source of truth for asset paths.
    ``ovrtx_root`` is discovered from the active Python import specification;
    we intentionally avoid ``import ovrtx`` here because importing the package
    runs ``register_schema_paths()`` which writes to ``PXR_PLUGINPATH_NAME``
    in the launch process's env and corrupts every child (notably
    ``assemble_scene``) with ``TfType::AddAlias`` collisions on
    ``ParticleField*``.
    """
    import importlib.util

    from launch.actions import TimerAction

    ovrtx_root_default = ""
    ovrtx_spec = importlib.util.find_spec("ovrtx")
    if ovrtx_spec is not None and ovrtx_spec.submodule_search_locations:
        for package_dir in ovrtx_spec.submodule_search_locations:
            candidate = os.path.realpath(os.path.join(package_dir, "bin"))
            if os.path.isfile(os.path.join(candidate, "libovrtx-dynamic.so")):
                ovrtx_root_default = candidate
                break

    render_env = {}
    if ovrtx_root_default:
        inherited_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        render_env["LD_LIBRARY_PATH"] = os.pathsep.join(
            part for part in (ovrtx_root_default, inherited_library_path) if part
        )

    ovrtx_parameters: list = []
    if plugins_params_path:
        ovrtx_parameters.append(plugins_params_path)
    ovrtx_parameters.append(
        {
            "stage_manifest": stage_manifest,
            "render_fps": 30.0,
            "ovrtx_root": ovrtx_root_default,
            **common_param,
        }
    )
    print(
        f"{MSG_COLOR}render_ovrtx stage_manifest: {stage_manifest} "
        f"ovrtx_root: {ovrtx_root_default or '(unset — ovrtx package not found on sys.path)'}{RESET}"
    )
    return TimerAction(
        period=period,
        actions=[
            Node(
                package="genie_sim_render",
                executable="genie_sim_render_node",
                name="render_ovrtx",
                output="both",
                parameters=ovrtx_parameters,
                remappings=[("~/free_cam_pose", f"/{physics_node_name}/viewer/camera_pose")],
                arguments=list(ros_log_args),
                additional_env=render_env,
            )
        ],
    )
