# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""
Sub-launch owning the physics backend. Selector is
``launcher.physics.engine`` in the launcher YAML; the active engine's
package/executable/name are pulled from
``launcher.physics.engines.<engine>``. Normally included by app.launch.py.

Spawns:
  * ``assemble_robot`` (xacro -> URDF -> USD, cache-gated) when a robot is
    configured in the scene YAML
  * ``assemble_scene`` (manifest.json + camera intrinsics, always runs)
  * The Isaac Sim physics ``Node`` (gated behind the assemble stages)
  * Optional ``industrial_bridge`` node when configured in launcher YAML
  * Optional ``interaction_tools`` node when ``interaction_tools:=true``
  * Optional ``render_ovrtx`` or ``render_isaacsim`` node when listed in launcher YAML renders

State-sharing contract: re-resolves the scene YAML, assets folder, stage
dir, launcher config and robot params from the same ``LaunchConfiguration``
namespace as the composer (no IPC, no global context). Helpers are pure
I/O — re-parsing scene YAML 2-3 times per run is negligible cost.
"""

import os, sys
import importlib.util
import json
from pathlib import Path

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import OpaqueFunction, DeclareLaunchArgument, TimerAction

# Load sibling utils.py via importlib (rather than a plain ``import utils``)
# so an unrelated ``utils`` package elsewhere on sys.path can't shadow it.
_lu_spec = importlib.util.spec_from_file_location(
    "_launch_utils", os.path.join(os.path.dirname(os.path.realpath(__file__)), "utils.py")
)
lu = importlib.util.module_from_spec(_lu_spec)
_lu_spec.loader.exec_module(lu)
MSG_COLOR = lu.MSG_COLOR
ERR_COLOR = lu.ERR_COLOR
RESET = lu.RESET


def generate_launch_description():
    # Only declare args that belong to this sub-launch.
    # Common args (scene, interaction_tools, launcher_config, fake_slam,
    # remap_tf, log_level, use_sim_time, robot_model, body, arm, gripper, …)
    # are declared by the composer context and pulled via lu.perform().
    declared_arguments = [
        DeclareLaunchArgument(
            name="physics_hz",
            default_value="100.0",
            description="Isaac Sim physics step rate (Hz)",
        ),
        DeclareLaunchArgument(
            name="render_hz",
            default_value="30.0",
            description="Isaac Sim render target rate (Hz, decoupled from physics)",
        ),
        DeclareLaunchArgument(
            name="headless",
            default_value="true",
            choices=["true", "false"],
            description="Run Isaac Sim physics in headless mode (no GUI)",
        ),
        DeclareLaunchArgument(
            name="always_regenerate_robot_usd",
            default_value="false",
            choices=["true", "false"],
            description=(
                "If true, force the URDF->USD robot conversion (assemble_robot) and the "
                "downstream scene assembly (assemble_scene) to run even when a cached "
                "manifest.json already exists in the per-scene stage dir. Useful when "
                "iterating on the robot URDF/xacro and you want each launch to pick up "
                "edits without manually deleting the cache. The cache is treated as a "
                "miss for this run only; the regenerated artifacts are written back in "
                "place so subsequent launches with the flag off will see a fresh hit."
            ),
        ),
        DeclareLaunchArgument(
            name="physics_engine",
            default_value="isaac_physx",
            choices=["isaac_physx", "isaac_newton", "newton"],
            description=(
                "'isaac_physx' (default), 'isaac_newton' (Isaac wrapper, "
                "XPBD/MuJoCo), 'newton' (direct ModelBuilder, VBD cloth)."
                " 'libuipc' selects the independent native IPC backend."
            ),
        ),
        DeclareLaunchArgument(
            name="physics_solver",
            default_value="mujoco-warp",
            choices=["mujoco-warp"],
            description=(
                "Rigid-body solver. The only supported value is "
                "``mujoco-warp`` (MuJoCo-Warp). Honored by "
                "``physics_engine=isaac_newton`` and ignored by "
                "``isaac_physx`` (PhysX has its own solver) and ``newton`` "
                "(newton-standalone hardcodes Featherstone for rigid; "
                "reads its cloth solver from the scene yaml's "
                "``newton.solver.prefer``). XPBD / Featherstone / "
                "SemiImplicit are NOT exposed at the launch level: "
                "Isaac Sim 6.x's ``isaacsim.physics.newton`` v0.6.0 only "
                "supports ``SolverMuJoCo`` and ``SolverXPBD`` and we pin "
                "``isaac_newton`` to MuJoCo-Warp specifically. For cloth "
                "or alternative rigid solvers, use ``physics_engine:=newton`` "
                "and configure via the scene yaml."
                " The independent libuipc backend additionally accepts 'ipc'."
            ),
        ),
        DeclareLaunchArgument(
            name="physics_solver_substep",
            default_value="0",
            description=(
                "Solver substeps per physics frame. ``0`` means 'engine picks "
                "its own default' — newton-standalone uses 10 (franka demo cadence); "
                "isaac_newton leaves Newton's NewtonConfig default alone "
                "(typically 1, with the iteration count carrying the convergence). "
                "Set explicitly to override either."
            ),
        ),
        DeclareLaunchArgument(
            name="physics_solver_iterations",
            default_value="0",
            description=(
                "Solver iterations per substep. ``0`` means 'engine picks its "
                "own default' — newton-standalone uses 5 (franka VBD demo); "
                "isaac_newton leaves Newton's NewtonConfig default (typically "
                "100). Forcing newton-standalone's 5 onto isaac_newton's MuJoCo/XPBD "
                "solver will under-converge stiff cloth and explode garments on "
                "the first frame."
            ),
        ),
        DeclareLaunchArgument(
            name="render_mode",
            default_value="raster",
            choices=["raster", "pathtrace", "offline"],
            description=(
                "RTX submode. Maps to ``/rtx/rendermode`` via RENDER_MODE_MAP "
                "in _isaacsim_runtime.py. Approximate per-frame cost on a 5090 "
                "with the fr3+cloth scene: "
                "raster ~16ms (RaytracedLighting; default), "
                "pathtrace ~49ms (RealTimePathTracing; kit default), "
                "offline 100-300ms (PathTracing; for screenshots). "
                "Storm isn't supported — the Pixar delegate isn't bundled in "
                "isaacsim.exp.base.python.kit which our newton-standalone path "
                "uses; ``vp.set_hd_engine('Storm')`` spams "
                "``unable to find suitable engine`` warnings and falls back "
                "to RTX. Use ``raster`` for the cheapest debug viewport."
            ),
        ),
        DeclareLaunchArgument(
            name="physics_engine_visualizer",
            default_value="",
            description=(
                "Visualizer backend (physics_engine=newton only). 'none' | "
                "'newton' (Newton GL ViewerGL) | 'ovrtx' (in-process OVRtx) | "
                "'rerun' (placeholder, not implemented). Empty (default) means "
                "'don't override' — the "
                "launcher yaml's ``physics_engine_visualizer`` key wins. When "
                "the user passes ``physics_engine_visualizer:=<value>`` on the "
                "CLI, this gets pushed into the node parameter dict via "
                "``cli_overrides`` so it OVERRIDES the launcher yaml. Ignored "
                "by isaac_physx / isaac_newton (those use the Kit viewport "
                "gated by ``headless:=false``)."
            ),
        ),
        DeclareLaunchArgument(
            name="realtime_factor",
            default_value="1.0",
            description=(
                "Realtime factor: 1.0 = realtime (default), 0.1 = 10× slower, 2.0 = 2× faster. "
                "Scales the wall-clock period between physics steps and the render cadence "
                "proportionally so sim_time stays consistent. Can also be set in the launcher "
                "yaml's ros__parameters block."
            ),
        ),
        DeclareLaunchArgument(
            name="num_envs",
            default_value="1",
            description=(
                "Native multi-world replicate count (physics_engine=newton only; "
                "isaac_physx/isaac_newton ignore it). 1 (default) = single env, "
                "byte-identical to the pre-multi-world path. >1 replicates the "
                "single-env ModelBuilder template into N GPU worlds via "
                "``builder.replicate`` before finalize."
            ),
        ),
    ]

    # Extend the existing declarations without replacing their original choices.
    for argument in declared_arguments:
        if argument.name == "physics_engine":
            argument.choices.append("libuipc")
        elif argument.name == "physics_solver":
            argument.choices.append("ipc")

    def parse_args(context):
        # ---- pull common args from the composer context ------------------
        use_sim_time = lu.perform(context, "use_sim_time")
        remap_tf = lu.perform(context, "remap_tf")
        fake_slam = lu.perform(context, "fake_slam")
        log_level = lu.perform(context, "log_level").strip()
        common_param = {"use_sim_time": True if "true" == use_sim_time else False}
        ros_log_args = ["--ros-args", "--log-level", log_level]

        # ---- launcher YAML metadata (engine binding + plugin params) -----
        # Loaded EARLY so the per-arg launcher-yaml override logic below
        # can see whether a given key is provided by the yaml's
        # ``ros__parameters`` block.
        launcher_config_arg = lu.perform(context, "launcher_config").strip()
        plugins_cfg_path = lu.resolve_bringup_config_file(launcher_config_arg) if launcher_config_arg else ""
        launcher_section: dict = {}
        plugins_params_path = ""
        if plugins_cfg_path:
            launcher_section, _plug_yaml, plugins_params_path = lu.load_launcher_yaml(plugins_cfg_path)

        _engine_cfg = {}
        _physics_section = launcher_section.get("physics") or {}
        if isinstance(_physics_section, dict):
            _engine_selector = str(_physics_section.get("engine") or "").strip()
            _engines_section = _physics_section.get("engines") or {}
            if isinstance(_engines_section, dict):
                _engine_cfg = _engines_section.get(_engine_selector) or {}
        if not isinstance(_engine_cfg, dict):
            print(f"{ERR_COLOR}[physics_backend] engine config must be a mapping{RESET}")
            sys.exit(1)

        # ---- pull own args -----------------------------------------------
        # ``physics_engine`` is a plain launch arg with the standard
        # launcher-yaml ``ros__parameters`` fallback (the ``cli_overrides``
        # logic below handles CLI > yaml > DeclareLaunchArgument default).
        # Assembly routing is selected independently by the engine's
        # ``assemble_style`` metadata.
        physics_hz = float(lu.perform(context, "physics_hz"))
        render_hz = float(lu.perform(context, "render_hz"))
        headless = lu.perform(context, "headless")
        physics_engine = lu.perform(context, "physics_engine").strip().lower()
        physics_solver = lu.perform(context, "physics_solver").strip().lower()
        physics_solver_substep = lu.perform(context, "physics_solver_substep").strip()
        physics_solver_iterations = lu.perform(context, "physics_solver_iterations").strip()
        render_mode = lu.perform(context, "render_mode").strip().lower()
        physics_engine_visualizer = lu.perform(context, "physics_engine_visualizer").strip().lower()
        realtime_factor = float(lu.perform(context, "realtime_factor"))
        num_envs = lu.perform(context, "num_envs").strip()
        always_regenerate_robot_usd = lu.perform(context, "always_regenerate_robot_usd").strip().lower() == "true"
        print(
            f"{MSG_COLOR}[physics_backend] physics_hz: {physics_hz}, "
            f"render_hz: {render_hz}, headless: {headless}, "
            f"physics_engine: {physics_engine}, physics_solver: {physics_solver}, "
            f"render_mode: {render_mode}, "
            f"realtime_factor: {realtime_factor}, num_envs: {num_envs}, "
            f"physics_engine_visualizer: {physics_engine_visualizer!r}{RESET}"
        )

        nodes: list = []  # runtime nodes — gated behind the assemble stages below.

        # ---------------------------------------------------------------
        # Scene YAML is the primary source for robot_model/arm/body/gripper;
        # launch CLI args only fill keys left empty in robot.robot_source.
        # ``init_joint_pos`` is forwarded as a JSON-encoded node parameter
        # (``init_joint_pos_json``) — kept out of the stage manifest
        # deliberately so editing this block does not invalidate the
        # assemble_robot / assemble_scene cache.
        # Units: degrees for revolute joints, metres for prismatic.
        # ---------------------------------------------------------------
        scene_info = lu.resolve_scene_yaml_robot_params(
            context,
            required=True,
            robot_model=lu.perform(context, "robot_model"),
            body=lu.perform(context, "body"),
            arm=lu.perform(context, "arm"),
            gripper=lu.perform(context, "gripper"),
        )
        scene_config_resolved = scene_info["scene_resolved"]
        _init_joint_pos = scene_info["init_joint_pos"]
        _viewer_camera = scene_info["viewer_camera"]
        _resolved = scene_info["resolved"]
        _src = scene_info["robot_source"]
        robot_model = _resolved["robot_model"]
        body = _resolved["body"]
        arm = _resolved["arm"]
        gripper = _resolved["gripper"]
        # Two-gripper scenes may split via robot_source.urdf.{lgripper,rgripper}.
        # Simple configs mirror the resolved gripper onto both halves.
        lgripper = _resolved.get("lgripper", gripper)
        rgripper = _resolved.get("rgripper", gripper)

        print(f"{MSG_COLOR}[physics_backend] scene: {scene_config_resolved}{RESET}")
        print(f"{MSG_COLOR}[physics_backend] robot_model: {robot_model}{RESET}")
        print(f"{MSG_COLOR}[physics_backend] body: {body}, arm: {arm}, gripper: {gripper}{RESET}")
        print(f"{MSG_COLOR}[physics_backend] remap_tf: {remap_tf}{RESET}")
        print(f"{MSG_COLOR}[physics_backend] fake_slam: {fake_slam}{RESET}")

        # ---- launcher YAML metadata (engine binding + plugin params) -----
        # ``launcher_section`` and ``plugins_params_path`` were already loaded
        # above so the per-arg launcher-default fallback can use them; here
        # we just pull out the legacy ``physics`` / ``renders`` fields.
        industrial_bridge = ""
        renders_from_yaml: list = []
        if plugins_cfg_path:
            physics_section = launcher_section.get("physics") or {}
            if isinstance(physics_section, dict):
                industrial_bridge = str(physics_section.get("industrial_bridge") or "").strip()
            raw_renders = launcher_section.get("renders") or []
            if isinstance(raw_renders, list):
                renders_from_yaml = [str(r).strip() for r in raw_renders if str(r).strip()]
            print(
                f"{MSG_COLOR}[physics_backend] launcher_config: {plugins_cfg_path} "
                f"(industrial_bridge={industrial_bridge!r}, renders={renders_from_yaml}){RESET}"
            )
        elif launcher_config_arg:
            print(
                f"{ERR_COLOR}launcher_config file not found: {launcher_config_arg}\n"
                f"  Try cwd-relative path or a name under share/genie_sim_bringup/config/{RESET}"
            )
            sys.exit(1)

        # ---- resolve physics engine (package / executable / node name) ---
        try:
            engine_id, physics_package, physics_executable, physics_executable_gui, physics_node_name = (
                lu.resolve_physics_engine(
                    launcher_section,
                    default_engine="genie_sim_engine_isaacsim",
                )
            )
        except ValueError as exc:
            print(f"{ERR_COLOR}{exc}{RESET}")
            sys.exit(1)

        physics_launch_executable = physics_executable
        if headless != "true" and physics_executable_gui:
            physics_launch_executable = physics_executable_gui
        print(
            f"{MSG_COLOR}[physics_backend] engine: {engine_id} "
            f"(package={physics_package}, executable={physics_launch_executable}, name={physics_node_name}){RESET}"
        )

        # ---- resolve physics scene (may differ from scene_config) --------
        physics_scene_from_yaml = lu.resolve_physics_scene(launcher_section, engine_id)
        if physics_scene_from_yaml:
            physics_scene_resolved = lu.resolve_bringup_config_file(physics_scene_from_yaml)
            if not physics_scene_resolved:
                print(
                    f"{ERR_COLOR}[physics_backend] physics scene from YAML not found: "
                    f"{physics_scene_from_yaml}{RESET}"
                )
                sys.exit(1)
            print(
                f"{MSG_COLOR}[physics_backend] physics scene (from launcher YAML): " f"{physics_scene_resolved}{RESET}"
            )
        else:
            physics_scene_resolved = scene_config_resolved

        # ---- assets folder + stage dir -----------------------------------
        assets_folder = lu.resolve_assets_folder()
        print(f"{MSG_COLOR}[physics_backend] assets_folder: {assets_folder}{RESET}")
        scene_stem = Path(physics_scene_resolved).stem
        stage_dir = os.path.abspath(os.path.join(assets_folder, "scenes", scene_stem))
        stage_manifest = os.path.join(stage_dir, "manifest.json")

        manifest_present = os.path.isfile(stage_manifest)
        if manifest_present:
            print(
                f"{MSG_COLOR}[physics_backend] stage manifest present: {stage_manifest}\n"
                f"  assemble_scene will overwrite it (manifest is never cached).{RESET}"
            )
        else:
            print(
                f"{MSG_COLOR}[physics_backend] stage manifest absent: {stage_manifest} "
                f"— assemble_scene will create it.{RESET}"
            )

        # Snapshot the input scene yaml alongside manifest.json on EVERY
        # launch so post-mortem debugging always has the exact yaml that
        # drove this run, even after the user edits it.
        lu.stage_yaml_snapshot(stage_dir, physics_scene_resolved)
        print(f"{MSG_COLOR}[physics_backend] stage_manifest: {stage_manifest}{RESET}")

        # ---- remappings --------------------------------------------------
        remapping_physics = [
            ("/imu", "/imu/livox_back"),
        ]
        if "true" == remap_tf:
            remapping_physics.append(("/tf", "/tf_sim"))

        # ---- physics node parameters -------------------------------------
        # ROS 2's parameter resolution applies later entries in the
        # ``parameters=`` list ON TOP OF earlier ones. To make the launcher
        # YAML (passed as ``plugins_params_path``) provide defaults and CLI
        # ``key:=value`` overrides them, we:
        #
        #   1. Append ``plugins_params_path`` first (its
        #      ``genie_sim_engine.ros__parameters`` block lands as the
        #      engine's params).
        #   2. Append a dict of values for keys the user explicitly typed
        #      on the ros2 launch CLI. Anything we DIDN'T put in the dict
        #      falls through to whatever's in the yaml.
        #
        # The ``_OVERRIDABLE_FROM_YAML`` set is the keys we want yaml-
        # overridable. Everything else (``stage_manifest``,
        # ``init_joint_pos_json``, ``use_sim_time``, ``fake_slam``) is
        # always-set: those are derived in the launch file from other args
        # (the scene yaml or composer parameters), so they have no
        # meaningful yaml default to fall through to.
        _OVERRIDABLE_FROM_YAML = (
            "physics_hz",
            "render_hz",
            "realtime_factor",
            "num_envs",
            "physics_engine",
            "physics_solver",
            "physics_solver_substep",
            "physics_solver_iterations",
            "render_mode",
            "physics_engine_visualizer",
        )
        _resolved_values = {
            "physics_hz": physics_hz,
            "render_hz": render_hz,
            "realtime_factor": realtime_factor,
            "num_envs": num_envs,
            "physics_engine": physics_engine,
            "physics_solver": physics_solver,
            "physics_solver_substep": physics_solver_substep,
            "physics_solver_iterations": physics_solver_iterations,
            "render_mode": render_mode,
            "physics_engine_visualizer": physics_engine_visualizer,
        }
        cli_overrides: dict = {}
        for key in _OVERRIDABLE_FROM_YAML:
            if lu._cli_explicit(key):
                cli_overrides[key] = _resolved_values[key]

        physics_params: list = []
        if plugins_params_path:
            physics_params.append(plugins_params_path)
        physics_params.append(
            {
                "stage_manifest": stage_manifest,
                "fake_slam": "true" == fake_slam,
                # Gripper model from the scene yaml, forwarded so engine-side
                # consumers can select the matching end-effector handling. The
                # split lgripper/rgripper mirror when the scene doesn't
                # author separate hands.
                "lgripper": lgripper,
                "rgripper": rgripper,
                # JSON-encode so it survives launch's flat-string parameter
                # pipeline. Empty dict serializes to ``"{}"`` which the
                # consumer treats as "no override". See
                # ``parse_init_joint_pos`` in _isaacsim_params.py.
                "init_joint_pos_json": json.dumps(_init_joint_pos or {}),
                # Viewport camera pose for the Newton GL viewer / OVRtx
                # FreeCam — comes from the scene YAML's ``viewer_camera``
                # block.  JSON-encoded for the same reason as
                # ``init_joint_pos_json`` (launch_ros's parameter pipeline
                # mangles list-typed values).  Empty dict ``"{}"`` =
                # "no override, use Newton's default pose".
                "viewer_camera_json": json.dumps(_viewer_camera or {}),
                # Path to the launcher YAML that produced this session's
                # ros__parameters (or ``""`` when no launcher was
                # provided).  ``plugins_params_path`` is the temp file
                # bringup materialised from the launcher YAML at
                # ``load_launcher_yaml`` above; threading its path
                # through as its own ROS parameter lets engine-side
                # consumers reference the exact file that produced their
                # parameters.
                "launcher_yaml_path": plugins_params_path or "",
                **common_param,
                **cli_overrides,
            }
        )
        if cli_overrides:
            print(
                f"{MSG_COLOR}[physics_backend] CLI overrides applied "
                f"on top of launcher yaml: {sorted(cli_overrides.keys())}{RESET}"
            )

        # Application flags must precede --ros-args; rclpy's lifecycle
        # listener also parses this argv and rejects --gui in the ROS section.
        physics_args = []
        if headless != "true":
            physics_args.append("--gui")
        physics_args.extend(ros_log_args)

        nodes.append(
            TimerAction(
                period=2.0,
                actions=[
                    Node(
                        package=physics_package,
                        executable=physics_launch_executable,
                        name=physics_node_name,
                        output="both",
                        parameters=physics_params,
                        remappings=remapping_physics,
                        arguments=physics_args,
                    )
                ],
            )
        )

        # ---- industrial_bridge ------------------------------------------
        print(f"{MSG_COLOR}[physics_backend] industrial_bridge: {industrial_bridge!r}{RESET}")
        if industrial_bridge:
            nodes.append(
                lu.make_industrial_bridge_node(
                    industrial_bridge=industrial_bridge,
                    common_param=common_param,
                    remap_tf=remap_tf == "true",
                    ros_log_args=ros_log_args,
                )
            )

        # ---- interaction_tools ------------------------------------------
        interaction_tools = lu.perform(context, "interaction_tools")
        print(f"{MSG_COLOR}[physics_backend] interaction_tools: {interaction_tools}{RESET}")
        if interaction_tools == "true":
            nodes.append(
                Node(
                    package="genie_sim_tools",
                    executable="interaction_tools.py",
                    name="interaction_tools",
                    output="both",
                    parameters=[
                        {
                            "scene": physics_scene_resolved,
                            **common_param,
                        }
                    ],
                    arguments=ros_log_args,
                )
            )

        # ---- render_ovrtx -----------------------------------------------
        active_renders = lu.resolve_active_renders(renders_from_yaml)
        print(f"{MSG_COLOR}[physics_backend] render (active={sorted(active_renders)}){RESET}")

        if "render_ovrtx" in active_renders:
            nodes.append(
                lu.make_render_ovrtx_node(
                    stage_manifest=stage_manifest,
                    physics_node_name=physics_node_name,
                    plugins_params_path=plugins_params_path,
                    common_param=common_param,
                    ros_log_args=ros_log_args,
                )
            )

        if "render_isaacsim" in active_renders:
            nodes.append(
                lu.make_render_isaacsim_node(
                    stage_manifest=stage_manifest,
                    physics_node_name=physics_node_name,
                    plugins_params_path=plugins_params_path,
                    common_param=common_param,
                    ros_log_args=ros_log_args,
                )
            )

        # ---------------------------------------------------------------
        # Prebaked-pipeline hook. When the scene yaml declares a
        # ``robot.robot_source.prebaked.pipeline`` block, plumb it into
        # the standard pipeline as an extra stage that runs after
        # assemble_newton and before the runtime nodes. The hook is
        # responsible for engine-specific artefacts the standard stages
        # don't produce (e.g. ``robot_runtime.xml`` for the mujoco
        # backend). ``assemble_scene`` / ``assemble_newton`` still run —
        # single source of truth for render_layer.usda + manifest.json —
        # and ``assemble_robot`` is skipped when ``prebaked.usda`` is
        # set (see ``lu.needs_assemble_robot``).
        # ---------------------------------------------------------------
        try:
            prebaked_pipeline = lu.resolve_prebaked_pipeline(_src)
        except ValueError as exc:
            print(f"{ERR_COLOR}{exc}{RESET}")
            sys.exit(1)

        # Launcher-YAML override:
        # ``launcher.physics.engines.<engine>.assemble_hook`` lets a
        # backend declare the assemble script it wants for the current
        # scene, WITHOUT the scene YAML having to name the backend.
        # Same block shape as ``robot.robot_source.prebaked.pipeline``
        # (package / executable / name).  When present, takes
        # precedence over the scene's ``prebaked.pipeline`` — the
        # scene stays backend-neutral; the launcher owns "how to
        # assemble for me".
        _launcher_hook = _engine_cfg.get("assemble_hook") or {}
        if isinstance(_launcher_hook, dict) and _launcher_hook.get("executable"):
            launcher_hook = {
                "package": str(_launcher_hook.get("package") or "").strip(),
                "executable": str(_launcher_hook.get("executable") or "").strip(),
                "name": str(_launcher_hook.get("name") or "").strip() or Path(_launcher_hook.get("executable")).stem,
            }
            missing = [k for k in ("package", "executable") if not launcher_hook[k]]
            if missing:
                print(
                    f"{ERR_COLOR}[physics_backend] launcher.assemble_hook missing " f"required fields {missing}{RESET}"
                )
                sys.exit(1)
            if prebaked_pipeline:
                print(
                    f"{MSG_COLOR}[physics_backend] launcher.assemble_hook overrides "
                    f"scene.prebaked.pipeline (was "
                    f"{prebaked_pipeline['package']}/{prebaked_pipeline['executable']}, "
                    f"now {launcher_hook['package']}/{launcher_hook['executable']}){RESET}"
                )
            prebaked_pipeline = launcher_hook

        if prebaked_pipeline:
            print(
                f"{MSG_COLOR}[physics_backend] prebaked.pipeline: "
                f"package={prebaked_pipeline['package']}, "
                f"executable={prebaked_pipeline['executable']}, "
                f"name={prebaked_pipeline['name']}{RESET}"
            )

        # ---------------------------------------------------------------
        # Final orchestration: gate ALL runtime ``nodes`` behind the
        # assemble pipeline so they only start once the stage is ready.
        # ``assemble_scene`` ALWAYS runs (manifest is never cached);
        # ``assemble_robot`` runs only on a cache miss and produces the
        # ``robot.usda`` that ``assemble_scene`` then consumes.
        #
        #   * scene only           -> [a_scene, gate_runtime]            (1 stage)
        #   * robot + scene        -> [a_robot, gate_scene, gate_runtime] (2 stages)
        #
        # IMPORTANT: when the robot pipeline is active, ``a_scene`` is
        # spawned by ``gate_scene`` (OnProcessExit(a_robot) -> a_scene),
        # so it MUST NOT appear at the top level — otherwise launch
        # raises ``executed more than once``.
        #
        # NOTE: ``make_assemble_pipeline`` snapshots ``runtime_nodes`` at
        # call time via ``list(...)``, so it MUST be called AFTER all
        # ``nodes.append(...)`` above — otherwise late-appended nodes
        # (physics engine, render_ovrtx, etc.) silently never spawn.
        # ---------------------------------------------------------------
        # ``assemble_style`` and ``assemble_pipeline`` are data-driven
        # dispatch fields from the active engine's launcher-YAML block.
        # Empty style falls through to the legacy built-in mapping;
        # downstream backends either provide a standalone hook or name
        # their registered core pipeline and import modules here.
        assemble_style = ""
        assemble_pipeline = {}
        if isinstance(_engine_cfg, dict):
            assemble_style = str(_engine_cfg.get("assemble_style") or "").strip()
            raw_assemble_pipeline = _engine_cfg.get("assemble_pipeline")
            assemble_pipeline = {} if raw_assemble_pipeline is None else raw_assemble_pipeline
        if not isinstance(assemble_pipeline, dict):
            print(f"{ERR_COLOR}[physics_backend] assemble_pipeline must be a mapping{RESET}")
            sys.exit(1)

        if assemble_pipeline and assemble_style != "registered_pipeline":
            print(
                f"{ERR_COLOR}[physics_backend] assemble_pipeline is only valid with "
                f"assemble_style='registered_pipeline'{RESET}"
            )
            sys.exit(1)

        if assemble_style == "registered_pipeline":
            raw_robot_stage = assemble_pipeline.get("robot_stage")
            if raw_robot_stage is not None and not isinstance(raw_robot_stage, str):
                print(f"{ERR_COLOR}[physics_backend] assemble_pipeline.robot_stage must be a string{RESET}")
                sys.exit(1)
            robot_stage = (raw_robot_stage or "").strip()
            needs_robot_assemble = bool(robot_stage) and lu.uses_urdf_assemble_pipeline(_src)
        else:
            needs_robot_assemble = lu.needs_assemble_robot(_src)

        a_scene, a_robot, gate_scene, gate_runtime = lu.make_assemble_pipeline(
            scene_resolved=physics_scene_resolved,
            stage_dir=stage_dir,
            assets_folder=assets_folder,
            always_regenerate_robot_usd=always_regenerate_robot_usd,
            runtime_nodes=nodes,
            physics_engine=engine_id,
            launcher_config=launcher_config_arg,
            assemble_style=assemble_style,
            assemble_pipeline=assemble_pipeline,
            assemble_robot_artefacts=(
                [
                    os.path.join(stage_dir, "robot.urdf"),
                    os.path.join(stage_dir, "robot.usda"),
                ]
                if needs_robot_assemble
                else None
            ),
            urdf_text=(
                lu.build_robot_description(robot_source=_src, resolved=_resolved)
                if lu.uses_urdf_assemble_pipeline(_src)
                else None
            ),
            pre_runtime_hook=prebaked_pipeline,
        )

        if a_robot is not None:
            return [a_robot, gate_scene, gate_runtime]
        return [a_scene, gate_runtime]

    return LaunchDescription(
        [
            *declared_arguments,
            OpaqueFunction(function=parse_args),
        ]
    )
