# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""
app.launch.py

Unified bringup composer. The launcher YAML (``launcher_config``) is the
single source of truth for which physics engine and render backends to start.
Physics is delegated to ``physics_backend.launch.py``.

``launcher_config`` is **required** — the launch aborts with a red error if
it is not provided.

Owns the stage-independent nodes (RSP, rviz, teleop, navigation) and
delegates the stage-dependent slice to the physics sub-launch.
"""

import os
import sys
import importlib.util

from launch import LaunchDescription
from launch.actions import OpaqueFunction, DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

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
    # Build common args, then override launcher_config to be required (no default).
    declared_arguments = [a for a in lu.common_declared_arguments() if a.name != "launcher_config"]
    declared_arguments.append(
        DeclareLaunchArgument(
            name="launcher_config",
            default_value="",
            description=(
                "Launcher YAML (required). Controls physics engine, render backends, "
                "and industrial_bridge. See share/genie_sim_bringup/config/launcher*.yaml."
            ),
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            name="scene",
            default_value="",
            description=(
                "Scene config YAML (required). Supplies robot_model / body / arm / gripper "
                "and scene layout. Resolved against cwd and share/genie_sim_bringup/config/."
            ),
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            name="physics_hz",
            default_value="100.0",
            description="Isaac Sim physics step rate (Hz)",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            name="render_hz",
            default_value="30.0",
            description="Isaac Sim render target rate (Hz, decoupled from physics)",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            name="headless",
            default_value="true",
            choices=["true", "false"],
            description="Run physics engine in headless mode (no GUI)",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            name="always_regenerate_robot_usd",
            default_value="false",
            choices=["true", "false"],
            description=(
                "If true, force the URDF->USD robot conversion (assemble_robot) and the "
                "downstream scene assembly (assemble_scene) to run even when a cached "
                "manifest.json already exists in the per-scene stage dir."
            ),
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            name="physics_engine",
            default_value="isaac_physx",
            choices=["isaac_physx", "isaac_newton", "newton"],
            description=(
                "'isaac_physx' (default), 'isaac_newton' (Isaac wrapper, "
                "XPBD/MuJoCo), 'newton' (direct ModelBuilder, VBD cloth)."
            ),
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            name="physics_solver",
            default_value="mujoco-warp",
            choices=["mujoco-warp", "xpbd", "featherstone", "semiImplicit"],
            description=(
                "Solver for the Newton engine (ignored when physics_engine=physx). "
                "'mujoco-warp' = MuJoCo-Warp (default). 'xpbd' = Extended Position-Based Dynamics. "
                "'featherstone' = recursive Newton-Euler. 'semiImplicit' = semi-implicit Euler."
            ),
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            name="physics_solver_substep",
            default_value="10",
            description=(
                "sim_substeps for physics_engine=newton. Matches franka demo's "
                "`self.sim_substeps`. Default 10 (franka). The solver dt is "
                "computed exactly like franka: sim_dt = (1/physics_hz) / "
                "sim_substeps. physics_hz is the main control variable; "
                "leave substeps at 10 unless you have a specific reason."
            ),
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            name="physics_solver_iterations",
            default_value="5",
            description=(
                "VBD iterations per substep (physics_engine=newton). Matches "
                "franka demo's `self.iterations`. Default 5 (franka). "
                "Total iterations per frame = sim_substeps × iterations = 50 "
                "at defaults (franka convergence budget)."
            ),
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            name="render_mode",
            default_value="raster",
            choices=["raster", "pathtrace", "offline"],
            description=(
                "RTX submode for the live viewport. "
                "raster = ~16ms/frame (RaytracedLighting; default debug "
                "viewport — raster + RT shadows + RT denoise). "
                "pathtrace = ~49ms/frame (RealTimePathTracing; kit default "
                "but eats physics_hz budget). "
                "offline = 100-300ms/frame (PathTracing; offline-quality, for "
                "screenshots / videos). "
                "Storm isn't supported (not bundled in the base kit our "
                "newton-standalone path uses)."
            ),
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            name="physics_engine_visualizer",
            default_value="",
            description=(
                "Visualizer backend for physics_engine=newton (genie_sim_engine_newton_node.py). "
                "'none' headless, 'newton' Newton GL ViewerGL, 'ovrtx' in-process OVRtx, "
                "'rerun' placeholder (not implemented). Empty (default) means 'don't "
                "override' — falls through to the "
                "launcher yaml's physics_engine_visualizer key. Honored only by "
                "physics_engine=newton; ignored by isaac_physx / isaac_newton (those use "
                "the Kit viewport, gated by headless:=false)."
            ),
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            name="realtime_factor",
            default_value="1.0",
            description=(
                "Realtime factor: 1.0 = realtime (default), 0.1 = 10× slower, 2.0 = 2× faster. "
                "Scales the wall-clock period between physics steps and the render cadence "
                "proportionally, keeping sim_time consistent. Can also be set in the launcher "
                "yaml's ros__parameters block."
            ),
        )
    )

    def parse_args(context):
        # ---- launcher_config is required --------------------------------
        launcher_config_arg = lu.perform(context, "launcher_config").strip()
        if not launcher_config_arg:
            print(
                f"\n{ERR_COLOR}❌  launcher_config is required for app.launch.py.\n"
                f"   Provide a launcher YAML that declares the physics engine:\n"
                f"   ros2 launch genie_sim_bringup app.launch.py \\\n"
                f"       launcher_config:=launcher_physx.yaml scene:=<your_scene.yaml>\n"
                f"   See share/genie_sim_bringup/config/launcher*.yaml for examples.{RESET}\n"
            )
            sys.exit(1)

        plugins_cfg_path = lu.resolve_bringup_config_file(launcher_config_arg)
        if not plugins_cfg_path:
            print(
                f"\n{ERR_COLOR}❌  launcher_config file not found: {launcher_config_arg}\n"
                f"   Try a cwd-relative path or a name under "
                f"share/genie_sim_bringup/config/{RESET}\n"
            )
            sys.exit(1)

        # ---- experimental launcher warning ------------------------------
        # launcher_ovrtx_isaac_newton uses Isaac Sim's in-development Newton
        # wrapper; launcher_newton_*.yaml use the standalone Newton physics
        # engine. Both are still maturing — make the user acknowledge before
        # we go further so they don't mistake unknown failures for bugs in
        # their scene/config.
        _cfg_stem = os.path.splitext(os.path.basename(plugins_cfg_path))[0]
        if _cfg_stem == "launcher_ovrtx_isaac_newton" or _cfg_stem.startswith("launcher_newton_"):
            WARN_COLOR = "\033[33m"
            print(
                f"\n{WARN_COLOR}"
                f"╔════════════════════════════════════════════════════════════════╗\n"
                f"║  ⚠  EXPERIMENTAL launcher: {_cfg_stem + '.yaml':<35} ║\n"
                f"╠════════════════════════════════════════════════════════════════╣\n"
                f"║  This launcher uses an in-development physics backend          ║\n"
                f"║  (Isaac's Newton wrapper, or standalone Newton). You may       ║\n"
                f"║  encounter unknown issues: crashes, instability, missing       ║\n"
                f"║  features.                                                     ║\n"
                f"║                                                                ║\n"
                f"║  For stable IsaacSim simualtion use launcher_ovrtx_isaac_physx ║\n"
                f"╚════════════════════════════════════════════════════════════════╝"
                f"{RESET}\n"
            )
            try:
                input(f"{WARN_COLOR}Press [ENTER] to continue, or Ctrl+C to abort...{RESET}")
            except (KeyboardInterrupt, EOFError):
                print(f"\n{ERR_COLOR}Aborted by user.{RESET}\n")
                sys.exit(130)

        launcher_section, _plug_yaml, plugins_params_path = lu.load_launcher_yaml(plugins_cfg_path)

        # ---- resolve physics engine from YAML ---------------------------
        try:
            physics_engine, physics_package, physics_executable, _physics_executable_gui, physics_node_name = (
                lu.resolve_physics_engine(launcher_section, default_engine="genie_sim_engine_isaacsim")
            )
        except ValueError as exc:
            print(f"{ERR_COLOR}❌  {exc}{RESET}")
            sys.exit(1)

        print(
            f"{MSG_COLOR}launcher_config: {plugins_cfg_path}{RESET}\n"
            f"{MSG_COLOR}physics engine:  {physics_engine} "
            f"(package={physics_package}, executable={physics_executable}){RESET}"
        )

        # ---- common args ------------------------------------------------
        use_sim_time = lu.perform(context, "use_sim_time")
        launch_robot_model = lu.perform(context, "robot_model")
        launch_body = lu.perform(context, "body")
        launch_arm = lu.perform(context, "arm")
        launch_gripper = lu.perform(context, "gripper")
        remap_tf = lu.perform(context, "remap_tf")
        rviz_config_file = lu.perform(context, "rviz_config_file")
        tf_prefix = lu.perform(context, "tf_prefix")
        debug = lu.perform(context, "debug")
        common_param = {"use_sim_time": use_sim_time == "true"}
        log_level = lu.perform(context, "log_level").strip()
        ros_log_args = ["--ros-args", "--log-level", log_level]

        # ---- scene YAML -------------------------------------------------
        scene_info = lu.resolve_scene_yaml_robot_params(
            context,
            required=True,
            robot_model=launch_robot_model,
            body=launch_body,
            arm=launch_arm,
            gripper=launch_gripper,
        )
        scene_config_resolved = scene_info["scene_resolved"]
        _src = scene_info["robot_source"]
        _resolved = scene_info["resolved"]
        robot_model = _resolved["robot_model"]
        body = _resolved["body"]
        arm = _resolved["arm"]
        gripper = _resolved["gripper"]
        # Two-gripper scenes may split via robot.robot_source.urdf.{lgripper,rgripper}.
        # SceneConfig.resolve_robot_params defaults each half to the resolved simple
        # gripper so single-gripper configs still behave the same way as before.
        lgripper = _resolved.get("lgripper", gripper)
        rgripper = _resolved.get("rgripper", gripper)

        print(f"{MSG_COLOR}scene:        {scene_config_resolved}{RESET}")
        print(f"{MSG_COLOR}robot_model:  {robot_model}{RESET}")
        print(f"{MSG_COLOR}body: {body}  arm: {arm}  gripper: {gripper}{RESET}")
        print(f"{MSG_COLOR}remap_tf: {remap_tf}  tf_prefix: '{tf_prefix}'  debug: {debug}{RESET}")
        print(f"{MSG_COLOR}common_param: {common_param}{RESET}")

        robot_description = lu.build_robot_description(
            robot_source=_src,
            resolved=_resolved,
        )
        if not robot_description:
            return

        publish_frequency = 100.0
        actions = []

        # ---- stage-independent nodes (start immediately) ----------------
        actions.append(
            lu.make_robot_state_publisher_node(
                robot_description=robot_description,
                publish_frequency=publish_frequency,
                tf_prefix=tf_prefix,
                common_param=common_param,
                remap_tf=remap_tf == "true",
                ros_log_args=ros_log_args,
            )
        )

        rviz_config_path = lu.resolve_rviz_config(rviz_config_file, robot_model)
        print(f"{MSG_COLOR}rviz config: {rviz_config_path}{RESET}")
        if lu.perform(context, "rviz") != "false":
            actions.append(
                lu.make_rviz_node(
                    rviz_config_file=rviz_config_path,
                    common_param=common_param,
                    ros_log_args=ros_log_args,
                )
            )

        teleop = lu.perform(context, "teleop")
        print(f"{MSG_COLOR}teleop: {teleop}{RESET}")
        navigation = lu.perform(context, "navigation")
        navigation_mode = lu.perform(context, "navigation_mode")
        print(f"{MSG_COLOR}navigation: {navigation} (mode={navigation_mode}){RESET}")

        # Chassis servo + teleop + navigation are mobile-base nodes
        # that assume a Genie-family four-wheel-steer chassis
        # (cmd_twist -> cmd_4ws -> per-wheel drive/steer joints).
        # Other robot families
        # (agilex/arx/franka/universal_robots) ship without a chassis,
        # so skip the whole block — even if teleop/navigation flags are
        # set, they'd just spew "no joints found" errors against an
        # arm-only URDF.
        #
        # Accept ``robot_model == "genie"`` (legacy family alias used
        # by older scene YAMLs) or any ``robot_model`` whose upper-case
        # form begins with ``G2`` (case-insensitive family prefix).
        # Chassis topology (2WD-diagonal vs 4WD) comes from the scene
        # yaml's ``robot.robot_source.chassis`` key; defaults to 4WD.
        # yaml's ``robot.robot_source.chassis`` key; defaults to 4WD.
        _rm_upper = robot_model.upper()
        _chassis = _src.get("chassis", "4wd") if isinstance(_src, dict) else "4wd"
        if robot_model == "genie" or _rm_upper.startswith("G2"):
            # Chassis servo node — required by both teleop and navigation
            # (they publish cmd_twist / steering targets that the servo
            # node translates into per-wheel drive/steer joint commands).
            # Launched once when either flag is true so we don't double-spawn.
            if teleop == "true" or navigation == "true":
                actions.append(
                    lu.make_chassis_controller_node(
                        chassis=_chassis,
                        common_param=common_param,
                        ros_log_args=ros_log_args,
                    )
                )

            if teleop == "true":
                actions.extend(
                    lu.make_teleop_nodes(
                        publish_frequency=publish_frequency,
                        common_param=common_param,
                        ros_log_args=ros_log_args,
                    )
                )

            if navigation == "true":
                actions.append(
                    lu.make_navigation_node(
                        navigation_mode=navigation_mode,
                        common_param=common_param,
                        ros_log_args=ros_log_args,
                    )
                )
        elif teleop == "true" or navigation == "true":
            print(
                f"{MSG_COLOR}skipping chassis / teleop / navigation: "
                f"robot_model='{robot_model}' has no Genie-style chassis{RESET}"
            )

        # ---- physics sub-launch (engine-specific, stage-gated) ----------
        here = os.path.dirname(os.path.realpath(__file__))
        sub_launch = os.path.join(here, "physics_backend.launch.py")
        sub_args = {
            "physics_hz": LaunchConfiguration("physics_hz"),
            "render_hz": LaunchConfiguration("render_hz"),
            "headless": LaunchConfiguration("headless"),
            "always_regenerate_robot_usd": LaunchConfiguration("always_regenerate_robot_usd"),
            "physics_engine": LaunchConfiguration("physics_engine"),
            "physics_solver": LaunchConfiguration("physics_solver"),
            "physics_solver_substep": LaunchConfiguration("physics_solver_substep"),
            "physics_solver_iterations": LaunchConfiguration("physics_solver_iterations"),
            "render_mode": LaunchConfiguration("render_mode"),
            "physics_engine_visualizer": LaunchConfiguration("physics_engine_visualizer"),
            "realtime_factor": LaunchConfiguration("realtime_factor"),
        }

        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(sub_launch),
                launch_arguments=sub_args.items(),
            )
        )

        return actions

    return LaunchDescription(
        [
            *declared_arguments,
            OpaqueFunction(function=parse_args),
        ]
    )
