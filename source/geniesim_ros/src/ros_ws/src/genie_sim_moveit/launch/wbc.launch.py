# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import importlib.util
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def _load_moveit_launch_utils():
    here = os.path.dirname(os.path.realpath(__file__))
    spec = importlib.util.spec_from_file_location("_moveit_launch_utils", os.path.join(here, "moveit_launch_utils.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _launch_setup(context):
    robot_model = context.perform_substitution(LaunchConfiguration("robot_model"))
    arm = context.perform_substitution(LaunchConfiguration("arm"))
    gripper = context.perform_substitution(LaunchConfiguration("gripper"))
    use_human_priors = context.perform_substitution(LaunchConfiguration("use_human_priors")).lower() in (
        "true",
        "1",
        "yes",
        "on",
    )

    use_ros2_control = context.perform_substitution(LaunchConfiguration("use_ros2_control")).lower() in (
        "true",
        "1",
        "yes",
        "on",
    )

    # A/B switch for the GenieBioIK "human prior" goals (torso-straight,
    # chassis-pin, head LookAt). kinematics.yaml ships the priors enabled;
    # kinematics_vanilla.yaml zeroes every weight so the plugin falls back
    # to upstream BioIK behaviour. Both files share identical plugin
    # selection / IK links / timeouts so the only experimental variable is
    # the goal stack.
    kinematics_file = "config/kinematics.yaml" if use_human_priors else "config/kinematics_vanilla.yaml"
    print(
        f"[wbc.launch.py] use_human_priors={use_human_priors} -> kinematics file '{kinematics_file}'",
        flush=True,
    )

    if gripper == "none":
        urdf_file = f"$(find genie_sim_robot_model)/urdf/genie_{robot_model}_{arm}.urdf"
    else:
        urdf_file = f"$(find genie_sim_robot_model)/urdf/genie_{robot_model}_{arm}_{gripper}.urdf"

    moveit_config = (
        MoveItConfigsBuilder("genie", package_name="genie_sim_moveit")
        .robot_description(
            file_path="config/genie.urdf.xacro",
            mappings={
                "ros2_control_hardware_plugin": "genie_sim_control/GenieSimRobotInterface",
                "urdf_file": urdf_file,
            },
        )
        .robot_description_semantic(
            file_path="config/genie.srdf.xacro",
            mappings={"gripper": gripper if gripper != "none" else "swiftpicker"},
        )
        .robot_description_kinematics(file_path=kinematics_file)
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .joint_limits(file_path="config/joint_limits.yaml")
        .to_moveit_configs()
    )

    # Humble expects one planner and a space-separated request-adapter chain.
    # Its time parameterization adapter runs on requests; Jazzy uses responses.
    if os.environ.get("ROS_DISTRO", "").strip().lower() == "humble":
        ompl = moveit_config.planning_pipelines["ompl"]
        planners = ompl.pop("planning_plugins", None)
        if isinstance(planners, (list, tuple)) and planners:
            ompl["planning_plugin"] = planners[0]
        elif isinstance(planners, str) and planners:
            ompl["planning_plugin"] = planners
        ompl["request_adapters"] = (
            "default_planner_request_adapters/AddTimeOptimalParameterization "
            "default_planner_request_adapters/ResolveConstraintFrames "
            "default_planner_request_adapters/FixWorkspaceBounds "
            "default_planner_request_adapters/FixStartStateBounds "
            "default_planner_request_adapters/FixStartStateCollision "
            "default_planner_request_adapters/FixStartStatePathConstraints"
        )
        ompl.pop("response_adapters", None)
        print("[wbc.launch.py] using Humble OMPL planner and request-adapter parameters", flush=True)

    # Widen URDF joint <limit> by a small pad so MoveIt's CheckStartStateBounds
    # tolerates the sub-mrad drift that MuJoCo's soft joint limit lets through
    # under contact load.  CheckStartStateBounds reads URDF limits directly
    # via the robot model, so the per-pipeline ``start_state_max_bounds_error``
    # (which only the FixStartStateBounds adapter consults) doesn't help here.
    # Mutating ``moveit_config.robot_description`` before ``to_dict()`` is the
    # only point where the URDF that move_group ingests can be patched without
    # editing the cached static URDF files in genie_sim_robot_model/urdf/.
    # Helper lives in genie_sim_robot_model (a shared dep of both this package
    # and genie_sim_bringup) to avoid a bringup<->moveit dependency loop.
    try:
        from genie_sim_robot_model.urdf_utils import pad_urdf_joint_limits  # type: ignore

        _urdf = moveit_config.robot_description.get("robot_description", "")
        if _urdf:
            _widened = pad_urdf_joint_limits(_urdf, 0.01, 0.001)
            if _widened is not _urdf:
                moveit_config.robot_description["robot_description"] = _widened
                print(
                    "[wbc.launch.py] widened MoveIt URDF joint <limit> by "
                    "+/-0.01 rad / +/-0.001 m to absorb MuJoCo soft-limit drift",
                    flush=True,
                )
    except Exception as exc:  # noqa: BLE001
        print(f"[wbc.launch.py] joint-limit widening skipped: {exc!r}", flush=True)

    mlu = _load_moveit_launch_utils()
    nodes = [mlu.moveit_joint_states_bridge_node(use_sim_time=False)]

    nodes.append(
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=[
                moveit_config.to_dict(),
                {
                    "publish_robot_description_semantic": True,
                    "allow_trajectory_execution": True,
                    "publish_planning_scene": True,
                    "publish_geometry_updates": True,
                    "publish_state_updates": True,
                    "publish_transforms_updates": True,
                    "monitor_dynamics": False,
                    "use_sim_time": False,
                    "trajectory_execution.allowed_start_tolerance": 0.0,
                    # Wide default workspace half-extent (meters) for OMPL planar-joint bounds.
                    # Without this, FixWorkspaceBounds clamps planar_joint/trans_x and planar_joint/trans_y
                    # to a narrow default box that excludes the actual map -> base_link pose,
                    # producing "Skipping invalid start state (invalid bounds)" for the
                    # chassis group. 200 m matches the simulator's planar workspace.
                    #
                    # NOTE: `FixWorkspaceBounds::initialize` reads this parameter from the
                    # planning-pipeline namespace (i.e. `ompl.default_workspace_bounds` for
                    # the "ompl" pipeline), NOT from the move_group root. The flat key is
                    # kept as a belt-and-braces fallback for tooling that reads either path.
                    "ompl.default_workspace_bounds": 200.0,
                    "default_workspace_bounds": 200.0,
                    # Allow FixStartStateBounds to nudge joints up to 0.2 rad (~11.5 deg)
                    # back into bounds. The simulator publishes joint values with measurable
                    # noise / encoder discretisation that occasionally pushes idx02_body_joint2
                    # (and others) past the URDF <limit> by more than the default 0.05 rad,
                    # which OMPL then rejects as "invalid bounds" for any composite group
                    # containing that joint (wbc_headless, wbc, mobile_base_manipulator).
                    "ompl.start_state_max_bounds_error": 0.2,
                    # Tell jazzy's CheckStartStateBounds adapter to actively NORMALISE
                    # joint values that are slightly outside their URDF limits (e.g.
                    # idx02_body_joint2 drifting past <limit lower/upper> by sub-mrad
                    # under contact load), instead of just rejecting the request.
                    # Without this, planning aborts on every grasp attempt that lands
                    # the arm near a hard stop.  The adapter's renormalisation pass
                    # (check_start_state_bounds.cpp:136-147 in moveit_ros 2.12.4) only
                    # writes the fixed state back when this flag is true; upstream
                    # default is false.  Lives on the move_group root namespace, not
                    # under the ompl pipeline namespace, because the request adapter
                    # is shared across pipelines.
                    "fix_start_state": True,
                },
            ],
            remappings=mlu.MOVEIT_MOVE_GROUP_REMAPPINGS,
            additional_env={"DISPLAY": os.environ.get("DISPLAY", "")},
        )
    )

    nodes.append(
        Node(
            package="rviz2",
            executable="rviz2",
            name="moveit_rviz",
            output="screen",
            arguments=[
                "-d",
                str(moveit_config.package_path / "config" / "moveit.rviz"),
            ],
            parameters=[
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,
                moveit_config.robot_description_kinematics,
                moveit_config.planning_pipelines,
                moveit_config.joint_limits,
                {"use_sim_time": False},
            ],
            condition=IfCondition(LaunchConfiguration("use_rviz")),
        )
    )

    nodes.append(
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="moveit_robot_state_publisher",
            output="screen",
            parameters=[
                moveit_config.robot_description,
                {"use_sim_time": False},
            ],
            remappings=[("/robot_description", "/moveit_robot_description")],
        )
    )

    # When the arm is driven directly via /joint_command (e.g. an external
    # motion driver), the ros2_control node + controllers are not just
    # unnecessary but harmful: the GenieSim hardware interface republishes
    # /joint_command at the CM update_rate, fighting the external publisher
    # (the arm jitters, payloads
    # get flung). use_ros2_control:=false leaves move_group only (the
    # /compute_ik, /compute_fk services), so /joint_command has a single source.
    if not use_ros2_control:
        print(
            "[wbc.launch.py] use_ros2_control=false → move_group only (no ros2_control "
            "node / controllers); drive the arm via /joint_command.",
            flush=True,
        )
        return nodes

    nodes.append(
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            parameters=[
                # Pass the URDF as a node parameter — controller_manager
                # checks this first.  ``moveit_config.robot_description``
                # is a dict ``{"robot_description": "<urdf xml>"}``.
                moveit_config.robot_description,
                str(moveit_config.package_path / "config" / "ros2_controllers.yaml"),
                {"use_sim_time": False},
            ],
            # Belt + braces: ALSO remap the topic ``robot_description``
            # so a sim-side publisher (or moveit's own) can be the source
            # if the parameter wasn't applied.  In ROS 2 Jazzy,
            # ``ros2_control_node`` subscribes to the GLOBAL
            # ``/robot_description`` topic regardless of whether the
            # parameter is set — and logs the spurious "Waiting for
            # data" warning unless a publisher exists.  Remapping to
            # ``/moveit_robot_description`` (the topic the launch's
            # moveit_robot_state_publisher writes to) gives the CM a
            # publisher to lock onto.  The remap key is ``robot_description``
            # (relative) NOT ``~/robot_description`` — the CM subscribes
            # to the global topic, not the node's private namespace.
            remappings=[("robot_description", "/moveit_robot_description")],
            output="screen",
        )
    )

    nodes.append(
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                "joint_state_broadcaster",
                "simple_arm_l_controller",
                "simple_arm_r_controller",
                "simple_waist_controller",
                "simple_torso_controller",
                # simple_body_controller is configured but NOT auto-spawned:
                # it claims the same idx01..idx05_body_joint position
                # interfaces as simple_waist_controller + simple_torso_controller,
                # so spawning all three at once produces a resource-claim
                # conflict in ros2_control's CM. Activate manually with
                # `ros2 control switch_controllers --activate simple_body_controller
                #  --deactivate simple_waist_controller simple_torso_controller`
                # when you want to drive the full body as a single chain.
                "simple_head_controller",
                "gripper_l_controller",
                "gripper_r_controller",
                "chassis_controller",
                "chassis_servo_controller",
            ],
            output="screen",
        )
    )

    return nodes


def generate_launch_description():
    ld = LaunchDescription()

    ld.add_action(
        DeclareLaunchArgument(
            "use_ros2_control",
            default_value="true",
            description=(
                "Start the ros2_control node + controllers (true, default) for MoveIt "
                "trajectory execution. Set false to launch move_group ONLY (the "
                "/compute_ik, /compute_fk services) when you drive the arm directly via "
                "/joint_command (e.g. an external motion driver) — avoids two publishers "
                "fighting on /joint_command."
            ),
        )
    )

    ld.add_action(
        DeclareLaunchArgument(
            "robot_model",
            default_value="g2",
            description=(
                "Robot body model.  Combined with ``arm`` and ``gripper`` to "
                "build the URDF path "
                "``$(find genie_sim_robot_model)/urdf/genie_<robot_model>_<arm>_<gripper>.urdf``.  "
                "Available combinations in genie_sim_robot_model/urdf/: g2."
            ),
        ),
    )
    ld.add_action(
        DeclareLaunchArgument(
            "arm",
            default_value="crsB",
            description="Arm model (e.g. crs, crsB) — second URDF filename token.",
        ),
    )
    ld.add_action(
        DeclareLaunchArgument(
            "gripper",
            default_value="swiftpicker",
            description=(
                "Gripper model (e.g. swiftpicker, omnipicker, none) — third "
                "URDF filename token.  ``none`` drops the gripper segment "
                "from the filename.  Also selects the SRDF fragment loaded "
                "from ``config/srdf_grippers/<gripper>.srdf.xacro`` (which "
                "carries the gripper-specific planning groups, passive mimic "
                "joints, and disable_collisions pairs).  ``none`` falls back "
                "to the swiftpicker fragment for SRDF purposes."
            ),
        ),
    )
    ld.add_action(
        DeclareLaunchArgument("use_rviz", default_value="true"),
    )
    ld.add_action(
        DeclareLaunchArgument(
            "use_human_priors",
            default_value="true",
            description=(
                "If true (default), load config/kinematics.yaml so the GenieBioIK plugin "
                "applies the human-like priors (torso-straight, chassis-pin, head LookAt). "
                "If false, load config/kinematics_vanilla.yaml -- every prior weight is "
                "zeroed and the plugin reverts to upstream BioIK behaviour. Use this "
                "for A/B comparison without rebuilding."
            ),
        ),
    )

    ld.add_action(OpaqueFunction(function=_launch_setup))

    return ld
