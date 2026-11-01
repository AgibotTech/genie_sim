# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pick up an Ethernet plug and insert it into a network device."""

from dataclasses import dataclass

from client.llm_gen.robot_api import (
    FollowTrajectoryConfig,
    GraspConfig,
    MoveConfig,
    RobotAPI,
)
import numpy as np
from scipy.spatial.transform import Rotation


PLUG_BODY_PART = "body"
PLUG_CABLE_PART = "part_00/visual"
PLUG_ANCHOR_PART = "body/EthernetPlugAnchor"
PORT_ANCHOR_PART = "EthernetPortAnchor"
AXIAL_FIX_THRESHOLD = 0.01
LATERAL_FIX_THRESHOLD = 0.005
ATTRACTOR_AXIAL_THRESHOLD = 0.001
ATTRACTOR_LATERAL_THRESHOLD = 0.005
POST_INSERT_STABILITY_THRESHOLD = 0.002
BASE_GRASP_WXYZ = np.array(
    [
        0.6708123124346509,
        0.22363103873585646,
        0.6708123124346481,
        -0.22363103873585732,
    ],
    dtype=float,
)


def mirror_grasp_quaternion(quaternion_wxyz) -> np.ndarray:
    mirror = np.diag([1.0, -1.0, 1.0])
    mirrored = mirror @ quat_wxyz_to_matrix(quaternion_wxyz) @ mirror
    return matrix_to_quat_wxyz(mirrored)

DESCRIPTION = {
    "task_name": 'Plug an Ethernet cable into a network device',
    "english_task_name": (
        'Plug an Ethernet cable into a network device'
    ),
    "init_scene_text": (
        "A network device and an Ethernet plug rest on a table."
    ),
}


@dataclass(frozen=True)
class InsertionTrajectory:
    preinsert_anchor_pose: np.ndarray
    anchor_waypoints: np.ndarray
    body_waypoints: np.ndarray


@dataclass(frozen=True)
class AnchorError:
    axial: float
    lateral: float


def quat_wxyz_to_matrix(quaternion) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(quaternion, dtype=float)
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()


def matrix_to_quat_wxyz(matrix) -> np.ndarray:
    qx, qy, qz, qw = Rotation.from_matrix(
        np.asarray(matrix, dtype=float)
    ).as_quat()
    return np.array([qw, qx, qy, qz])


def make_pose(position, quaternion_wxyz) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = quat_wxyz_to_matrix(quaternion_wxyz)
    pose[:3, 3] = np.asarray(position, dtype=float)
    return pose


def pose_to_xyz_wxyz(pose) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    return np.concatenate(
        (pose[:3, 3], matrix_to_quat_wxyz(pose[:3, :3]))
    )


def build_grasp_poses(
    x_offsets,
    y_offsets,
    z_angles_deg,
    base_quaternion=BASE_GRASP_WXYZ,
) -> np.ndarray:
    base_rotation = quat_wxyz_to_matrix(base_quaternion)
    poses = []
    for x_offset in x_offsets:
        for y_offset in y_offsets:
            for angle_deg in z_angles_deg:
                pose = np.eye(4)
                pose[:3, :3] = Rotation.from_euler(
                    "z", angle_deg, degrees=True
                ).as_matrix() @ base_rotation
                pose[:3, 3] = [x_offset, y_offset, 0.0]
                poses.append(pose)
    return np.asarray(poses)


def build_insertion_trajectory(
    world_from_port,
    body_from_plug_anchor,
    insertion_distance: float,
    waypoint_count: int,
) -> InsertionTrajectory:
    if insertion_distance <= 0.0:
        raise ValueError("insertion_distance must be positive")
    if waypoint_count < 2:
        raise ValueError("waypoint_count must be at least two")

    world_from_port = np.asarray(world_from_port, dtype=float)
    body_from_plug_anchor = np.asarray(body_from_plug_anchor, dtype=float)
    insertion_axis = world_from_port[:3, 0].copy()
    insertion_axis /= np.linalg.norm(insertion_axis)

    preinsert_anchor_pose = world_from_port.copy()
    preinsert_anchor_pose[:3, 3] -= insertion_axis * insertion_distance
    positions = np.linspace(
        preinsert_anchor_pose[:3, 3],
        world_from_port[:3, 3],
        waypoint_count,
    )
    anchor_waypoints = []
    body_waypoints = []
    anchor_from_body = np.linalg.inv(body_from_plug_anchor)
    for position in positions:
        world_from_anchor = world_from_port.copy()
        world_from_anchor[:3, 3] = position
        anchor_waypoints.append(world_from_anchor)
        body_waypoints.append(world_from_anchor @ anchor_from_body)
    return InsertionTrajectory(
        preinsert_anchor_pose=preinsert_anchor_pose,
        anchor_waypoints=np.asarray(anchor_waypoints),
        body_waypoints=np.asarray(body_waypoints),
    )


def build_preinsert_candidates(
    world_from_port,
    distances=np.linspace(0.04, 0.06, 10),
) -> np.ndarray:
    """Build fixed-orientation candidates outside the port insertion axis."""
    world_from_port = np.asarray(world_from_port, dtype=float)
    insertion_axis = world_from_port[:3, 0].copy()
    insertion_axis /= np.linalg.norm(insertion_axis)
    candidates = []
    for distance in np.asarray(distances, dtype=float):
        if distance <= 0.0:
            raise ValueError("pre-insertion distances must be positive")
        candidate = world_from_port.copy()
        candidate[:3, 3] -= insertion_axis * distance
        candidates.append(candidate)
    return np.asarray(candidates)


def build_dynamic_insertion_trajectory(
    world_from_port,
    world_from_current_anchor,
    body_from_plug_anchor,
    insertion_waypoint_count: int = 5,
) -> InsertionTrajectory:
    """First return to the insertion line, then insert to the fixed port."""
    if insertion_waypoint_count < 1:
        raise ValueError("insertion_waypoint_count must be positive")
    world_from_port = np.asarray(world_from_port, dtype=float)
    world_from_current_anchor = np.asarray(
        world_from_current_anchor, dtype=float
    )
    body_from_plug_anchor = np.asarray(body_from_plug_anchor, dtype=float)
    insertion_axis = world_from_port[:3, 0].copy()
    insertion_axis /= np.linalg.norm(insertion_axis)

    current_delta = (
        world_from_current_anchor[:3, 3] - world_from_port[:3, 3]
    )
    current_axial_offset = float(np.dot(current_delta, insertion_axis))
    aligned_anchor_pose = world_from_port.copy()
    aligned_anchor_pose[:3, 3] += insertion_axis * current_axial_offset

    insertion_positions = np.linspace(
        aligned_anchor_pose[:3, 3],
        world_from_port[:3, 3],
        insertion_waypoint_count + 1,
    )[1:]
    anchor_waypoints = [aligned_anchor_pose]
    for position in insertion_positions:
        waypoint = world_from_port.copy()
        waypoint[:3, 3] = position
        anchor_waypoints.append(waypoint)
    anchor_waypoints = np.asarray(anchor_waypoints)
    anchor_from_body = np.linalg.inv(body_from_plug_anchor)
    body_waypoints = anchor_waypoints @ anchor_from_body
    return InsertionTrajectory(
        preinsert_anchor_pose=aligned_anchor_pose,
        anchor_waypoints=anchor_waypoints,
        body_waypoints=body_waypoints,
    )


def measure_anchor_error(
    plug_position,
    port_position,
    insertion_axis,
) -> AnchorError:
    delta = np.asarray(plug_position, dtype=float) - np.asarray(
        port_position, dtype=float
    )
    axis = np.asarray(insertion_axis, dtype=float).copy()
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        raise ValueError("insertion_axis must be non-zero")
    axis /= norm
    signed_axial = float(np.dot(delta, axis))
    lateral_vector = delta - signed_axial * axis
    return AnchorError(
        axial=abs(signed_axial),
        lateral=float(np.linalg.norm(lateral_vector)),
    )


def should_retry_insertion(error: AnchorError) -> bool:
    """Return whether the real anchor is still outside the attractor region."""
    return (
        error.axial > ATTRACTOR_AXIAL_THRESHOLD
        or error.lateral > ATTRACTOR_LATERAL_THRESHOLD
    )


def find_scene_objects(scene_info):
    """Find sibling task assets from semantic metadata, not instance IDs."""
    found = {}
    for object_id, object_info in scene_info.get("layout", {}).items():
        keywords = {
            str(keyword).lower()
            for keyword in object_info.get("keywords", [])
        }
        usd = str(object_info.get("usd", "")).lower()

        if "ethernet_router" in keywords or usd.endswith(
            "benchmark_network_device_000"
        ):
            found["router"] = object_id
        elif "ethernet_plug" in keywords or usd.endswith(
            "benchmark_network_device_001"
        ):
            found["plug"] = object_id
        elif "table" in keywords or usd.endswith("table_000"):
            found["table"] = object_id

    missing = [name for name in ("router", "plug") if name not in found]
    if missing:
        raise ValueError(f"Ethernet scene is missing required objects: {missing}")
    return found


def build_default_grasp_poses(arm="right") -> np.ndarray:
    """Return body-local grasps with position variation only along X."""
    return build_grasp_poses(
        x_offsets=np.linspace(0.0, -0.005, 3),
        y_offsets=[0.0],
        z_angles_deg=np.linspace(-30.0, 30.0, 5),
        base_quaternion=BASE_GRASP_WXYZ,
    )


def select_arm(scene_info) -> str:
    for object_info in scene_info.get("layout", {}).values():
        keywords = {str(value).lower() for value in object_info.get("keywords", [])}
        if "left_hand" in keywords:
            return "left"
    return "right"


def _read_pose(robot, actor, articulation_part=None) -> np.ndarray:
    position, quaternion = robot.get_object_pose(
        actor,
        articulation_part=articulation_part,
    )
    return make_pose(position, quaternion)


def _move_plug_anchor(robot, plug_id, target_pose, unmovable_objects, text, arm="right"):
    move_config = MoveConfig(
        unmovable_objects=unmovable_objects,
        action_description=text,
        extra_params={"skip_check": True},
    )
    target_pose = np.asarray(target_pose, dtype=float)
    if target_pose.ndim == 2:
        target_pose = target_pose[np.newaxis, ...]
    return robot.move_actor_to_target_pose(
        actor=plug_id,
        target_pose=target_pose,
        arm=arm,
        move_config=move_config,
        part=PLUG_ANCHOR_PART,
    )


def _move_plug_anchor_with_retries(
    robot,
    plug_id,
    target_pose,
    unmovable_objects,
    text,
    max_attempts=3,
    arm="right",
):
    for attempt in range(1, max_attempts + 1):
        if _move_plug_anchor(
            robot,
            plug_id,
            target_pose,
            unmovable_objects,
            text,
            arm=arm,
        ):
            return True
        print(
            f"[ethernet] pre-insertion planning attempt "
            f"{attempt}/{max_attempts} failed"
        )
    return False


def _draw_insertion_debug_targets(
    robot,
    world_from_port,
    preinsert_anchor_pose,
    body_from_plug_anchor,
):
    """Draw the exact anchor and plug-body poses used by the move stages."""
    anchor_from_body = np.linalg.inv(body_from_plug_anchor)
    targets = (
        ("ethernet_port_anchor_target", world_from_port, [1.0, 0.0, 0.0]),
        (
            "ethernet_preinsert_anchor_target",
            preinsert_anchor_pose,
            [1.0, 1.0, 0.0],
        ),
        (
            "ethernet_preinsert_body_target",
            preinsert_anchor_pose @ anchor_from_body,
            [1.0, 0.0, 1.0],
        ),
    )
    for name, pose, color in targets:
        pose = np.asarray(pose, dtype=float)
        try:
            robot.draw_debug_cube(
                name=name,
                pose_4x4=pose,
                size=0.018,
                color=color,
            )
            print(
                f"[ethernet-debug] {name}: "
                f"xyz={pose[:3, 3].tolist()}"
            )
        except Exception as error:
            print(f"[ethernet-debug] failed to draw {name}: {error}")


def _anchor_error(robot, plug_id, router_id):
    world_from_port = _read_pose(
        robot,
        router_id,
        articulation_part=PORT_ANCHOR_PART,
    )
    world_from_plug = _read_pose(
        robot,
        plug_id,
        articulation_part=PLUG_ANCHOR_PART,
    )
    error = measure_anchor_error(
        plug_position=world_from_plug[:3, 3],
        port_position=world_from_port[:3, 3],
        insertion_axis=world_from_port[:3, 0],
    )
    return error, world_from_plug[:3, 3]


def verify_final_alignment(robot, plug_id, router_id) -> bool:
    """Wait for the attractor, then verify alignment and short-term stability."""
    if not robot.wait(0.50):
        print("[ethernet] failed while waiting for the attachment trigger")
        return False
    try:
        first_error, first_position = _anchor_error(
            robot, plug_id, router_id
        )
    except Exception as error:
        print(f"[ethernet] failed to measure final anchor error: {error}")
        return False

    if (
        first_error.axial > AXIAL_FIX_THRESHOLD
        or first_error.lateral > LATERAL_FIX_THRESHOLD
    ):
        print(
            "[ethernet] final anchor error exceeds trigger thresholds: "
            f"axial={first_error.axial:.6f}, "
            f"lateral={first_error.lateral:.6f}"
        )
        return False

    if not robot.wait(0.25):
        print("[ethernet] failed during the stability wait")
        return False
    try:
        second_error, second_position = _anchor_error(
            robot, plug_id, router_id
        )
    except Exception as error:
        print(f"[ethernet] failed to recheck anchor stability: {error}")
        return False

    drift = float(np.linalg.norm(second_position - first_position))
    stable = (
        second_error.axial <= AXIAL_FIX_THRESHOLD
        and second_error.lateral <= LATERAL_FIX_THRESHOLD
        and drift <= POST_INSERT_STABILITY_THRESHOLD
    )
    print(
        "[ethernet] final alignment: "
        f"axial={second_error.axial:.6f}, "
        f"lateral={second_error.lateral:.6f}, drift={drift:.6f}"
    )
    return stable


def insert_plug(robot, plug_id, router_id, unmovable_objects, arm="right") -> bool:
    """Align the internal plug anchor, then follow a straight insertion path."""
    try:
        world_from_port = _read_pose(
            robot,
            router_id,
            articulation_part=PORT_ANCHOR_PART,
        )
        world_from_plug_anchor = _read_pose(
            robot,
            plug_id,
            articulation_part=PLUG_ANCHOR_PART,
        )
        world_from_plug_body = _read_pose(
            robot,
            plug_id,
            articulation_part=PLUG_BODY_PART,
        )
    except Exception as error:
        print(f"[ethernet] failed to read anchor poses: {error}")
        return False

    body_from_plug_anchor = (
        np.linalg.inv(world_from_plug_body) @ world_from_plug_anchor
    )
    preinsert_candidates = build_preinsert_candidates(world_from_port)

    # The attached plug is intentionally approaching the router from this
    # point onward. Keep PhysX contact enabled, but exclude the router from
    # Curobo before solving the pre-insertion move and insertion trajectory.
    router_prim_path = robot.get_object_prim_path(router_id)
    robot.remove_objs_from_obstacle([router_prim_path])

    _draw_insertion_debug_targets(
        robot,
        world_from_port=world_from_port,
        preinsert_anchor_pose=preinsert_candidates[0],
        body_from_plug_anchor=body_from_plug_anchor,
    )
    if not _move_plug_anchor_with_retries(
        robot,
        plug_id,
        preinsert_candidates,
        unmovable_objects,
        {
            "zh": "将网线插头对准网口",
            "en": "aligns the Ethernet plug with the port",
        },
        arm=arm,
    ):
        print("[ethernet] failed to reach the pre-insertion pose")
        return False

    try:
        world_from_current_anchor = _read_pose(
            robot,
            plug_id,
            articulation_part=PLUG_ANCHOR_PART,
        )
    except Exception as error:
        print(f"[ethernet] failed to read the reached pre-insertion pose: {error}")
        return False
    trajectory = build_dynamic_insertion_trajectory(
        world_from_port=world_from_port,
        world_from_current_anchor=world_from_current_anchor,
        body_from_plug_anchor=body_from_plug_anchor,
        insertion_waypoint_count=5,
    )

    anchor_waypoints = np.asarray(
        [pose_to_xyz_wxyz(pose) for pose in trajectory.anchor_waypoints]
    )
    trajectory_config = FollowTrajectoryConfig(
        noise_range=[0.0, 0.002, 0.002, 0.0, 0.0, 0.0],
        noise_frame="local",
        sample_count=10,
        unmovable_objects=unmovable_objects,
        action_description={
            "zh": "沿网口方向插入网线",
            "en": "inserts the Ethernet plug along the port axis",
        },
    )
    if not robot.follow_trajectory(
        actor=plug_id,
        waypoints=anchor_waypoints[np.newaxis, ...],
        arm=arm,
        trajectory_config=trajectory_config,
        part=PLUG_ANCHOR_PART,
    ):
        print("[ethernet] failed to follow the insertion trajectory")
        return False

    # Physical tracking can stop several millimetres before the commanded
    # endpoint even when all IK waypoints solve. Re-read the real plug anchor
    # and issue a short corrective trajectory to the same fixed port pose.
    # This does not deepen or move the target; it only closes execution error.
    for correction_attempt in range(1, 3):
        try:
            correction_error, _ = _anchor_error(robot, plug_id, router_id)
            if not should_retry_insertion(correction_error):
                break
            world_from_current_anchor = _read_pose(
                robot,
                plug_id,
                articulation_part=PLUG_ANCHOR_PART,
            )
        except Exception as error:
            print(f"[ethernet] failed to prepare insertion correction: {error}")
            return False
        print(
            f"[ethernet] insertion correction {correction_attempt}/2: "
            f"axial={correction_error.axial:.6f}, "
            f"lateral={correction_error.lateral:.6f}"
        )
        correction = build_dynamic_insertion_trajectory(
            world_from_port=world_from_port,
            world_from_current_anchor=world_from_current_anchor,
            body_from_plug_anchor=body_from_plug_anchor,
            insertion_waypoint_count=5,
        )
        correction_waypoints = np.asarray(
            [pose_to_xyz_wxyz(pose) for pose in correction.anchor_waypoints]
        )
        if not robot.follow_trajectory(
            actor=plug_id,
            waypoints=correction_waypoints[np.newaxis, ...],
            arm=arm,
            trajectory_config=trajectory_config,
            part=PLUG_ANCHOR_PART,
        ):
            print("[ethernet] failed to follow the insertion correction")
            return False
    return verify_final_alignment(robot, plug_id, router_id)


def execute_task(robot: RobotAPI) -> bool:
    """Grasp the Ethernet plug; later stages perform anchor-guided insertion."""
    try:
        object_ids = find_scene_objects(robot.scene_info)
    except ValueError as error:
        print(f"[ethernet] {error}")
        return False

    router_id = object_ids["router"]
    plug_id = object_ids["plug"]
    arm = select_arm(robot.scene_info)
    table_id = object_ids.get("table")
    unmovable_objects = [router_id]
    if table_id is not None:
        unmovable_objects.append(table_id)

    robot.client.set_task_task_basic_info(DESCRIPTION)

    # Keep the deformable cable fully physical in PhysX, but exclude its long
    # mesh from Curobo's obstacle/attached-object model.  This mirrors the
    # phone charging task's runtime obstacle removal and prevents the cable
    # resting on the table from making the post-grasp lift unsolvable.
    cable_prim_path = robot.get_object_prim_path(
        plug_id,
        articulation_part=PLUG_CABLE_PART,
    )
    robot.remove_objs_from_obstacle([cable_prim_path])

    grasp_config = GraspConfig(
        grasp_offset=0.005,
        grasp_pose=build_default_grasp_poses(arm=arm),
        grasp_pose_frame="relative",
        pick_up_distance=0.05,
        disable_upside_down=True,
        flip_grasp=True,
        unmovable_objects=unmovable_objects,
        action_description={
            "zh": "抓起网线插头",
            "en": "picks up the Ethernet plug",
        },
    )
    if not robot.pick_actor(
        actor=plug_id,
        arm=arm,
        part=PLUG_BODY_PART,
        grasp_config=grasp_config,
    ):
        print("[ethernet] failed to pick up the plug")
        return False

    return insert_plug(
        robot=robot,
        plug_id=plug_id,
        router_id=router_id,
        unmovable_objects=unmovable_objects,
        arm=arm,
    )
