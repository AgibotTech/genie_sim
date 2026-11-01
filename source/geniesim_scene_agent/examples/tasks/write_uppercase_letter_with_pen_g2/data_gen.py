# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pick up the writing pen and write an uppercase letter on the paper."""

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
from scipy.spatial.transform import Rotation


PEN_GRASP_POINT = np.array(
    [1.8596235662471372e-15, 0.03840777570972542, 3.4833975589079746e-7]
)
PEN_TIP_LOCAL = np.array(
    [0.0, -0.07184357772142377, -6.515861400178732e-7]
)
# Distance from the pen tip to the paper's top surface while drawing, in metres.
PEN_TIP_PAPER_DISTANCE = 0.002
PEN_LIFT_HEIGHT = 0.05
LETTER_WIDTH = 0.06
LETTER_HEIGHT = 0.08
STROKE_POINT_SPACING = 0.002
LETTER_PLACEMENT_Y_OFFSETS = (-0.01, 0.0, 0.01)
LETTER_STROKES = {
    "A": (
        ((-0.5, -0.5), (0.0, 0.5), (0.5, -0.5)),
        ((-0.25, 0.0), (0.25, 0.0)),
    ),
    "B": (
        ((-0.5, 0.5), (-0.5, -0.5)),
        (
            (-0.5, 0.5),
            (0.1, 0.5),
            (0.5, 0.38),
            (0.5, 0.18),
            (0.1, 0.0),
            (-0.5, 0.0),
            (0.1, 0.0),
            (0.5, -0.18),
            (0.5, -0.38),
            (0.1, -0.5),
            (-0.5, -0.5),
        ),
    ),
    "C": (
        (
            (0.38, 0.38),
            (0.22, 0.47),
            (0.0, 0.5),
            (-0.22, 0.47),
            (-0.4, 0.36),
            (-0.5, 0.18),
            (-0.5, 0.0),
            (-0.5, -0.18),
            (-0.4, -0.36),
            (-0.22, -0.47),
            (0.0, -0.5),
            (0.22, -0.47),
            (0.38, -0.38),
        ),
    ),
    "D": (
        ((-0.5, 0.5), (-0.5, -0.5)),
        (
            (-0.5, 0.5),
            (-0.15, 0.5),
            (0.15, 0.46),
            (0.38, 0.34),
            (0.5, 0.16),
            (0.5, 0.0),
            (0.5, -0.16),
            (0.38, -0.34),
            (0.15, -0.46),
            (-0.15, -0.5),
            (-0.5, -0.5),
        ),
    ),
    "E": (
        ((-0.5, 0.5), (-0.5, -0.5)),
        ((-0.5, 0.5), (0.5, 0.5)),
        ((-0.5, 0.0), (0.25, 0.0)),
        ((-0.5, -0.5), (0.5, -0.5)),
    ),
    "F": (
        ((-0.5, 0.5), (-0.5, -0.5)),
        ((-0.5, 0.5), (0.5, 0.5)),
        ((-0.5, 0.0), (0.25, 0.0)),
    ),
    "G": (
        (
            (0.38, 0.38),
            (0.22, 0.47),
            (0.0, 0.5),
            (-0.22, 0.47),
            (-0.4, 0.36),
            (-0.5, 0.18),
            (-0.5, 0.0),
            (-0.5, -0.18),
            (-0.4, -0.36),
            (-0.22, -0.47),
            (0.0, -0.5),
            (0.22, -0.47),
            (0.38, -0.38),
            (0.48, -0.25),
            (0.5, -0.08),
            (0.1, -0.08),
        ),
    ),
    "H": (
        ((-0.5, 0.5), (-0.5, -0.5)),
        ((0.5, 0.5), (0.5, -0.5)),
        ((-0.5, 0.0), (0.5, 0.0)),
    ),
    "I": (
        ((-0.4, 0.5), (0.4, 0.5)),
        ((0.0, 0.5), (0.0, -0.5)),
        ((-0.4, -0.5), (0.4, -0.5)),
    ),
    "J": (
        ((-0.2, 0.5), (0.5, 0.5)),
        (
            (0.3, 0.5),
            (0.3, -0.25),
            (0.25, -0.4),
            (0.1, -0.5),
            (-0.15, -0.5),
            (-0.35, -0.4),
            (-0.45, -0.25),
        ),
    ),
    "K": (
        ((-0.5, 0.5), (-0.5, -0.5)),
        ((-0.5, 0.0), (0.5, 0.5)),
        ((-0.5, 0.0), (0.5, -0.5)),
    ),
    "L": (
        ((-0.5, 0.5), (-0.5, -0.5)),
        ((-0.5, -0.5), (0.5, -0.5)),
    ),
    "M": (
        ((-0.5, -0.5), (-0.5, 0.5), (0.0, 0.0), (0.5, 0.5), (0.5, -0.5)),
    ),
    "N": (
        ((-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)),
    ),
    "O": (
        (
            (0.0, 0.5),
            (-0.22, 0.47),
            (-0.4, 0.36),
            (-0.5, 0.18),
            (-0.5, 0.0),
            (-0.5, -0.18),
            (-0.4, -0.36),
            (-0.22, -0.47),
            (0.0, -0.5),
            (0.22, -0.47),
            (0.4, -0.36),
            (0.5, -0.18),
            (0.5, 0.0),
            (0.5, 0.18),
            (0.4, 0.36),
            (0.22, 0.47),
            (0.0, 0.5),
        ),
    ),
    "P": (
        ((-0.5, 0.5), (-0.5, -0.5)),
        (
            (-0.5, 0.5),
            (0.0, 0.5),
            (0.3, 0.45),
            (0.48, 0.3),
            (0.5, 0.2),
            (0.48, 0.1),
            (0.3, 0.02),
            (0.0, 0.0),
            (-0.5, 0.0),
        ),
    ),
    "Q": (
        (
            (0.0, 0.5),
            (-0.22, 0.47),
            (-0.4, 0.36),
            (-0.5, 0.18),
            (-0.5, 0.0),
            (-0.5, -0.18),
            (-0.4, -0.36),
            (-0.22, -0.47),
            (0.0, -0.5),
            (0.22, -0.47),
            (0.4, -0.36),
            (0.5, -0.18),
            (0.5, 0.0),
            (0.5, 0.18),
            (0.4, 0.36),
            (0.22, 0.47),
            (0.0, 0.5),
        ),
        ((0.1, -0.2), (0.5, -0.5)),
    ),
    "R": (
        ((-0.5, 0.5), (-0.5, -0.5)),
        (
            (-0.5, 0.5),
            (0.0, 0.5),
            (0.3, 0.45),
            (0.48, 0.3),
            (0.5, 0.2),
            (0.48, 0.1),
            (0.3, 0.02),
            (0.0, 0.0),
            (-0.5, 0.0),
        ),
        ((-0.05, 0.0), (0.5, -0.5)),
    ),
    "S": (
        (
            (0.4, 0.38),
            (0.2, 0.48),
            (-0.15, 0.5),
            (-0.4, 0.38),
            (-0.5, 0.2),
            (-0.4, 0.05),
            (-0.15, 0.0),
            (0.15, 0.0),
            (0.4, -0.05),
            (0.5, -0.2),
            (0.4, -0.38),
            (0.15, -0.5),
            (-0.2, -0.48),
            (-0.4, -0.38),
        ),
    ),
    "T": (
        ((-0.5, 0.5), (0.5, 0.5)),
        ((0.0, 0.5), (0.0, -0.5)),
    ),
    "U": (
        (
            (-0.5, 0.5),
            (-0.5, -0.2),
            (-0.4, -0.4),
            (-0.2, -0.5),
            (0.0, -0.5),
            (0.2, -0.5),
            (0.4, -0.4),
            (0.5, -0.2),
            (0.5, 0.5),
        ),
    ),
    "V": (
        ((-0.5, 0.5), (0.0, -0.5), (0.5, 0.5)),
    ),
    "W": (
        ((-0.5, 0.5), (-0.3, -0.5), (0.0, 0.1), (0.3, -0.5), (0.5, 0.5)),
    ),
    "X": (
        ((-0.5, 0.5), (0.5, -0.5)),
        ((0.5, 0.5), (-0.5, -0.5)),
    ),
    "Y": (
        ((-0.5, 0.5), (0.0, 0.0)),
        ((0.5, 0.5), (0.0, 0.0)),
        ((0.0, 0.0), (0.0, -0.5)),
    ),
    "Z": (
        ((-0.5, 0.5), (0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)),
    ),
}


def target_letter_from_scene_info(scene_info) -> str:
    """Read the required scene letter; fail rather than write a different glyph."""
    letter = scene_info.get("target_letter")
    if not isinstance(letter, str) or letter not in LETTER_STROKES:
        raise ValueError("scene_info.target_letter must be one uppercase letter A-Z")
    return letter


def description_for_letter(letter) -> dict:
    """Use the scene's letter in both task-description fields."""
    name = f"Pick up the pen and write an uppercase letter {letter} on the paper"
    return {
        "task_name": name,
        "english_task_name": name,
        "init_scene_text": "A pen stands in a holder beside a sheet of paper on a writing desk.",
    }


def find_scene_objects(scene_info):
    """Find writing-scene instances from semantic metadata, not instance IDs."""
    found = {}
    for object_id, object_info in scene_info.get("layout", {}).items():
        keywords = {str(item).lower() for item in object_info.get("keywords", [])}
        usd = str(object_info.get("usd", "")).lower()

        if "writing_pen" in keywords or usd == "benchmark_pen_009":
            found["pen"] = object_id
        elif "paper" in keywords or usd == "benchmark_paper_000":
            found["paper"] = object_id
        elif "pen_holder" in keywords or usd == "benchmark_stationery_021":
            found["pen_holder"] = object_id
        elif "writing_table" in keywords or usd == "table_000":
            found["table"] = object_id

    missing = [name for name in ("pen", "paper") if name not in found]
    if missing:
        raise ValueError(f"writing scene is missing required objects: {missing}")
    return found


def build_grasp_poses() -> np.ndarray:
    """Return 144 body-local grasp candidates for the pen."""
    poses = []
    for base_quat in ([1, 0, 0, 0], [0, 0, 0, 1]):
        base_rotation = quat2mat_wxyz(np.asarray(base_quat, dtype=float))
        for y_offset in np.linspace(-0.01, 0.01, 3):
            for angle in np.linspace(-np.pi, np.pi, 24, endpoint=False):
                pose = np.eye(4)
                pose[:3, :3] = base_rotation @ Rotation.from_euler(
                    "y", angle
                ).as_matrix()
                pose[:3, 3] = PEN_GRASP_POINT + [0.0, y_offset, 0.0]
                poses.append(pose)
    return np.asarray(poses)


def get_letter_strokes(letter):
    """Return normalized 2D polylines for one supported letter."""
    key = str(letter).upper()
    if key not in LETTER_STROKES:
        raise ValueError(f"unsupported letter: {letter}")
    return tuple(np.asarray(stroke, dtype=float) for stroke in LETTER_STROKES[key])


def sample_polyline(points, spacing) -> np.ndarray:
    """Sample a 2D polyline so adjacent output points are at most spacing apart."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("polyline must contain at least two 2D points")
    if spacing <= 0:
        raise ValueError("spacing must be positive")

    sampled = [points[0]]
    for start, end in zip(points[:-1], points[1:]):
        segment = end - start
        steps = max(1, int(np.ceil(np.linalg.norm(segment) / spacing)))
        sampled.extend(
            start + segment * ratio
            for ratio in np.linspace(0.0, 1.0, steps + 1)[1:]
        )
    return np.asarray(sampled)


def glyph_points_to_paper_xy(points) -> np.ndarray:
    """Map upright glyph coordinates into the paper axes seen by the robot."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("glyph points must have shape (N, 2)")
    return np.column_stack(
        (
            points[:, 1] * LETTER_HEIGHT,
            -points[:, 0] * LETTER_WIDTH,
        )
    )


def build_letter_contact_strokes(
    letter, paper_pose: np.ndarray, paper_size
):
    """Return a list of (candidate, point, xyz+wxyz) contact trajectories."""
    paper_size = np.asarray(paper_size, dtype=float)
    pen_rotation = Rotation.from_euler("x", np.pi / 2).as_matrix()
    pen_quaternion = mat2quat_wxyz(pen_rotation)
    surface_local_z = paper_size[2] / 2 + PEN_TIP_PAPER_DISTANCE
    contact_strokes = []

    for normalized_stroke in get_letter_strokes(letter):
        # The robot stands on world -X and faces +X: robot-view up is paper
        # +X, while robot-view right is paper -Y.
        paper_stroke = glyph_points_to_paper_xy(normalized_stroke)
        sampled_stroke = sample_polyline(paper_stroke, STROKE_POINT_SPACING)
        candidates = []
        for y_offset in LETTER_PLACEMENT_Y_OFFSETS:
            path = []
            for x, y in sampled_stroke:
                tip_local = np.array([x, y + y_offset, surface_local_z])
                tip_target = paper_pose[:3, :3] @ tip_local + paper_pose[:3, 3]
                actor_position = tip_target - pen_rotation @ PEN_TIP_LOCAL
                path.append([*actor_position, *pen_quaternion])
            candidates.append(path)
        contact_strokes.append(np.asarray(candidates))
    return contact_strokes


def stroke_endpoint_poses(
    contact_candidates, waypoint_index, lift_height=0.0
) -> np.ndarray:
    """Convert one endpoint from every contact candidate to lifted actor poses."""
    contact_candidates = np.asarray(contact_candidates, dtype=float)
    poses = []
    for candidate in contact_candidates:
        waypoint = candidate[waypoint_index].copy()
        waypoint[2] += lift_height
        poses.append(get_pose_wxyz(waypoint[:3], waypoint[3:]))
    return np.asarray(poses)


def write_letter(robot, pen_id, paper_id, contact_strokes) -> bool:
    """Execute explicit hover, descent, writing, and lift actions per stroke."""
    for stroke_index, candidates in enumerate(contact_strokes, start=1):
        hover_config = MoveConfig(
            unmovable_objects=[paper_id],
            action_description={
                "zh": f"将笔移动到第{stroke_index}笔起点上方",
                "en": f"moves the pen above stroke {stroke_index}",
            },
        )
        if not robot.move_actor_to_target_pose(
            actor=pen_id,
            target_pose=stroke_endpoint_poses(
                candidates, waypoint_index=0, lift_height=PEN_LIFT_HEIGHT
            ),
            arm="right",
            move_config=hover_config,
        ):
            print(f"[write-letter] failed to move above stroke {stroke_index}")
            return False

        descend_config = MoveConfig(
            unmovable_objects=[paper_id],
            action_description={
                "zh": f"将笔下降到第{stroke_index}笔起点",
                "en": f"lowers the pen to the start of stroke {stroke_index}",
            },
        )
        if not robot.move_actor_to_target_pose(
            actor=pen_id,
            target_pose=stroke_endpoint_poses(candidates, waypoint_index=0),
            arm="right",
            move_config=descend_config,
        ):
            print(f"[write-letter] failed to descend for stroke {stroke_index}")
            return False

        trajectory_config = FollowTrajectoryConfig(
            # Pen local Y is the paper normal; only perturb within local XZ.
            noise_range=[0.003, 0.0, 0.003, 0.0, np.pi, 0.0],
            noise_frame="local",
            sample_count=32,
            unmovable_objects=[paper_id],
            action_description={
                "zh": f"书写字母的第{stroke_index}笔",
                "en": f"writes stroke {stroke_index} of the letter",
            },
        )
        if not robot.follow_trajectory(
            actor=pen_id,
            waypoints=candidates,
            arm="right",
            trajectory_config=trajectory_config,
        ):
            print(f"[write-letter] failed to write stroke {stroke_index}")
            return False

        lift_config = MoveConfig(
            unmovable_objects=[paper_id],
            action_description={
                "zh": f"写完第{stroke_index}笔后抬起笔",
                "en": f"lifts the pen after stroke {stroke_index}",
            },
        )
        if not robot.move_actor_to_target_pose(
            actor=pen_id,
            target_pose=stroke_endpoint_poses(
                candidates, waypoint_index=-1, lift_height=PEN_LIFT_HEIGHT
            ),
            arm="right",
            move_config=lift_config,
        ):
            print(f"[write-letter] failed to lift after stroke {stroke_index}")
            return False
    return True


def execute_task(robot: RobotAPI) -> bool:
    """Grasp the pen and write the uppercase letter specified by this scene."""
    try:
        target_letter = target_letter_from_scene_info(robot.scene_info)
        object_ids = find_scene_objects(robot.scene_info)
    except ValueError as error:
        print(f"[write-letter] {error}")
        return False

    pen_id = object_ids["pen"]
    paper_id = object_ids["paper"]
    robot.client.set_task_task_basic_info(description_for_letter(target_letter))

    grasp_poses = build_grasp_poses()
    grasp_config = GraspConfig(
        grasp_pose=grasp_poses,
        grasp_pose_frame="relative",
        grasp_width=np.full(len(grasp_poses), 0.02),
        grasp_offset=-0.005,
        pre_grasp_offset=0.03,
        pick_up_distance=0.15,
        disable_upside_down=True,
        flip_grasp=False,
        unmovable_objects=[paper_id],
        action_description={"zh": "抓起笔", "en": "picks up the pen"},
    )
    if not robot.pick_actor(pen_id, arm="right", grasp_config=grasp_config):
        print("[write-letter] failed to pick up the pen")
        return False

    try:
        paper_position, paper_quaternion = robot.get_object_pose(paper_id)
        paper_pose = get_pose_wxyz(paper_position, paper_quaternion)
        paper_size = robot.get_object_size_in_world_frame(paper_id)
        paper_prim_path = robot.get_object_prim_path(paper_id)
    except Exception as error:
        print(f"[write-letter] failed to read the paper geometry: {error}")
        return False

    robot.remove_objs_from_obstacle([paper_prim_path])
    contact_strokes = build_letter_contact_strokes(
        target_letter, paper_pose, paper_size
    )
    if not write_letter(
        robot, pen_id, paper_id, contact_strokes
    ):
        return False

    return bool(robot.back_to_origin(arm="right"))
