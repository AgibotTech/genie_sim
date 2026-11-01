# Observe a block sequence and press buttons in the same order

**Tags:** `press` `articulation` `table_with_articulation` `gripper_only` `move_actor_to_target_pose` `path_constraint` `from_current_pose` `goal_offset` `remove_objs_from_obstacle` `articulation_part` `extra_checkers` `unmovable_objects` `multiple_candidate_poses` `close_gripper` `set_object_parameter_action` `xformOp:translate` `observation_event` `cue_then_act` `keywords_position_tag` `sequential_press` `get_ee_pose` `return_to_initial_pose` `delay` `MoveConfig` `judge_arm_to_use` `euler2mat` `rotate_along_axis`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find the three buttons in `scene_info["layout"]` by their `<position>_button` keyword (left/middle/right) and the building block by `hint_block`. Read the block's initial position from its `in_front_of_<position>_button` keyword — that is the first observation position.
2. **Visit order:** start = initial block position; the other two positions are randomized. The robot must remember the order and replay it on the buttons.
3. **Observation phase (the robot watches, no manipulation):** For each position in visit order, call `set_object_parameter_action(prim_path, "xformOp:translate", [x,y,z], delay=BLOCK_DWELL_SEC)` to relocate the block in front of that button. `delay` is applied BEFORE the new translate is written, so a positive value (e.g. 2.5 s) keeps the block visible at its CURRENT position for that duration before the next jump. The teleport is **not** a robot action — its `description` is framed as "**observes** the block in front of the X button". The first step also issues a `set_object_parameter_action` (target = canonical `(button_x + offset, button_y, block_z)`) so all observation events are uniform.
4. **Clear cue:** Move the block to `(20, 20, z)`. Description framed as "observes the cue being removed".
5. **Press phase:** After ALL block-observation events have finished (not before — gripper-close events should not interleave with the cues), close BOTH grippers via `close_gripper(arm="left"/"right")` and capture each arm's initial ee pose with `get_ee_pose`. The grippers stay **closed** for the rest of the task. For each position in visit order, run the standard press-button pattern (`close_gripper` → multi-candidate poses above the button → `remove_objs_from_obstacle` → `from_current_pose` + `goal_offset=[0,0,-0.05]` + `path_constraint=[0.1,0.1,0.1,0.1,0.1,0.0]` + `check_button_pressed`). After each press, **do NOT** call `open_gripper` or `back_to_origin`. Instead return the used arm via two `move_actor_to_target_pose` calls: first a +0.10 m straight-up lift to clear the just-pressed button, then a move to the recorded initial ee pose. Both calls pass `unmovable_objects=[all button ids]` so the planner steers around every button.

---

# INPUT

**scene_description:** A table (`table_000`) with three same-size but different-color buttons arranged in a row at the table center (left / middle / right). One building block sits on the table in front of one of the three buttons.

**task description:** Watch the block being shown in front of each of the three buttons in some order, then press the buttons in the same order.

**code to generate the scene:**

```python
from helper import *
import numpy as np
import random

"""
scene_name: block_in_front_of_one_of_three_buttons
description: Three identical-size but different-color buttons placed in a row at
the table center (left / middle / right). A single building block is placed in
front of one randomly chosen button (same y as that button, x further toward the
robot). The chosen button carries a "has_block_in_front" tag and the block carries
a "in_front_of_<position>_button" tag so data_gen.py can recover the pairing.

User-specified coordinate ranges (relative to table center):
  - left   button: x in (-0.05, -0.03), y in (-0.12, -0.08)
  - middle button: x in (-0.05, -0.03), y in (-0.02,  0.02)
  - right  button: x in (-0.05, -0.03), y in ( 0.08,  0.12)
  - block (placed at chosen button's y, on +X side): y = chosen button y, x in (0.18, 0.22)
"""

# Three same-size buttons with different primary colors (dimensions 0.0998 x 0.0675 x 0.0998 m).
BUTTON_CANDIDATES = [
    ("benchmark_button_000", "red"),
    ("benchmark_button_012", "green"),
    ("benchmark_button_013", "blue"),
]

BUILDING_BLOCK_CANDIDATES = [
    "benchmark_building_blocks_073",
    "benchmark_building_blocks_085",
    "benchmark_building_blocks_089",
]

# X range shared by all three button slots (relative to table center).
BUTTON_X_RANGE = (-0.05, -0.03)

# Y range of the middle button (left/right are derived from it + a center
# spacing so that the gap between adjacent buttons is at least 0.05 m).
MIDDLE_Y_RANGE = (-0.02, 0.02)

# Center-to-center spacing between adjacent buttons (m). The buttons have a
# Y-extent of ~0.0998 m, so 0.15–0.18 m of center spacing gives a gap of
# roughly 0.05–0.08 m between neighbours.
SIDE_SPACING_RANGE = (0.15, 0.18)

# Per-button yaw jitter (degrees).
BUTTON_YAW_DEG_RANGE = (-5.0, 5.0)

BLOCK_X_RANGE = (0.18, 0.22)

# Minimum surface-to-surface gap (m) between the hint block and the target
# button after both have been placed (accounts for the buttons' actual rotated
# AABB plus the block's diagonal half-extent under random yaw).
BLOCK_TO_BUTTON_MIN_GAP = 0.02


def _place_button(button_oid: str, color: str, position_tag: str,
                  has_block_in_front: bool,
                  x_local: float, y_local: float,
                  table_center, table_top_z: float) -> Shape:
    keywords = [
        f"{position_tag}_button",
        "button",
        "control",
        color,
        position_tag,
        button_oid,
    ]
    if has_block_in_front:
        keywords.append("has_block_in_front")
        keywords.append("target")
    btn = library_call("usd", oid=button_oid, keywords=keywords)
    btn_center_local = compute_shape_center(btn)
    yaw = np.deg2rad(random.uniform(*BUTTON_YAW_DEG_RANGE))
    btn = transform_shape(
        btn, rotation_matrix(yaw, (0, 0, 1), tuple(btn_center_local))
    )
    info = get_object_info(btn)
    cur_center = info["center"]
    btn_half_h = info["size"][2] / 2.0
    target = (
        table_center[0] + x_local,
        table_center[1] + y_local,
        table_top_z + btn_half_h,
    )
    delta = (target[0] - cur_center[0],
             target[1] - cur_center[1],
             target[2] - cur_center[2])
    return transform_shape(btn, translation_matrix(delta))


def _place_block(in_front_of_position: str, x_local: float, y_local: float,
                 table_center, table_top_z: float) -> Shape:
    block_oid = random.choice(BUILDING_BLOCK_CANDIDATES)
    block = library_call(
        "usd",
        oid=block_oid,
        keywords=[
            "hint_block",
            "building_block",
            "cube",
            f"in_front_of_{in_front_of_position}_button",
            in_front_of_position,
            "target_indicator",
        ],
    )
    block_center = compute_shape_center(block)
    yaw = random.uniform(-np.pi, np.pi)
    block = transform_shape(
        block, rotation_matrix(yaw, (0, 0, 1), tuple(block_center))
    )

    info = get_object_info(block)
    cur_center = info["center"]
    block_half_h = info["size"][2] / 2.0
    target = (
        table_center[0] + x_local,
        table_center[1] + y_local,
        table_top_z + block_half_h,
    )
    delta = (target[0] - cur_center[0],
             target[1] - cur_center[1],
             target[2] - cur_center[2])
    return transform_shape(block, translation_matrix(delta))


@register()
def three_buttons_with_hint_block() -> Shape:
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "wood", "rectangular", "furniture"],
    )
    table_info = get_object_info(table_shape)
    table_top_z = table_info["max"][2]
    table_center = table_info["center"]

    target_idx = random.randint(0, 2)
    button_assignment = random.sample(BUTTON_CANDIDATES, k=3)

    middle_y = random.uniform(*MIDDLE_Y_RANGE)
    left_spacing = random.uniform(*SIDE_SPACING_RANGE)
    right_spacing = random.uniform(*SIDE_SPACING_RANGE)
    slots = [
        ("left",   random.uniform(*BUTTON_X_RANGE), middle_y + left_spacing),
        ("middle", random.uniform(*BUTTON_X_RANGE), middle_y),
        ("right",  random.uniform(*BUTTON_X_RANGE), middle_y - right_spacing),
    ]

    shapes = [table_shape]
    chosen_y_local = chosen_position = chosen_button_shape = None
    for idx, ((position_tag, x_local, y_local),
              (button_oid, color)) in enumerate(zip(slots, button_assignment)):
        is_target = (idx == target_idx)
        placed_button = _place_button(
            button_oid, color, position_tag, is_target,
            x_local, y_local, table_center, table_top_z,
        )
        shapes.append(placed_button)
        if is_target:
            chosen_y_local = y_local
            chosen_position = position_tag
            chosen_button_shape = placed_button

    btn_world_info = get_object_info(chosen_button_shape)
    btn_max_x_world = btn_world_info["max"][0]
    block_diag_half = (0.04 * np.sqrt(2.0)) / 2.0
    block_x_min_world = btn_max_x_world + BLOCK_TO_BUTTON_MIN_GAP + block_diag_half
    block_x_min_local = block_x_min_world - table_center[0]
    block_x_lo, block_x_hi = BLOCK_X_RANGE
    block_x_lo = max(block_x_lo, block_x_min_local)
    block_x_hi = max(block_x_hi, block_x_lo)
    block_x_local = random.uniform(block_x_lo, block_x_hi)
    shapes.append(
        _place_block(
            chosen_position, block_x_local, chosen_y_local,
            table_center, table_top_z,
        )
    )

    return concat_shapes(*shapes)


@register()
def root_scene() -> Shape:
    return library_call("three_buttons_with_hint_block")
```

---

# OUTPUT

```python
from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os
import random


def execute_task(robot: RobotAPI):
    objects_list = {
        "benchmark_button_000": {"has_interactions": True, "object_type": "button"},
        "benchmark_button_012": {"has_interactions": True, "object_type": "button"},
        "benchmark_button_013": {"has_interactions": True, "object_type": "button"},
        "benchmark_building_blocks_073": {"has_interactions": False, "object_type": "building_block"},
        "benchmark_building_blocks_085": {"has_interactions": False, "object_type": "building_block"},
        "benchmark_building_blocks_089": {"has_interactions": False, "object_type": "building_block"},
        "table_000": {"has_interactions": False, "object_type": "table"},
    }

    scene_info = robot.scene_info
    success = True

    description_json = {
        "task_name": "依次观察积木被展示在三个按钮前的顺序，然后按相同顺序按下对应按钮",
        "english_task_name": "observe the order in which the building block is shown in front of each button, then press the buttons in the same order",
        "init_scene_text": "The robot is in front of a table. Three same-size buttons with different colors are arranged left/middle/right at the table center. A single building block is initially placed in front of one of the three buttons. The block will be shown in front of each button one at a time; the robot must remember the order and then press the buttons in the same order.",
    }
    robot.client.set_task_task_basic_info(description_json)

    POSITIONS = ["left", "middle", "right"]
    button_id_by_pos = {}
    block_id = None
    initial_block_position = None

    for oid, info in scene_info["layout"].items():
        kws = info.get("keywords", [])
        usd = info.get("usd", "")
        if "hint_block" in kws:
            block_id = oid
            for kw in kws:
                if kw.startswith("in_front_of_") and kw.endswith("_button"):
                    initial_block_position = kw[len("in_front_of_"):-len("_button")]
                    break
            continue
        if "button" in kws or "benchmark_button" in usd:
            for pos in POSITIONS:
                if f"{pos}_button" in kws:
                    button_id_by_pos[pos] = oid
                    break

    if block_id is None or initial_block_position is None:
        return False
    if set(button_id_by_pos.keys()) != set(POSITIONS):
        return False

    remaining = [p for p in POSITIONS if p != initial_block_position]
    random.shuffle(remaining)
    visit_order = [initial_block_position] + remaining

    table_id = None
    for oid, info in scene_info["layout"].items():
        if info.get("usd", "") == "table_000" or "table" in info.get("keywords", []):
            table_id = oid
            break
    table_pos, _ = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)
    table_top_z = table_pos[2] + 0.5 * table_size[2]

    block_size = robot.get_object_size_in_world_frame(block_id)
    block_z = table_top_z + 0.5 * block_size[2] + 0.005
    block_prim_path = robot.get_object_prim_path(block_id)

    BLOCK_X_OFFSET_IN_FRONT = 0.24
    POS_ZH = {"left": "左侧", "middle": "中间", "right": "右侧"}
    all_button_ids = list(button_id_by_pos.values())

    BLOCK_DWELL_SEC = 2.5  # delay before each block teleport (= dwell at prev pose)

    # Observation phase: NOT a robot action; the robot watches the cue.
    # Every step (including the first) issues set_object_parameter_action so
    # all observation events are uniform.
    for visit_idx, pos in enumerate(visit_order):
        btn_id = button_id_by_pos[pos]
        btn_pos, _ = robot.get_object_pose(btn_id)
        target_xyz = [
            float(btn_pos[0] + BLOCK_X_OFFSET_IN_FRONT),
            float(btn_pos[1]),
            float(block_z),
        ]
        ok = robot.set_object_parameter_action(
            block_prim_path, "xformOp:translate", target_xyz, delay=BLOCK_DWELL_SEC,
            description={
                "zh": f"观察积木出现在{POS_ZH[pos]}按钮前",
                "en": f"observes the building block in front of the {pos} button",
            },
        )
        if not ok:
            return False

    far_xyz = [20.0, 20.0, float(block_z)]
    ok = robot.set_object_parameter_action(
        block_prim_path, "xformOp:translate", far_xyz, delay=BLOCK_DWELL_SEC,
        description={
            "zh": "观察积木被移走，准备按按钮",
            "en": "observes the building block being removed, ready to press buttons",
        },
    )
    if not ok:
        return False

    # After all block-observation events finish, close both grippers and
    # capture each arm's initial ee pose so the used arm can be moved back
    # without back_to_origin.
    robot.close_gripper(arm="left")
    robot.close_gripper(arm="right")
    initial_ee_pose = {
        "left": robot.get_ee_pose(arm="left"),
        "right": robot.get_ee_pose(arm="right"),
    }

    for pos in visit_order:
        used_arm = _press_button(robot, button_id_by_pos[pos], pos)
        if used_arm is None:
            return False
        # Lift +0.10 m to clear the just-pressed button — gripper stays closed.
        cur_ee = robot.get_ee_pose(arm=used_arm)
        lift_pose = cur_ee.copy()
        lift_pose[2, 3] += 0.10
        arm_zh = "左臂" if used_arm == "left" else "右臂"
        lift_cfg = MoveConfig(
            unmovable_objects=all_button_ids,
            action_description={"zh": "抬起", "en": "lifts up"},
        )
        if not robot.move_actor_to_target_pose(
            "gripper", np.array([lift_pose]), arm=used_arm, move_config=lift_cfg
        ):
            return False
        # Move used arm back to its captured initial ee pose.
        home_cfg = MoveConfig(
            unmovable_objects=all_button_ids,
            action_description={"zh": "归位", "en": "returns to initial pose"},
        )
        if not robot.move_actor_to_target_pose(
            "gripper", np.array([initial_ee_pose[used_arm]]),
            arm=used_arm, move_config=home_cfg,
        ):
            return False

    return success


def _press_button(robot: RobotAPI, button_id: str, position_tag: str):
    button_pos, _ = robot.get_object_pose(button_id)
    button_origin_pos = button_pos.copy()
    part_pos_before, _ = robot.get_object_pose(button_id, articulation_part="part_00")
    button_size = robot.get_object_size_in_world_frame(button_id)
    arm = robot.judge_arm_to_use(button_pos)
    button_top_z = button_pos[2] + 0.5 * button_size[2]
    target_height = button_top_z + 0.05

    target_poses = []
    down_z_rotation = euler2mat([np.pi, 0, 0])
    for dx in np.linspace(-0.001, 0.001, 3):
        for dy in np.linspace(-0.001, 0.001, 3):
            position = np.array([button_pos[0] + dx, button_pos[1] + dy, target_height])
            target_pose = np.eye(4)
            target_pose[:3, 3] = position
            target_pose[:3, :3] = down_z_rotation
            for angle in np.linspace(180, 270, 12, endpoint=False):
                rotated_target_pose = rotate_along_axis(target_pose, angle, rot_axis="z", use_local=False)
                for angle_x in np.random.uniform(-10, 10, 10):
                    rotated_target_pose = rotate_along_axis(rotated_target_pose, angle_x, rot_axis="x", use_local=False)
                    for angle_y in np.random.uniform(-10, 10, 10):
                        rotated_target_pose = rotate_along_axis(rotated_target_pose, angle_y, rot_axis="y", use_local=False)
                target_poses.append(rotated_target_pose)
    target_poses = np.array(target_poses)

    robot.close_gripper(arm=arm)
    POS_ZH = {"left": "左侧", "middle": "中间", "right": "右侧"}
    approach_cfg = MoveConfig(
        action_description={
            "zh": f"将夹爪移动到{POS_ZH[position_tag]}按钮上方",
            "en": f"moves the gripper above the {position_tag} button",
        },
    )
    if not robot.move_actor_to_target_pose("gripper", target_poses, arm, move_config=approach_cfg):
        return None

    def check_button_pressed():
        cur_pos, _ = robot.get_object_pose(button_id)
        if cur_pos is None or np.linalg.norm(cur_pos - button_origin_pos) > 0.01:
            return False
        part_pos_after, _ = robot.get_object_pose(button_id, articulation_part="part_00")
        return part_pos_after[2] <= part_pos_before[2] - 0.001

    press_cfg = MoveConfig(
        extra_params={
            "from_current_pose": True,
            "goal_offset": [0, 0, -0.05, 1, 0, 0, 0],
            "path_constraint": [0.1, 0.1, 0.1, 0.1, 0.1, 0.0],
            "offset_and_constraint_in_goal_frame": False,
        },
        extra_checkers=[check_button_pressed],
        action_description={
            "zh": f"按压{POS_ZH[position_tag]}按钮",
            "en": f"presses the {position_tag} button",
        },
        unmovable_objects=[button_id],
    )
    robot.client.remove_objs_from_obstacle([robot.get_object_prim_path(button_id)])
    if not robot.move_actor_to_target_pose("gripper", target_poses, arm, move_config=press_cfg):
        return None
    return arm
```
