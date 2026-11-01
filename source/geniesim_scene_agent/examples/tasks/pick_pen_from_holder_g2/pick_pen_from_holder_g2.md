# Pick a pen from its holder

**Tags:** `pick_only` `pick_actor` `pen` `pen_holder` `GraspConfig` `passive_primitive` `grasp_lower_percentile` `pick_up_distance` `judge_arm_to_use` `usd_match` `keywords` `unmovable_objects`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find table (usd table_000 or keywords "table"), pen_holder (usd benchmark_stationery_021 or keywords "pen_holder"), pen (usd containing "benchmark_pen_"). Iterate layout once and assign table_id, pen_holder_id, pen_id.
2. **Arm:** arm = judge_arm_to_use(pen_pos). Get pen pose; fail if None.
3. **Pick:** GraspConfig(grasp_lower_percentile=70, passive_primitive="default", pick_up_distance=0.15, unmovable_objects=[table_id, pen_holder_id]). pick_actor(pen_id, arm, grasp_config). No place step—task is pick only.

---

# INPUT

**scene_description:** A desk with a pen holder (benchmark_stationery_021) on the table at random position and rotation, and a pen (one of benchmark_pen_000/001/002/003/005/006) placed inside the holder.

**task description:** Pick up the pen from the pen holder.

**code to generate the scene:**

```python
from helper import *
import numpy as np

"""
scene_name: desk_with_pen_holder
description: A pen holder with a pen placed on a table
"""


def get_pen_holder_position() -> Tuple[P, float]:
    """
    Generates random position and rotation for the pen holder on the table.

    Returns:
        Tuple[P, float]: Position (x, y, 0) and random z-rotation angle
    """
    x_pos = np.random.uniform(-0.15, 0.15)
    y_pos = np.random.uniform(-0.2, 0.2)
    z_rotation = np.random.uniform(0, 2 * np.pi)
    return (x_pos, y_pos, 0), z_rotation


def get_random_pen() -> str:
    """
    Randomly selects a pen from the available pen assets (excluding benchmark_pen_004).

    Returns:
        str: The object_id of the selected pen
    """
    pen_ids = [
        "benchmark_pen_000",
        "benchmark_pen_001",
        "benchmark_pen_002",
        "benchmark_pen_003",
        "benchmark_pen_005",
        "benchmark_pen_006",
    ]
    return np.random.choice(pen_ids)


@register()
def pen_in_holder(pen_holder_shape: Shape) -> Shape:
    """
    Places a randomly selected pen inside the pen holder.
    The pen is positioned at the center, slightly above, and rotated 90 degrees around x-axis.
    """
    holder_info = get_object_info(pen_holder_shape)
    holder_center = holder_info["center"]
    pen_oid = get_random_pen()
    pen_shape = library_call(
        "usd",
        oid=pen_oid,
        keywords=["pen", "writing_tool", "cylinder", "long"],
    )
    pen_size = compute_shape_sizes(pen_shape)
    pen_offset_z = holder_center[2] + pen_size[2] / 2.0 + 0.05
    pen_shape = transform_shape(
        pen_shape,
        rotation_matrix(np.pi / 2, direction=(1, 0, 0), point=holder_center),
    )
    pen_shape = transform_shape(
        pen_shape,
        translation_matrix((holder_center[0], holder_center[1], pen_offset_z)),
    )
    return pen_shape


@register()
def place_pen_holder_on_table(table_shape: Shape) -> Shape:
    """
    Places the pen holder on the table with random position and rotation.
    """
    pos, z_rotation = get_pen_holder_position()
    pen_holder_shape = library_call(
        "usd",
        oid="benchmark_stationery_021",
        keywords=["pen_holder", "stationery", "container", "cylinder"],
    )
    holder_info = get_object_info(pen_holder_shape)
    holder_min = compute_shape_min(pen_holder_shape)
    holder_max = compute_shape_max(pen_holder_shape)
    holder_height = holder_max[2] - holder_min[2]
    table_info = get_object_info(table_shape)
    table_top_z = table_info["center"][2] + get_subpart_info(object_id="table_000", subpart_id="desktop")["xyz_max"][2]
    pen_holder_shape = transform_shape(
        pen_holder_shape,
        translation_matrix((pos[0], pos[1], table_top_z)),
    )
    pen_holder_shape = transform_shape(
        pen_holder_shape,
        rotation_matrix(z_rotation, direction=(0, 0, 1), point=pos),
    )
    pen_shape = pen_in_holder(pen_holder_shape)
    return concat_shapes(table_shape, pen_holder_shape, pen_shape)


@register()
def desk_with_pen_holder() -> Shape:
    """
    Main scene composition: table with pen holder containing a pen.
    """
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "white", "desktop_with_legs"],
    )
    return place_pen_holder_on_table(table_shape)


@register()
def root_scene() -> Shape:
    return desk_with_pen_holder()
```

---

# OUTPUT

```python
from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    # Initialize objects_list based on provided interaction data [1] and scene description
    objects_list = {
        "benchmark_pen_000": {"has_interactions": True, "object_type": "pen"},
        "benchmark_pen_001": {"has_interactions": True, "object_type": "pen"},
        "benchmark_pen_002": {"has_interactions": True, "object_type": "pen"},
        "benchmark_pen_003": {"has_interactions": True, "object_type": "pen"},
        "benchmark_pen_005": {"has_interactions": True, "object_type": "pen"},
        "benchmark_pen_006": {"has_interactions": True, "object_type": "pen"},
        "benchmark_stationery_021": {"has_interactions": False, "object_type": "pen_holder"},
        "table_000": {"has_interactions": False, "object_type": "table"}
    }

    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    # MANDATORY: Set task description before first action
    description_json = {
        "task_name": "把笔从笔筒中拿出来",
        "english_task_name": "pick up the pen from the pen holder",
        "init_scene_text": "The robot is in front of the table, there is a pen holder on the table with a pen inside"
    }
    robot.client.set_task_task_basic_info(description_json)

    # 1. Find objects dynamically
    table_id = None
    pen_holder_id = None
    pen_id = None

    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        if usd == "table_000" or any("table" in kw.lower() for kw in keywords):
            table_id = obj_id
        elif usd == "benchmark_stationery_021" or any("pen_holder" in kw.lower() for kw in keywords):
            pen_holder_id = obj_id
        elif "benchmark_pen_" in usd:
            pen_id = obj_id

    if not pen_id:
        print("Pen not found")
        return False
    if not pen_holder_id:
        print("Pen holder not found")
        return False
    if not table_id:
        print("Table not found")
        return False

    print(f"Found pen: {pen_id}, pen holder: {pen_holder_id}, table: {table_id}")

    # 2. Determine which arm to use based on pen position
    pen_pos, _ = robot.get_object_pose(pen_id)
    if pen_pos is None:
        print("Cannot get pen position")
        return False

    arm = robot.judge_arm_to_use(pen_pos)
    print(f"Using arm: {arm}")

    # 3. Prepare grasp configuration
    # Interaction data [1] confirms the pen assets have a 'default' grasp primitive
    # We protect the table and pen holder from moving during the grasp
    unmovable_objects = [table_id, pen_holder_id]

    grasp_config = GraspConfig(
        grasp_lower_percentile=70,
        passive_primitive="default",  # Use 'default' primitive from interaction data [1]
        pick_up_distance=0.15,         # Lift 15cm to ensure it clears the pen holder
        unmovable_objects=unmovable_objects,
        action_description={"zh": "拿起笔筒里的笔", "en": "picks up the pen from the pen holder"}
    )

    # 4. Execute the pick action
    success = robot.pick_actor(pen_id, arm=arm, grasp_config=grasp_config)

    if success:
        print(f"Successfully picked up pen {pen_id}")
    else:
        print(f"Failed to pick up pen {pen_id}")

    return success
```
