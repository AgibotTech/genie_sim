# Assemble a gear or bearing

**Tags:** `assembly` `gearbox` `gear` `bearing` `place_actor_to_target_actor` `passive_primitive` `small_cylinder` `right_cylinder` `extra_checkers` `interactions_passive_place` `part_size_mapping` `GraspConfig` `PlaceConfig` `pick_and_place_actor_to_target_actor`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find gearbox by usd "benchmark_gearbox_000". Find the part (gear or bearing) by matching usd to part_candidates (benchmark_bearing_001/000, benchmark_gear_001/000); exclude gearbox. Set part_type (bearing/gear) and part_size: gear_001 or bearing_000 → "small"; gear_000 or bearing_001 → "large".
2. **Target primitive:** target_primitive = "small_cylinder" if part_size == "small" else "right_cylinder". Get passive place points from interactions["benchmark_gearbox_000"]["passive"]["place"][target_primitive] for optional extra_checkers.
3. **Place config:** GraspConfig with passive_primitive="default", set_grasp_along_axis="y", unmovable_objects=[gearbox_id]. PlaceConfig with active_primitive="normal_place", passive_primitive=target_primitive, pre_insert_offset=0.1, extra_params={"insert_depth": 0.05}, extra_checkers to verify part is on target cylinder (transform primitive local xyz to world, check distance < 2 cm and part height).
4. **Execution:** robot.pick_and_place_actor_to_target_actor(actor=part_id, target_actor=gearbox_id, arm, grasp_config, place_config).

---

# INPUT

**scene_description:** A table with a gearbox (benchmark_gearbox_000) in the center and a gear or bearing (benchmark_gear_000/001, benchmark_bearing_000/001) placed nearby. The gearbox has placement primitives (e.g. small_cylinder, right_cylinder). The task is to pick the part and mount it on the corresponding column of the gearbox.

**task description:** Pick up the part (gear or bearing) and mount it on the corresponding column of the gearbox. Small part goes to small_cylinder; large part to right_cylinder (or back_cylinder).

**code to generate the scene:**

```python
from helper import *
import numpy as np

"""
scene_name: table_with_gearbox_and_part
description: A table with a gearbox in the center and a gear or bearing placed nearby, avoiding collision.
"""


@register()
def table_scene() -> Shape:
    """
    Generates a scene with a table, a centered gearbox, and a randomly placed gear/bearing.
    """
    # 1. Load the table
    table_shape = library_call("usd", oid="table_000", keywords=["main_table", "furniture", "table_000"])
    table_info = get_object_info(table_shape)
    desktop_z = table_info["max"][2] + 0.1  # Top surface height

    # 2. Place the Gearbox (Center)
    # Random position in range x(-0.1, 0.05), y(-0.05, 0.05)
    g_x = np.random.uniform(-0.1, 0.05)
    g_y = np.random.uniform(-0.05, 0.05)

    # Random rotation within 5 degrees around Z
    g_rot_deg = np.random.uniform(-5.0 + 180.0, 5.0 + 180.0)

    gearbox_shape = library_call(
        "usd",
        oid="benchmark_gearbox_000",
        keywords=["gearbox_center", "mechanical_part", "placed_middle"],
    )
    # Transform Gearbox
    # First translate to position (z=0 for now, will lift later)
    gearbox_shape = transform_shape(gearbox_shape, translation_matrix((g_x, g_y, 0)))
    # Then rotate around its center
    g_center = compute_shape_center(gearbox_shape)
    gearbox_shape = transform_shape(
        gearbox_shape,
        rotation_matrix(np.radians(g_rot_deg), (0, 0, 1), g_center),
    )

    # 3. Place Side Part (Gear or Bearing)
    # Candidates list
    parts_list = [
        "benchmark_bearing_001",  # Small bearing
        "benchmark_bearing_000",  # Large bearing
        "benchmark_gear_001",  # Small gear
        "benchmark_gear_000",  # Large gear
    ]
    side_oid = np.random.choice(parts_list)

    # Find valid position satisfying distance constraint
    # Range x(-0.2, 0.15), y (-0.2, 0.2)
    # Distance >= 0.2 from gearbox center

    valid_pos_found = False
    s_x, s_y = 0.0, 0.0
    max_attempts = 100

    for _ in range(max_attempts):
        s_x = np.random.uniform(-0.2, 0.15)
        s_y = np.random.uniform(-0.2, 0.2)

        dist = np.sqrt((s_x - g_x) ** 2 + (s_y - g_y) ** 2)
        if dist >= 0.2:
            valid_pos_found = True
            break

    # If no position found after attempts, keep the last sampled one (though logic suggests it should find one quickly)
    # Random rotation arbitrary angle around Z
    s_rot_deg = np.random.uniform(0.0, 360.0)

    side_shape = library_call(
        "usd",
        oid=side_oid,
        keywords=[f"part_{side_oid}", "mechanical", "random_placement", "side_item"],
    )
    # Transform Side Part
    side_shape = transform_shape(side_shape, translation_matrix((s_x, s_y, 0)))
    s_center = compute_shape_center(side_shape)
    side_shape = transform_shape(
        side_shape,
        rotation_matrix(np.radians(s_rot_deg), (0, 0, 1), s_center),
    )

    # 4. Combine objects and lift to table height
    objects_on_table = concat_shapes(gearbox_shape, side_shape)
    # Lift objects so their bottom sits on desktop_z
    objects_on_table = transform_shape(objects_on_table, translation_matrix((0, 0, desktop_z)))

    # Final scene
    return concat_shapes(table_shape, objects_on_table)


@register()
def root_scene() -> Shape:
    return library_call("table_scene")
```

---

# OUTPUT

```python
from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    ## Initialize objects_list based on provided interaction data
    ## Note: benchmark_gearbox_000 interaction data is provided in the prompt text
    objects_list = {
        "benchmark_gearbox_000": {
            "has_interactions": True,
            "object_type": "gearbox",
            "interaction_type": "container"
        },
        "benchmark_bearing_001": {
            "has_interactions": True,
            "object_type": "bearing",
            "size": "small"
        },
        "benchmark_bearing_000": {
            "has_interactions": True,
            "object_type": "bearing",
            "size": "large"
        },
        "benchmark_gear_001": {
            "has_interactions": True,
            "object_type": "gear",
            "size": "small"
        },
        "benchmark_gear_000": {
            "has_interactions": True,
            "object_type": "gear",
            "size": "large"
        }
    }

    ## Runtime-only data
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    ## MANDATORY: Set task description before first action
    description_json = {
        "task_name": "拿起零件并将其安装到齿轮盒上对应的柱子上",
        "english_task_name": "pick up the part and mount it on the corresponding column of the gearbox",
        "init_scene_text": "The robot is in front of the table, there is a gearbox in the center and a gear or bearing placed nearby"
    }
    robot.client.set_task_task_basic_info(description_json)

    ## 1. Find the gearbox (benchmark_gearbox_000)
    gearbox_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        if "benchmark_gearbox_000" in usd:
            gearbox_id = obj_id
            break

    if not gearbox_id:
        print("Gearbox not found")
        return False

    print(f"Found gearbox: {gearbox_id}")

    ## 2. Find the gear or bearing (side item)
    part_candidates = [
        "benchmark_bearing_001", "benchmark_bearing_000",
        "benchmark_gear_001", "benchmark_gear_000"
    ]

    part_id = None
    part_usd = None
    part_type = None
    part_size = None

    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        # Check if it's one of the candidate parts
        if any(candidate in usd for candidate in part_candidates):
            # Ensure it's not the gearbox (though keywords usually distinguish)
            if "gearbox" not in keywords:
                part_id = obj_id
                part_usd = usd
                # Determine type and size
                if "bearing" in usd:
                    part_type = "bearing"
                elif "gear" in usd:
                    part_type = "gear"

                # Determine size based on asset ID suffix
                if "gear_001" in usd or "bearing_000" in usd:
                    part_size = "small"
                elif "gear_000" in usd or "bearing_001" in usd:
                    part_size = "large"
                break

    if not part_id:
        print("Gear or bearing not found")
        return False

    print(f"Found part: {part_id} (type: {part_type}, size: {part_size})")

    ## 3. Determine target primitive based on size
    ## Small gear/bearing -> small_cylinder
    ## Large gear/bearing -> right_cylinder (or back_cylinder, choosing right_cylinder as primary)
    part_name_zh = "轴承" if part_type == "bearing" else "齿轮"
    part_name_en = "bearing" if part_type == "bearing" else "gear"
    if part_size == "small":
        target_primitive = "small_cylinder"
        place_desc_zh = f"将{part_name_zh}放置到齿轮盒的小柱子上"
        place_desc_en = f"places the {part_name_en} on the small column of the gearbox"
    else: # large
        target_primitive = "right_cylinder"
        place_desc_zh = f"将{part_name_zh}放置到齿轮盒的右侧柱子上"
        place_desc_en = f"places the {part_name_en} on the right column of the gearbox"
    ## 4. Get part position to determine arm
    part_position, _ = robot.get_object_pose(part_id)
    if part_position is None:
        print(f"Cannot get position for part {part_id}")
        return False

    arm = robot.judge_arm_to_use(part_position)
    print(f"Using arm: {arm}")

    ## 5. Define success checker
    ## Verify that the part is actually placed on the target cylinder
    def create_part_mounted_checker(part_id, gearbox_id, target_primitive):
        def checker():
            # Get current poses
            part_pos, _ = robot.get_object_pose(part_id)
            gear_pos, gear_rot = robot.get_object_pose(gearbox_id)

            if part_pos is None or gear_pos is None or gear_rot is None:
                return False

            # Get target cylinder coordinates from interaction data (in object local frame)
            gearbox_interaction = interactions.get("benchmark_gearbox_000", {})
            passive_place = gearbox_interaction.get("passive", {}).get("place", {})

            if target_primitive not in passive_place:
                print(f"Target primitive {target_primitive} not found in interaction data")
                return False

            # There should be at least one point for the primitive
            primitive_points = passive_place[target_primitive]
            if len(primitive_points) == 0:
                return False

            target_local_xyz = np.array(primitive_points[0]["xyz"])

            # Transform local target to world coordinates
            # World = Gear_Transform * Local
            gear_rot_matrix = quat2mat_wxyz(gear_rot)
            target_world_offset = np.dot(gear_rot_matrix, target_local_xyz)
            target_world_pos = gear_pos + target_world_offset

            # Check distance (tolerance: 2cm)
            distance = np.linalg.norm(part_pos[:2] - target_world_pos[:2])
            is_close = distance < 0.02

            if not is_close:
                print(f"Part distance to target {target_primitive}: {distance:.3f}m (threshold: 0.02m)")

            if part_pos[2] > target_world_pos[2] - 0.04:
                print(f"Part is not on target height: {part_pos[2]} > {target_world_pos[2]}")
                return False

            return is_close

        return checker

    ## 6. Execute Pick and Place
    grasp_config = GraspConfig(
        passive_primitive="default",  # Parts have default grasp
        set_grasp_along_axis="y",
        # set_grasp_pose_center=True,
        pick_up_distance=0.18,
        unmovable_objects=[gearbox_id],
        action_description={"zh": f"拿起{part_name_zh}", "en": f"picks up the {part_name_en}"}
    )

    place_config = PlaceConfig(
        active_primitive="normal_place",  # Parts use normal_place for active placement
        passive_primitive=target_primitive,  # Target specific cylinder
        pre_insert_offset=0.1,
        extra_params={
            "insert_depth": 0.05
        },
        unmovable_objects=[gearbox_id],
        extra_checkers=[create_part_mounted_checker(part_id, gearbox_id, target_primitive)],
        action_description={"zh": place_desc_zh, "en": place_desc_en}
    )

    print(f"Attempting to place part on {target_primitive}")
    current_success = robot.pick_and_place_actor_to_target_actor(
        actor=part_id,
        target_actor=gearbox_id,
        arm=arm,
        grasp_config=grasp_config,
        place_config=place_config
    )

    if not current_success:
        print(f"Failed to place part {part_id}")
        success = False
    else:
        print(f"Successfully placed part {part_id} on {target_primitive}")

    return success
```
