# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os
import random

def execute_task(robot: RobotAPI):
    # Map asset IDs to names.
    OBJECT_NAMES = {
        "benchmark_beverage_bottle_001": {"zh": "可口可乐", "en": "coke cola"},
        "benchmark_beverage_bottle_003": {"zh": "芬达", "en": "fanta"},
        "benchmark_beverage_bottle_004": {"zh": "乌龙茶饮料", "en": "oolong tea beverage"},
        "genie_beverage_bottle_009": {"zh": "葡萄汁", "en": "grape juice"},
        "benchmark_toy_car_018": {"zh": "玩具车", "en": "toy car"},
        "benchmark_building_blocks_072": {"zh": "积木", "en": "building blocks"},
        "benchmark_cup_043": {"zh": "马克杯", "en": "mug"},
        "benchmark_cup_105": {"zh": "纸杯", "en": "paper cup"},
        "benchmark_toiletries_009": {"zh": "洗面奶", "en": "face cleanser"},
        "benchmark_stationery_009": {"zh": "黄色订书机", "en": "yellow stapler"},
        "benchmark_mouse_000": {"zh": "鼠标", "en": "mouse"},
        "benchmark_bottle_017": {"zh": "保温杯", "en": "water bottle"},
        "benchmark_apple_002": {"zh": "苹果", "en": "apple"},
        "benchmark_orange_004": {"zh": "橙子", "en": "orange"},
        "benchmark_lemon_030": {"zh": "柠檬", "en": "lemon"},
        "benchmark_peach_021": {"zh": "桃子", "en": "peach"},
        "benchmark_green_apple_001": {"zh": "青苹果", "en": "green apple"},
        "benchmark_bowl_sponge_000": {"zh": "洗碗海绵", "en": "bowl sponge"},
        "benchmark_clock_044": {"zh": "闹钟", "en": "clock"},
        "benchmark_tape_measure_007": {"zh": "卷尺", "en": "tape measure"},
        "benchmark_toy_plane_029": {"zh": "玩具飞机", "en": "toy plane"},
        "benchmark_rubik_cube_001": {"zh": "魔方", "en": "rubik cube"},
        "benchmark_medicine_bottle_000": {"zh": "药瓶", "en": "medicine bottle"},
        "benchmark_garage_kit_064": {"zh": "小火龙摆件", "en": "charmander figure"}
    }

    # Initialize objects_list; all assets have interaction data.
    objects_list = {}
    for obj_id in OBJECT_NAMES.keys():
        objects_list[obj_id] = {"has_interactions": True}

    # Runtime data.
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    # 1. Locate table_000.
    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        if usd == "table_000" or "table" in keywords or "workspace" in keywords:
            table_id = obj_id
            break

    if not table_id:
        print("Table not found")
        return False

    print(f"Found table: {table_id}")

    # 2. Locate all tabletop objects.
    desktop_objects = []
    for obj_id, obj_info in scene_info["layout"].items():
        keywords = obj_info.get("keywords", [])
        usd = obj_info.get("usd", "")

        # Exclude the table itself.
        if obj_id == table_id:
            continue

        # Check tabletop membership; scene generation marks these objects with "on_desk".
        if any("on_desk" in kw for kw in keywords):
            pos, rot = robot.get_object_pose(obj_id)
            if pos is not None:
                # Get the name.
                names = OBJECT_NAMES.get(usd, {"zh": usd, "en": usd})

                desktop_objects.append({
                    "id": obj_id,
                    "usd": usd,
                    "position": pos,
                    "rotation": rot,
                    "keywords": keywords,
                    "chinese_name": names["zh"],
                    "english_name": names["en"]
                })

    # If no tabletop objects were found, try another method.
    if len(desktop_objects) == 0:
        print("No objects found on desk using 'on_desk' keyword, trying alternative search...")
        table_pos, _ = robot.get_object_pose(table_id)
        if table_pos is not None:
            for obj_id, obj_info in scene_info["layout"].items():
                if obj_id == table_id:
                    continue

                pos, rot = robot.get_object_pose(obj_id)
                if pos is not None:
                    # Check proximity to the table by z coordinate.
                    if abs(pos[2] - table_pos[2]) < 0.2:
                        usd = obj_info.get("usd", "")
                        names = OBJECT_NAMES.get(usd, {"zh": usd, "en": usd})

                        desktop_objects.append({
                            "id": obj_id,
                            "usd": usd,
                            "position": pos,
                            "rotation": rot,
                            "keywords": obj_info.get("keywords", []),
                            "chinese_name": names["zh"],
                            "english_name": names["en"]
                        })

    if len(desktop_objects) < 2:
        print(f"Found only {len(desktop_objects)} objects on desk, need at least 2")
        return False

    print(f"Found {len(desktop_objects)} objects on desk:")
    for i, obj in enumerate(desktop_objects):
        print(f"  {i+1}. {obj['id']} - {obj['chinese_name']} at position: {obj['position']}")

    # 3. Helper functions.

    def check_object_already_in_direction(obj_a_pos, obj_b_pos, direction):
        """Check whether A is already in the requested direction from B, allowing orthogonal offset."""
        dx = obj_a_pos[0] - obj_b_pos[0]  # x difference.
        dy = obj_a_pos[1] - obj_b_pos[1]  # y difference.

        main_threshold = 0.05  # Main-direction threshold: 5cm.
        ortho_threshold = 0.03  # Orthogonal-direction threshold: 3cm.

        if direction == "up":
            # Above: positive x difference with small y difference.
            return dx > main_threshold and abs(dy) < ortho_threshold

        elif direction == "down":
            # Below: negative x difference with small y difference.
            return dx < -main_threshold and abs(dy) < ortho_threshold

        elif direction == "left":
            # Left: positive y difference with small x difference.
            return dy > main_threshold and abs(dx) < ortho_threshold

        elif direction == "right":
            # Right: negative y difference with small x difference.
            return dy < -main_threshold and abs(dx) < ortho_threshold

        elif direction == "top_left":
            # Upper left: both x and y differences positive.
            return dx > main_threshold and dy > main_threshold

        elif direction == "bottom_left":
            # Lower left: x difference negative, y difference positive.
            return dx < -main_threshold and dy > main_threshold

        elif direction == "top_right":
            # Upper right: x difference positive, y difference negative.
            return dx > main_threshold and dy < -main_threshold

        elif direction == "bottom_right":
            # Lower right: both x and y differences negative.
            return dx < -main_threshold and dy < -main_threshold

        return False

    def get_object_half(position_y):
        """Determine the object half from its y coordinate."""
        if position_y > -0.1:  # y > -0.1: left half.
            return "left"
        elif position_y < 0.1:  # y < 0.1: right half.
            return "right"
        else:  # -0.1 <= y <= 0.1: either half.
            return "both"

    def check_same_half(half1, half2):
        """Check whether two halves match, including the both case."""
        if half1 == "both" or half2 == "both":
            return True
        return half1 == half2

    def check_position_collision(position, excluded_objects=None, min_distance=0.1):
        """Check collision with other objects at this position."""
        if excluded_objects is None:
            excluded_objects = []

        for obj in desktop_objects:
            if obj["id"] in excluded_objects:
                continue

            obj_pos = obj["position"]
            distance = np.linalg.norm(position[:2] - obj_pos[:2])

            if distance < min_distance:
                return True

        return False

    def check_position_in_bounds(position):
        """Check whether the position is within bounds."""
        # Check x range (-0.2, 0.15).
        if position[0] < -0.2 or position[0] > 0.15:
            return False

        return True

    def generate_base_positions(obj_b_pos, obj_a_size, obj_b_size, direction, margin=0.05):
        """Generate base positions for the direction."""
        base_offset_x = (obj_a_size[0] + obj_b_size[0]) / 2.0 + margin
        base_offset_y = (obj_a_size[1] + obj_b_size[1]) / 2.0 + margin

        base_positions = []

        if direction == "up":
            # Above (increasing x): several x positions.
            for dx in np.linspace(-0.02, 0.02, 5):
                pos = obj_b_pos.copy()
                pos[0] += base_offset_x + dx
                # Keep y approximately unchanged.
                base_positions.append(pos)

        elif direction == "down":
            # Below (decreasing x): several x positions.
            for dx in np.linspace(-0.02, 0.02, 5):
                pos = obj_b_pos.copy()
                pos[0] -= base_offset_x + dx
                # Keep y approximately unchanged.
                base_positions.append(pos)

        elif direction == "left":
            # Left (increasing y): several y positions.
            for dy in np.linspace(-0.02, 0.02, 5):
                pos = obj_b_pos.copy()
                pos[1] += base_offset_y + dy
                # Keep x approximately unchanged.
                base_positions.append(pos)

        elif direction == "right":
            # Right (decreasing y): several y positions.
            for dy in np.linspace(-0.02, 0.02, 5):
                pos = obj_b_pos.copy()
                pos[1] -= base_offset_y + dy
                # Keep x approximately unchanged.
                base_positions.append(pos)

        elif direction == "top_left":
            # Upper left: several diagonal positions.
            for i in range(5):
                pos = obj_b_pos.copy()
                pos[0] += base_offset_x * 0.7 + np.random.uniform(-0.02, 0.02)
                pos[1] += base_offset_y * 0.7 + np.random.uniform(-0.02, 0.02)
                base_positions.append(pos)

        elif direction == "bottom_left":
            # Lower left: several diagonal positions.
            for i in range(5):
                pos = obj_b_pos.copy()
                pos[0] -= base_offset_x * 0.7 + np.random.uniform(-0.02, 0.02)
                pos[1] += base_offset_y * 0.7 + np.random.uniform(-0.02, 0.02)
                base_positions.append(pos)

        elif direction == "top_right":
            # Upper right: several diagonal positions.
            for i in range(5):
                pos = obj_b_pos.copy()
                pos[0] += base_offset_x * 0.7 + np.random.uniform(-0.02, 0.02)
                pos[1] -= base_offset_y * 0.7 + np.random.uniform(-0.02, 0.02)
                base_positions.append(pos)

        elif direction == "bottom_right":
            # Lower right: several diagonal positions.
            for i in range(5):
                pos = obj_b_pos.copy()
                pos[0] -= base_offset_x * 0.7 + np.random.uniform(-0.02, 0.02)
                pos[1] -= base_offset_y * 0.7 + np.random.uniform(-0.02, 0.02)
                base_positions.append(pos)

        return base_positions

    def generate_candidate_poses(base_positions, base_rotation, obj_a_size, table_top_z):
        """Generate candidate poses from base positions."""
        candidate_poses = []

        for base_pos in base_positions:
            # Set z to the tabletop placement height.
            margin_z = 0.01  # Safety margin.
            position = base_pos.copy()
            position[2] = table_top_z + margin_z + obj_a_size[2] / 2.0

            # Convert orientation using quat2mat_wxyz.
            base_rot_matrix = quat2mat_wxyz(base_rotation)

            # Base pose.
            pose = np.eye(4)
            pose[:3, :3] = base_rot_matrix
            pose[:3, 3] = position
            candidate_poses.append(pose)

            # Add poses with slight orientation variation.
            for angle in [0, np.pi/12, -np.pi/12, np.pi/6, -np.pi/6]:
                rot_z = euler2mat([0, 0, angle])
                final_rot = np.dot(base_rot_matrix, rot_z)

                pose_variation = np.eye(4)
                pose_variation[:3, :3] = final_rot
                pose_variation[:3, 3] = position
                candidate_poses.append(pose_variation)

            # Add small position offsets.
            for dx in np.linspace(-0.005, 0.005, 3):
                for dy in np.linspace(-0.005, 0.005, 3):
                    if dx == 0 and dy == 0:
                        continue

                    offset_position = position + np.array([dx, dy, 0])

                    pose_offset = np.eye(4)
                    pose_offset[:3, :3] = base_rot_matrix
                    pose_offset[:3, 3] = offset_position
                    candidate_poses.append(pose_offset)

        return np.array(candidate_poses)

    # 4. Try generating a feasible instruction.
    directions = [
        "up",           # Above.
        "down",         # Below.
        "left",         # Left.
        "right",        # Right.
        "top_left",     # Upper left.
        "bottom_left",  # Lower left.
        "top_right",    # Upper right.
        "bottom_right"  # Lower right.
    ]

    random.seed(os.getpid())

    # Maximum attempts.
    max_attempts = 100
    found_valid_instruction = False
    valid_base_positions = []
    selected_objects = {}
    selected_direction = ""

    for attempt in range(max_attempts):
        print(f"\nAttempt {attempt+1} to find valid instruction...")

        # Randomly select reference object B.
        obj_b = random.choice(desktop_objects)

        # Randomly select a direction.
        selected_direction = random.choice(directions)

        # Select object A, distinct from B.
        other_objects = [obj for obj in desktop_objects if obj["id"] != obj_b["id"]]
        if len(other_objects) == 0:
            continue

        obj_a = random.choice(other_objects)

        obj_a_id = obj_a["id"]
        obj_b_id = obj_b["id"]
        obj_a_pos = obj_a["position"]
        obj_b_pos = obj_b["position"]

        print(f"Testing: Object A={obj_a['chinese_name']}, Object B={obj_b['chinese_name']}, Direction={selected_direction}")
        print(f"  Position differences: dx={obj_a_pos[0]-obj_b_pos[0]:.3f}, dy={obj_a_pos[1]-obj_b_pos[1]:.3f}")

        # Check whether A is already in the requested direction from B.
        if check_object_already_in_direction(obj_a_pos, obj_b_pos, selected_direction):
            print(f"  Skipped: Object A is already in the {selected_direction} of Object B")
            continue

        # Get object dimensions.
        obj_a_size = robot.get_object_size_in_world_frame(obj_a_id)
        obj_b_size = robot.get_object_size_in_world_frame(obj_b_id)

        if obj_a_size is None or obj_b_size is None:
            continue

        # Determine A's starting half.
        obj_a_start_half = get_object_half(obj_a_pos[1])

        # Generate base positions.
        base_positions = generate_base_positions(obj_b_pos, obj_a_size, obj_b_size, selected_direction, margin=0.05)

        if len(base_positions) == 0:
            continue

        # Filter valid base positions.
        valid_positions = []
        for pos in base_positions:
            # Check boundary conditions.
            if not check_position_in_bounds(pos):
                continue

            # Check collisions.
            if check_position_collision(pos, excluded_objects=[obj_a_id, obj_b_id], min_distance=0.1):
                continue

            # Check half consistency.
            target_half = get_object_half(pos[1])
            if not check_same_half(obj_a_start_half, target_half):
                continue

            # Passed all checks.
            valid_positions.append(pos)

        print(f"  Generated {len(base_positions)} base positions, {len(valid_positions)} valid positions")

        # Save the instruction if enough valid positions exist.
        if len(valid_positions) >= 3:  # Require at least three valid positions.
            selected_objects = {
                "obj_a": obj_a,
                "obj_b": obj_b,
                "obj_a_id": obj_a_id,
                "obj_b_id": obj_b_id,
                "obj_a_pos": obj_a_pos,
                "obj_b_pos": obj_b_pos,
                "obj_a_size": obj_a_size,
                "obj_b_size": obj_b_size,
                "obj_a_name_zh": obj_a["chinese_name"],
                "obj_b_name_zh": obj_b["chinese_name"],
                "obj_a_name_en": obj_a["english_name"],
                "obj_b_name_en": obj_b["english_name"]
            }
            valid_base_positions = valid_positions
            found_valid_instruction = True
            print(f"✓ Found valid instruction with {len(valid_positions)} valid positions!")
            break

    if not found_valid_instruction:
        print(f"\n❌ Failed to find valid instruction after {max_attempts} attempts")
        return False

    # 5. Get table information.
    table_pos, _ = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)

    if table_pos is None or table_size is None:
        print("Cannot get table position/size")
        return False

    table_top_z = table_pos[2] + table_size[2] / 2.0

    # 6. Generate candidate poses from valid base positions.
    candidate_poses = generate_candidate_poses(
        valid_base_positions,
        selected_objects["obj_a"]["rotation"],
        selected_objects["obj_a_size"],
        table_top_z
    )

    print(f"Generated {len(candidate_poses)} candidate poses from {len(valid_base_positions)} valid base positions")

    # Map direction descriptions.
    direction_descriptions_chinese = {
        "up": "上边",
        "down": "下边",
        "left": "左边",
        "right": "右边",
        "top_left": "左上方",
        "bottom_left": "左下方",
        "top_right": "右上方",
        "bottom_right": "右下方"
    }

    direction_descriptions_english = {
        "up": "top",
        "down": "bottom",
        "left": "left",
        "right": "right",
        "top_left": "top-left",
        "bottom_left": "bottom-left",
        "top_right": "top-right",
        "bottom_right": "bottom-right"
    }

    dir_desc_zh = direction_descriptions_chinese.get(selected_direction, "旁边")
    dir_desc_en = direction_descriptions_english.get(selected_direction, "side")

    # MANDATORY: Set task description dynamically before first action (after determining objects and direction)
    task_description = {
        "task_name": f"place the {selected_objects['obj_a_name_en']} to the {dir_desc_en} of the {selected_objects['obj_b_name_en']}",
        "english_task_name": f"place the {selected_objects['obj_a_name_en']} to the {dir_desc_en} of the {selected_objects['obj_b_name_en']}",
        "init_scene_text": "The robot is in front of the table, there are objects on the table"
    }
    robot.client.set_task_task_basic_info(task_description)

    # 7. Execute movement.
    arm = robot.judge_arm_to_use(selected_objects["obj_a_pos"])
    print(f"\nUsing arm: {arm}")

    # Prepare the unmovable object list.
    unmovable_objects = [table_id, selected_objects["obj_b_id"]]

    # Create grasp configuration.
    grasp_config = GraspConfig(
        unmovable_objects=unmovable_objects,
        action_description={
            "zh": f"拿起{selected_objects['obj_a_name_zh']}",
            "en": f"pick up the {selected_objects['obj_a_name_en']}"
        }
    )

    # Define the checker.
    def create_check_function(obj_a_id, obj_b_id, selected_direction):
        def check_placement_success():
            current_pos, _ = robot.get_object_pose(obj_a_id)
            obj_b_pos, _ = robot.get_object_pose(obj_b_id)

            if current_pos is None or obj_b_pos is None:
                return False

            # Check that the object is on the tabletop.
            if current_pos[2] < table_top_z:
                return False

            # Check position relative to object B.
            dx = current_pos[0] - obj_b_pos[0]  # x difference.
            dy = current_pos[1] - obj_b_pos[1]  # y difference.

            main_threshold = 0.02  # Main-direction threshold: 2cm.
            ortho_threshold = 0.03  # Orthogonal-direction threshold: 3cm.

            if selected_direction == "up":
                return dx > main_threshold and abs(dy) < ortho_threshold
            elif selected_direction == "down":
                return dx < -main_threshold and abs(dy) < ortho_threshold
            elif selected_direction == "left":
                return dy > main_threshold and abs(dx) < ortho_threshold
            elif selected_direction == "right":
                return dy < -main_threshold and abs(dx) < ortho_threshold
            elif selected_direction == "top_left":
                return dx > main_threshold and dy > main_threshold
            elif selected_direction == "bottom_left":
                return dx < -main_threshold and dy > main_threshold
            elif selected_direction == "top_right":
                return dx > main_threshold and dy < -main_threshold
            elif selected_direction == "bottom_right":
                return dx < -main_threshold and dy < -main_threshold

            return False

        return check_placement_success

    # Create placement configuration.
    place_config = PlaceConfig(
        unmovable_objects=unmovable_objects,
        extra_checkers=[create_check_function(
            selected_objects["obj_a_id"],
            selected_objects["obj_b_id"],
            selected_direction
        )],
        action_description={
            "zh": f"将{selected_objects['obj_a_name_zh']}放置到{selected_objects['obj_b_name_zh']}的{dir_desc_zh}",
            "en": f"place the {selected_objects['obj_a_name_en']} to the {dir_desc_en} of the {selected_objects['obj_b_name_en']}"
        }
    )

    # Execute grasp and placement.
    print(f"\nExecuting: Move {selected_objects['obj_a_name_zh']} to {dir_desc_zh} of {selected_objects['obj_b_name_zh']}")
    current_success = robot.pick_and_place_actor_to_target_pose(
        actor=selected_objects["obj_a_id"],
        target_pose=candidate_poses,
        arm=arm,
        grasp_config=grasp_config,
        place_config=place_config
    )

    if not current_success:
        print(f"Failed to place {selected_objects['obj_a_name_zh']}")
        success = False
    else:
        print(f"Successfully placed {selected_objects['obj_a_name_zh']} to {dir_desc_zh} of {selected_objects['obj_b_name_zh']}")

    # 8. Final verification.
    if success:
        final_pos, _ = robot.get_object_pose(selected_objects["obj_a_id"])
        if final_pos is not None:
            print(f"\nFinal position: {final_pos}")
            print(f"Task completed successfully!")

    # Return the arm to its initial position.
    robot.back_to_origin(arm=arm)

    print(f"\n=== Task summary ===")
    if success:
        print(f"✅ SUCCESS: Moved {selected_objects['obj_a_name_zh']} to {dir_desc_zh} of {selected_objects['obj_b_name_zh']}")
    else:
        print(f"❌ FAILED: Could not complete the task")

    return success
