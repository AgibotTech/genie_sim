# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import random


def execute_task(robot: RobotAPI):
    # Initialize objects_list for digit-block assets.
    objects_list = {}
    for i in range(10):  # Digits 0-9.
        oid = f"benchmark_building_blocks_{34+i:03d}"  # 034-043
        objects_list[oid] = {
            "has_interactions": True,
            "object_type": "number_block",
            "digit": i,
        }
    objects_list["table_000"] = {"has_interactions": False, "object_type": "table"}

    # Runtime data.
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    # 1. Locate the table.
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

    # 2. Locate all digit blocks.
    number_blocks = []
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        # Check digit-block IDs 034-043.
        if "benchmark_building_blocks_0" in usd and usd[27:].isdigit():
            digit_str = usd[27:]
            if digit_str.isdigit():
                digit = int(digit_str) - 34  # Map asset suffix to digit: 034->0, 035->1, etc.
                if 0 <= digit <= 9:
                    pos, rot = robot.get_object_pose(obj_id)
                    if pos is not None:
                        size = robot.get_object_size_in_world_frame(obj_id)
                        number_blocks.append(
                            {
                                "id": obj_id,
                                "usd": usd,
                                "digit": digit,
                                "initial_position": pos,
                                "initial_rotation": rot,  # Save initial orientation.
                                "size": size,
                                "keywords": keywords,
                            }
                        )
                        print(f"Found number block: {obj_id}, digit: {digit}")

    if len(number_blocks) < 4:
        print(f"Found only {len(number_blocks)} number blocks, need at least 4")
        return False

    print(f"Total number blocks found: {len(number_blocks)}")

    # 3. Count available copies of each digit.
    digit_counts = {}
    for block in number_blocks:
        digit = block["digit"]
        digit_counts[digit] = digit_counts.get(digit, 0) + 1

    print("Available digit counts:")
    for digit in sorted(digit_counts.keys()):
        print(f"  Digit {digit}: {digit_counts[digit]} blocks")

    # 4. Randomly generate a one-to-three-digit target from available digits.
    def generate_target_number(digit_counts):
        """Generate a one-to-three-digit target number."""
        # Get all available digits.
        available_digits = [d for d, count in digit_counts.items() if count > 0]
        if not available_digits:
            return None

        # Randomly choose one to three digits.
        num_digits = random.randint(1, 3)

        # Ensure enough distinct digits exist.
        if num_digits > len(available_digits):
            num_digits = len(available_digits)

        # Randomly choose available digits for the target.
        selected_digits = random.sample(available_digits, num_digits)

        # First digit cannot be zero.
        if selected_digits[0] == 0 and num_digits > 1:
            # Find a nonzero leading digit.
            non_zero_digits = [d for d in selected_digits if d != 0]
            if non_zero_digits:
                first_digit = non_zero_digits[0]
                # Reorder with a nonzero digit first.
                selected_digits.remove(first_digit)
                selected_digits.insert(0, first_digit)
            else:
                # If only zero exists, use single-digit zero.
                selected_digits = [0]
                num_digits = 1

        # Compose the number.
        target_num = 0
        for digit in selected_digits:
            target_num = target_num * 10 + digit

        # Keep the number within a reasonable range.
        if 0 <= target_num <= 999:
            return target_num, selected_digits

        return None, None

    # Generate the target number.
    target_number, target_digits_list = generate_target_number(digit_counts)

    if target_number is None:
        most_available_digit = max(digit_counts.items(), key=lambda x: x[1])[0]
        target_number = most_available_digit
        target_digits_list = [most_available_digit]

    print(f"Target number to represent: {target_number}")
    print(f"Target digits: {target_digits_list}")

    # 5. Generate an expression yielding that number.
    def generate_equation(target_num):
        """Generate an expression whose result is target_num."""
        # Possible operators.
        operators = ["+", "-", "*", "/"]
        if target_num == 0:
            operators = ["-", "*", "/"]
        max_attempts = 50
        for _ in range(max_attempts):
            op = random.choice(operators)

            if op == "+":
                a = random.randint(1, min(target_num - 1, 50))
                b = target_num - a
                if b > 0:
                    return f"{a} + {b}", [a, b], "+"

            elif op == "-":
                a = random.randint(target_num + 1, target_num + 50)
                b = a - target_num
                if b > 0:
                    return f"{a} - {b}", [a, b], "-"

            elif op == "*":
                # Search for factors.
                factors = []
                for i in range(1, min(target_num + 1, 50)):
                    if target_num % i == 0:
                        factors.append(i)
                if factors:
                    a = random.choice(factors)
                    b = target_num // a
                    if b > 0:
                        return f"{a} × {b}", [a, b], "*"

            elif op == "/":
                b = random.randint(2, 5)
                a = target_num * b
                if a <= 100:
                    return f"{a} ÷ {b}", [a, b], "/"

        # Fall back to simple addition if all methods fail.
        a = max(1, target_num // 2)
        b = target_num - a
        return f"{a} + {b}", [a, b], "+"

    equation_str, equation_numbers, operator = generate_equation(target_number)

    # MANDATORY: Set task description before first action
    description_json = {
        "task_name": f"calculate and arrange the result of {equation_str.replace('×', '×').replace('÷', '÷')}",
        "english_task_name": f"calculate and arrange the result of {equation_str.replace('×', '×').replace('÷', '÷')}",
        "init_scene_text": "The robot is in front of a table with several number blocks (0-9) scattered on it",
    }
    robot.client.set_task_task_basic_info(description_json)

    print(f"Generated equation: {equation_str} = {target_number}")
    print(f"Equation numbers: {equation_numbers}")

    # 6. Convert the target number to a digit list.
    target_digits = []
    if target_number == 0:
        target_digits = [0]
    else:
        temp = target_number
        while temp > 0:
            target_digits.insert(0, temp % 10)
            temp //= 10

    print(f"Target digits to arrange: {target_digits}")

    # 7. Check enough digit blocks exist to form the result.
    required_digits = {}
    for digit in target_digits:
        required_digits[digit] = required_digits.get(digit, 0) + 1

    # Check availability.
    missing_digits = []
    for digit, required_count in required_digits.items():
        available_count = digit_counts.get(digit, 0)
        if available_count < required_count:
            missing_digits.append((digit, required_count, available_count))

    if missing_digits:
        print("Warning: Not enough blocks for some digits:")
        for digit, required, available in missing_digits:
            print(f"  Digit {digit}: need {required}, have {available}")

        # Adjust target to available digits.
        print("Adjusting target number to available digits...")
        # Build available digit list, sorted by quantity.
        available_digits_sorted = sorted(
            [(d, c) for d, c in digit_counts.items() if c > 0],
            key=lambda x: x[1],
            reverse=True,
        )

        # Choose a combination of the most available digits.
        if available_digits_sorted:
            # Use the most available digit.
            target_digits = [available_digits_sorted[0][0]]
            target_number = target_digits[0]
            required_digits = {target_digits[0]: 1}
            print(f"Adjusted target: {target_number}, digits: {target_digits}")

    # 8. Select a concrete block for each target digit.
    selected_blocks = []
    remaining_blocks = number_blocks.copy()

    for digit in target_digits:
        # Find the first available block of that digit.
        found = False
        for i, block in enumerate(remaining_blocks):
            if block["digit"] == digit:
                selected_blocks.append(block)
                remaining_blocks.pop(i)
                found = True
                break

        if not found:
            print(f"Cannot find block for digit {digit}")
            # Try replacing it with the nearest available digit.
            for i, block in enumerate(remaining_blocks):
                # Select any available digit.
                selected_blocks.append(block)
                remaining_blocks.pop(i)
                print(f"Using digit {block['digit']} instead of {digit}")
                found = True
                break

        if not found:
            print(f"Failed to find any block for digit {digit}")
            # Reassign one of the selected blocks to this digit.
            for i, selected in enumerate(selected_blocks):
                if (
                    selected["digit"] in digit_counts
                    and digit_counts[selected["digit"]] > 1
                ):
                    # Find a substitute block.
                    for j, remaining in enumerate(remaining_blocks):
                        if remaining["digit"] == digit:
                            selected_blocks[i] = remaining
                            remaining_blocks.pop(j)
                            found = True
                            break

    if not selected_blocks:
        print("Failed to select any blocks")
        return False

    print(f"Selected {len(selected_blocks)} blocks for arrangement")

    # 9. Get table information for placement.
    table_pos, table_rot = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)

    if table_pos is None or table_size is None:
        print("Cannot get table position/size")
        return False

    table_top_z = table_pos[2] + table_size[2] / 2.0

    # 10. Calculate placement parameters.
    # Center region: x=(-0.13 to -0.15), y=(-0.03, 0.03).
    target_x = random.uniform(-0.15, -0.13)

    # Number of digits.
    num_digits = len(selected_blocks)

    # Calculate each block's width along y.
    block_widths = []
    for block in selected_blocks:
        block_size = block["size"]
        # Use the y dimension as width.
        width = block_size[1]
        block_widths.append(width)

    total_width = sum(block_widths)

    # Spacing: 0.05m between digits.
    spacing = 0.05
    total_spacing = spacing * (num_digits - 1)
    total_needed_width = total_width + total_spacing

    print(f"Number of digits: {num_digits}")
    print(f"Total needed width: {total_needed_width:.3f}m")

    # Starting position uses center y.
    center_y = random.uniform(-0.03, 0.03)

    # Calculate starting y to distribute blocks about center.
    if num_digits == 1:
        start_y = center_y
    else:
        # Calculate total center-to-left-edge distance.
        half_width = total_needed_width / 2.0
        start_y = center_y + half_width - (block_widths[0] / 2.0)

    # 11. Calculate each digit's target position.
    target_positions = []
    current_y = start_y

    for i, (block, width) in enumerate(zip(selected_blocks, block_widths)):
        block_id = block["id"]
        block_size = block["size"]
        digit = block["digit"]
        initial_rotation = block["initial_rotation"]  # Get initial orientation.

        # Calculate y position, starting at the left.
        current_y = current_y - width / 2.0
        target_y = current_y

        # Calculate z: tabletop + safety margin + half block height.
        margin = 0.01  # 1cm safety margin avoids gripper collisions.
        target_z = table_top_z + margin + block_size[2] / 2.0

        # Target position.
        target_position = np.array([target_x, target_y, target_z])
        target_positions.append(
            {
                "id": block_id,
                "digit": digit,
                "position": target_position,
                "size": block_size,
                "width": width,
                "initial_rotation": initial_rotation,  # Save initial orientation.
                "index": i,
            }
        )

        print(
            f"Block {i+1} (digit {digit}) target position: [{target_x:.3f}, {target_y:.3f}, {target_z:.3f}], width: {width:.3f}m"
        )

        # Update current y for the next block.
        current_y = current_y - width / 2.0 - spacing

    # 12. Generate candidate orientations satisfying digit-placement requirements.
    # Local y parallel to world z is already satisfied by upward-facing y on the table.
    # Require local x parallel to negative world x.
    # Rotate initial orientation about world z to align local x with negative world x.
    def calculate_target_rotation(initial_rotation):
        """Calculate the target rotation matrix.
        1. The block already has local y upward, parallel to world z.
        2. Rotate about world z so local x is parallel to negative world x.
        """
        # Convert initial quaternion to rotation matrix.
        initial_rot_mat = quat2mat_wxyz(initial_rotation)  # Initial rotation matrix.

        # Get local x direction in world coordinates.
        # The first rotation-matrix column gives initial local x direction.
        local_x_axis = initial_rot_mat[:, 0]  # [r11, r21, r31]

        # Target: negative world x, [-1, 0, 0].
        target_direction = np.array([-1.0, 0.0, 0.0])

        # Project current local x onto the XY plane.
        local_x_proj = local_x_axis[:2]  # [x, y] components.
        local_x_proj = local_x_proj / (np.linalg.norm(local_x_proj) + 1e-10)

        # Project target onto XY; already [-1, 0].
        target_proj = np.array([-1.0, 0.0])
        target_proj = target_proj / np.linalg.norm(target_proj)

        # Calculate required world-z rotation angle.
        # Use dot and cross products to calculate angle.
        cos_angle = np.dot(local_x_proj, target_proj)
        sin_angle = np.cross(local_x_proj, target_proj)  # 2D cross product.

        # Ensure angle lies in valid range.
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        # Determine rotation sign.
        if sin_angle < 0:
            angle = -angle

        # Construct world-z rotation matrix.
        rot_z = euler2mat([0, 0, angle])

        # Apply rotation to initial matrix.
        target_rot_mat = np.dot(rot_z, initial_rot_mat)

        return target_rot_mat

    def generate_digit_rotations_for_block(initial_rotation):
        """Generate candidate rotation matrices for a particular block."""
        candidate_rotations = []

        # Calculate reference rotation.
        base_rotation = calculate_target_rotation(initial_rotation)
        candidate_rotations.append(base_rotation)

        # Add small world-z variations of +/-5 degrees.
        for angle_deg in [-5, -2.5, 0, 2.5, 5]:
            angle_rad = np.deg2rad(angle_deg)
            rot_z_small = euler2mat([0, 0, angle_rad])
            rotated = np.dot(rot_z_small, base_rotation)
            candidate_rotations.append(rotated)

        # Deduplicate.
        unique_rotations = []
        for rot in candidate_rotations:
            is_unique = True
            for existing in unique_rotations:
                if np.allclose(rot, existing, atol=1e-6):
                    is_unique = False
                    break
            if is_unique:
                unique_rotations.append(rot)

        return unique_rotations

    # 13. Execute placement.
    placed_blocks = []
    last_arm = ""

    for i, target_info in enumerate(target_positions):
        block_id = target_info["id"]
        digit = target_info["digit"]
        base_position = target_info["position"]
        block_size = target_info["size"]
        initial_rotation = target_info["initial_rotation"]

        print(
            f"\n=== Processing block {i+1}/{len(target_positions)}: {block_id} (digit {digit}) ==="
        )

        # Get current block position.
        block_position, block_rotation = robot.get_object_pose(block_id)
        if block_position is None:
            print(f"Warning: Cannot get position for block {block_id}")
            continue

        # Choose the arm.
        block_arm = robot.judge_arm_to_use(block_position)
        print(f"Using arm: {block_arm}")

        # Return to origin when switching arms.
        if last_arm != block_arm and last_arm != "":
            robot.back_to_origin(arm=last_arm)
        last_arm = block_arm

        # Create grasp configuration.
        unmovable_objects = [table_id]
        unmovable_objects.extend([b["id"] for b in placed_blocks])

        grasp_config = GraspConfig(
            is_from_up_side=True,
            grasp_offset=-0.003,
            unmovable_objects=unmovable_objects,
            action_description={
                "zh": f"拿起数字{digit}的积木",
                "en": f"picks up the block with digit {digit}",
            },
        )

        # Generate candidate orientations for this block.
        candidate_rotations = generate_digit_rotations_for_block(initial_rotation)
        print(
            f"Generated {len(candidate_rotations)} candidate rotations for block {block_id}"
        )

        # Generate many candidate poses.
        target_poses = []

        # Small position offsets improve the chance of success.
        x_offsets = np.linspace(-0.002, 0.002, 3)  # Small x offset.
        y_offsets = np.linspace(-0.002, 0.002, 3)  # Small y offset.
        z_offsets = np.linspace(0, 0.004, 3)  # Small z offset.

        # Generate all candidate-pose combinations.
        for x_offset in x_offsets:
            for y_offset in y_offsets:
                for z_offset in z_offsets:
                    position = base_position + np.array([x_offset, y_offset, z_offset])
                    for rot_mat in candidate_rotations:
                        pose = np.eye(4)
                        pose[:3, :3] = rot_mat
                        pose[:3, 3] = position
                        target_poses.append(pose)

        target_poses = np.array(target_poses)
        print(f"Generated {len(target_poses)} candidate poses for block {block_id}")

        # Define a checker for successful block placement.
        def create_check_function(block_id, expected_position, digit, table_top_z):
            def check_block_placed():
                current_pos, current_rot = robot.get_object_pose(block_id)
                if current_pos is None or current_rot is None:
                    return False

                # XY position check with 3cm tolerance.
                xy_distance = np.linalg.norm(current_pos[:2] - expected_position[:2])
                tolerance = 0.03  # 3cm
                if xy_distance > tolerance:
                    print(
                        f"Block {block_id} (digit {digit}) position error: {xy_distance:.3f}m > {tolerance:.3f}m"
                    )
                    return False

                # Check that it is on the table.
                if current_pos[2] < table_top_z:
                    print(f"Block {block_id} (digit {digit}) below table surface, current_pos: {current_pos}, table_top_z: {table_top_z}")
                    return False

                # Check local x approximately faces negative world x.
                current_rot_mat = quat2mat_wxyz(current_rot)
                local_x_axis = current_rot_mat[:, 0]
                target_direction = np.array([-1.0, 0.0, 0.0])

                # Calculate angular difference in the XY plane only.
                x_proj = local_x_axis[:2]
                x_proj = x_proj / (np.linalg.norm(x_proj) + 1e-10)
                target_proj = np.array([-1.0, 0.0])

                dot_product = np.dot(x_proj, target_proj)
                angle_error = np.arccos(np.clip(dot_product, -1.0, 1.0))

                if angle_error > np.deg2rad(15):  # Allow 15 degrees of error.
                    print(
                        f"Block {block_id} rotation error: {np.rad2deg(angle_error):.1f}°"
                    )
                    return False

                return True

            return check_block_placed

        # Create the checker.
        check_function = create_check_function(
            block_id, base_position, digit, table_top_z
        )

        # Configure placement action_description.
        if num_digits == 1:
            place_desc_zh = f"将数字{digit}的积木放置到桌子中央偏左位置"
            place_desc_en = (
                f"places the block with digit {digit} to the left center of the table"
            )
        elif i == 0:
            place_desc_zh = f"将数字{digit}的积木放置到结果排列的最左侧"
            place_desc_en = f"places the block with digit {digit} to the leftmost position of the result arrangement"
        elif i == num_digits - 1:
            place_desc_zh = f"将数字{digit}的积木放置到结果排列的最右侧"
            place_desc_en = f"places the block with digit {digit} to the rightmost position of the result arrangement"
        else:
            place_desc_zh = f"将数字{digit}的积木放置到结果排列的第{i+1}个位置"
            place_desc_en = f"places the block with digit {digit} to the {i+1}th position of the result arrangement"

        place_config = PlaceConfig(
            disable_upside_down=False,
            unmovable_objects=unmovable_objects,
            extra_checkers=[check_function],
            action_description={"zh": place_desc_zh, "en": place_desc_en},
        )

        # Execute grasp and placement.
        print(
            f"Attempting to pick and place block {block_id} with {len(target_poses)} candidate poses"
        )
        current_success = robot.pick_and_place_actor_to_target_pose(
            actor=block_id,
            target_pose=target_poses,
            arm=block_arm,
            grasp_config=grasp_config,
            place_config=place_config,
        )

        if not current_success:
            print(f"Failed to place block {block_id}")
            success = False
            return False

        print(f"Successfully placed block {block_id} (digit {digit})")

        # Add to placed list.
        placed_blocks.append(
            {"id": block_id, "digit": digit, "position": base_position}
        )

    # 14. Final verification.
    print("\n=== Performing final verification ===")

    # Check every target block was placed.
    placed_digits = [block["digit"] for block in placed_blocks]
    print(f"Placed digits: {placed_digits}")
    print(f"Target digits: {target_digits}")

    # Check arrangement linearity.
    if len(placed_blocks) >= 1:
        positions = []
        rotations = []
        for block_info in placed_blocks:
            current_pos, current_rot = robot.get_object_pose(block_info["id"])
            if current_pos is not None and current_rot is not None:
                positions.append(current_pos)
                rotations.append(current_rot)

        if len(positions) >= 2:
            # Check consistent x coordinates on one vertical line.
            x_values = [pos[0] for pos in positions]
            x_mean = np.mean(x_values)
            x_std = np.std(x_values)

            print(f"\nX-coordinates analysis:")
            print(f"  Mean x: {x_mean:.3f}m")
            print(f"  Std dev: {x_std:.3f}m")

            if x_std > 0.01:  # x standard deviation exceeds 1cm.
                print(f"  Warning: Blocks not aligned vertically")
            else:
                print(f"  ✓ Blocks are well aligned vertically")

            # Check y spacing.
            y_values = [pos[1] for pos in positions]
            y_values_sorted = sorted(y_values)

            print(f"\nY-coordinates of placed blocks (from left to right):")
            for i, y in enumerate(y_values_sorted):
                print(f"  Position {i+1}: y = {y:.3f}m")

            # Calculate spacing.
            if len(y_values_sorted) > 1:
                spacings = []
                for i in range(len(y_values_sorted) - 1):
                    spacing_actual = abs(y_values_sorted[i] - y_values_sorted[i + 1])
                    spacings.append(spacing_actual)

                if spacings:
                    avg_spacing = np.mean(spacings)
                    min_spacing = min(spacings)
                    max_spacing = max(spacings)

                    print(f"\nSpacing analysis:")
                    print(f"  Target spacing: {spacing:.3f}m")
                    print(f"  Average spacing: {avg_spacing:.3f}m")
                    print(f"  Min spacing: {min_spacing:.3f}m")
                    print(f"  Max spacing: {max_spacing:.3f}m")

                    if abs(avg_spacing - spacing) > 0.02:  # Allow 2cm error.
                        print(
                            f"  Warning: Spacing deviation from target: {abs(avg_spacing - spacing):.3f}m"
                        )
                    else:
                        print(f"  ✓ Spacing is close to target")

        # Check correct orientation.
        print(f"\nRotation verification:")
        for j, (pos, rot) in enumerate(zip(positions, rotations)):
            rot_mat = quat2mat_wxyz(rot)
            local_x_axis = rot_mat[:, 0]
            target_direction = np.array([-1.0, 0.0, 0.0])

            # Calculate angular difference.
            x_proj = local_x_axis[:2]
            if np.linalg.norm(x_proj) > 0.1:  # Avoid division by zero.
                x_proj = x_proj / np.linalg.norm(x_proj)
                dot_product = np.dot(x_proj, target_direction[:2])
                angle_error = np.arccos(np.clip(dot_product, -1.0, 1.0))
                angle_error_deg = np.rad2deg(angle_error)

                if angle_error_deg < 15:
                    print(
                        f"  Block {j+1}: ✓ Rotation correct (error: {angle_error_deg:.1f}°)"
                    )
                else:
                    print(f"  Block {j+1}: ⚠️ Rotation error: {angle_error_deg:.1f}°")

    # Return the arm to its initial position.
    if last_arm != "":
        robot.back_to_origin(arm=last_arm)

    # 15. Output final result.
    print("\n" + "=" * 50)
    print("TASK SUMMARY:")
    print(f"Generated equation: {equation_str}")
    print(f"Target result: {target_number}")

    arranged_number = 0
    for digit in placed_digits:
        arranged_number = arranged_number * 10 + digit

    print(f"Arranged result: {arranged_number} (digits: {placed_digits})")

    if arranged_number == target_number:
        print(f"✓ SUCCESS: Correctly arranged {target_number}")
    else:
        print(f"⚠️ Result mismatch: Expected {target_number}, got {arranged_number}")
        success = False

    # Final task state.
    if success:
        print("✅ TASK COMPLETED SUCCESSFULLY")
        return True
    else:
        print("⚠️ TASK COMPLETED WITH ISSUES")
        return False
