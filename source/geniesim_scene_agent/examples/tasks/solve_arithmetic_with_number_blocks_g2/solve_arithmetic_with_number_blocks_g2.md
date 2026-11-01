# Solve arithmetic with number blocks

**Tags:** `reasoning` `digit_blocks` `generate_target_number` `random_1_to_3_digits` `digit_counts` `dynamic_task_description` `place_actor_to_target_pose` `linear_arrangement` `spacing` `GraspConfig` `PlaceConfig` `pick_and_place_actor_to_target_pose` `extra_checkers`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find digit blocks (0–9). Count available digits. Generate target number at runtime: 1–3 digits, first digit != 0 if length > 1. Select blocks needed for each digit of target (e.g. from available, may need to pick same digit block if multiple).
2. **Order and positions:** Place digits in line in order of target number (left to right). target_positions[i] = (start_x - i*spacing, start_y, table_top_z + margin + size/2). spacing e.g. 0.05 m.
3. **Task description:** Set dynamically: e.g. "arrange the number blocks to form {target_number}".
4. **Execution:** For each digit in target number: pick corresponding block, place_at_target_pose. pick_and_place_actor_to_target_pose with extra_checkers (position/order).

---

# INPUT

**scene_description:** A table with multiple number blocks (digits 0–9, benchmark_building_blocks_034–043). At least 4 blocks; digit counts vary. The task is to arrange a subset of blocks to form a target number (1–3 digits) chosen at runtime based on available digits.

**task description:** Arrange the number blocks so they form the target number (e.g. "123" or "7"). The target number is generated at runtime (1–3 digits) from available digits; first digit cannot be 0 if length > 1. Place blocks in a line with correct order and spacing.

**code to generate the scene:**

```python
from helper import *
import random

"""
scene_name: table_with_number_blocks
description: A table with 4-7 randomly placed number blocks (digits 0-9), with at least 3 different digits,
             placed in region x(-0.1, 0.15), y(-0.3, 0.3), with minimum 5cm spacing and rotation between 160-200 degrees.
"""


def get_table_top_info(table_shape: Shape) -> Tuple[P, P]:
    """
    Get the table top surface center position and size in world coordinates.
    """
    table_info = get_object_info(table_shape)
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")

    # Table top center in world coordinates
    table_top_center = table_info["center"] + desktop_subpart_info["center"]

    # Table top surface z-coordinate (top of desktop)
    table_top_z = table_info["center"][2] + desktop_subpart_info["xyz_max"][2]

    table_top_pos = np.array([table_top_center[0], table_top_center[1], table_top_z])
    table_top_size = desktop_subpart_info["size"]

    return table_top_pos, table_top_size


def generate_random_digits(num_blocks: int) -> List[int]:
    """
    Generate random digits ensuring at least 3 different digits.
    """
    if num_blocks < 3:
        # This shouldn't happen given our constraints (4-7 blocks)
        digits = random.choices(range(10), k=num_blocks)
    else:
        # Ensure at least 3 different digits
        unique_digits = random.sample(range(10), 3)
        remaining_count = num_blocks - 3
        remaining_digits = random.choices(range(10), k=remaining_count)
        digits = unique_digits + remaining_digits
        random.shuffle(digits)

    return digits


def get_block_object_id(digit: int) -> str:
    """
    Map digit to corresponding object ID.
    """
    base_id = 34  # benchmark_building_blocks_034 is digit 0
    return f"benchmark_building_blocks_{base_id + digit:03d}"


def blocks_overlap(pos1: P, pos2: P, min_distance: float = 0.05) -> bool:
    """
    Check if two blocks overlap (distance less than minimum required).
    """
    distance = np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2]))
    return distance < min_distance


@register()
def number_block(digit: int, block_index: int) -> Shape:
    """
    Create a single number block with specified digit.
    """
    object_id = get_block_object_id(digit)
    block_shape = library_call(
        "usd",
        oid=object_id,
        keywords=[f"number_block_{digit}", "building_block", "colored", f"digit_{digit}", f"block_{block_index}"],
    )

    # Get block size for proper placement
    block_info = get_object_info(block_shape)
    block_size = block_info["size"]

    return block_shape


@register()
def place_number_blocks_on_table(table_shape: Shape) -> Shape:
    """
    Place 4-7 number blocks on the table with specified constraints.
    """
    # Get table top information
    table_top_pos, table_top_size = get_table_top_info(table_shape)

    # Determine number of blocks (4-7)
    num_blocks = random.randint(4, 7)

    # Generate digits ensuring at least 3 different ones
    digits = generate_random_digits(num_blocks)

    # Placement region relative to table center
    x_min, x_max = -0.1, 0.15
    y_min, y_max = -0.3, 0.3

    # Store placed block positions to avoid collisions
    placed_positions = []
    all_blocks = []

    for i in range(num_blocks):
        digit = digits[i]
        block_shape = library_call("number_block", digit=digit, block_index=i)
        block_info = get_object_info(block_shape)
        block_size = block_info["size"]

        # Try to find a valid position (avoid collisions)
        max_attempts = 100
        position_found = False

        for attempt in range(max_attempts):
            # Generate random position in the specified region
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            z = table_top_pos[2]  # Place on table top

            candidate_pos = [x, y, z]

            # Check for collisions with already placed blocks
            collision = False
            for placed_pos in placed_positions:
                if blocks_overlap(candidate_pos, placed_pos, min_distance=0.05):
                    collision = True
                    break

            if not collision:
                position_found = True
                placed_positions.append(candidate_pos)
                break

        # If no valid position found after max attempts, use the last candidate
        if not position_found:
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            candidate_pos = [x, y, table_top_pos[2]]
            placed_positions.append(candidate_pos)

        # Apply translation to place block on table
        block_translated = transform_shape(
            block_shape,
            translation_matrix(candidate_pos)
        )

        # Apply random rotation (160-200 degrees around z-axis)
        rotation_angle = random.uniform(160, 200) * (math.pi / 180.0)  # Convert to radians
        block_center = compute_shape_center(block_translated)
        block_rotated = transform_shape(
            block_translated,
            rotation_matrix(rotation_angle, direction=(0, 0, 1), point=block_center)
        )

        all_blocks.append(block_rotated)

    # Combine table and all blocks
    return concat_shapes(table_shape, *all_blocks)


@register()
def table_with_number_blocks() -> Shape:
    """
    Create the complete scene with table and number blocks.
    """
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "white", "desktop_with_legs"],
    )
    return place_number_blocks_on_table(table_shape)


@register()
def root_scene() -> Shape:
    return table_with_number_blocks()
```


---

# OUTPUT

```python
from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import random


def execute_task(robot: RobotAPI):
    # 初始化objects_list - 数字积木资产
    objects_list = {}
    for i in range(10):  # 数字0-9
        oid = f"benchmark_building_blocks_{34+i:03d}"  # 034-043
        objects_list[oid] = {
            "has_interactions": True,
            "object_type": "number_block",
            "digit": i,
        }
    objects_list["table_000"] = {"has_interactions": False, "object_type": "table"}

    # 运行时数据
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    # 1. 查找桌子
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

    # 2. 查找所有数字积木
    number_blocks = []
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        # 检查是否是数字积木 (034-043)
        if "benchmark_building_blocks_0" in usd and usd[27:].isdigit():
            digit_str = usd[27:]
            if digit_str.isdigit():
                digit = int(digit_str) - 34  # 计算对应数字 (034->0, 035->1, ...)
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
                                "initial_rotation": rot,  # 保存初始旋转
                                "size": size,
                                "keywords": keywords,
                            }
                        )
                        print(f"Found number block: {obj_id}, digit: {digit}")

    if len(number_blocks) < 4:
        print(f"Found only {len(number_blocks)} number blocks, need at least 4")
        return False

    print(f"Total number blocks found: {len(number_blocks)}")

    # 3. 统计每个数字的可用数量
    digit_counts = {}
    for block in number_blocks:
        digit = block["digit"]
        digit_counts[digit] = digit_counts.get(digit, 0) + 1

    print("Available digit counts:")
    for digit in sorted(digit_counts.keys()):
        print(f"  Digit {digit}: {digit_counts[digit]} blocks")

    # 4. 根据可用数字随机生成1-3位数的目标结果
    def generate_target_number(digit_counts):
        """生成1-3位数的目标数字"""
        # 获取所有可用的数字
        available_digits = [d for d, count in digit_counts.items() if count > 0]
        if not available_digits:
            return None

        # 随机决定位数 (1-3位)
        num_digits = random.randint(1, 3)

        # 确保有足够的不同数字
        if num_digits > len(available_digits):
            num_digits = len(available_digits)

        # 从可用数字中随机选择构成目标数字
        selected_digits = random.sample(available_digits, num_digits)

        # 首位不能是0
        if selected_digits[0] == 0 and num_digits > 1:
            # 找到非0数字作为首位
            non_zero_digits = [d for d in selected_digits if d != 0]
            if non_zero_digits:
                first_digit = non_zero_digits[0]
                # 重新排序：非0数字在前
                selected_digits.remove(first_digit)
                selected_digits.insert(0, first_digit)
            else:
                # 只有0，则只能是一位数0
                selected_digits = [0]
                num_digits = 1

        # 组成数字
        target_num = 0
        for digit in selected_digits:
            target_num = target_num * 10 + digit

        # 确保数字在合理范围内
        if 0 <= target_num <= 999:
            return target_num, selected_digits

        return None, None

    # 生成目标数字
    target_number, target_digits_list = generate_target_number(digit_counts)

    if target_number is None:
        most_available_digit = max(digit_counts.items(), key=lambda x: x[1])[0]
        target_number = most_available_digit
        target_digits_list = [most_available_digit]

    print(f"Target number to represent: {target_number}")
    print(f"Target digits: {target_digits_list}")

    # 5. 生成一个结果为该数字的算式
    def generate_equation(target_num):
        """生成结果为target_num的算式"""
        # 可能的操作符
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
                # 寻找因数
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

        # 如果以上都失败，使用简单的加法
        a = max(1, target_num // 2)
        b = target_num - a
        return f"{a} + {b}", [a, b], "+"

    equation_str, equation_numbers, operator = generate_equation(target_number)

    # MANDATORY: Set task description before first action
    description_json = {
        "task_name": f"算出并摆出{equation_str}的结果",
        "english_task_name": f"calculate and arrange the result of {equation_str.replace('×', '×').replace('÷', '÷')}",
        "init_scene_text": "The robot is in front of a table with several number blocks (0-9) scattered on it",
    }
    robot.client.set_task_task_basic_info(description_json)

    print(f"Generated equation: {equation_str} = {target_number}")
    print(f"Equation numbers: {equation_numbers}")

    # 6. 将目标数字转换为数字列表
    target_digits = []
    if target_number == 0:
        target_digits = [0]
    else:
        temp = target_number
        while temp > 0:
            target_digits.insert(0, temp % 10)
            temp //= 10

    print(f"Target digits to arrange: {target_digits}")

    # 7. 检查是否有足够的数字积木来组成结果
    required_digits = {}
    for digit in target_digits:
        required_digits[digit] = required_digits.get(digit, 0) + 1

    # 检查可用性
    missing_digits = []
    for digit, required_count in required_digits.items():
        available_count = digit_counts.get(digit, 0)
        if available_count < required_count:
            missing_digits.append((digit, required_count, available_count))

    if missing_digits:
        print("Warning: Not enough blocks for some digits:")
        for digit, required, available in missing_digits:
            print(f"  Digit {digit}: need {required}, have {available}")

        # 调整目标数字为可用数字
        print("Adjusting target number to available digits...")
        # 构建可用数字列表，按数量排序
        available_digits_sorted = sorted(
            [(d, c) for d, c in digit_counts.items() if c > 0],
            key=lambda x: x[1],
            reverse=True,
        )

        # 选择最多可用的数字组合
        if available_digits_sorted:
            # 使用最多可用的数字
            target_digits = [available_digits_sorted[0][0]]
            target_number = target_digits[0]
            required_digits = {target_digits[0]: 1}
            print(f"Adjusted target: {target_number}, digits: {target_digits}")

    # 8. 为每个目标数字选择具体的积木块
    selected_blocks = []
    remaining_blocks = number_blocks.copy()

    for digit in target_digits:
        # 找到第一个可用的该数字积木
        found = False
        for i, block in enumerate(remaining_blocks):
            if block["digit"] == digit:
                selected_blocks.append(block)
                remaining_blocks.pop(i)
                found = True
                break

        if not found:
            print(f"Cannot find block for digit {digit}")
            # 尝试用其他数字替代（选择最接近的可用数字）
            for i, block in enumerate(remaining_blocks):
                # 选择任意可用数字
                selected_blocks.append(block)
                remaining_blocks.pop(i)
                print(f"Using digit {block['digit']} instead of {digit}")
                found = True
                break

        if not found:
            print(f"Failed to find any block for digit {digit}")
            # 从已选择的块中移除一个给这个数字用
            for i, selected in enumerate(selected_blocks):
                if (
                    selected["digit"] in digit_counts
                    and digit_counts[selected["digit"]] > 1
                ):
                    # 找一个替代的积木
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

    # 9. 获取桌子信息用于计算摆放位置
    table_pos, table_rot = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)

    if table_pos is None or table_size is None:
        print("Cannot get table position/size")
        return False

    table_top_z = table_pos[2] + table_size[2] / 2.0

    # 10. 计算摆放位置参数
    # 中间位置：x(-0.13 ~ -0.15), y(-0.03, 0.03)
    target_x = random.uniform(-0.15, -0.13)

    # 数字数量
    num_digits = len(selected_blocks)

    # 计算每个积木的y方向宽度
    block_widths = []
    for block in selected_blocks:
        block_size = block["size"]
        # 使用y方向尺寸作为宽度
        width = block_size[1]
        block_widths.append(width)

    total_width = sum(block_widths)

    # 间隔：每个数字间隔0.05米
    spacing = 0.05
    total_spacing = spacing * (num_digits - 1)
    total_needed_width = total_width + total_spacing

    print(f"Number of digits: {num_digits}")
    print(f"Total needed width: {total_needed_width:.3f}m")

    # 起始位置（中间位置y坐标）
    center_y = random.uniform(-0.03, 0.03)

    # 计算起始y位置（从中心向两侧排列）
    if num_digits == 1:
        start_y = center_y
    else:
        # 计算从中心到左侧边缘的总距离
        half_width = total_needed_width / 2.0
        start_y = center_y + half_width - (block_widths[0] / 2.0)

    # 11. 为每个数字计算目标位置
    target_positions = []
    current_y = start_y

    for i, (block, width) in enumerate(zip(selected_blocks, block_widths)):
        block_id = block["id"]
        block_size = block["size"]
        digit = block["digit"]
        initial_rotation = block["initial_rotation"]  # 获取初始旋转

        # 计算y位置（从左侧开始）
        current_y = current_y - width / 2.0
        target_y = current_y

        # 计算z位置：桌子顶部 + 安全边距 + 半个积木高度
        margin = 0.01  # 安全边距1cm，避免夹爪碰撞
        target_z = table_top_z + margin + block_size[2] / 2.0

        # 目标位置
        target_position = np.array([target_x, target_y, target_z])
        target_positions.append(
            {
                "id": block_id,
                "digit": digit,
                "position": target_position,
                "size": block_size,
                "width": width,
                "initial_rotation": initial_rotation,  # 保存初始旋转
                "index": i,
            }
        )

        print(
            f"Block {i+1} (digit {digit}) target position: [{target_x:.3f}, {target_y:.3f}, {target_z:.3f}], width: {width:.3f}m"
        )

        # 更新当前y位置，为下一个积木准备
        current_y = current_y - width / 2.0 - spacing

    # 12. 根据数字积木放置要求生成候选旋转姿态
    # 要求：局部y轴与世界坐标z轴平行（已经满足，积木在桌子上y轴朝上）
    # 要求：局部x轴与世界坐标x轴负方向平行
    # 方法：获取初始旋转，然后绕世界坐标z轴旋转使局部x轴与世界坐标x轴负方向平行
    def calculate_target_rotation(initial_rotation):
        """
        计算目标旋转矩阵
        1. 积木已经在桌子上，y轴朝上（局部y轴平行于世界z轴）
        2. 需要绕世界z轴旋转，使局部x轴平行于世界x轴负方向
        """
        # 将初始四元数转换为旋转矩阵
        initial_rot_mat = quat2mat_wxyz(initial_rotation)  # 初始旋转矩阵

        # 获取局部x轴在世界坐标系中的方向
        # 局部x轴在初始姿态下的方向是旋转矩阵的第一列
        local_x_axis = initial_rot_mat[:, 0]  # [r11, r21, r31]

        # 目标方向：世界x轴负方向 [-1, 0, 0]
        target_direction = np.array([-1.0, 0.0, 0.0])

        # 计算当前局部x轴在xy平面上的投影
        local_x_proj = local_x_axis[:2]  # [x, y]分量
        local_x_proj = local_x_proj / (np.linalg.norm(local_x_proj) + 1e-10)

        # 计算目标方向在xy平面上的投影（已经是[-1, 0]）
        target_proj = np.array([-1.0, 0.0])
        target_proj = target_proj / np.linalg.norm(target_proj)

        # 计算需要绕世界z轴旋转的角度
        # 使用点积和叉积计算角度
        cos_angle = np.dot(local_x_proj, target_proj)
        sin_angle = np.cross(local_x_proj, target_proj)  # 2D叉积

        # 确保角度在有效范围内
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        # 确定旋转方向
        if sin_angle < 0:
            angle = -angle

        # 创建绕世界z轴的旋转矩阵
        rot_z = euler2mat([0, 0, angle])

        # 应用旋转到初始旋转矩阵
        target_rot_mat = np.dot(rot_z, initial_rot_mat)

        return target_rot_mat

    def generate_digit_rotations_for_block(initial_rotation):
        """为特定积木生成候选旋转矩阵"""
        candidate_rotations = []

        # 计算基准旋转
        base_rotation = calculate_target_rotation(initial_rotation)
        candidate_rotations.append(base_rotation)

        # 生成一些小的旋转变化（绕z轴±5度）
        for angle_deg in [-5, -2.5, 0, 2.5, 5]:
            angle_rad = np.deg2rad(angle_deg)
            rot_z_small = euler2mat([0, 0, angle_rad])
            rotated = np.dot(rot_z_small, base_rotation)
            candidate_rotations.append(rotated)

        # 去重
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

    # 13. 执行摆放任务
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

        # 获取当前积木位置
        block_position, block_rotation = robot.get_object_pose(block_id)
        if block_position is None:
            print(f"Warning: Cannot get position for block {block_id}")
            continue

        # 判断使用哪只手臂
        block_arm = robot.judge_arm_to_use(block_position)
        print(f"Using arm: {block_arm}")

        # 切换手臂时需要返回原点
        if last_arm != block_arm and last_arm != "":
            robot.back_to_origin(arm=last_arm)
        last_arm = block_arm

        # 创建抓取配置
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

        # 为这个积木生成候选旋转姿态
        candidate_rotations = generate_digit_rotations_for_block(initial_rotation)
        print(
            f"Generated {len(candidate_rotations)} candidate rotations for block {block_id}"
        )

        # 生成大量候选姿态
        target_poses = []

        # 位置变化参数（小范围偏移，增加成功机会）
        x_offsets = np.linspace(-0.002, 0.002, 3)  # x方向小偏移
        y_offsets = np.linspace(-0.002, 0.002, 3)  # y方向小偏移
        z_offsets = np.linspace(0, 0.004, 3)  # z方向小偏移

        # 生成所有候选姿态的组合
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

        # 定义检查函数：验证积木是否成功放置
        def create_check_function(block_id, expected_position, digit, table_top_z):
            def check_block_placed():
                current_pos, current_rot = robot.get_object_pose(block_id)
                if current_pos is None or current_rot is None:
                    return False

                # 位置检查（xy平面容差3cm）
                xy_distance = np.linalg.norm(current_pos[:2] - expected_position[:2])
                tolerance = 0.03  # 3cm
                if xy_distance > tolerance:
                    print(
                        f"Block {block_id} (digit {digit}) position error: {xy_distance:.3f}m > {tolerance:.3f}m"
                    )
                    return False

                # 检查是否在桌子上
                if current_pos[2] < table_top_z:
                    print(f"Block {block_id} (digit {digit}) below table surface")
                    return False

                # 检查旋转是否正确（局部x轴是否大致指向世界x轴负方向）
                current_rot_mat = quat2mat_wxyz(current_rot)
                local_x_axis = current_rot_mat[:, 0]
                target_direction = np.array([-1.0, 0.0, 0.0])

                # 计算x轴方向与目标方向的角度差（只考虑xy平面）
                x_proj = local_x_axis[:2]
                x_proj = x_proj / (np.linalg.norm(x_proj) + 1e-10)
                target_proj = np.array([-1.0, 0.0])

                dot_product = np.dot(x_proj, target_proj)
                angle_error = np.arccos(np.clip(dot_product, -1.0, 1.0))

                if angle_error > np.deg2rad(15):  # 允许15度误差
                    print(
                        f"Block {block_id} rotation error: {np.rad2deg(angle_error):.1f}°"
                    )
                    return False

                return True

            return check_block_placed

        # 创建检查函数
        check_function = create_check_function(
            block_id, base_position, digit, table_top_z
        )

        # 为放置配置action_description
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

        # 执行抓取和放置
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

        # 添加到已放置列表
        placed_blocks.append(
            {"id": block_id, "digit": digit, "position": base_position}
        )

    # 14. 最终验证
    print("\n=== Performing final verification ===")

    # 检查所有目标积木是否已放置
    placed_digits = [block["digit"] for block in placed_blocks]
    print(f"Placed digits: {placed_digits}")
    print(f"Target digits: {target_digits}")

    # 检查排列的线性
    if len(placed_blocks) >= 1:
        positions = []
        rotations = []
        for block_info in placed_blocks:
            current_pos, current_rot = robot.get_object_pose(block_info["id"])
            if current_pos is not None and current_rot is not None:
                positions.append(current_pos)
                rotations.append(current_rot)

        if len(positions) >= 2:
            # 检查x坐标是否一致（在同一条竖线上）
            x_values = [pos[0] for pos in positions]
            x_mean = np.mean(x_values)
            x_std = np.std(x_values)

            print(f"\nX-coordinates analysis:")
            print(f"  Mean x: {x_mean:.3f}m")
            print(f"  Std dev: {x_std:.3f}m")

            if x_std > 0.01:  # x方向标准偏差大于1cm
                print(f"  Warning: Blocks not aligned vertically")
            else:
                print(f"  ✓ Blocks are well aligned vertically")

            # 检查y坐标间距
            y_values = [pos[1] for pos in positions]
            y_values_sorted = sorted(y_values)

            print(f"\nY-coordinates of placed blocks (from left to right):")
            for i, y in enumerate(y_values_sorted):
                print(f"  Position {i+1}: y = {y:.3f}m")

            # 计算间距
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

                    if abs(avg_spacing - spacing) > 0.02:  # 允许2cm误差
                        print(
                            f"  Warning: Spacing deviation from target: {abs(avg_spacing - spacing):.3f}m"
                        )
                    else:
                        print(f"  ✓ Spacing is close to target")

        # 检查旋转是否正确
        print(f"\nRotation verification:")
        for j, (pos, rot) in enumerate(zip(positions, rotations)):
            rot_mat = quat2mat_wxyz(rot)
            local_x_axis = rot_mat[:, 0]
            target_direction = np.array([-1.0, 0.0, 0.0])

            # 计算角度差
            x_proj = local_x_axis[:2]
            if np.linalg.norm(x_proj) > 0.1:  # 避免除零
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

    # 返回手臂到初始位置
    if last_arm != "":
        robot.back_to_origin(arm=last_arm)

    # 15. 输出最终结果
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

    # 最终任务状态
    if success:
        print("✅ TASK COMPLETED SUCCESSFULLY")
        return True
    else:
        print("⚠️ TASK COMPLETED WITH ISSUES")
        return False
```
