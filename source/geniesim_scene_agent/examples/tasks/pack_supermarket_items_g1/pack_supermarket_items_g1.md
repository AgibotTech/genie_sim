# Pack supermarket items

**Tags:** `place_into_container` `handbag` `product` `fruit` `beverage` `pick_actor` `place_actor_to_target_pose` `check_box_containment` `judge_arm_to_use` `GraspConfig` `PlaceConfig` `usd_match` `multiple_candidate_poses` `quat2mat_wxyz` `rotate_along_axis` `extra_checkers` `pre_insert_offset` `interactions` `dynamic_product_description`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find table (usd table_000 or keywords "table"), bag (benchmark_handbag_004 or "handbag"), product (benchmark_apple_001, benchmark_green_apple_000, benchmark_orange_004, benchmark_fruit_021, benchmark_beverage_bottle_073). Get poses and sizes.
2. **Arm:** arm = judge_arm_to_use(product_pos).
3. **Product description:** From product usd set product_desc_zh/en (苹果/apple, 青苹果/green apple, 橙子/orange, 香蕉/fruit, 红牛/red bull) for GraspConfig and PlaceConfig action_description.
4. **Pick:** GraspConfig(pick_up_distance=0.1, unmovable_objects=[table_id, bag_id]). pick_actor(product_id, arm, grasp_config).
5. **Place into bag:** Generate target_poses inside bag (x/y/z offsets from bag_pos, bag_top + z_offset + product_size[2]/2; multiple rotations). PlaceConfig(pre_insert_offset=0.1, unmovable_objects, extra_checkers=create_product_in_bag_checker). place_actor_to_target_pose(product_id, arm, target_poses, place_config). The check_box_containment checker verifies placement inside the bag.
6. **Optional final verification:** After placement, check_box_containment again (ignore_axes=["z"] optional). back_to_origin(arm).

---

# INPUT

**scene_description:** A desk with a handbag in the left half area (x 0.02–0.15, y 0.1–0.15) and a product (fruit or beverage) in the right half area (x -0.2–0.15, y -0.1–-0.25), with minimum distance 0.25 between bag and product centers.

**task description:** Place the product (fruit or beverage) into the handbag. Pick the product with one arm, then place it inside the bag. Verify that the product is placed inside the bag.

**code to generate the scene:**

```python
from helper import *
import numpy as np
import random

"""
scene_name: desk_with_product_and_bag
description: A product and a handbag placed on a desk with specific positioning constraints
- Bag in left half area: x (-0.15, 0.1), y (0.07, 0.15)
- Product in right half area: x (-0.2, 0.15), y (-0.1, -0.2)
- Minimum distance between bag and product center >= 0.25
"""


def find_desktop_top_surface(desk_shape: Shape) -> Tuple[P, P]:
    desk_info = get_object_info(desk_shape)
    try:
        desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")
    except:
        desktop_subpart_info = {
            "center": [0, 0, 0],
            "size": desk_info["size"],
            "xyz_max": [desk_info["size"][0] / 2, desk_info["size"][1] / 2, desk_info["size"][2] / 2],
        }
    desktop_center = desk_info["center"] + desktop_subpart_info["center"]
    desktop_top_z = desk_info["center"][2] + desktop_subpart_info["xyz_max"][2] - 0.05
    desktop_pos = np.array([desktop_center[0], desktop_center[1], desktop_top_z])
    desktop_size = desktop_subpart_info["size"]
    return desktop_pos, desktop_size


@register()
def select_product() -> Shape:
    product_options = [
        ("benchmark_orange_004", "orange", "orange"),
        ("benchmark_apple_001", "red", "apple"),
        ("benchmark_green_apple_000", "green", "green_apple"),
        ("benchmark_fruit_021", "yellow", "banana"),
        ("benchmark_beverage_bottle_073", "red", "red_bull"),
    ]
    oid, color, category = random.choice(product_options)
    product_shape = library_call(
        "usd",
        oid=oid,
        keywords=["product", color, category, "on_desk"],
    )
    return product_shape


@register()
def place_bag_on_desk(desk_shape: Shape) -> Shape:
    workspace, margin = find_desktop_top_surface(desk_shape)
    bag_x = np.random.uniform(0.02, 0.15)
    bag_y = np.random.uniform(0.1, 0.15)
    bag_shape = library_call(
        "usd",
        oid="benchmark_handbag_004",
        keywords=["handbag", "bag", "left_area", "on_desk"],
    )
    random_angle = np.random.uniform(-np.pi, np.pi)
    bag_shape = transform_shape(
        bag_shape,
        translation_matrix((workspace[0] + bag_x, workspace[1] + bag_y, workspace[2])),
    )
    bag_shape = transform_shape(
        bag_shape,
        rotation_matrix(random_angle, direction=(0, 0, 1), point=compute_shape_center(bag_shape)),
    )
    return bag_shape


@register()
def place_product_on_desk(desk_shape: Shape, bag_shape: Shape) -> Shape:
    workspace, margin = find_desktop_top_surface(desk_shape)
    bag_info = get_object_info(bag_shape)
    bag_center = bag_info["center"]
    min_distance = 0.25
    max_attempts = 100
    product_shape = None
    for attempt in range(max_attempts):
        product_x = np.random.uniform(-0.2, 0.15)
        product_y = np.random.uniform(-0.1, -0.25)
        potential_product_center = np.array([workspace[0] + product_x, workspace[1] + product_y, workspace[2]])
        distance = np.linalg.norm(potential_product_center[:2] - bag_center[:2])
        if distance >= min_distance:
            product_shape = select_product()
            random_angle = np.random.uniform(-np.pi, np.pi)
            product_shape = transform_shape(
                product_shape,
                translation_matrix(
                    (potential_product_center[0], potential_product_center[1], potential_product_center[2])
                ),
            )
            product_shape = transform_shape(
                product_shape,
                rotation_matrix(random_angle, direction=(0, 0, 1), point=compute_shape_center(product_shape)),
            )
            break
    if product_shape is None:
        product_shape = select_product()
        fallback_x = -0.2
        fallback_y = -0.15
        product_shape = transform_shape(
            product_shape,
            translation_matrix((workspace[0] + fallback_x, workspace[1] + fallback_y, workspace[2])),
        )
        random_angle = np.random.uniform(-np.pi, np.pi)
        product_shape = transform_shape(
            product_shape,
            rotation_matrix(random_angle, direction=(0, 0, 1), point=compute_shape_center(product_shape)),
        )
    return product_shape


@register()
def desk_with_product_and_bag() -> Shape:
    desk_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "white", "desktop_with_legs"],
    )
    bag_shape = place_bag_on_desk(desk_shape)
    product_shape = place_product_on_desk(desk_shape, bag_shape)
    return concat_shapes(desk_shape, bag_shape, product_shape)


@register()
def root_scene() -> Shape:
    return desk_with_product_and_bag()
```

---

# OUTPUT

```python
from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    # 基于提供的交互数据初始化 objects_list
    objects_list = {
        "benchmark_apple_001": {"has_interactions": True, "object_type": "fruit", "product": True},
        "benchmark_green_apple_000": {"has_interactions": True, "object_type": "fruit", "product": True},
        "benchmark_orange_004": {"has_interactions": True, "object_type": "fruit", "product": True},
        "benchmark_fruit_021": {"has_interactions": True, "object_type": "fruit", "product": True},
        "benchmark_beverage_bottle_073": {"has_interactions": True, "object_type": "beverage", "product": True},
        "benchmark_handbag_004": {"has_interactions": False, "object_type": "handbag", "container": True}
    }

    # 运行时数据
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    # MANDATORY: Set task description before first action
    description_json = {
        "task_name": "把商品装进袋子里",
        "english_task_name": "place the product into the bag",
        "init_scene_text": "The robot is in front of the table, there is a handbag and a fruit product on the table"
    }
    robot.client.set_task_task_basic_info(description_json)

    # 1. 查找桌子
    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        if usd == "table_000" or any("table" in kw.lower() for kw in keywords):
            table_id = obj_id
            break

    if not table_id:
        print("Table not found")
        return False

    print(f"Found table: {table_id}")

    # 2. 查找袋子（手包）
    bag_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        if "benchmark_handbag_004" in usd or any("handbag" in kw.lower() for kw in keywords):
            bag_id = obj_id
            print(f"Found bag: {obj_id}")
            break

    if not bag_id:
        print("Bag not found")
        return False

    # 3. 查找商品（水果）
    product_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        # 检查是否是已知的水果产品
        if any(product_usd in usd for product_usd in ["benchmark_apple_001", "benchmark_green_apple_000", "benchmark_orange_004", "benchmark_fruit_021", "benchmark_beverage_bottle_073"]):
            product_id = obj_id
            print(f"Found product: {obj_id}")
            break

    if not product_id:
        print("Product not found")
        return False

    # 4. 判断使用哪只手臂
    product_pos, product_rot = robot.get_object_pose(product_id)
    arm = robot.judge_arm_to_use(product_pos)
    print(f"Using arm: {arm}")

    # 5. 检查交互数据
    bag_interaction = interactions.get("benchmark_handbag_004", {})

    # 更安全的交互数据获取方式
    for obj_id, obj_info in scene_info["layout"].items():
        if obj_id == product_id:
            product_usd = obj_info.get("usd", "")
            product_interaction = interactions.get(product_usd, {})
            break

    bag_has_place = "passive" in bag_interaction and "place" in bag_interaction.get("passive", {})
    product_has_grasp = "passive" in product_interaction and "grasp" in product_interaction.get("passive", {})

    print(f"Bag has passive.place: {bag_has_place}")
    print(f"Product has passive.grasp: {product_has_grasp}")

    # 6. 准备不可移动对象列表
    unmovable_objects = [table_id, bag_id]

    # 7. 配置抓取参数
    product_desc_zh = ""
    product_desc_en = ""

    # 确定商品类型描述
    for obj_id, obj_info in scene_info["layout"].items():
        if obj_id == product_id:
            usd = obj_info.get("usd", "")
            if "apple" in usd.lower():
                product_desc_zh = "苹果"
                product_desc_en = "apple"
            elif "green" in usd.lower():
                product_desc_zh = "青苹果"
                product_desc_en = "green apple"
            elif "orange" in usd.lower():
                product_desc_zh = "橙子"
                product_desc_en = "orange"
            elif "fruit_021" in usd.lower():
                product_desc_zh = "香蕉"
                product_desc_en = "fruit"
            elif "beverage" in usd.lower():
                product_desc_zh = "红牛"
                product_desc_en = "red bull"
            break

    grasp_config = GraspConfig(
        pick_up_distance=0.1,
        unmovable_objects=unmovable_objects,
        action_description={"zh": f"拿起{product_desc_zh}", "en": f"picks up the {product_desc_en}"}
    )

    # 8. 定义检查函数：验证商品是否成功放入袋子
    def create_product_in_bag_checker(product_id, bag_id):
        def checker():
            product_pos, product_rot = robot.get_object_pose(product_id)
            product_size = robot.get_object_size_in_local_frame(product_id)
            bag_pos, bag_rot = robot.get_object_pose(bag_id)
            bag_size = robot.get_object_size_in_local_frame(bag_id)

            if product_pos is None or bag_pos is None:
                return False

            # 检查包含关系，忽略 z 轴（只检查 xy 平面投影，允许不同高度）
            from common.base_utils.transform_utils import check_box_containment
            bag_contains_product, _ = check_box_containment(
                bag_pos, bag_rot, bag_size,
                product_pos, product_rot, product_size,
                overlap_threshold=0.7
            )

            return bag_contains_product

        return checker

    # 9. 执行放置操作
    place_desc_zh = f"将{product_desc_zh}装进袋子里"
    place_desc_en = f"places the {product_desc_en} into the bag"
    print(f"Using place_actor_to_target_pose for bag {bag_id}")

    bag_pos, bag_rot = robot.get_object_pose(bag_id)
    bag_size = robot.get_object_size_in_world_frame(bag_id)
    product_size = robot.get_object_size_in_world_frame(product_id)

    # 生成多个候选放置位置（在袋子内部）
    target_poses = []
    margin = 0.01  # 安全边距避免夹爪碰撞

    # 生成袋子内部的多个位置（考虑袋子开口大小）
    x_offsets = np.linspace(-0.05, 0.05,  10)
    y_offsets = np.linspace(-0.05, 0.0, 5)
    z_offsets = np.linspace(0.008, 0.012, 3)

    rot_matrix = quat2mat_wxyz(bag_rot)
    product_rot_matrix = quat2mat_wxyz(product_rot)
    for x_offset in x_offsets:
        for y_offset in y_offsets:
            for z_offset in z_offsets:
                target_position = bag_pos.copy()
                target_position[0] += x_offset
                target_position[1] += y_offset

                # 计算放置高度：袋子底部 + 安全边距 + 商品半高
                bag_top = bag_pos[2] + bag_size[2] / 2.0
                target_position[2] = bag_top  + z_offset + product_size[2] / 2.0

                # 为每个位置生成多个旋转姿态（随机旋转）
                for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
                    pose = np.eye(4)
                    pose[:3, :3] = product_rot_matrix
                    pose[:3, 3] = target_position
                    pose = rotate_along_axis(pose, angle, "z", use_local=False)
                    target_poses.append(pose)

    if len(target_poses) == 0:
        print(f"No valid target poses generated for product {product_id}")
        success = False
        return success

    target_poses = np.array(target_poses)
    print(f"Generated {len(target_poses)} candidate poses for product {product_id}")

    place_config = PlaceConfig(
        pre_insert_offset=0.1,
        unmovable_objects=unmovable_objects,
        extra_checkers=[create_product_in_bag_checker(product_id, bag_id)],
        action_description={"zh": place_desc_zh, "en": place_desc_en}
    )
    pick_success = robot.pick_actor(product_id, arm=arm, grasp_config=grasp_config)
    if not pick_success:
        print(f"Failed to pick product {product_id}")
        success = False
        return success
    else:
        print(f"Successfully picked product {product_id}")

    place_success = robot.place_actor_to_target_pose(product_id, arm=arm, target_pose=target_poses, place_config=place_config)
    if not place_success:
        print(f"Failed to place product {product_id} into bag {bag_id}")
        success = False
        return success
    else:
        print(f"Successfully placed product {product_id} into bag {bag_id}")

    # 返回到初始位置
    robot.back_to_origin(arm=arm)

    # 10. 最终验证
    print("\n=== Performing final verification ===")

    final_product_pos, _ = robot.get_object_pose(product_id)
    if final_product_pos is not None:
        in_bag = False
        bag_pos, bag_rot = robot.get_object_pose(bag_id)
        bag_size = robot.get_object_size_in_local_frame(bag_id)
        product_size = robot.get_object_size_in_local_frame(product_id)

        bag_contains_product, _ = check_box_containment(
            bag_pos, bag_rot, bag_size,
            final_product_pos, _, product_size,
            ignore_axes=["z"],
            overlap_threshold=0.7
        )

        if bag_contains_product:
            print(f"✅ Verification PASSED: Product is inside the bag")
        else:
            print(f"⚠️ Verification FAILED: Product may not be fully inside the bag")
            success = False
    else:
        print(f"⚠️ Cannot verify product position after placement")
        success = False

    if success:
        print("\n🎉 SUCCESS: Product successfully placed into the bag")
    else:
        print("\n❌ FAILED: Task execution encountered issues")

    return success
```
