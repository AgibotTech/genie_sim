# Store chess pieces in a box

**Tags:** `place_into_container` `chess` `chessboard` `keywords_position_tag` `regex_parse` `position_x_y_in_keywords` `storage_box` `place_actor_to_target_actor` `pick_and_place` `graph_relations` `pieces_inside_box` `GraspConfig` `PlaceConfig` `check_box_containment` `unmovable_objects` `pre_insert_offset` `extra_checkers`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find chessboard by usd "benchmark_chessboard_001". Find target piece by keyword containing "position_x_" and "_y_" (world coordinates of the reserved spot inside the board). Parse target_x, target_y with regex r"position_x_([-+]?\d*\.?\d+)_y_([-+]?\d*\.?\d+)".
2. **Target position:** target_position = (target_x, target_y, target_z). target_z = board_top_z + margin + piece height/2; board_top_z = board_pos[2] + 0.5*board_size[2]. Margin 0.01 m.
3. **Pieces inside box:** Optionally find pieces already inside the board via check_box_containment (board_pos, board_rot, board_local_size, piece_pos, piece_rot, piece_local_size, ignore_axes=["z"], overlap_threshold=0.6). Add them to unmovable_objects with board_id.
4. **Candidate poses:** 72 rotations around world Z (linspace 0 to 2*pi); final_rot = rot_z @ initial_rot_mat; same target_position for all.
5. **Verification:** extra_checkers: current piece position within 3 cm of target_position[:2].
6. **Execution:** Set task description; pick_and_place_actor_to_target_pose(target_piece_id, target_poses, ...) with pre_insert_offset=0.1, unmovable_objects=[board_id] + pieces_inside_box.

---

# INPUT

**scene_description:** A table with a chessboard (benchmark_chessboard_001) and chess pieces. One target piece is placed outside the board; its keywords contain a position tag `position_x_{value}_y_{value}` (world coordinates of the reserved internal spot). Other pieces may be inside the board. Put the target piece into the board (storage box).

**task description:** Place the chess piece on the table (the one marked by the position tag in keywords) into the storage box (chessboard).

**code to generate the scene:**

```python
from helper import *
import random
import numpy as np

"""
scene_name: chessboard_with_target_piece_v3
description: A chessboard (benchmark_chessboard_001) is placed randomly on a table (table_000).
             A target piece position is reserved inside the board's container area (Local Coords).
             The target piece is placed outside the board (Global Coords).
             The target piece is tagged with the GLOBAL world coordinates of the reserved internal spot.
             Constraints:
             - Board placed on table center: x(-0.1, 0.05), y(-0.1, 0.1).
             - Internal placement area (Local): x(-0.12, 0.12), y(-0.04, 0.04).
             - Target external placement area (Global): x(-0.2, 0.1), y(-0.25, 0.25).
             - Target must be >= 0.05m from board edges and other pieces.
"""

CHESS_PIECES = [
    "benchmark_chess_000", "benchmark_chess_001", "benchmark_chess_002",
    "benchmark_chess_003", "benchmark_chess_004", "benchmark_chess_005",
    "benchmark_chess_006", "benchmark_chess_007", "benchmark_chess_008",
    "benchmark_chess_009", "benchmark_chess_010", "benchmark_chess_011"
]

def get_random_piece_id():
    return random.choice(CHESS_PIECES)

@register()
def chessboard_scene() -> Shape:
    # 1. Setup Table
    table_shape = library_call("usd", oid="table_000", keywords=["table", "furniture", "workplace"])
    table_info = get_object_info(table_shape)
    table_top_z = table_info["max"][2]

    # 2. Setup Chessboard (Container)
    # Random position on table center area: x(-0.1, 0.05), y(-0.1, 0.1)
    board_pos_x = random.uniform(-0.1, 0.05)
    board_pos_y = random.uniform(-0.1, 0.1)
    board_pos = np.array([board_pos_x, board_pos_y, table_top_z])

    # Random rotation around Z axis
    board_rot_z = random.uniform(0, 2 * np.pi)

    # Load board and apply transforms
    base_board_shape = library_call("usd", oid="benchmark_chessboard_001", keywords=["chessboard", "container", "wooden"])

    # Rotation matrix (around Z, at origin, then translate)
    rot_mat = rotation_matrix(board_rot_z, direction=(0, 0, 1), point=(0, 0, 0))
    trans_mat = translation_matrix(board_pos)

    board_shape = transform_shape(base_board_shape, rot_mat)
    board_shape = transform_shape(board_shape, trans_mat)

    # Get board world info
    board_world_info = get_object_info(board_shape)
    board_surface_z = board_world_info["max"][2] # Top surface Z

    # Board AABB in world coordinates (XY plane)
    board_min_x = board_world_info["min"][0]
    board_max_x = board_world_info["max"][0]
    board_min_y = board_world_info["min"][1]
    board_max_y = board_world_info["max"][1]

    # Helper: Local to World conversion
    def local_to_world(lx, ly):
        wx = board_pos_x + lx * np.cos(board_rot_z) - ly * np.sin(board_rot_z)
        wy = board_pos_y + lx * np.sin(board_rot_z) + ly * np.cos(board_rot_z)
        return wx, wy

    # 3. Define Reserved Target Position (Inside the box - Local Coords)
    # Internal range: x(-0.120, 0.120), y(-0.040, 0.040)
    local_x_min, local_x_max = -0.120, 0.120
    local_y_min, local_y_max = -0.040, 0.040

    target_local_x = random.uniform(local_x_min, local_x_max)
    target_local_y = random.uniform(local_y_min, local_y_max)

    # Calculate World Coordinates for the Tag (Tag must be GLOBAL coordinates)
    target_world_x, target_world_y = local_to_world(target_local_x, target_local_y)

    # Tag string using GLOBAL world coordinates
    pos_tag = f"position_x_{target_world_x:.3f}_y_{target_world_y:.3f}"

    # 4. Place Internal Pieces (Other pieces, avoiding the reserved spot)
    # Total pieces 1-5. One is target (placed outside). Remaining are placed inside.
    num_total_pieces = random.randint(1, 5)
    num_internal_pieces = num_total_pieces - 1

    internal_pieces_shapes = []
    internal_piece_world_pos = [] # Store (x, y) for collision check

    piece_width = 0.04
    safe_distance = 0.05 # Minimum distance between pieces

    for _ in range(num_internal_pieces):
        pid = get_random_piece_id()
        piece_shape = library_call("usd", oid=pid, keywords=["chess_piece", "internal", "random"])

        placed = False
        while not placed:
            lx = random.uniform(local_x_min, local_x_max)
            ly = random.uniform(local_y_min, local_y_max)

            wx, wy = local_to_world(lx, ly)

            # Check collision with reserved target spot (must leave this empty)
            dist_to_target = np.sqrt((wx - target_world_x)**2 + (wy - target_world_y)**2)
            if dist_to_target < safe_distance:
                continue

            # Check collision with other internal pieces
            collision = False
            for (ox, oy) in internal_piece_world_pos:
                dist = np.sqrt((wx - ox)**2 + (wy - oy)**2)
                if dist < safe_distance:
                    collision = True
                    break

            if not collision:
                internal_piece_world_pos.append((wx, wy))
                placed = True

        # Transform piece to world
        piece_rot_z = random.uniform(0, 2 * np.pi)
        p_rot_mat = rotation_matrix(piece_rot_z, direction=(0, 0, 1), point=(0, 0, 0))
        p_trans_mat = translation_matrix((wx, wy, board_surface_z))
        final_piece = transform_shape(piece_shape, p_rot_mat)
        final_piece = transform_shape(final_piece, p_trans_mat)
        internal_pieces_shapes.append(final_piece)

    # 5. Place Target Piece (Outside the board - Global Coords)
    # Constraints:
    # a. Global Boundary: x (-0.2, 0.1), y (-0.25, 0.25)
    # b. Must be outside the board (distance to board edges >= 0.05)
    # c. Distance to other pieces (internal) >= 0.05

    target_found = False
    target_wx_final, target_wy_final = 0.0, 0.0

    # Global External Bounds
    global_ext_x_min, global_ext_x_max = -0.2, 0.1
    global_ext_y_min, global_ext_y_max = -0.25, 0.25

    max_attempts = 1000

    while not target_found and max_attempts > 0:
        max_attempts -= 1

        # Sample in Global Coordinates
        wx = random.uniform(global_ext_x_min, global_ext_x_max)
        wy = random.uniform(global_ext_y_min, global_ext_y_max)

        # Check Distance to Board Edges
        # Calculate distance from point (wx, wy) to board AABB
        dx = max(board_min_x - wx, 0, wx - board_max_x)
        dy = max(board_min_y - wy, 0, wy - board_max_y)
        dist_edge = np.sqrt(dx*dx + dy*dy)

        if dist_edge < 0.05:
            continue

        # Check Distance to Internal Pieces
        valid_dist = True
        for (px, py) in internal_piece_world_pos:
            dist = np.sqrt((wx - px)**2 + (wy - py)**2)
            if dist < 0.05:
                valid_dist = False
                break

        if valid_dist:
            target_found = True
            target_wx_final, target_wy_final = wx, wy

    if not target_found:
        # Fallback: place it further out if random search failed
        # Place it at a safe distance along the X axis
        target_wx_final = board_max_x + 0.15
        target_wy_final = board_pos_y

    # Create Target Shape with the Tag (tag points to the internal GLOBAL spot)
    target_pid = get_random_piece_id()
    target_shape = library_call("usd", oid=target_pid, keywords=["chess_piece", "target", "white", "highlighted", pos_tag])

    # Transform Target to its External Position
    target_rot_z = random.uniform(0, 2 * np.pi)
    t_rot_mat = rotation_matrix(target_rot_z, direction=(0, 0, 1), point=(0, 0, 0))
    t_trans_mat = translation_matrix((target_wx_final, target_wy_final, board_surface_z))

    final_target = transform_shape(target_shape, t_rot_mat)
    final_target = transform_shape(final_target, t_trans_mat)

    # 6. Combine Shapes
    return concat_shapes(table_shape, board_shape, *internal_pieces_shapes, final_target)

@register()
def root_scene() -> Shape:
    return chessboard_scene()
```

---

# OUTPUT

```python
from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os
import re

def execute_task(robot: RobotAPI):
    ## Initialize objects_list based on interaction data
    objects_list = {
        "benchmark_chessboard_001": {"has_interactions": True, "object_type": "chessboard"},
        "benchmark_chess_000": {"has_interactions": True, "object_type": "chess_piece"},
        "benchmark_chess_001": {"has_interactions": True, "object_type": "chess_piece"},
        "benchmark_chess_002": {"has_interactions": True, "object_type": "chess_piece"},
        "benchmark_chess_003": {"has_interactions": True, "object_type": "chess_piece"},
        "benchmark_chess_004": {"has_interactions": True, "object_type": "chess_piece"},
        "benchmark_chess_005": {"has_interactions": True, "object_type": "chess_piece"},
        "benchmark_chess_006": {"has_interactions": True, "object_type": "chess_piece"},
        "benchmark_chess_007": {"has_interactions": True, "object_type": "chess_piece"},
        "benchmark_chess_008": {"has_interactions": True, "object_type": "chess_piece"},
        "benchmark_chess_009": {"has_interactions": True, "object_type": "chess_piece"},
        "benchmark_chess_010": {"has_interactions": True, "object_type": "chess_piece"},
        "benchmark_chess_011": {"has_interactions": True, "object_type": "chess_piece"}
    }
    ## Runtime-only data
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    ## MANDATORY: Set task description before first action
    description_json = {
        "task_name": "将桌上的棋子放入收纳盒中",
        "english_task_name": "place the chess piece on the table into the storage box",
        "init_scene_text": "The robot is in front of the table, there is a chessboard with a target chess piece outside"
    }
    robot.client.set_task_task_basic_info(description_json)

    ## 1. Find chessboard (benchmark_chessboard_001)
    board_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        if "benchmark_chessboard_001" in usd:
            board_id = obj_id
            break

    if not board_id:
        print("Chessboard not found")
        return False

    print(f"Found chessboard: {board_id}")

    ## 2. Find target chess piece (by tag: position_x_{xxx}_y_{xxx})
    target_piece_id = None
    target_position_tag = None

    for obj_id, obj_info in scene_info["layout"].items():
        keywords = obj_info.get("keywords", [])
        for kw in keywords:
            if "position_x_" in kw and "_y_" in kw:
                target_piece_id = obj_id
                target_position_tag = kw
                print(f"Found target piece: {obj_id} with tag: {kw}")
                break
        if target_piece_id:
            break

    if not target_piece_id or not target_position_tag:
        print("Target chess piece not found or no position tag")
        return False

    ## 3. Parse target position from tag
    ## Tag format: position_x_{value}_y_{value}, where value is world coordinate
    try:
        # Extract x and y values using regex
        match = re.search(r"position_x_([-+]?\d*\.?\d+)_y_([-+]?\d*\.?\d+)", target_position_tag)
        if match:
            target_x = float(match.group(1))
            target_y = float(match.group(2))
        else:
            # Fallback: split by underscore
            parts = target_position_tag.split('_')
            target_x = float(parts[2])
            target_y = float(parts[4])

        print(f"Target position from tag: ({target_x:.4f}, {target_y:.4f})")
    except Exception as e:
        print(f"Failed to parse position tag: {e}")
        return False

    ## 4. Get board info to calculate target z height
    board_pos, board_rot = robot.get_object_pose(board_id)
    board_size = robot.get_object_size_in_world_frame(board_id)

    if board_pos is None or board_size is None:
        print("Cannot get board position/size")
        return False

    ## 5. Get piece size and initial pose
    piece_pos, piece_quat = robot.get_object_pose(target_piece_id)
    piece_size = robot.get_object_size_in_world_frame(target_piece_id)

    if piece_pos is None or piece_quat is None or piece_size is None:
        print("Cannot get piece pose/size")
        return False

    ## [NEW] Logic: Find all chess pieces already inside the storage box
    pieces_inside_box = []

    for obj_id, obj_info in scene_info["layout"].items():
        # Skip the target piece and the board itself
        if obj_id == target_piece_id or obj_id == board_id:
            continue

        usd = obj_info.get("usd", "")

        # Check if this object is a chess piece (from our list)
        is_chess_piece = False
        for piece_usd in objects_list.keys():
            if piece_usd == usd and piece_usd != "benchmark_chessboard_001":
                is_chess_piece = True
                break

        if is_chess_piece:
            curr_piece_pos, curr_piece_rot = robot.get_object_pose(obj_id)
            curr_piece_size = robot.get_object_size_in_local_frame(obj_id)
            board_local_size = robot.get_object_size_in_local_frame(board_id)

            if curr_piece_pos is not None and curr_piece_rot is not None:
                # Check if the piece is inside the board's area (XY plane projection)
                board_contains_piece, _ = check_box_containment(
                    board_pos, board_rot, board_local_size,
                    curr_piece_pos, curr_piece_rot, curr_piece_size,
                    ignore_axes=["z"],  # Only check XY plane
                    overlap_threshold=0.6  # 60% overlap required to be considered "inside"
                )

                if board_contains_piece:
                    pieces_inside_box.append(obj_id)
                    print(f"Found piece inside box: {obj_id}")

    # Combine board and internal pieces into unmovable_objects list
    unmovable_objects_list = [board_id] + pieces_inside_box
    print(f"Unmovable objects: {unmovable_objects_list}")

    ## 6. Calculate target z height
    ## Board top surface: board_pos[2] + 0.5 * board_size[2]
    ## CRITICAL: Always add safety margin (0.01m) to avoid gripper collision
    margin = 0.01  # Safety margin in meters (1cm)
    board_top_z = board_pos[2] + 0.5 * board_size[2]
    target_z = board_top_z + margin + 0.5 * piece_size[2]

    target_position = np.array([target_x, target_y, target_z])

    print(f"Target position: [{target_x:.4f}, {target_y:.4f}, {target_z:.4f}]")

    ## 7. Extract initial rotation and generate candidate poses
    ## Convert quaternion to rotation matrix (wxyz format)
    initial_rot_mat = quat2mat_wxyz(piece_quat)

    ## Generate 36 rotation angles around world z axis (0 to 360 degrees, step 10 degrees)
    target_poses = []
    z_angles = np.linspace(0, 2*np.pi, 72, endpoint=False)  # 72 samples as per user request

    for z_angle in z_angles:
        ## Rotation matrix for z axis rotation (world frame)
        rot_z = euler2mat([0, 0, z_angle])

        ## Apply rotation: new_rot = rot_z * initial_rot (rotate around world z axis)
        final_rot = np.dot(rot_z, initial_rot_mat)

        ## Build pose matrix
        pose = np.eye(4)
        pose[:3, :3] = final_rot
        pose[:3, 3] = target_position
        target_poses.append(pose)

    target_poses = np.array(target_poses)
    print(f"Generated {len(target_poses)} candidate poses (72 rotations around world z axis)")

    ## 8. Determine which arm to use
    arm = robot.judge_arm_to_use(piece_pos)
    print(f"Using arm: {arm}")

    ## 9. Execute pick and place
    grasp_config = GraspConfig(
        is_from_up_side=True,
        pick_up_distance=0.12,
        unmovable_objects=unmovable_objects_list,  # Use the updated list
        action_description={"zh": "拿起桌上的棋子", "en": "picks up the chess piece on the table"}
    )

    ## 10. Define checker to verify placement success
    def check_piece_placed_at_target():
        """Check if piece is placed at target position"""
        current_pos, _ = robot.get_object_pose(target_piece_id)
        if current_pos is None:
            return False

        ## Check if position matches target (with 2cm tolerance)
        pos_distance = np.linalg.norm(current_pos[:2] - target_position[:2])

        if pos_distance < 0.03:
            return True
        print(f"Piece not placed at target position: {pos_distance:.4f}m")
        return False

    place_config = PlaceConfig(
        pre_insert_offset=0.1,
        unmovable_objects=unmovable_objects_list,  # Use the updated list
        extra_checkers=[check_piece_placed_at_target],
        action_description={"zh": "将棋子放置到收纳盒中", "en": "places the chess piece on the table into the storage box"}
    )

    print("Attempting to pick and place target chess piece...")
    success = robot.pick_and_place_actor_to_target_pose(
        actor=target_piece_id,
        target_pose=target_poses,
        arm=arm,
        grasp_config=grasp_config,
        place_config=place_config
    )

    if success:
        print("Successfully placed target chess piece to target position")
    else:
        print("Failed to place target chess piece")

    robot.back_to_origin(arm=arm)

    return success
```
