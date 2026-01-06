# Task Configuration Guide

This document describes how to create task configuration files (JSON format) for defining robot simulation tasks. Configuration files are parsed by `client/layout/task_generate.py` to generate specific task instances.

## Table of Contents

1. [Basic Structure](#basic-structure)
2. [Scene Origin (origin)](#scene-origin-origin)
3. [Object Configuration (objects)](#object-configuration-objects)
4. [Scene Configuration (scene)](#scene-configuration-scene)
5. [Robot Configuration (robot)](#robot-configuration-robot)
6. [Task Stages (stages)](#task-stages-stages)
7. [Other Configurations](#other-configurations)
8. [Complete Examples](#complete-examples)
9. [Quick Start](#quick-start)

---

## Basic Structure

A task configuration file is a JSON file containing the following main sections:

```json
{
    "task": "任务名称",
    "origin": { ... },           // Scene origin (optional)
    "objects": { ... },           // Object configuration (required)
    "scene": { ... },             // Scene configuration (required)
    "robot": { ... },             // Robot configuration (optional)
    "stages": [ ... ],            // Task stages (required)
    "recording_setting": { ... }, // Recording settings (optional)
    "task_description": { ... },  // Task description (optional)
    "task_metric": { ... }        // Task evaluation metrics (optional)
}
```

---

## Scene Origin (origin)

The `origin` field defines the global origin position of the scene. **All object and robot positions will be transformed relative to this origin**.

### Structure

```json
{
    "origin": {
        "position": [2.91, 0.76, 0.0],
        "quaternion": [1, 0, 0, 0]
    }
}
```

### Field Description

- **position** (required): Origin position in world coordinates [x, y, z] (unit: meters)
- **quaternion** (required): Origin rotation (quaternion format w, x, y, z)

### Important Notes

- If `origin` is configured, all workspace, object, and robot positions will be transformed relative to `origin` first, then converted to world coordinates
- If `origin` is not configured, it defaults to `position: [0, 0, 0]` and `quaternion: [1, 0, 0, 0]`
- **Recommendation for choosing origin position**: Usually choose the midpoint of the robot's operating space in front of it, such as the center of a table. This simplifies the configuration of subsequent workspaces and object positions

---

## Object Configuration (objects)

The `objects` field defines the configuration of all objects in the scene, containing the following sub-fields:

### Structure

```json
{
    "objects": {
        "task_related_objects": [ ... ],  // Task-related objects (required)
        "scene_objects": [ ... ],         // Scene objects (optional)
        "attach_objects": [ ... ],        // Attached objects (optional)
        "fix_objects": [ ... ],           // Fixed position objects (optional)
        "constraints": null              // Constraints (optional)
    }
}
```

### 1. task_related_objects (Task-Related Objects)

Main objects involved in task execution. These objects will be placed with priority.

#### Basic Format

```json
{
    "object_id": "unique_object_id",
    "data_info_dir": "objects/path/to/object",
    "workspace_id": "work_table",
    "mass": 0.05
}
```

#### Candidate Objects (candidate_objects)

If the same `object_id` needs to be randomly selected from multiple candidates, use `candidate_objects`:

```json
{
    "object_id": "geniesim_2025_storage_box_2",
    "candidate_objects": [
        {
            "data_info_dir": "objects/benchmark/storage_box/benchmark_storage_box_006/",
            "object_id": "geniesim_2025_storage_box_006_green"
        },
        {
            "data_info_dir": "objects/benchmark/storage_box/benchmark_storage_box_007/",
            "object_id": "geniesim_2025_storage_box_007_blue"
        }
    ],
    "mass": 1000,
    "workspace_id": "box_poses"
}
```

**Notes**:
- The system will randomly select one from `candidate_objects`
- External fields (such as `mass`, `workspace_id`) will override corresponding fields in candidate objects
- The `object_id` used is the one defined externally, not the one in candidate objects

#### Workspace Relative Position

If an object needs to be placed relative to a workspace:

```json
{
    "object_id": "geniesim_2025_storage_box_2",
    "workspace_id": "box_poses",
    "workspace_relative_position": [0.0, 0.0, 0.0],
    "workspace_relative_orientation": [0.5, 0.5, 0.5, 0.5]
}
```

### 2. scene_objects (Scene Objects)

Background objects used to increase scene complexity, supporting sampling configuration.

#### Sampling Configuration

```json
{
    "sample": {
        "min_num": 2,      // Minimum sampling count
        "max_num": 7,      // Maximum sampling count
        "max_repeat": 1    // Maximum repeat count
    },
    "workspace_id": "work_table",
    "available_objects": [
        {
            "data_info_dir": "objects/benchmark/apple/benchmark_apple_000/",
            "object_id": "geniesim_2025_apple_000",
            "mass": 0.05
        },
        {
            "data_info_dir": "objects/benchmark/orange/benchmark_orange_001/",
            "object_id": "geniesim_2025_orange_001",
            "mass": 0.05
        }
    ]
}
```

**Notes**:
- `sample` defines sampling rules
- `available_objects` is the candidate object list
- The system will randomly sample `min_num` to `max_num` objects from `available_objects`
- Nested sampling is supported (`available_objects` can also contain object groups with `sample`)

### 3. attach_objects (Attached Objects)

Objects attached to other objects, such as items placed in boxes.

```json
{
    "sample": {
        "min_num": 1,
        "max_num": 2,
        "max_repeat": 2
    },
    "anchor_info": {
        "anchor_object": "geniesim_2025_target_storage_box",  // Anchor object ID
        "position": [0, 0.07, 0.04],                          // Relative position
        "quaternion": [1, 0, 0, 0],                          // Relative rotation (optional)
        "random_range": [0.04, 0.0, 0.04]                    // Random range (optional)
    },
    "workspace_id": "work_table",
    "available_objects": [ ... ]
}
```

**Notes**:
- `anchor_object` specifies the object ID to attach to
- `position` and `quaternion` define the pose relative to the anchor object
- `random_range` allows random offset within the specified range

### 4. fix_objects (Fixed Position Objects)

Fixed position objects. Format is the same as `task_related_objects`, but position is directly specified by configuration.

### Field Description

All object types support the following fields:

- **object_id** (required): Unique identifier for the object
- **data_info_dir** (required): Object data directory path, relative to the `$SIM_ASSETS` environment variable
- **workspace_id** (optional): Workspace ID, specifying the workspace where the object is placed
- **mass** (optional): Object mass (kg)
- **workspace_relative_position** (optional): Position relative to workspace [x, y, z]
- **workspace_relative_orientation** (optional): Rotation relative to workspace (quaternion w, x, y, z)
- **chinese_semantic_name** (optional): Chinese semantic name, can be a list (random selection)
- **english_semantic_name** (optional): English semantic name, can be a list (random selection)
- **allow_duplicate** (optional): Whether to allow duplicates (default `false`)

---

## Scene Configuration (scene)

The `scene` field defines the basic information and workspaces of the scene.

### Structure

```json
{
    "scene": {
        "scene_id": "background/home_b/",
        "scene_info_dir": "background/home_b/",
        "scene_usd": "background/home_b/home_b_00.usda",  // or list
        "function_space_objects": { ... }
    }
}
```

### Field Description

- **scene_id** (required): Scene ID, used to match robot initial pose
- **scene_info_dir** (required): Scene information directory path
- **scene_usd** (required): USD scene file path, can be a string or list (random selection when list)
- **function_space_objects** (optional): Workspace definitions

### function_space_objects (Workspaces)

Defines workspace areas for object placement. Depending on the configuration method, the system uses two different layout generators (refer to `GeneratorType.SPACE` and `GeneratorType.SAMPLE`):

#### 1. Workspace Area Mode (SPACE)

When the workspace **does not contain** the `poses` field, the system uses `GeneratorType.SPACE` mode:

```json
{
    "work_table": {
        "position": [0.0, 0.01, 0.9],
        "quaternion": [1, 0, 0, 0],
        "size": [0.5, 0.8, 0.2]
    }
}
```

**Characteristics**:
- Multiple objects will be **simultaneously arranged** in this area
- The system uses a 2D layout solver (`LayoutSolver2D`) to automatically calculate positions
- Positions are selected within the area with **collision avoidance** as the principle
- Suitable for scenarios where multiple objects need to be randomly distributed in the same area (e.g., fruits to be grasped placed on a table, can be randomly arranged within the area)

**Field Description**:
- `position`: Workspace area center position [x, y, z]
- `quaternion`: Workspace area rotation (quaternion w, x, y, z)
- `size`: Workspace area dimensions [x, y, z] (unit: meters)
- `blocked_zone` (optional): Prohibited placement areas

#### 2. Candidate Position Mode (SAMPLE)

When the workspace **contains** the `poses` field, the system uses `GeneratorType.SAMPLE` mode:

```json
{
    "box_poses": {
        "poses": [
            {
                "position": [0.11, 0.09, 0.9],
                "quaternion": [1, 0, 0, 0],
                "random": {
                    "delta_position": [0.03, 0.0, 0]
                }
            },
            {
                "position": [0.11, -0.1, 0.9],
                "quaternion": [1, 0, 0, 0],
                "random": {
                    "delta_position": [0.03, 0.0, 0]
                }
            }
        ]
    }
}
```

**Characteristics**:
- Multiple objects will **sample positions** from candidate positions in the `poses` list
- **Different objects will not be placed at the same position** (each pose is assigned to at most one object)
- If the number of objects exceeds the number of poses, the system will report an error
- Each object randomly selects one from candidate positions, and can only perform small random offsets near the point through `random.delta_position`
- Suitable for scenarios where objects need to be placed at specific position points (e.g., multiple boxes need to be placed left and right on a table, need to sample placement points from two positions)

**Field Description**:
- `poses`: Candidate position list, each element contains:
  - `position`: Position [x, y, z]
  - `quaternion`: Rotation (quaternion w, x, y, z)
  - `random` (optional): Random offset configuration
    - `delta_position`: Position random range [x, y, z]
    - `delta_angle`: Angle random range (radians)
  - `chinese_position_semantic` (optional): Chinese position semantic name
  - `english_position_semantic` (optional): English position semantic name

**Comparison of Two Modes**:

| Feature | SPACE (Workspace Area) | SAMPLE (Candidate Positions) |
|---------|----------------------|----------------------------|
| Configuration | Contains `size`, does not contain `poses` | Contains `poses` array |
| Object Placement | Simultaneously arranged in area, can be random within area | Sampled from candidate positions, can only be random near points |
| Position Calculation | 2D layout solver automatically calculates | Selected from predefined positions |
| Collision Detection | Automatically avoids collisions | Does not reuse the same position |
| Use Case | Fruits to be grasped placed on table, can be randomly arranged in area | Multiple boxes need to be placed left and right on table, need to sample placement points from two positions |

---

## Robot Configuration (robot)

The `robot` field defines the robot configuration and initial state.

### Structure

```json
{
    "robot": {
        "arm": "dual",              // "left", "right", "dual"
        "robot_id": "G2",           // "G1", "G2"
        "robot_cfg": "G2_omnipicker_fixed_dual.json",
        "robot_init_pose": { ... },
        "init_arm_pose": { ... },
        "init_arm_pose_noise": { ... }
    }
}
```

### robot_init_pose (Robot Initial Pose)

```json
{
    "robot_init_pose": {
        "position": [-0.66, 0.0, 0.0],
        "quaternion": [1, 0, 0, 0],
        "random": {
            "delta_position": [0.06, 0.06, 0]
        }
    }
}
```

**Notes**:
- `position` and `quaternion` define the initial pose of the robot base
- `random.delta_position` allows random position offset within the specified range

### init_arm_pose (Arm Initial Joint Angles)

Define the initial angles (radians) of each robot joint:

```json
{
    "init_arm_pose": {
        "idx21_arm_l_joint1": 0.739033,
        "idx22_arm_l_joint2": -0.717023,
        "idx61_arm_r_joint1": -0.739033,
        "idx62_arm_r_joint2": -0.717023,
        // ... other joints
        "idx41_gripper_l_outer_joint1": 0.85,
        "idx81_gripper_r_outer_joint1": 0.85
    }
}
```

### init_arm_pose_noise (Joint Angle Noise)

Add random noise to joint angles:

```json
{
    "init_arm_pose_noise": {
        ".*_arm_.*": {
            "noise_type": "uniform",
            "low": -0.05,
            "high": 0.05
        }
    }
}
```

**Notes**:
- Keys are regular expressions matching joint names
- `noise_type` can be `"uniform"` (uniform distribution)
- `low` and `high` define the noise range

---

## Task Stages (stages)

`stages` is an array that defines the various stages of task execution.

### Basic Structure

```json
{
    "stages": [
        {
            "action": "pick",           // Action type
            "action_description": { ... },
            "active": { ... },          // Active object
            "passive": { ... },         // Passive object
            "extra_params": { ... },    // Extra parameters
            "checker": [ ... ]          // Checker (optional)
        }
    ]
}
```

### Action Types

- **pick**: Grasp object
- **place**: Place object
- **rotate**: Rotate object
- **insert**: Insert object
- **reset**: Reset arm

### active and passive

```json
{
    "active": {
        "object_id": "gripper",     // or specific object ID
        "primitive": null           // Grasp primitive (optional)
    },
    "passive": {
        "object_id": "geniesim_2025_target_grasp_object",
        "primitive": null
    }
}
```

**Notes**:
- `active` is the object performing the action (usually `"gripper"`)
- `passive` is the object being operated on
- `object_id` can be a string, list, or dictionary (when dictionary, selected based on `arm`)

### extra_params (Extra Parameters)

Different action types have different parameters. For detailed parameter descriptions, please refer to [Action Extra Parameters Detailed Description](#action-extra-parameters-detailed-description).

#### pick Action

```json
{
    "extra_params": {
        "arm": "auto",                    // "left", "right", "auto"
        "disable_upside_down": true,      // Disable upside-down grasping
        "flip_grasp": true,               // Flip grasp
        "grasp_offset": 0.01,            // Grasp offset
        "pick_up_distance": 0.1,          // Lift distance
        "grasp_upper_percentile": 75     // Grasp upper percentile
    }
}
```

#### place Action

```json
{
    "extra_params": {
        "arm": "auto",
        "place_with_origin_orientation": true
    }
}
```

#### insert Action

```json
{
    "extra_params": {
        "arm": "auto",
        "pre_insert_offset": 0.1,
        "gripper_state": "open"
    }
}
```

#### rotate Action

```json
{
    "extra_params": {
        "arm": "auto",
        "place_up_axis": "y",
        "pick_up_distance": 0.05
    }
}
```

#### reset Action

```json
{
    "extra_params": {
        "arm": "auto",
        "plan_type": "AvoidObs"           // "Simple", "AvoidObs"
    }
}
```

**Note**: For complete parameter lists and descriptions of each action, please refer to [Action Extra Parameters Detailed Description](#action-extra-parameters-detailed-description).

### action_description (Action Description)

Used to generate task description text:

```json
{
    "action_description": {
        "action_text": "{左/右}臂拿起桌面上的苹果",
        "english_action_text": "{Left/Right} arm picks up the apple on the table"
    }
}
```

**Placeholders**:
- `{左/右}` or `{Left/Right}` or `{left/right}`: Automatically replaced based on the arm used
- `{object:object_id}`: Replaced with the object's Chinese/English semantic name
- `{position:object_id}`: Replaced with the object's position semantic name

### checker (Checker)

Used to verify whether a stage has been successfully completed. For detailed checker parameter descriptions, please refer to [Runtime Checker Description](./common/data_filter/README.md#runtime-checker-runtime-checker).

**Basic Structure**:
```json
{
    "checker": [
        {
            "checker_name": "distance_to_target",
            "params": {
                "object_id": "geniesim_2025_target_grasp_object",
                "target_id": "gripper",
                "rule": "lessThan",
                "value": 0.08
            }
        }
    ]
}
```

**Available Checkers**:
- `distance_to_target`: Check distance between object and target
- `local_axis_angle`: Check angle between object's local axis and target vector

For detailed parameter descriptions and examples, please refer to [Runtime Checker Description](./common/data_filter/README.md#runtime-checker-runtime-checker).

---

## Other Configurations

### recording_setting (Recording Settings)

```json
{
    "recording_setting": {
        "camera_list": [
            "/G2/head_link3/head_front_Camera",
            "/G2/gripper_r_base_link/Right_Camera"
        ],
        "fps": 30,
        "num_of_episode": 8,
        "noised_probability": 0.1
    }
}
```

### task_description (Task Description)

```json
{
    "task_description": {
        "task_name": "将苹果放入对应的收纳盒中",
        "english_task_name": "sort the apple into the corresponding storage box",
        "init_scene_text": "机器人在桌面前，桌面上放着一个水果和两个盛有不同水果的收纳盒"
    }
}
```

Supports placeholders (same as `action_description`).

### task_metric (Task Evaluation Metrics)

Defines data filtering rules used to verify whether collected data meets quality requirements after data collection. For detailed filtering rule descriptions, please refer to [Data Filter Rules Description](./common/data_filter/README.md#data-filter-rules-filter-rules).

**Basic Structure**:
```json
{
    "task_metric": {
        "filter_rules": [
            {
                "rule_name": "is_gripper_in_view",
                "params": {
                    "camera": "head",
                    "gripper": "right",
                    "out_view_allow_time": 0.2
                },
                "result_code": 4
            },
            {
                "rule_name": "is_object_relative_position_in_target",
                "params": {
                    "objects": ["geniesim_2025_target_grasp_object"],
                    "target": "geniesim_2025_target_storage_box",
                    "relative_position_range": [[-0.06, 0.06], [-0.05, 0.05], [-0.12, 0.12]]
                },
                "result_code": 1
            }
        ]
    }
}
```

**Available Filter Rules**:
- `is_object_reach_target`: Check if object reaches target area
- `is_object_pose_similar2start`: Check if object pose is similar to initial
- `is_object_in_view`: Check if object is within image bounds
- `is_gripper_in_view`: Check if gripper is in camera view
- `is_object_end_pose_up`: Check if object's final pose is vertically upward
- `is_object_end_higher_than_start`: Check if object's final position is higher than initial position
- `is_objects_distance_greater_than`: Check distance between objects
- `is_object_end_in_region`: Check if object's final position is in specified region
- `is_object_grasped_by_gripper`: Check if object is grasped by gripper
- `is_object_relative_position_in_target`: Check object's relative position to target object

For detailed parameter descriptions and examples, please refer to [Data Filter Rules Description](./common/data_filter/README.md#data-filter-rules-filter-rules).

---

## Complete Examples

For complete task configuration examples, please refer to configuration files in the `tasks/geniesim_2025/` directory, for example:
- `tasks/geniesim_2025/sort_fruit/g2/sort_the_fruit_into_the_box_apple_g2.json` - Fruit sorting task
- Other task configuration files

---

## Quick Start

### Creating Task Configuration from Scratch

This section uses `tasks/geniesim_2025/sort_fruit/g2/sort_the_fruit_into_the_box_apple_g2.json` as an example to describe in detail how to create a task configuration from scratch.

#### 1. Configure Scene and Origin (origin)

**Design Rationale**: First, you need to select a scene file and determine the scene origin. The origin is usually chosen as the midpoint of the operating space in front of the robot (such as the center of a table), which simplifies the configuration of subsequent workspaces and object positions.

```json
{
    "origin": {
        "position": [2.91, 0.76, 0.0],
        "quaternion": [1, 0, 0, 0]
    },
    "scene": {
        "scene_id": "background/home_b/",
        "scene_info_dir": "background/home_b/",
        "scene_usd": ["background/home_b/home_b_00.usda"],
        "function_space_objects": {
            "work_table": {
                "position": [-0.15, 0.01, 0.9],
                "quaternion": [1, 0, 0, 0],
                "size": [0.24, 0.8, 0.2]
            },
            "box_poses": {
                "poses": [
                    {
                        "position": [0.11, 0.09, 0.9],
                        "quaternion": [1, 0, 0, 0],
                        "random": {
                            "delta_position": [0.03, 0.0, 0]
                        }
                    },
                    {
                        "position": [0.11, -0.1, 0.9],
                        "quaternion": [1, 0, 0, 0],
                        "random": {
                            "delta_position": [0.03, 0.0, 0]
                        }
                    }
                ]
            }
        }
    }
}
```

**Notes**:
- `work_table` uses SPACE mode (contains `size`), used for placing fruits to be grasped, can be randomly arranged within the area
- `box_poses` uses SAMPLE mode (contains `poses`), used for placing two storage boxes, sampling from two candidate positions

#### 2. Configure Robot

```json
{
    "robot": {
        "arm": "dual",
        "robot_id": "G2",
        "robot_cfg": "G2_omnipicker_fixed_dual.json",
        "robot_init_pose": {
            "position": [-0.66, 0.0, 0.0],
            "quaternion": [1, 0, 0, 0],
            "random": {
                "delta_position": [0.06, 0.06, 0]
            }
        },
        "init_arm_pose": {
            "idx21_arm_l_joint1": 0.739033,
            "idx22_arm_l_joint2": -0.717023,
            // ... other joints
        },
        "init_arm_pose_noise": {
            ".*_arm_.*": {
                "noise_type": "uniform",
                "low": -0.05,
                "high": 0.05
            }
        }
    }
}
```

#### 3. Configure Task-Related Objects (task_related_objects)

**Design Rationale**: The task needs to grasp an apple and place it into the corresponding storage box. Therefore, we need to configure:
- Target grasp object: Randomly select one from multiple apple assets
- Two storage boxes: Randomly select from multiple storage box assets, use SAMPLE mode to sample placement from two candidate positions

```json
{
    "objects": {
        "task_related_objects": [
            {
                "object_id": "geniesim_2025_target_grasp_object",
                "candidate_objects": [
                    {
                        "data_info_dir": "objects/benchmark/apple/benchmark_apple_000/",
                        "object_id": "geniesim_2025_apple_000"
                    },
                    {
                        "data_info_dir": "objects/benchmark/apple/benchmark_apple_001/",
                        "object_id": "geniesim_2025_apple_001"
                    },
                    {
                        "data_info_dir": "objects/benchmark/apple/benchmark_apple_002/",
                        "object_id": "geniesim_2025_apple_002"
                    },
                    {
                        "data_info_dir": "objects/benchmark/apple/benchmark_apple_003/",
                        "object_id": "geniesim_2025_apple_003"
                    }
                ],
                "mass": 0.05,
                "workspace_id": "work_table"
            },
            {
                "object_id": "geniesim_2025_target_storage_box",
                "candidate_objects": [
                    {
                        "data_info_dir": "objects/benchmark/storage_box/benchmark_storage_box_006/",
                        "object_id": "geniesim_2025_storage_box_006_green"
                    },
                    {
                        "data_info_dir": "objects/benchmark/storage_box/benchmark_storage_box_007/",
                        "object_id": "geniesim_2025_storage_box_007_blue"
                    },
                    {
                        "data_info_dir": "objects/benchmark/storage_box/benchmark_storage_box_008/",
                        "object_id": "geniesim_2025_storage_box_008_white"
                    },
                    {
                        "data_info_dir": "objects/benchmark/storage_box/benchmark_storage_box_009/",
                        "object_id": "geniesim_2025_storage_box_009_red"
                    },
                    {
                        "data_info_dir": "objects/benchmark/storage_box/benchmark_storage_box_010/",
                        "object_id": "geniesim_2025_storage_box_010_grey"
                    },
                    {
                        "data_info_dir": "objects/benchmark/storage_box/benchmark_storage_box_011/",
                        "object_id": "geniesim_2025_storage_box_011_black"
                    }
                ],
                "mass": 1000,
                "workspace_id": "box_poses",
                "workspace_relative_position": [0.0, 0.0, 0.0],
                "workspace_relative_orientation": [0.5, 0.5, 0.5, 0.5]
            },
            {
                "object_id": "geniesim_2025_storage_box_2",
                "candidate_objects": [
                    {
                        "data_info_dir": "objects/benchmark/storage_box/benchmark_storage_box_006/",
                        "object_id": "geniesim_2025_storage_box_006_green"
                    },
                    // ... other storage box options
                ],
                "mass": 1000,
                "workspace_id": "box_poses",
                "workspace_relative_position": [0.0, 0.0, 0.0],
                "workspace_relative_orientation": [0.5, 0.5, 0.5, 0.5]
            }
        ]
    }
}
```

#### 4. Configure Attached Objects (attach_objects)

**Design Rationale**: To increase scene realism and complexity, we need to place 1-2 apples in the target storage box, and 1-2 other types of fruits (peaches, oranges, lemons, pomegranates, etc.) in another storage box. This demonstrates nested sampling: outer layer samples from multiple fruit groups, inner layer samples from multiple assets in each fruit group.

```json
{
    "objects": {
        "attach_objects": [
            {
                "sample": {
                    "min_num": 1,
                    "max_num": 2,
                    "max_repeat": 2
                },
                "anchor_info": {
                    "anchor_object": "geniesim_2025_target_storage_box",
                    "position": [0, 0.07, 0.04],
                    "quaternion": [1, 0, 0, 0],
                    "random_range": [0.04, 0.0, 0.04]
                },
                "workspace_id": "work_table",
                "available_objects": [
                    {
                        "data_info_dir": "objects/benchmark/apple/benchmark_apple_000/",
                        "object_id": "geniesim_2025_apple_000"
                    },
                    {
                        "data_info_dir": "objects/benchmark/apple/benchmark_apple_001/",
                        "object_id": "geniesim_2025_apple_001"
                    },
                    {
                        "data_info_dir": "objects/benchmark/apple/benchmark_apple_002/",
                        "object_id": "geniesim_2025_apple_002"
                    },
                    {
                        "data_info_dir": "objects/benchmark/apple/benchmark_apple_003/",
                        "object_id": "geniesim_2025_apple_003"
                    }
                ]
            },
            {
                "sample": {
                    "min_num": 1,
                    "max_num": 1,
                    "max_repeat": 1
                },
                "workspace_id": "work_table",
                "anchor_info": {
                    "anchor_object": "geniesim_2025_storage_box_2",
                    "position": [0, 0.12, 0],
                    "quaternion": [1, 0, 0, 0],
                    "random_range": [0.04, 0.0, 0.04]
                },
                "available_objects": [
                    {
                        "sample": {
                            "min_num": 1,
                            "max_num": 2,
                            "max_repeat": 2
                        },
                        "workspace_id": "work_table",
                        "available_objects": [
                            {
                                "data_info_dir": "objects/benchmark/peach/benchmark_peach_000/",
                                "object_id": "geniesim_2025_peach_000"
                            },
                            {
                                "data_info_dir": "objects/benchmark/peach/benchmark_peach_019/",
                                "object_id": "geniesim_2025_peach_019"
                            },
                            {
                                "data_info_dir": "objects/benchmark/peach/benchmark_peach_020/",
                                "object_id": "geniesim_2025_peach_020"
                            },
                            {
                                "data_info_dir": "objects/benchmark/peach/benchmark_peach_021/",
                                "object_id": "geniesim_2025_peach_021"
                            }
                        ]
                    },
                    {
                        "sample": {
                            "min_num": 1,
                            "max_num": 2,
                            "max_repeat": 2
                        },
                        "workspace_id": "work_table",
                        "available_objects": [
                            {
                                "data_info_dir": "objects/benchmark/orange/benchmark_orange_001/",
                                "object_id": "geniesim_2025_orange_001"
                            },
                            {
                                "data_info_dir": "objects/benchmark/orange/benchmark_orange_002/",
                                "object_id": "geniesim_2025_orange_002"
                            },
                            {
                                "data_info_dir": "objects/benchmark/orange/benchmark_orange_004/",
                                "object_id": "geniesim_2025_orange_004"
                            }
                        ]
                    },
                    {
                        "sample": {
                            "min_num": 1,
                            "max_num": 2,
                            "max_repeat": 2
                        },
                        "workspace_id": "work_table",
                        "available_objects": [
                            {
                                "data_info_dir": "objects/benchmark/lemon/benchmark_lemon_027/",
                                "object_id": "geniesim_2025_lemon_027"
                            },
                            {
                                "data_info_dir": "objects/benchmark/lemon/benchmark_lemon_028/",
                                "object_id": "geniesim_2025_lemon_028"
                            },
                            {
                                "data_info_dir": "objects/benchmark/lemon/benchmark_lemon_029/",
                                "object_id": "geniesim_2025_lemon_029"
                            },
                            {
                                "data_info_dir": "objects/benchmark/lemon/benchmark_lemon_030/",
                                "object_id": "geniesim_2025_lemon_030"
                            }
                        ]
                    }
                ]
            }
        ]
    }
}
```

#### 5. Configure Task Stages (stages)

**Design Rationale**: The task consists of three stages: grasp apple, place into target storage box, reset arm. Each stage needs to configure action type, action description, checker, etc.

```json
{
    "stages": [
        {
            "action": "pick",
            "action_description": {
                "action_text": "{左/右}臂拿起桌面上的苹果",
                "english_action_text": "{Left/Right} arm picks up the apple on the table"
            },
            "active": {
                "object_id": "gripper",
                "primitive": null
            },
            "passive": {
                "object_id": "geniesim_2025_target_grasp_object",
                "primitive": null
            },
            "extra_params": {
                "arm": "auto",
                "disable_upside_down": true,
                "flip_grasp": true,
                "grasp_offset": 0.01,
                "pick_up_distance": 0.1,
                "grasp_upper_percentile": 75
            },
            "checker": [
                {
                    "checker_name": "distance_to_target",
                    "params": {
                        "object_id": "geniesim_2025_target_grasp_object",
                        "target_id": "gripper",
                        "rule": "lessThan",
                        "value": 0.08
                    }
                }
            ]
        },
        {
            "action": "place",
            "action_description": {
                "action_text": "将拿着的苹果归类到桌面上对应的收纳筐中",
                "english_action_text": "Place the apple into the corresponding storage bin"
            },
            "active": {
                "object_id": "geniesim_2025_target_grasp_object",
                "primitive": null
            },
            "passive": {
                "object_id": "geniesim_2025_target_storage_box",
                "primitive": null
            },
            "extra_params": {
                "arm": "auto",
                "place_with_origin_orientation": true
            },
            "checker": [
                {
                    "checker_name": "distance_to_target",
                    "params": {
                        "object_id": "geniesim_2025_target_grasp_object",
                        "target_id": "geniesim_2025_target_storage_box",
                        "target_offset": {
                            "frame": "world",
                            "position": [0, 0, 0.02]
                        },
                        "rule": "lessThan",
                        "value": 0.12
                    }
                }
            ]
        },
        {
            "action": "reset",
            "action_description": {
                "action_text": "{左/右}臂复位",
                "english_action_text": "{Left/Right} arm resets"
            },
            "active": {
                "object_id": "gripper",
                "primitive": null
            },
            "passive": {
                "object_id": "gripper",
                "primitive": null
            },
            "extra_params": {
                "arm": "auto"
            }
        }
    ]
}
```

#### 6. Configure Task Evaluation Metrics (task_metric)

**Design Rationale**: After data collection, we need to verify the quality of collected data, such as checking if the gripper is in view, if objects are successfully placed at target positions, etc.

```json
{
    "task_metric": {
        "filter_rules": [
            {
                "rule_name": "is_gripper_in_view",
                "params": {
                    "camera": "head",
                    "gripper": "right",
                    "out_view_allow_time": 0.2
                },
                "result_code": 4
            },
            {
                "rule_name": "is_gripper_in_view",
                "params": {
                    "camera": "head",
                    "gripper": "left",
                    "out_view_allow_time": 0.1
                },
                "result_code": 4
            },
            {
                "rule_name": "is_object_relative_position_in_target",
                "params": {
                    "objects": ["geniesim_2025_target_grasp_object"],
                    "target": "geniesim_2025_target_storage_box",
                    "relative_position_range": [[-0.06, 0.06], [-0.05, 0.05], [-0.12, 0.12]]
                },
                "result_code": 1
            }
        ]
    }
}
```

#### 7. Configure Other Settings

```json
{
    "task": "sort_the_fruit_into_the_box_apple_g2",
    "task_description": {
        "task_name": "将苹果放入对应的收纳盒中",
        "english_task_name": "sort the apple into the corresponding storage box",
        "init_scene_text": "机器人在桌面前，桌面上放着一个水果和两个盛有不同水果的收纳盒"
    },
    "recording_setting": {
        "camera_list": [
            "/G2/head_link3/head_front_Camera",
            "/G2/gripper_r_base_link/Right_Camera",
            "/G2/gripper_l_base_link/Left_Camera",
            "/G2/head_link3/head_right_Camera",
            "/G2/head_link3/head_left_Camera"
        ],
        "fps": 30,
        "num_of_episode": 8,
        "noised_probability": 0.1
    }
}
```

### Modifying Existing Configuration

If you already have a similar task configuration, you can find a similar task in the `tasks/geniesim_2025/` directory as a template, then modify the corresponding configuration items according to your needs.

---

## Frequently Asked Questions

### Q: How to implement random object selection?

A: Use the `candidate_objects` field, and the system will randomly select one from the candidates.

### Q: How to attach objects to other objects?

A: Use `attach_objects`, and specify the anchor object and relative position through `anchor_info`.

### Q: How to define multiple optional positions?

A: Use the `poses` array in workspace configuration. Note: Workspaces containing `poses` use SAMPLE mode, where multiple objects sample positions from candidate positions, and different objects will not be placed at the same position. If you need multiple objects arranged in the same area, use the workspace area mode (SPACE) containing `size`.

### Q: How to implement sampling multiple objects?

A: Use `sample` configuration in `scene_objects` or `attach_objects`, specifying `min_num` and `max_num`.

### Q: How to specify object paths?

A: `data_info_dir` is a path relative to the `$SIM_ASSETS` environment variable, for example `"objects/benchmark/apple/benchmark_apple_000/"`.

---

## Action Extra Parameters Detailed Description

Action extra parameters are defined in the `extra_params` field of `stages` to control the specific behavior of actions.

### pick Action

**Parameters**:
- `arm` (str, optional): Arm to use, `"left"`, `"right"` or `"auto"`, default `"right"`
- `grasp_offset` (float, optional): Grasp offset (unit: meters), default `0.03`
- `pre_grasp_offset` (float, optional): Pre-grasp offset (unit: meters), default `0.0`
- `grasp_lower_percentile` (float, optional): Grasp lower percentile (0-100), default `0`
- `grasp_upper_percentile` (float, optional): Grasp upper percentile (0-100), default `100`
- `disable_upside_down` (bool, optional): Disable upside-down grasping, default `false`
- `flip_grasp` (bool, optional): Flip grasp (180 degrees around z-axis), default `false`
- `pick_up_distance` (float, optional): Lift distance (unit: meters), default `0.12`
- `pick_up_type` (str, optional): Lift type, `"Simple"` or `"AvoidObs"`, default `"Simple"`
- `use_near_point` (bool, optional): Whether to use nearby point, default `false`
- `error_data` (dict, optional): Error data configuration
  - `type` (str): Error type, such as `"RandomPerturbations"`, `"MissGrasp"`, `"WrongTarget"`, `"KeepClose"`
  - `params` (dict): Error parameters

**Example**:
```json
{
    "extra_params": {
        "arm": "auto",
        "disable_upside_down": true,
        "flip_grasp": true,
        "grasp_offset": 0.01,
        "pick_up_distance": 0.1,
        "grasp_upper_percentile": 75
    }
}
```

### place Action

**Parameters**:
- `arm` (str, optional): Arm to use, `"left"`, `"right"` or `"auto"`, default `"right"`
- `place_with_origin_orientation` (bool, optional): Whether to use original orientation for placement, default `true`
- `disable_upside_down` (bool, optional): Disable upside-down placement, default `false`
- `use_pre_place` (bool, optional): Whether to use pre-placement, default `false`
- `pre_place_offset` (float, optional): Pre-placement offset (unit: meters), default `0.12`
- `pre_place_direction` (str, optional): Pre-placement direction, default `"z"`
- `pre_pose_noise` (dict, optional): Pre-pose noise configuration
  - `position_noise` (float): Position noise
  - `rotation_noise` (float): Rotation noise
- `gripper_state` (str | None, optional): Gripper state, `"open"`, `"close"` or `None`, default `"open"`
- `post_place_action` (list[dict], optional): Post-placement action list
  - `gripper_cmd` (str, optional): Gripper command
  - `distance` (float, optional): Movement distance (unit: meters), default `0.02`
  - `direction` (list[float], optional): Movement direction (in passive object's local coordinate system), default `[0, 0, 1]`
- `use_near_point` (bool, optional): Whether to use nearby point, default `false`
- `error_data` (dict, optional): Error data configuration (same format as pick)

**Example**:
```json
{
    "extra_params": {
        "arm": "auto",
        "place_with_origin_orientation": true,
        "use_pre_place": true,
        "pre_place_offset": 0.12
    }
}
```

### insert Action

Inherits from `place` action, all parameters are the same as `place` action.

**Example**:
```json
{
    "extra_params": {
        "arm": "auto",
        "use_pre_place": true,
        "pre_place_offset": 0.1,
        "gripper_state": "open"
    }
}
```

### rotate Action

**Parameters**:
- `arm` (str, optional): Arm to use, `"left"`, `"right"` or `"auto"`, default `"right"`
- `place_up_axis` (str, optional): Placement upward axis, `"x"`, `"y"` or `"z"`, default `"y"`
- `pick_up_distance` (float, optional): Lift distance (unit: meters), default `0.0`
- `pick_up_direction` (str, optional): Lift direction, `"x"`, `"y"` or `"z"`, default `"z"`
- `place_origin_position` (bool, optional): Whether to place at original position, default `true`

**Example**:
```json
{
    "extra_params": {
        "arm": "auto",
        "place_up_axis": "y",
        "pick_up_distance": 0.05
    }
}
```

### reset Action

**Parameters**:
- `arm` (str, optional): Arm to use, `"left"`, `"right"` or `"auto"`, default `"right"`
- `plan_type` (str, optional): Planning type, `"Simple"` or `"AvoidObs"`, default `"AvoidObs"`

**Example**:
```json
{
    "extra_params": {
        "arm": "auto",
        "plan_type": "AvoidObs"
    }
}
```
