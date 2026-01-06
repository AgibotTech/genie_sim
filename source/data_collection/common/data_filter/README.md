# Data Filter and Checker Documentation

This document describes the parameters and usage of checkers and data filter rules used in task configurations.

## Table of Contents

1. [Runtime Checker](#runtime-checker-runtime-checker)
2. [Data Filter Rules](#data-filter-rules-filter-rules)

---

## Runtime Checker

Runtime checkers are used to verify whether stages have been successfully completed during task execution. Checkers are defined in the `checker` field of `stages`.

### Basic Structure

```json
{
    "checker": [
        {
            "checker_name": "checker_name",
            "params": {
                // Checker-specific parameters
            }
        }
    ]
}
```

### 1. distance_to_target

Checks whether the distance between an object and a target meets the condition.

**Parameters**:
- `object_id` (str, required): Object ID to check
- `target_id` (str, required): Target object ID (can be an object ID or "left"/"right" for gripper)
- `value` (float, required): Distance threshold (unit: meters)
- `rule` (str, required): Comparison rule, options:
  - `"lessThan"`: Distance is less than threshold
  - `"greaterThan"`: Distance is greater than threshold
  - `"equalTo"`: Distance equals threshold
- `object_offset` (dict, optional): Object offset configuration
  - `frame` (str): Coordinate frame, `"world"` or `"local"`
  - `position` (list[float]): Position offset [x, y, z]
  - `orientation` (list[float]): Rotation offset (quaternion w, x, y, z)
- `target_offset` (dict, optional): Target offset configuration (same format as `object_offset`)
- `ignore_axises` (list[str], optional): Ignored coordinate axes, e.g., `["z"]` means only calculate distance in x and y directions
- `is_local` (bool, optional): Whether to calculate distance using the target object's local coordinate system (default `false`)

**Example**:
```json
{
    "checker_name": "distance_to_target",
    "params": {
        "object_id": "geniesim_2025_target_grasp_object",
        "target_id": "gripper",
        "rule": "lessThan",
        "value": 0.08,
        "target_offset": {
            "frame": "world",
            "position": [0, 0, 0.02]
        }
    }
}
```

### 2. local_axis_angle

Checks whether the angle difference between the object's local specified axis and the target vector meets the condition.

**Parameters**:
- `object_id` (str, required): Object ID to check
- `axis` (str, required): Local axis to check, `"x"`, `"y"` or `"z"`
- `target_vector` (list[float], required): Target vector [x, y, z] (world coordinate system)
- `value` (float, required): Angle threshold (unit: radians)
- `rule` (str, required): Comparison rule, options:
  - `"lessThan"`: Angle difference is less than threshold
  - `"greaterThan"`: Angle difference is greater than threshold
  - `"equalTo"`: Angle difference equals threshold
- `object_offset` (dict, optional): Object offset configuration (same format as `distance_to_target`)

**Example**:
```json
{
    "checker_name": "local_axis_angle",
    "params": {
        "object_id": "geniesim_2025_target_grasp_object",
        "axis": "y",
        "target_vector": [0, 0, 1],
        "rule": "lessThan",
        "value": 0.1745
    }
}
```

---

## Data Filter Rules

Data filter rules are used to verify whether collected data meets quality requirements after data collection. Filter rules are defined in the `filter_rules` field of `task_metric`.

### Basic Structure

```json
{
    "task_metric": {
        "filter_rules": [
            {
                "rule_name": "rule_name",
                "params": {
                    // Rule-specific parameters
                },
                "result_code": 1
            }
        ]
    }
}
```

**Notes**:
- `rule_name`: Filter rule name
- `params`: Rule-specific parameters
- `result_code`: Result code returned if the rule check fails

### 1. is_object_reach_target

Checks whether an object finally reaches the target area.

**Parameters**:
- `objects` (list[str], required): List of object IDs to check
- `target` (str | list[float], required): Target position
  - If string: Target object ID
  - If list: Length 3, representing target point coordinates [x, y, z] in world coordinate system
- `target_scope` (list[list[float]], required): Allowed range for target point, shape (3, 2), each row represents [min, max]

**Example**:
```json
{
    "rule_name": "is_object_reach_target",
    "params": {
        "objects": ["geniesim_2025_target_grasp_object"],
        "target": "geniesim_2025_target_storage_box",
        "target_scope": [[-0.06, 0.06], [-0.05, 0.05], [-0.12, 0.12]]
    },
    "result_code": 1
}
```

### 2. is_object_pose_similar2start

Checks whether interactive objects that need to remain fixed have collided (pose offset).

**Parameters**:
- `objects` (list[tuple[str, str]], required): List of objects to check, each element is (object type, object name)
  - Object type: `"object"` (scene object name), `"camera"` (scene camera name), `"gripper"` ("left" or "right")
- `pos_threshold` (list[float], optional): Position threshold [x, y, z] (unit: meters), default `[0.1, 0.1, 0.1]`
- `euler_threshold` (list[float], optional): Euler angle threshold [x, y, z] (unit: degrees), default `[5, 5, 5]`
- `check_exist` (bool, optional): Whether to check object existence, default `true`

**Example**:
```json
{
    "rule_name": "is_object_pose_similar2start",
    "params": {
        "objects": [["object", "geniesim_2025_storage_box_2"]],
        "pos_threshold": [0.05, 0.05, 0.05],
        "euler_threshold": [3, 3, 3]
    },
    "result_code": 1
}
```

### 3. is_object_in_view

Checks whether an object is within image bounds.

**Parameters**:
- `objects` (list[str], required): List of object IDs to check
- `camera` (str, optional): Camera name to check, default `"head"`
- `downsample_ratio` (float, optional): Downsampling ratio (< 1.0), used to speed up checking, default `0.2`
- `refresh_rate` (int, optional): Actual frame rate used for simulation, default `30`
- `out_view_allow_time` (float, optional): Allowed time for object to leave camera view (unit: seconds), default `0.5`
- `camear_z_reverse` (bool, optional): Whether z-axis is reversed, G1 robot head view z-axis points backward by default, default `true`

**Example**:
```json
{
    "rule_name": "is_object_in_view",
    "params": {
        "objects": ["geniesim_2025_target_grasp_object"],
        "camera": "head",
        "downsample_ratio": 0.2,
        "out_view_allow_time": 0.5
    },
    "result_code": 1
}
```

### 4. is_gripper_in_view

Checks whether the gripper is in camera view (similar to `is_object_in_view`).

**Parameters**:
- `gripper` (str, optional): Gripper, `"left"` or `"right"`, default `"right"`
- `camera` (str, optional): Camera name to check, default `"head"`
- `downsample_ratio` (float, optional): Downsampling ratio, default `0.2`
- `refresh_rate` (int, optional): Frame rate, default `30`
- `out_view_allow_time` (float, optional): Allowed time to leave view (unit: seconds), default `0.5`
- `camear_z_reverse` (bool, optional): Whether z-axis is reversed, default `true`
- `zoom_factor` (float, optional): Zoom factor, default `1.05`

**Example**:
```json
{
    "rule_name": "is_gripper_in_view",
    "params": {
        "camera": "head",
        "gripper": "right",
        "downsample_ratio": 0.2,
        "out_view_allow_time": 0.2
    },
    "result_code": 4
}
```

### 5. is_object_end_pose_up

Checks whether an object's final pose is vertically upward.

**Parameters**:
- `objects` (list[str], required): List of object IDs to check
- `objects_up_axis` (list[str] | list[list[float]], required): Upward axis for each object
  - Can be string: `"x"`, `"y"` or `"z"`
  - Or list of length 3: [x, y, z] representing upward direction vector
- `thresholds` (list[float], required): Angle threshold for each object (unit: radians), representing the angle difference threshold between object's upward axis and world coordinate system z-axis direction

**Example**:
```json
{
    "rule_name": "is_object_end_pose_up",
    "params": {
        "objects": ["geniesim_2025_target_grasp_object"],
        "objects_up_axis": ["y"],
        "thresholds": [0.1745]
    },
    "result_code": 1
}
```

### 6. is_object_end_higher_than_start

Checks whether the target object's final state position is higher than the initial value on the z-axis by `delta_z`, allowing tolerance error.

**Parameters**:
- `objects` (list[str], required): List of object IDs to check
- `delta_z` (float, optional): Expected height difference in z-axis direction (unit: meters), default `0.08`
- `tolerance` (float, optional): Tolerance (unit: meters), default `0.03`

**Example**:
```json
{
    "rule_name": "is_object_end_higher_than_start",
    "params": {
        "objects": ["geniesim_2025_target_grasp_object"],
        "delta_z": 0.15,
        "tolerance": 0.05
    },
    "result_code": 1
}
```

### 7. is_objects_distance_greater_than

Checks whether the distance between specified objects is greater than the threshold.

**Parameters**:
- `objects` (list[str], required): List of object IDs to check (at least 2)
- `min_distance` (float, optional): Minimum distance threshold (unit: meters), default `0.3`
- `check_frame` (str, optional): Frame to check, options:
  - `"first"`: First frame
  - `"last"`: Last frame
  - `"all"`: All frames
  Default `"last"`

**Example**:
```json
{
    "rule_name": "is_objects_distance_greater_than",
    "params": {
        "objects": ["geniesim_2025_target_grasp_object", "geniesim_2025_storage_box_2"],
        "min_distance": 0.2,
        "check_frame": "last"
    },
    "result_code": 1
}
```

### 8. is_object_end_in_region

Checks whether an object's final position is within the specified region.

**Parameters**:
- `objects` (list[str], required): List of object IDs to check
- `region_center` (list[float], required): Region center point coordinates (world coordinate system) [x, y, z]
- `region_size` (list[float], required): Region dimensions (length, width, height) [x, y, z]

**Example**:
```json
{
    "rule_name": "is_object_end_in_region",
    "params": {
        "objects": ["geniesim_2025_target_grasp_object"],
        "region_center": [0.0, 0.0, 0.9],
        "region_size": [0.5, 0.8, 0.2]
    },
    "result_code": 1
}
```

### 9. is_object_grasped_by_gripper

Checks whether an object is grasped by the gripper and determines if there is unexpected dropping.

**Parameters**:
- `objs` (list[str], required): List of object IDs to check
- `gripper` (str, optional): Gripper, `"left"` or `"right"`, default `"right"`
- `active_gripper_joint` (str, optional): Active gripper joint name, default `"idx81_gripper_r_outer_joint1"`
- `grasp_time_threshold` (float, optional): Grasp time threshold (unit: seconds), default `1.0`
- `check_unexpected_drop` (bool, optional): Whether to check for unexpected dropping, default `false`
- `object_gripper_move_threshold` (float, optional): Relative movement threshold between object and gripper (unit: meters), default `0.002`

**Example**:
```json
{
    "rule_name": "is_object_grasped_by_gripper",
    "params": {
        "objs": ["geniesim_2025_target_grasp_object"],
        "gripper": "right",
        "grasp_time_threshold": 1.0
    },
    "result_code": 1
}
```

### 10. is_object_relative_position_in_target

Checks the final positional relationship between an object and a target object, calculates the object's position in the target object's coordinate system, and checks whether the relative position is within range (used to check if an object enters a container).

**Parameters**:
- `objects` (list[str], required): List of object IDs to check
- `target` (str, required): Target object ID
- `relative_position_range` (list[list[float]], required): Relative position range, shape (3, 2), each row represents [min, max]

**Example**:
```json
{
    "rule_name": "is_object_relative_position_in_target",
    "params": {
        "objects": ["geniesim_2025_target_grasp_object"],
        "target": "geniesim_2025_target_storage_box",
        "relative_position_range": [[-0.06, 0.06], [-0.05, 0.05], [-0.12, 0.12]]
    },
    "result_code": 1
}
```

---

## References

- Checker implementation: `common/data_filter/runtime_checker/`
- Filter rules implementation: `common/data_filter/filter_rules/data_filter.py`
- Action implementation: `client/planner/action/`
