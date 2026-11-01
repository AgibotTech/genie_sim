# ADER (Action Domain Evaluation Rule) System Functions and Capabilities

## I. System Overview

ADER (Action Domain Evaluation Rule) is a benchmark evaluation framework for robot simulation environments. The system enables real-time monitoring, evaluation, and verification of robot task execution through structured action definitions and rule configurations. ADER adopts an Action Domain-based abstraction approach to organize evaluation logic and supports defining complex evaluation rules through JSON configuration files.

## II. Core Architecture

### 2.1 Main Components

**AderEnv (Environment Class)**
- Manages action executors (ActionManager)
- Provides interface with simulation API core (APICore)
- Supports action start, cancel, pause, and resume operations
- Maintains task execution state and time tracking

**AderTask (Task Class)**
- Parses task definitions from JSON configuration files
- Builds action execution trees
- Tracks task progress and state updates
- Supports task reset and dynamic updates

**ActionManager (Action Manager)**
- Manages multiple action slots, supporting concurrent execution
- Provides action lifecycle management (start, pause, resume, stop)
- Automatically cleans up completed actions
- Supports querying and checking action status by slot

**ActionBase (Action Base Class)**
- Defines action state machine: INIT (initialization) → RUNNING (running) → PAUSED (paused) → FINISHED (completed) → CANCELED (canceled)
- Supports event callback mechanism (STARTED, PAUSED, RESUMED, FINISHED, CANCELED)
- Provides progress information tracking and placeholder replacement functionality
- Implements non-blocking time update mechanism

### 2.2 Action Type System

**Base Action Classes**
- `ActionBase`: Base class for all actions, defining common behaviors
- `EvaluateAction`: Base class for evaluation actions, providing query interfaces for object pose, bounding box, joint information, etc.
- `EvalExitAction`: Base class for evaluation exit actions, automatically cancels evaluation upon completion

**Composite Action Classes**
- `ActionList`: Sequential action list execution, internal actions execute in order until all are completed
- `ActionSetWaitAny`: Conditional wait action set, exits when any internal action completes
- `ActionSetWaitAll`: Full wait action set, exits only when all internal actions complete
- `ActionSetWaitSome_N`: Partial wait action set, exits when at least `N` internal actions complete (the count is the integer suffix in the key, e.g. `ActionSetWaitSome_2`; default 1)

**Control Action Classes**
- `ActionWaitForTime`: Time wait action, similar to sleep but does not block threads
- `TimeOut`: Timeout verification action, checks if specified time limit is exceeded (scores 0 if timeout)
- `StepOut`: Step limit verification action, checks if specified step limit is exceeded (scores 0 if exceeded)

## III. Evaluation Action Capabilities

The checkers below are the complete, currently-supported set (the authority is
`plugins/ader/action/action_parsing.py`). Use only these keys. Arguments are
pipe-separated (`|`) unless stated otherwise. "N frames" means the condition must
hold for N consecutive simulation frames before it passes.

### 3.1 Spatial Relationship Evaluation

**Ontop (Above Relationship)**
- Function: Detects whether one object sits on top of another
- Syntax: `"Ontop": "active_obj|passive_obj"`
- Principle: Vertical gap between active bottom and passive top ≤ 0.02 m and XY projection overlap ≥ 50% (1 frame)

**Inside (Interior Relationship)**
- Function: Detects whether one object is inside another
- Syntax: `"Inside": "active_obj|passive_obj|scale_factor"`
- Principle: Rescales the passive object's bounding box by `scale_factor` about its center and checks the active object's center is inside it (2 consecutive frames)

**InBBox (Fixed Box Containment)**
- Function: Detects whether an object stays inside a fixed world-frame box
- Syntax: `"InBBox": "object_id|center_x,center_y,center_z|len_x,len_y,len_z"`
- Parameters: box center and full side lengths in world coordinates (2 consecutive frames)

**Cover (Coverage Relationship)**
- Function: Detects whether the active object covers the passive object
- Syntax: `"Cover": "active_obj|passive_obj"`
- Principle: Active center-of-gravity ≤ passive top + 0.002 m and XY overlap ratio ≥ 50% (1 frame)

**Stack (Stack Alignment)**
- Function: Detects whether multiple objects' XY centers are all aligned with the first object (stack-alignment tasks)
- Syntax: `"Stack": "[a,b,c]|[x,y]"`
- Parameters: `[a,b,c]` = object IDs (at least 2, first is the reference), `[x,y]` = XY center tolerance in meters (each defaults to 0.05 m if omitted) (2 consecutive frames)

**OnShelf (Shelf / Region Placement)**
- Function: Detects whether an object is placed within a region defined relative to a target object
- Syntax: `"OnShelf": "obj_id|target_id|[x_min,x_max,y_min,y_max,z_min,z_max]|height"`
- Parameters: bbox is a relative-offset range vs `target_id`; `height` is subtracted from the object Z before the z-range test (must hold >5 consecutive frames)

**Onfloor (Ground / Drop Detection)**
- Function: Fires when the object drops to the floor — a failure condition that cancels evaluation
- Syntax: `"Onfloor": "obj_id|ref_z"`
- Principle: Fires when the object Z is within 0.3 m of `ref_z`; sets SCORE = 0
- Type: cancel/exit action

**Upright (Upright Detection)**
- Function: Detects whether an object is upright
- Syntax: `"Upright": "obj_id|tilt_threshold_deg"` or `"obj_id|tilt_threshold_deg|allow_flipped"`
- Parameters: object's local +Y must align with world +Z within `tilt_threshold_deg` degrees (default 15); `allow_flipped=true` also accepts the 180°-flipped orientation (default false) (1 frame)

**RelativePosition (Directional / Alignment Relationship)**
- Function: Detects a spatial relation between two objects in the robot body frame
- Syntax: `"RelativePosition": "obj_A|obj_B|relation"` or `"obj_A|obj_B|relation|threshold"`
- Parameters: `relation` ∈ `leftof, rightof, topof, bottomof, aligned_x, aligned_y, aligned_z`; `threshold` (default 0.05) is used only by the `aligned_*` relations (2 consecutive frames)

### 3.2 Grasping and Manipulation Evaluation

**PickUpOnGripper (Grasp Detection)**
- Function: Detects whether the gripper has picked up an object
- Syntax: `"PickUpOnGripper": "object|gripper_id"`
- Parameters: `object` may be a single id or comma-separated list (passes if any is picked); `gripper_id` containing `right` → right gripper, otherwise left
- Principle: Object lifted strictly > 0.02 m above its first sampled Z and strictly < 0.2 m from the gripper reference (the robot config's `end_effector_prim_path`). SCORE = 1 on the first passing evaluator sample; no contact or closure is checked

**StableGrasp (Stable Grasp Detection)**
- Function: Detects whether an object is held stably and moves rigidly with the gripper
- Syntax: `"StableGrasp": "object|gripper_id"` (optional trailing `|distance|pos_diff|rot_diff_rad`)
- Parameters: defaults distance 0.1 m, pos_diff 0.02 m, rot_diff 0.1 rad; requires 2 consecutive stable frames while both object and gripper are moving

**Follow (Following Detection)**
- Function: Detects whether the gripper is within a bounding box around an object
- Syntax: `"Follow": "object|bbox|gripper_id"`
- Parameters: bbox is `"[x,y,z]"` box size; `object` may be comma-separated (passes if gripper is in any); `gripper_id` containing `right` → right else left (1 frame)

**Approach (Approach Detection)**
- Function: Detects whether the right gripper reaches a target world point
- Syntax: `"Approach": "x|y|z"`
- Principle: Right gripper center within 0.01 m of the target on every axis (1 frame)

**GripperPassing (Bimanual Handover)**
- Function: Detects an object handover between the two grippers
- Syntax: `"GripperPassing": "obj_prim|reverse"`
- Parameters: `reverse=true` = right→left handover, otherwise left→right; passes when the transfer occurs with the object on the line between the gripper centers

**LiftUp (Lift Detection)**
- Function: Detects whether an object has been lifted
- Syntax: `"LiftUp": "object|lift_threshold"`
- Parameters: object Z risen by at least `lift_threshold` meters above its first Z (default 0.05) (1 frame)

### 3.3 Joint, State and Navigation Evaluation

**PushPull (Joint Range Detection)**
- Function: Detects whether an articulated object's prismatic joint is within a range (drawer/door open or closed)
- Syntax: `"PushPull": "obj_id|thresh_min|thresh_max"` or `"obj_id|thresh_min|thresh_max|joint_index"`
- Parameters: `joint_index` selects the prismatic joint (default 0); passes when its position is within `[thresh_min, thresh_max]` (2 consecutive frames)

**TriggerAction (Trigger State)**
- Function: Detects whether a prim's trigger state matches an expected value
- Syntax: `"TriggerAction": "prim_path|expected_response"` (1 frame)

**ChassisAtTarget (Base Pose)**
- Function: Detects whether the robot chassis has reached a target pose
- Syntax: `"ChassisAtTarget": "[x,y,yaw]|[x_thresh,y_thresh,yaw_thresh]"`
- Parameters: yaw and yaw threshold in degrees; passes when within the XY box and yaw threshold (1 frame)

### 3.4 Special Scenario Evaluation

**FluidInside (Fluid Containment)**
- Function: Detects whether enough fluid particles are inside a container
- Syntax: `"FluidInside": "container_obj|object_info_dir"`
- Parameters: container AABB is read from `object_info_dir`; passes when > 50 particles are inside

**CheckParticleInBBox (Particle Count)**
- Function: Detects whether particles have left a region (count drops below a threshold)
- Syntax: `"CheckParticleInBBox": "threshold|x_min,y_min,z_min,x_max,y_max,z_max"`
- Parameters: passes when the particle count inside the box is **below** `threshold`

**CheckStainClean (Stain Cleaning)**
- Function: Detects whether stains have been cleaned
- Syntax: `"CheckStainClean": "stain_prim_path|threshold"`
- Parameters: passes when the number of visible stain meshes is **below** `threshold`

**PlaceOnRivet (Precise Pose Placement)**
- Function: Detects whether a workpiece is placed at a target relative pose and held still
- Syntax: `"PlaceOnRivet": "active_obj|passive_obj|rel_x,rel_y,rel_z|qw,qx,qy,qz|xy_tol|z_tol|orient_tol|still_thresh|still_steps"`
- Parameters: target relative position/orientation of `active_obj` vs `passive_obj`; tolerances optional with defaults xy_tol 0.02 m, z_tol 0.01 m, orient_tol 0.15 rad, still_thresh 0.02, still_steps 15

**VLM (Vision-Language Model Judgement)**
- Function: Uses a VLM to judge task completion from rendered observations
- Syntax: `"VLM": "task_id|interval"`
- Parameters: `task_id` selects the task description; the VLM scores every `interval` updates and passes when the (fractional) score reaches 1.0

### 3.5 Composite Rule Aggregation

**MixedRules (Multi-rule Aggregation)**
- Function: Aggregates several sub-checkers with OR (partial credit) or AND (all required) logic
- Syntax: a JSON object (not a pipe string):
  ```json
  "MixedRules": {
    "rules": [
      { "name": "Inside", "params": "cup|tray|1.0" },
      { "name": "Upright", "params": "cup|15" }
    ],
    "check_interval": 1,
    "mode": "or"
  }
  ```
- Parameters: `mode="or"` scores `passed/total` (e.g. 0.5); `mode="and"` scores 1.0 only when all pass; `check_interval` (default 1) evaluates every N updates. Supported sub-checker `name`s: `Inside, PushPull, Upright, RelativePosition, Ontop, Onfloor, Cover, LiftUp, InBBox, Stack, StableGrasp`.

## IV. Configuration Format

### 4.1 JSON Configuration File Structure Example

```json
{
  "Acts": [
    {
      "ActionList": [
        {
          "ActionSetWaitAny": [
            {
              "VLM": "task_id|30"
            },
            {
              "StepOut": 300
            }
          ]
        }
      ]
    }
  ],
  "Init": [],
  "Objects": [],
  "Problem": "pack_in_the_supermarket"
}
```

### 4.2 Configuration Field Descriptions

- **Acts**: Action definition array, containing the main execution logic of the task
- **Init**: Initialization configuration (optional)
- **Objects**: Object definitions (optional)
- **Problem**: Problem name, used to identify task type

### 4.3 Action Parameter Format

Most custom actions use pipe separator (`|`) to separate parameters:
- Object names: Use object ID directly, e.g., `"beverage_bottle_002"`
- Bounding boxes: Use list format string, e.g., `"[0.2,0.2,0.2]"`
- Numeric parameters: Use numbers directly, e.g., `"120"`, `"0.0"`, `"1"`
- Boolean parameters: Use strings `"true"` or `"false"`

### 4.4 Placeholder Support

Actions support placeholder replacement, format is `{@placeholder_name}`, which can be dynamically replaced with actual values at runtime.

## V. Execution Flow

### 5.1 Task Initialization

1. Search for configuration file based on task name and instance number (`problem{instance}.json`)
2. If file does not exist, use default configuration file (`default_problem.json`)
3. Parse JSON configuration, build action execution tree
4. Create AderTask object, associate with AderEnv

### 5.2 Action Execution

1. Call `do_eval_action()` to start evaluation action
2. Call `action_update()` in simulation loop to update action state
3. ActionManager manages lifecycle of all actions
4. Each action executes corresponding update logic based on its type and state

### 5.3 State Update Mechanism

- Each action receives time delta (delta_time) in the `update()` method
- Actions determine completion based on their own logic (`_is_done()`)
- Completed actions automatically trigger FINISHED event
- Composite actions determine their own completion state based on child actions' completion status

### 5.4 Progress Tracking

- Each action maintains a `progress_info` dictionary, recording execution state
- Supports STATUS (state) and SCORE (scoring) fields
- Task progress is updated in real-time through the `update_progress()` method

## VI. Use Cases

### 6.1 Typical Applications

1. **Grasping Task Evaluation**: Use PickUpOnGripper, LiftUp to detect successful grasping, use Onfloor to detect object dropping
2. **Placement Task Evaluation**: Use Inside, Ontop, OnShelf to detect object placement position
3. **Manipulation Task Evaluation**: Use PushPull to detect drawer opening/closing, use Follow to detect approach actions
4. **Complex Task Evaluation**: Combine multiple actions, use ActionSetWaitAny to implement multi-path success conditions

### 6.2 Best Practices

1. **Timeout Protection**: Use Timeout in critical action combinations to prevent infinite waiting
2. **Failure Detection**: Use cancel actions like Onfloor, StepOut to timely detect failure situations
3. **Conditional Combination**: Use ActionSetWaitAny to implement conditional logic like "success or timeout"
4. **Sequential Execution**: Use ActionList to ensure actions execute in correct order

## VII. Extension Capabilities

### 7.1 Custom Action Development

The system supports creating custom evaluation actions by inheriting from `EvaluateAction` or `ActionBase`:
- Implement `_is_done()` method to define completion conditions
- Implement `update()` method to define per-frame update logic
- Implement `update_progress()` method to update progress information
- Register new action parsing logic in `action_parsing.py`

### 7.2 Plugin System

The system is designed as an extensible plugin architecture, supporting dynamic addition of new evaluation rules and action types.

## VIII. Technical Features

1. **Non-blocking Execution**: All actions adopt non-blocking design, will not block the simulation main loop
2. **State Machine Management**: Complete action state machine, supporting pause, resume, cancel and other operations
3. **Real-time Evaluation**: Updates action state every frame, provides real-time feedback on task progress
4. **Flexible Composition**: Supports arbitrary nesting and combination of actions, building complex evaluation logic
5. **Fault Tolerance Mechanism**: Supports timeout, step limit and other fault tolerance detection, preventing task deadlock

## IX. Summary

ADER is a powerful, flexible, and extensible robot task evaluation framework. Through structured action definitions and rule configurations, it achieves real-time monitoring and evaluation of complex robot tasks. The system supports various spatial relationship detection, grasping operation verification, joint state checking and other evaluation capabilities, and provides rich composite action types, capable of meeting the requirements of various robot simulation evaluation scenarios.
