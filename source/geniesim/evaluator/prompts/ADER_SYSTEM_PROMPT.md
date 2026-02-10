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
- `ActionSetWaitSome`: Partial wait action set, exits when a specified number of internal actions complete

**Control Action Classes**
- `ActionWaitForTime`: Time wait action, similar to sleep but does not block threads
- `TimeOut`: Timeout verification action, checks if specified time limit is exceeded (scores 0 if timeout)
- `StepOut`: Step limit verification action, checks if specified step limit is exceeded (scores 0 if exceeded)

## III. Evaluation Action Capabilities

### 3.1 Spatial Relationship Evaluation

**Ontop (Above Relationship)**
- Function: Detects whether one object is located above another object
- Syntax: `"Ontop": "active_obj|passive_obj"`
- Principle: Calculates the intersection area of two objects' projections on the XY plane, determines if it exceeds the threshold (default 50%), while checking Z-axis height difference

**Inside (Interior Relationship)**
- Function: Detects whether one object is located inside another object
- Syntax: `"Inside": "active_obj|passive_obj|scale_factor"`
- Principle: Uses a scale factor to adjust the container's bounding box, checks if the active object's center point is within the adjusted bounding box, requires 3 consecutive frames to satisfy the condition

**Cover (Coverage Relationship)**
- Function: Detects whether object A covers object B
- Syntax: `"Cover": "active_obj|passive_obj"`

**OnShelf (Shelf Placement)**
- Function: Detects whether an object is located in a specific area
- Syntax: `"OnShelf": "obj_id|target_id|bbox|height"`

**OnShelfCurobo (Curobo Shelf Placement)**
- Function: Uses Curobo algorithm to detect whether an object is located in a specific area
- Syntax: `"OnShelfCurobo": "obj_id|target_id|bbox|height"`

**Onfloor (Ground Detection)**
- Function: Detects whether the specified object is below the reference height, exits evaluation (failure) if below
- Syntax: `"Onfloor": "obj_id|ref_z"`
- Type: ActionCancelBase (cancel action class)

**Upright (Upright Detection)**
- Function: Detects whether an object is upright
- Syntax: `"Upright": "obj_id|angle_thresh"`

### 3.2 Grasping and Manipulation Evaluation

**PickUpOnGripper (Grasp Detection)**
- Function: Detects whether the gripper successfully grasps an object
- Syntax: `"PickUpOnGripper": "object|gripper_id"`
- Principle: Two-phase detection
  - Phase 1: Detects if the object's Z-axis is lifted beyond threshold (0.05m)
  - Phase 2: Detects if the relative position and rotation between object and gripper are stable (2 consecutive frames satisfying position threshold 0.06m and rotation threshold 10 degrees)
- Scoring: Completing phase 2 scores 1 point, only completing phase 1 scores 0.5 points, incomplete scores 0 points

**Follow (Following Detection)**
- Function: Detects whether left/right gripper follows a specific object within the range defined by bounding box
- Syntax: `"Follow": "obj_id|bbox|gripper_id"`
- Principle: Creates an extended bounding box based on object bounding box and specified dimensions, checks if gripper center point is within the bounding box
- Parameters: bbox format is `"[x,y,z]"`, representing bounding box dimensions

**Approach (Approach Detection)**
- Function: Detects whether the gripper approaches the target position
- Syntax: `"Approach": "x|y|z"`

**GripperPassing (Gripper Passing)**
- Function: Detects object passing between grippers
- Syntax: `"GripperPassing": "obj_id|is_passing"`

**LiftUp (Lift Up Detection)**
- Function: Detects whether an object is lifted up
- Syntax: `"LiftUp": "obj_id|z_thresh"`

### 3.3 Joint and State Evaluation

**PushPull (Push-Pull Detection)**
- Function: Detects whether the sliding joint of a jointed object is within threshold range (used to determine if drawer-like objects are open or closed)
- Syntax: `"PushPull": "obj_id|thresh_min|thresh_max|joint_index"`
- Principle: Queries the object's prismatic joint position, checks if the value at the specified joint index is within [min, max] range, requires 2 consecutive frames to satisfy the condition

**TriggerAction (Trigger Action)**
- Function: Triggers a specific action
- Syntax: `"TriggerAction": "action_name|param"`

### 3.4 Special Scenario Evaluation

**FluidInside (Fluid Interior Detection)**
- Function: Detects whether fluid is inside a container
- Syntax: `"FluidInside": "fluid_obj|container_obj"`

**CheckParticleInBBox (Particle Bounding Box Detection)**
- Function: Detects whether particles are within the specified bounding box
- Syntax: `"CheckParticleInBBox": "particle_id|bbox_coords"`
- Parameters: bbox_coords format is `"x_min,y_min,z_min,x_max,y_max,z_max"`

**CheckStainClean (Stain Cleaning Detection)**
- Function: Detects whether surface stains are cleaned
- Syntax: `"CheckStainClean": "surface_id|threshold"`

**Pass2People (Pass to People)**
- Function: Detects whether an object is passed to a person
- Syntax: `"Pass2People": "obj_id|person_id"`

**VLM (Vision Language Model Detection)**
- Function: Determines whether a task is completed through vision language model
- Syntax: `"VLM": "task_id"`

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
              "VLM": "task_id|instruction"
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
