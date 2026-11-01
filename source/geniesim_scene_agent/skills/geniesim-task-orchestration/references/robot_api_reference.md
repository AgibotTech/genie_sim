# Complete RobotAPI parameter reference

This self-contained snapshot documents the inspected collection runtime API.
It requires no external prompt or source checkout to read. Execute tasks with
an installed compatible runtime, NumPy, and an Isaac Sim RPC server. The
asset library remains externally configured. The current installed version
may differ: check callable signatures locally when adapting a different version.

Import public objects from `client.llm_gen.robot_api`. The reference includes
every public RobotAPI method, the constructor, all nine action configuration
constructors, RobotAPIConfig, dual specifications, action extensions, and the
supporting interfaces needed by bundled task examples. Private underscore
helpers are implementation details. Type hints are preserved in signatures;
actual return differences are stated explicitly in each method's behavior.

## Contents

- [Shared conventions](#shared-conventions)
- [RobotAPI methods](#robotapi-methods)
- [Action configuration constructors](#action-configuration-constructors)
- [RobotAPIConfig](#robotapiconfig)
- [Dual specifications](#dual-specifications)
- [Supported extension parameters](#supported-extension-parameters)
- [Coordinate helpers](#coordinate-helpers)
- [Client interfaces used by tasks](#client-interfaces-used-by-tasks)

## Shared conventions

World position is xyz in meters; +Z is up. Scene metadata stores xyzw, while
runtime object pose and robot initialization use wxyz. `quat2mat` is xyzw;
`quat2mat_wxyz` is wxyz. Rotation utilities differ in angle units; see below.

An object ID is a layout instance key, an asset ID is layout[id].usd, and a
prim path is /World/Objects/<id>/entity[/<part>]. A prim origin is not guaranteed
to be a box center. Geometry/scale/interior/shelf measurements may need the
bundled mesh/support tools rather than metadata size getters.

Most high-level action methods return bool. `dual_act`, `run_stage`,
`dual_run_stage`, and `run_simple_action` return two-value tuples. `step`
returns a different two-value tuple. Void gripper methods are not booleans.
Action success still requires task-specific physical completion checks.

Pose goals are actor/object transforms `(N,4,4)`; actor="gripper" makes them
gripper goals. Bimanual rigid goals are leader-gripper targets. Ordered
trajectories instead use `(N,7)` / `(M,N,7)` xyz+wxyz arrays. For a held object,
`T_object_to_ee = inv(T_world_object) @ T_world_ee`, so an object target maps to
`T_world_ee_goal = T_world_object_goal @ T_object_to_ee`. Use the current grasp
transform, not a copied example transform.

Named configuration constructor defaults below are authoritative for API
calls; the stages sometimes have different defaults when used directly.
Several constructors use shared mutable list/dict defaults in this version.
Pass fresh configs and lists per action and do not mutate shared defaults.
`to_dict()` merges extra_params after explicit fields; unknown keys need a
documented consumer and do not automatically become supported capabilities.

Read [task contract](task-contract.md), [grasp selection](grasp-selection.md),
and [motion patterns](motion-patterns.md) for workflow and physical semantics.
All example links there resolve inside this repository.

## RobotAPI methods
Method index: `__init__`, `exit`, `stop_recording`, `get_object_pose`, `get_object_prim_path`, `get_articulation_joint_info`, `judge_arm_to_use`, `back_to_origin`, `wait`, `dual_act`, `dual_run_stage`, `handover_ownership_of_attached_object`, `pick_actor`, `place_actor_to_target_actor`, `place_actor_to_target_pose`, `pick_and_place_actor_to_target_actor`, `pick_and_place_actor_to_target_pose`, `rotate_actor`, `move_actor_to_target_pose`, `follow_trajectory`, `bimanual_rigid_move`, `shake`, `turn_to`, `joint_to`, `open`, `get_object_size_in_world_frame`, `get_object_size_in_local_frame`, `add_all_objects_info`, `run_simple_action`, `step`, `run_stage`, `get_grasp_stage`, `get_place_stage`, `get_rotate_stage`, `get_move_stage`, `get_bimanual_rigid_move_stage`, `get_fix_pose_object`, `get_object`, `get_all_objects`, `update_objects`, `add_object_to_objects`, `get_object_usd_id`, `open_gripper`, `close_gripper`, `get_ee_pose`, `get_joint_positions`, `set_joint_positions`, `add_object`, `get_cloth_nodal_positions`, `set_fem_pin`, `release_fem_pin`, `set_rigid_pin`, `release_rigid_pin`, `set_gripper_colliders`, `disable_gripper_colliders`, `enable_gripper_colliders`, `draw_debug_cube`, `clear_debug_cubes`, `attach_obj`, `detach_obj`, `remove_objs_from_obstacle`, `add_objs_to_obstacle`, `set_object_parameter_action`, `solve_ik`, `reset`.

### `RobotAPI.__init__`

```python
RobotAPI.__init__(config: RobotAPIConfig, scene_usd: str, scene_info: str, client_host: str='localhost:50051', from_robot=None)
```

Constructs RobotAPI, loads scene metadata/indexed objects and interactions, initializes the robot, resets client state, sets task descriptions/metrics, and starts recording if auto_record. Returns an instance; construction/import can fail for invalid assets, paths, robot profile or unavailable RPC. Initialization includes real simulator side effects.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `config` | `RobotAPIConfig` | `required` | RobotAPIConfig for construction, BimanualRigidMoveConfig for bimanual actions, or lifecycle dictionary for run_simple_action. See the corresponding section. |
| `scene_usd` | `str` | `required` | Path to the selected composed scene USD/USDA file; all asset references must resolve. |
| `scene_info` | `str` | `required` | Path to the selected scene_info.json file; the constructor loads it into robot.scene_info. |
| `client_host` | `str` | `'localhost:50051'` | Isaac Sim RPC endpoint as host:port. Robot execution requires a running compatible server. |
| `from_robot` | `RobotAPI or None` | `None` | Previous RobotAPI instance or None. When present, reuse robot initial pose/noise and body initialization information. |

### `RobotAPI.exit`

```python
RobotAPI.exit()
```

Returns None after requesting runtime exit. Runner-owned; do not exit the simulator inside an ordinary task phase.

### `RobotAPI.stop_recording`

```python
RobotAPI.stop_recording()
```

Returns None; stops recording only when this API instance has auto_record enabled.

### `RobotAPI.get_object_pose`

```python
RobotAPI.get_object_pose(actor: str, articulation_part: str=None) -> Tuple[np.ndarray, np.ndarray]
```

Returns `(position_xyz, quaternion_wxyz)` arrays. Reads the entity/part world prim transform, not mesh/bounding-box center. Articulated assets default to body. Missing index entries or prims may raise rather than return None.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `actor` | `str` | `required` | Manipulated instance object ID or supported entity prim path; `"gripper"` where documented. `pick_actor` also accepts OmniObject. An arbitrary mesh path is not an indexed object. |
| `articulation_part` | `str` | `None` | Relative part path below entity. None means the default reference; pose lookup defaults articulated assets to `body`. |

### `RobotAPI.get_object_prim_path`

```python
RobotAPI.get_object_prim_path(object_name: str, articulation_part: str=None) -> str
```

Returns the constructed string `/World/Objects/<id>/entity[/<part>]`; this does not validate prim existence.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `object_name` | `str` | `required` | Instance object ID used to construct the entity path; no existence check is performed. |
| `articulation_part` | `str` | `None` | Relative part path below entity. None means the default reference; pose lookup defaults articulated assets to `body`. |

### `RobotAPI.get_articulation_joint_info`

```python
RobotAPI.get_articulation_joint_info(articulation_object_prim_path: str, joint_prim_path: str=None) -> Dict
```

Returns server dictionary with joint_path, joint_prim_name, body0_path, body1_path, axis, current_angle_deg, lower_limit_deg, upper_limit_deg. Angle fields are degrees. Validate required keys and actual joint before use.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `articulation_object_prim_path` | `str` | `required` | Absolute entity prim path of the articulated object. |
| `joint_prim_path` | `str` | `None` | Absolute joint prim path or None to let the server select a joint. |

### `RobotAPI.judge_arm_to_use`

```python
RobotAPI.judge_arm_to_use(position: List[float]) -> str
```

Returns left/right based on target Y in the robot base frame. Outside +/-0.15 m it is deterministic; inside that band it samples a linear left-arm probability. This is a side heuristic, not an IK/collision test.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `position` | `List[float]` | `required` | World xyz position, shape `(3,)`, meters; for arm selection transformed into the robot base frame. |

### `RobotAPI.back_to_origin`

```python
RobotAPI.back_to_origin(arm: str='') -> bool
```

Returns bool. Empty arm resets left then right. Tries AvoidObs reset motion, then Simple fallback; this fallback is less restrictive. Reset movement can disturb a held object, so verify completion afterward.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `arm` | `str` | `''` | Arm selector: `"left"` or `"right"`; collider helpers additionally accept `"both"`. No length unit. |

### `RobotAPI.wait`

```python
RobotAPI.wait(time: float) -> bool
```

Returns True after advancing int(seconds*60) simulator steps. It waits simulation time, not wall-clock sleep.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `time` | `float` | `required` | Simulation wait duration in seconds, converted to int(time*60) steps. Use a nonnegative duration. |

### `RobotAPI.dual_act`

```python
RobotAPI.dual_act(left: Optional[DualSpec]=None, right: Optional[DualSpec]=None, action_description: Optional[Dict[str, str]]=None, extra_checkers: Optional[List[Callable[[], bool]]]=None) -> Tuple[bool, str]
```

Returns `(stage_success, store_name)`. Uses synchronized stage execution and shared rollback, not two independent threads. Per-arm Config checks and shared extra_checkers are separate. Both specs None returns `(True, current_state_name)` without motion. Unpack the result before testing success.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `left` | `Optional[DualSpec]` | `None` | Left-arm DualSpec or None to leave that arm inactive. |
| `right` | `Optional[DualSpec]` | `None` | Right-arm DualSpec or None to leave that arm inactive. |
| `action_description` | `Optional[Dict[str, str]]` | `None` | Nonempty action text mapping with `zh` and `en` keys. Ordinary actions add the arm prefix. Checkers use actual state, not this text. |
| `extra_checkers` | `Optional[List[Callable[[], bool]]]` | `None` | List of zero-argument callables returning bool. Read live state and avoid side effects. All required checks must pass; failures/exceptions fail the stage. |

### `RobotAPI.dual_run_stage`

```python
RobotAPI.dual_run_stage(left_stage: Optional[Stage], right_stage: Optional[Stage], objects: Dict[str, OmniObject], action_description: Optional[Dict[str, str]]=None, extra_checkers: Optional[List[Callable[[], bool]]]=None) -> Tuple[bool, str]
```

Returns `(stage_success, store_name)` from synchronized lower-level stages. Stores a common state and executes bounded attempts with coordinated dispatch and rollback. Here action_description is already recorder-shaped (`action_text`, `english_action_text`); dual_act converts its zh/en mapping before calling this executor. High-level authoring should normally use dual_act.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `left_stage` | `Optional[Stage]` | `required` | Already constructed left Stage or None; lower-level dual executor input. |
| `right_stage` | `Optional[Stage]` | `required` | Already constructed right Stage or None; lower-level dual executor input. |
| `objects` | `Dict[str, OmniObject]` | `required` | Mutable dictionary of cache ID to OmniObject, including gripper for stage operations. Pose snapshots are refreshed by execution. |
| `action_description` | `Optional[Dict[str, str]]` | `None` | Nonempty action text mapping with `zh` and `en` keys. Ordinary actions add the arm prefix. Checkers use actual state, not this text. |
| `extra_checkers` | `Optional[List[Callable[[], bool]]]` | `None` | List of zero-argument callables returning bool. Read live state and avoid side effects. All required checks must pass; failures/exceptions fail the stage. |

### `RobotAPI.handover_ownership_of_attached_object`

```python
RobotAPI.handover_ownership_of_attached_object(from_arm: str) -> bool
```

Opens the source arm, attaches `self.attached_obj_id` to the other arm, and returns True without checking the attach response. Requires the stored attached object ID to be valid. This wrapper is not a physical regrasp or transfer-success checker.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `from_arm` | `str` | `required` | Arm currently owning the attached object; ownership is transferred to the other arm. |

### `RobotAPI.pick_actor`

```python
RobotAPI.pick_actor(actor: Union[OmniObject, str], arm: str='right', grasp_config: GraspConfig=GraspConfig(), part: str=None) -> bool
```

Returns bool after grasp/contact/close and configured lift. Builds and runs PickStage, with optional part reference. Custom grasps can bypass missing annotations. Failure may roll back the stage; validate stable grasp beyond closed-gripper state.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `actor` | `Union[OmniObject, str]` | `required` | Manipulated instance object ID or supported entity prim path; `"gripper"` where documented. `pick_actor` also accepts OmniObject. An arbitrary mesh path is not an indexed object. |
| `arm` | `str` | `'right'` | Arm selector: `"left"` or `"right"`; collider helpers additionally accept `"both"`. No length unit. |
| `grasp_config` | `GraspConfig` | `GraspConfig()` | GraspConfig for contact candidates, filters, approach/lift, descriptions and checks. Pass a fresh instance per action. |
| `part` | `str` | `None` | Relative actor part path such as `body/hole`; changes the manipulated reference frame. None uses the object default. |

### `RobotAPI.place_actor_to_target_actor`

```python
RobotAPI.place_actor_to_target_actor(actor: str, arm: str, target_actor: str, place_config: PlaceConfig=PlaceConfig()) -> bool
```

Returns bool. Requires a held object and compatible receiving interaction labels; executes place/release or insertion chosen by PlaceConfig. Re-read a moving container pose before constructing the stage.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `actor` | `str` | `required` | Manipulated instance object ID or supported entity prim path; `"gripper"` where documented. `pick_actor` also accepts OmniObject. An arbitrary mesh path is not an indexed object. |
| `arm` | `str` | `required` | Arm selector: `"left"` or `"right"`; collider helpers additionally accept `"both"`. No length unit. |
| `target_actor` | `str` | `required` | Receiving indexed instance ID for high-level calls; OmniObject for stage construction. Its receiving annotations must match the goal. |
| `place_config` | `PlaceConfig` | `PlaceConfig()` | PlaceConfig for release/placement or insertion when pre_insert_offset is nonzero. Pass a fresh instance per action. |

### `RobotAPI.place_actor_to_target_pose`

```python
RobotAPI.place_actor_to_target_pose(actor: str, arm: str, target_pose: np.ndarray, place_config: PlaceConfig=PlaceConfig()) -> bool
```

Returns bool after placing/releasing the held actor at one valid world object pose candidate. Candidates describe the actor, not the gripper. The stage maps through the current grasp transform.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `actor` | `str` | `required` | Manipulated instance object ID or supported entity prim path; `"gripper"` where documented. `pick_actor` also accepts OmniObject. An arbitrary mesh path is not an indexed object. |
| `arm` | `str` | `required` | Arm selector: `"left"` or `"right"`; collider helpers additionally accept `"both"`. No length unit. |
| `target_pose` | `np.ndarray` | `required` | World actor target candidates `(N,4,4)`, except bimanual calls which target the leader end effector. Each matrix has a valid rotation and bottom row `[0,0,0,1]`. |
| `place_config` | `PlaceConfig` | `PlaceConfig()` | PlaceConfig for release/placement or insertion when pre_insert_offset is nonzero. Pass a fresh instance per action. |

### `RobotAPI.pick_and_place_actor_to_target_actor`

```python
RobotAPI.pick_and_place_actor_to_target_actor(actor: str, target_actor: str, arm: str='right', grasp_config: GraspConfig=GraspConfig(), place_config: PlaceConfig=PlaceConfig()) -> bool
```

Returns bool from sequential pick then annotation-based place. A place failure also restores the state saved before grasp. The task goal still needs a checker.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `actor` | `str` | `required` | Manipulated instance object ID or supported entity prim path; `"gripper"` where documented. `pick_actor` also accepts OmniObject. An arbitrary mesh path is not an indexed object. |
| `target_actor` | `str` | `required` | Receiving indexed instance ID for high-level calls; OmniObject for stage construction. Its receiving annotations must match the goal. |
| `arm` | `str` | `'right'` | Arm selector: `"left"` or `"right"`; collider helpers additionally accept `"both"`. No length unit. |
| `grasp_config` | `GraspConfig` | `GraspConfig()` | GraspConfig for contact candidates, filters, approach/lift, descriptions and checks. Pass a fresh instance per action. |
| `place_config` | `PlaceConfig` | `PlaceConfig()` | PlaceConfig for release/placement or insertion when pre_insert_offset is nonzero. Pass a fresh instance per action. |

### `RobotAPI.pick_and_place_actor_to_target_pose`

```python
RobotAPI.pick_and_place_actor_to_target_pose(actor: str, target_pose: np.ndarray, arm: str='right', grasp_config: GraspConfig=GraspConfig(), place_config: PlaceConfig=PlaceConfig()) -> bool
```

Returns bool from sequential pick then explicit pose placement. Empty/None target pool returns False. A place failure also restores the state saved before grasp.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `actor` | `str` | `required` | Manipulated instance object ID or supported entity prim path; `"gripper"` where documented. `pick_actor` also accepts OmniObject. An arbitrary mesh path is not an indexed object. |
| `target_pose` | `np.ndarray` | `required` | World actor target candidates `(N,4,4)`, except bimanual calls which target the leader end effector. Each matrix has a valid rotation and bottom row `[0,0,0,1]`. |
| `arm` | `str` | `'right'` | Arm selector: `"left"` or `"right"`; collider helpers additionally accept `"both"`. No length unit. |
| `grasp_config` | `GraspConfig` | `GraspConfig()` | GraspConfig for contact candidates, filters, approach/lift, descriptions and checks. Pass a fresh instance per action. |
| `place_config` | `PlaceConfig` | `PlaceConfig()` | PlaceConfig for release/placement or insertion when pre_insert_offset is nonzero. Pass a fresh instance per action. |

### `RobotAPI.rotate_actor`

```python
RobotAPI.rotate_actor(actor: str, target_pose: np.ndarray, arm: str='right', rotate_config: RotateConfig=RotateConfig())
```

Returns bool. Uses the candidate rotations to rotate a held actor under constrained-axis settings; translations are not placement goals. It does not perform a release.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `actor` | `str` | `required` | Manipulated instance object ID or supported entity prim path; `"gripper"` where documented. `pick_actor` also accepts OmniObject. An arbitrary mesh path is not an indexed object. |
| `target_pose` | `np.ndarray` | `required` | World actor target candidates `(N,4,4)`, except bimanual calls which target the leader end effector. Each matrix has a valid rotation and bottom row `[0,0,0,1]`. |
| `arm` | `str` | `'right'` | Arm selector: `"left"` or `"right"`; collider helpers additionally accept `"both"`. No length unit. |
| `rotate_config` | `RotateConfig` | `RotateConfig()` | RotateConfig for orientation constraints and checks. Candidate translations are not the rotation goal. |

### `RobotAPI.move_actor_to_target_pose`

```python
RobotAPI.move_actor_to_target_pose(actor: str, target_pose: np.ndarray, arm: str='right', move_config: MoveConfig=MoveConfig(), part: Optional[str]=None)
```

Returns bool. Held-object target poses preserve the current object-to-gripper transform; actor=gripper targets the EE directly. part changes the actor goal reference. Does not release unless a gripper command requests it. Empty/None targets return False.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `actor` | `str` | `required` | Manipulated instance object ID or supported entity prim path; `"gripper"` where documented. `pick_actor` also accepts OmniObject. An arbitrary mesh path is not an indexed object. |
| `target_pose` | `np.ndarray` | `required` | World actor target candidates `(N,4,4)`, except bimanual calls which target the leader end effector. Each matrix has a valid rotation and bottom row `[0,0,0,1]`. |
| `arm` | `str` | `'right'` | Arm selector: `"left"` or `"right"`; collider helpers additionally accept `"both"`. No length unit. |
| `move_config` | `MoveConfig` | `MoveConfig()` | MoveConfig for holding/gripper movement, path constraints and optional gripper command. Pass a fresh instance per action. |
| `part` | `Optional[str]` | `None` | Relative actor part path such as `body/hole`; changes the manipulated reference frame. None uses the object default. |

### `RobotAPI.follow_trajectory`

```python
RobotAPI.follow_trajectory(actor: str, waypoints: np.ndarray, arm: str='right', trajectory_config: FollowTrajectoryConfig=FollowTrajectoryConfig(), part: Optional[str]=None) -> bool
```

Returns bool from an ordered path execution, using alternate path candidates if supplied. Object/part waypoints are mapped to EE waypoints with the current grasp transform. Validates arm, config type, nonempty description and noise options; invalid inputs can raise ValueError. Preserves waypoint order and does not place/release automatically.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `actor` | `str` | `required` | Manipulated instance object ID or supported entity prim path; `"gripper"` where documented. `pick_actor` also accepts OmniObject. An arbitrary mesh path is not an indexed object. |
| `waypoints` | `np.ndarray` | `required` | Ordered world xyz+wxyz path `(N,7)`, or alternative paths `(M,N,7)`. Object/part goals preserve the current object-to-end-effector transform. |
| `arm` | `str` | `'right'` | Arm selector: `"left"` or `"right"`; collider helpers additionally accept `"both"`. No length unit. |
| `trajectory_config` | `FollowTrajectoryConfig` | `FollowTrajectoryConfig()` | FollowTrajectoryConfig controlling waypoint IK perturbations and post-action checks. A nonempty description is required. |
| `part` | `Optional[str]` | `None` | Relative actor part path such as `body/hole`; changes the manipulated reference frame. None uses the object default. |

### `RobotAPI.bimanual_rigid_move`

```python
RobotAPI.bimanual_rigid_move(target_pose: np.ndarray, config: BimanualRigidMoveConfig=BimanualRigidMoveConfig())
```

Returns bool. Both arms follow the leader goal with fixed initial relative EE transform and both-arm IK filtering. Establish intended contacts/attachments first. Empty/None targets return False. Does not detach/release automatically.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `target_pose` | `np.ndarray` | `required` | World actor target candidates `(N,4,4)`, except bimanual calls which target the leader end effector. Each matrix has a valid rotation and bottom row `[0,0,0,1]`. |
| `config` | `BimanualRigidMoveConfig` | `BimanualRigidMoveConfig()` | RobotAPIConfig for construction, BimanualRigidMoveConfig for bimanual actions, or lifecycle dictionary for run_simple_action. See the corresponding section. |

### `RobotAPI.shake`

```python
RobotAPI.shake(actor: str, angle_ranges: List[List[float]], shake_config: ShakeConfig=ShakeConfig(), arm: str='right', horizontal: bool=True, times: int=2)
```

Returns bool from wrist-twist cycles; currently asserts a G2 profile. Wrist ranges are radians and subject to joint limits. Action text/checks are required for a valid recorded action; shaking does not release the object.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `actor` | `str` | `required` | Manipulated instance object ID or supported entity prim path; `"gripper"` where documented. `pick_actor` also accepts OmniObject. An arbitrary mesh path is not an indexed object. |
| `angle_ranges` | `List[List[float]]` | `required` | List of two-value wrist angle offset ranges in radians, one range per candidate attempt. |
| `shake_config` | `ShakeConfig` | `ShakeConfig()` | ShakeConfig with action description and post-action checks; angle ranges belong to the method, not this config. |
| `arm` | `str` | `'right'` | Arm selector: `"left"` or `"right"`; collider helpers additionally accept `"both"`. No length unit. |
| `horizontal` | `bool` | `True` | True shakes wrist joint6; false uses joint7. This is a wrist choice, not a world-axis motion specification. |
| `times` | `int` | `2` | Number of shake cycles per attempted angle range; positive integer. |

### `RobotAPI.turn_to`

```python
RobotAPI.turn_to(target_point: List[float], turn_body_config: TurnBodyConfig=TurnBodyConfig()) -> bool
```

Returns bool. Currently G2 only: rotates body joint5 to face target X/Y, clamps its joint target to +/-3.2 radians and interpolates by step_deg. Missing description raises ValueError. Check reach/collisions after body motion.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `target_point` | `List[float]` | `required` | World xyz point `(3,)` in meters; turn_to uses X/Y to face it. |
| `turn_body_config` | `TurnBodyConfig` | `TurnBodyConfig()` | TurnBodyConfig defining angular interpolation and checks; nonempty description required. |

### `RobotAPI.joint_to`

```python
RobotAPI.joint_to(joint_names: Union[List[str], str, Dict[str, float]], target_joints: Optional[Dict[str, float]]=None, steps: int=50, delta: bool=False, action_description: Optional[Dict[str, str]]=None) -> bool
```

Returns bool after joint interpolation. Raises ValueError for missing/extra target keys, empty names/targets, dictionary plus redundant target_joints, or nonpositive steps. Configure action text for recorded execution; no implicit task completion check.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `joint_names` | `Union[List[str], str, Dict[str, float]]` | `required` | Robot joint name string/list, or a name-to-target dictionary. Dictionary form must omit target_joints. |
| `target_joints` | `Optional[Dict[str, float]]` | `None` | Name-to-target mapping matching joint_names exactly. Revolute positions are radians; prismatic positions use runtime length units. |
| `steps` | `int` | `50` | Positive interpolation step count for joint_to; no units. |
| `delta` | `bool` | `False` | True interprets joint targets as increments from current values; false uses absolute targets. |
| `action_description` | `Optional[Dict[str, str]]` | `None` | Nonempty action text mapping with `zh` and `en` keys. Ordinary actions add the arm prefix. Checkers use actual state, not this text. |

### `RobotAPI.open`

```python
RobotAPI.open(actor: str, joint_name: str=None, open_config: OpenConfig=OpenConfig(), arm: str='right')
```

Returns bool. Requests GetRevoluteJointTrajectory and follows it with the chosen hand. Grasp/close on the actual handle first. Empty trajectory returns False; missing action description raises ValueError. Does not support generic prismatic drawer opening. Verify joint angle and stationary body with a checker.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `actor` | `str` | `required` | Manipulated instance object ID or supported entity prim path; `"gripper"` where documented. `pick_actor` also accepts OmniObject. An arbitrary mesh path is not an indexed object. |
| `joint_name` | `str` | `None` | Joint prim path for articulation opening, not a robot joint name; None requests the default revolute joint. |
| `open_config` | `OpenConfig` | `OpenConfig()` | OpenConfig selecting revolute limit direction and trajectory/check settings. A nonempty description is required. |
| `arm` | `str` | `'right'` | Arm selector: `"left"` or `"right"`; collider helpers additionally accept `"both"`. No length unit. |

### `RobotAPI.get_object_size_in_world_frame`

```python
RobotAPI.get_object_size_in_world_frame(actor_name: str) -> List[float] | None
```

Returns world AABB extent `[sx,sy,sz]` (list) or None. Constructs box vertices from indexed local size and current rotation; does not scan mesh, interior or instance scale. Its implicit box center can differ from the mesh center.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `actor_name` | `str` | `required` | Indexed instance object ID for the size query. |

### `RobotAPI.get_object_size_in_local_frame`

```python
RobotAPI.get_object_size_in_local_frame(actor_name: str) -> List[float] | None
```

Returns an array of indexed local dimensions or None (despite a List annotation). Reads bbox/other shape metadata, its orientation and asset up_axis. Not a composed mesh measurement; missing metadata returns None.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `actor_name` | `str` | `required` | Indexed instance object ID for the size query. |

### `RobotAPI.add_all_objects_info`

```python
RobotAPI.add_all_objects_info()
```

Returns None. Registers indexed layout objects with the simulator and sets task-scene text; skips unknown asset IDs. Constructor helper with simulator side effects.

### `RobotAPI.run_simple_action`

```python
RobotAPI.run_simple_action(action: Callable, config: dict, objects: Dict[str, OmniObject])
```

Returns `(stage_success, store_name)` using a state snapshot and at most three attempts, subject to callback terminate. Calls zero-argument checks on success; a failed check is sampled again after 30 simulation steps. Failure restores the stored state. Callback config includes action_type, active_id, passive_id, if_attached, arm, action_description, unmovable_objects and extra_checkers. Its action_description is already recorder-shaped (`action_text`, `english_action_text`).

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `action` | `Callable` | `required` | Action instance for step; for run_simple_action a callable `(attempt_index, config, objects) -> (success, terminate)`. |
| `config` | `dict` | `required` | RobotAPIConfig for construction, BimanualRigidMoveConfig for bimanual actions, or lifecycle dictionary for run_simple_action. See the corresponding section. |
| `objects` | `Dict[str, OmniObject]` | `required` | Mutable dictionary of cache ID to OmniObject, including gripper for stage operations. Pose snapshots are refreshed by execution. |

### `RobotAPI.step`

```python
RobotAPI.step(action: Action, stage: Stage, objects: Dict[str, OmniObject], index: int=0, step_index: int=0) -> bool
```

Actual return is `(step_success, try_next_sequence)`, despite the bool annotation. Executes one action with preparation/planning/contact checks and recording state. Use only when extending the stage lifecycle.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `action` | `Action` | `required` | Action instance for step; for run_simple_action a callable `(attempt_index, config, objects) -> (success, terminate)`. |
| `stage` | `Stage` | `required` | Stage instance whose candidate buffer and checker configuration have been initialized or are initialized by run_stage. |
| `objects` | `Dict[str, OmniObject]` | `required` | Mutable dictionary of cache ID to OmniObject, including gripper for stage operations. Pose snapshots are refreshed by execution. |
| `index` | `int` | `0` | Attempt index passed to step; zero-based. |
| `step_index` | `int` | `0` | Index within the selected action sequence; zero-based. |

### `RobotAPI.run_stage`

```python
RobotAPI.run_stage(stage: Stage, objects: Dict[str, OmniObject]) -> bool
```

Actual return is `(stage_success, store_name)`, despite the bool annotation. Initializes candidates, stores state, attempts at most three sequences, checks protected objects and stage/extra checkers, and restores on failure. Failed extra checkers are re-sampled after 30 steps before rejecting the attempt.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `stage` | `Stage` | `required` | Stage instance whose candidate buffer and checker configuration have been initialized or are initialized by run_stage. |
| `objects` | `Dict[str, OmniObject]` | `required` | Mutable dictionary of cache ID to OmniObject, including gripper for stage operations. Pose snapshots are refreshed by execution. |

### `RobotAPI.get_grasp_stage`

```python
RobotAPI.get_grasp_stage(actor: OmniObject, objects: Dict[str, OmniObject], arm: str='right', grasp_config: GraspConfig=GraspConfig()) -> Stage
```

Returns a PickStage; it constructs but does not execute a grasp. Requires a nonempty GraspConfig action description.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `actor` | `OmniObject` | `required` | Manipulated instance object ID or supported entity prim path; `"gripper"` where documented. `pick_actor` also accepts OmniObject. An arbitrary mesh path is not an indexed object. |
| `objects` | `Dict[str, OmniObject]` | `required` | Mutable dictionary of cache ID to OmniObject, including gripper for stage operations. Pose snapshots are refreshed by execution. |
| `arm` | `str` | `'right'` | Arm selector: `"left"` or `"right"`; collider helpers additionally accept `"both"`. No length unit. |
| `grasp_config` | `GraspConfig` | `GraspConfig()` | GraspConfig for contact candidates, filters, approach/lift, descriptions and checks. Pass a fresh instance per action. |

### `RobotAPI.get_place_stage`

```python
RobotAPI.get_place_stage(actor: OmniObject, target_actor: OmniObject, objects: Dict[str, OmniObject], target_pose: Optional[np.ndarray]=None, arm: str='right', place_config: PlaceConfig=PlaceConfig()) -> Stage
```

Returns PlaceStage if pre_insert_offset is None/zero, otherwise InsertStage. Requires a nonempty description. target_pose bypasses placement labels where supported; None uses active/passive annotation alignment. Stage construction does not execute it.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `actor` | `OmniObject` | `required` | Manipulated instance object ID or supported entity prim path; `"gripper"` where documented. `pick_actor` also accepts OmniObject. An arbitrary mesh path is not an indexed object. |
| `target_actor` | `OmniObject` | `required` | Receiving indexed instance ID for high-level calls; OmniObject for stage construction. Its receiving annotations must match the goal. |
| `objects` | `Dict[str, OmniObject]` | `required` | Mutable dictionary of cache ID to OmniObject, including gripper for stage operations. Pose snapshots are refreshed by execution. |
| `target_pose` | `Optional[np.ndarray]` | `None` | World actor target candidates `(N,4,4)`, except bimanual calls which target the leader end effector. Each matrix has a valid rotation and bottom row `[0,0,0,1]`. |
| `arm` | `str` | `'right'` | Arm selector: `"left"` or `"right"`; collider helpers additionally accept `"both"`. No length unit. |
| `place_config` | `PlaceConfig` | `PlaceConfig()` | PlaceConfig for release/placement or insertion when pre_insert_offset is nonzero. Pass a fresh instance per action. |

### `RobotAPI.get_rotate_stage`

```python
RobotAPI.get_rotate_stage(actor: OmniObject, target_pose: np.ndarray, objects: Dict[str, OmniObject], arm: str='right', rotate_config: RotateConfig=RotateConfig())
```

Returns ConstraintRotateStage without executing. Requires nonempty action text and orientation candidates.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `actor` | `OmniObject` | `required` | Manipulated instance object ID or supported entity prim path; `"gripper"` where documented. `pick_actor` also accepts OmniObject. An arbitrary mesh path is not an indexed object. |
| `target_pose` | `np.ndarray` | `required` | World actor target candidates `(N,4,4)`, except bimanual calls which target the leader end effector. Each matrix has a valid rotation and bottom row `[0,0,0,1]`. |
| `objects` | `Dict[str, OmniObject]` | `required` | Mutable dictionary of cache ID to OmniObject, including gripper for stage operations. Pose snapshots are refreshed by execution. |
| `arm` | `str` | `'right'` | Arm selector: `"left"` or `"right"`; collider helpers additionally accept `"both"`. No length unit. |
| `rotate_config` | `RotateConfig` | `RotateConfig()` | RotateConfig for orientation constraints and checks. Candidate translations are not the rotation goal. |

### `RobotAPI.get_move_stage`

```python
RobotAPI.get_move_stage(actor: OmniObject, target_actor: OmniObject, target_pose: np.ndarray, objects: Dict[str, OmniObject], arm: str='right', move_config: MoveConfig=MoveConfig())
```

Returns MoveStage without executing; copies configs and optional target_pose. Requires nonempty action text. strip_arm_prefix extension affects recorded descriptions.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `actor` | `OmniObject` | `required` | Manipulated instance object ID or supported entity prim path; `"gripper"` where documented. `pick_actor` also accepts OmniObject. An arbitrary mesh path is not an indexed object. |
| `target_actor` | `OmniObject` | `required` | Receiving indexed instance ID for high-level calls; OmniObject for stage construction. Its receiving annotations must match the goal. |
| `target_pose` | `np.ndarray` | `required` | World actor target candidates `(N,4,4)`, except bimanual calls which target the leader end effector. Each matrix has a valid rotation and bottom row `[0,0,0,1]`. |
| `objects` | `Dict[str, OmniObject]` | `required` | Mutable dictionary of cache ID to OmniObject, including gripper for stage operations. Pose snapshots are refreshed by execution. |
| `arm` | `str` | `'right'` | Arm selector: `"left"` or `"right"`; collider helpers additionally accept `"both"`. No length unit. |
| `move_config` | `MoveConfig` | `MoveConfig()` | MoveConfig for holding/gripper movement, path constraints and optional gripper command. Pass a fresh instance per action. |

### `RobotAPI.get_bimanual_rigid_move_stage`

```python
RobotAPI.get_bimanual_rigid_move_stage(target_actor: OmniObject, target_pose: np.ndarray, objects: Dict[str, OmniObject], config: BimanualRigidMoveConfig=BimanualRigidMoveConfig()) -> Stage
```

Returns BimanualRigidMoveStage without executing; requires the bimanual action description and actor/target cache entries.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `target_actor` | `OmniObject` | `required` | Receiving indexed instance ID for high-level calls; OmniObject for stage construction. Its receiving annotations must match the goal. |
| `target_pose` | `np.ndarray` | `required` | World actor target candidates `(N,4,4)`, except bimanual calls which target the leader end effector. Each matrix has a valid rotation and bottom row `[0,0,0,1]`. |
| `objects` | `Dict[str, OmniObject]` | `required` | Mutable dictionary of cache ID to OmniObject, including gripper for stage operations. Pose snapshots are refreshed by execution. |
| `config` | `BimanualRigidMoveConfig` | `BimanualRigidMoveConfig()` | RobotAPIConfig for construction, BimanualRigidMoveConfig for bimanual actions, or lifecycle dictionary for run_simple_action. See the corresponding section. |

### `RobotAPI.get_fix_pose_object`

```python
RobotAPI.get_fix_pose_object(obj_id: str, position: List[float], direction: List[float], N_align: int=1) -> OmniObject
```

Returns a synthetic OmniObject with a tiny fixed-pose box and passive placement direction/sampling metadata. This is a planning target, not a spawned physical object.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `obj_id` | `str` | `required` | Cache ID for a synthetic fixed-pose target; use a distinct fix_pose name to follow update_objects conventions. |
| `position` | `List[float]` | `required` | World xyz position, shape `(3,)`, meters; for arm selection transformed into the robot base frame. |
| `direction` | `List[float]` | `required` | Direction vector `(3,)` for a synthetic placement annotation; dimensionless, nonzero. |
| `N_align` | `int` | `1` | Number of alignment-angle samples for the synthetic placement annotation; positive integer. |

### `RobotAPI.get_object`

```python
RobotAPI.get_object(actor: str, part: str=None)
```

Returns/caches an OmniObject from indexed asset metadata and current pose. Prefer an entity path and explicit part argument. Missing index/asset files can raise; a failed pose lookup may return None. Part mode preserves an entity parent path and part-specific pose prim.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `actor` | `str` | `required` | Manipulated instance object ID or supported entity prim path; `"gripper"` where documented. `pick_actor` also accepts OmniObject. An arbitrary mesh path is not an indexed object. |
| `part` | `str` | `None` | Relative actor part path such as `body/hole`; changes the manipulated reference frame. None uses the object default. |

### `RobotAPI.get_all_objects`

```python
RobotAPI.get_all_objects()
```

Returns None; populates self.objects with indexed layout objects and skips unknown IDs. Does not return the dictionary despite its name.

### `RobotAPI.update_objects`

```python
RobotAPI.update_objects(objects, arm='right')
```

Returns the updated dictionary. Requires a gripper entry, refreshes EE/object poses and articulated joint state. Synthetic fix_pose objects retain their fixed planning pose.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `objects` | `Dict[str, OmniObject]` | `required` | Mutable dictionary of cache ID to OmniObject, including gripper for stage operations. Pose snapshots are refreshed by execution. |
| `arm` | `str` | `'right'` | Arm selector: `"left"` or `"right"`; collider helpers additionally accept `"both"`. No length unit. |

### `RobotAPI.add_object_to_objects`

```python
RobotAPI.add_object_to_objects(object_id: str, object_element: OmniObject)
```

Returns None. Inserts an OmniObject into self.objects only if that key is absent; duplicate keys log and leave the existing value unchanged.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `object_id` | `str` | `required` | Instance ID in `scene_info.layout`, not the asset ID. Low-level cache helpers also use part-qualified IDs. |
| `object_element` | `OmniObject` | `required` | OmniObject instance to insert into the runtime cache; existing keys are not overwritten. |

### `RobotAPI.get_object_usd_id`

```python
RobotAPI.get_object_usd_id(object_id: str) -> str
```

Returns the layout asset ID when present in ASSETS_INDEX, otherwise an empty string and an error log. Input is an instance ID.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `object_id` | `str` | `required` | Instance ID in `scene_info.layout`, not the asset ID. Low-level cache helpers also use part-qualified IDs. |

### `RobotAPI.open_gripper`

```python
RobotAPI.open_gripper(arm: str='right', opened_width: float=1.0) -> None
```

Returns None, not a success bool; forwards normalized opening to the robot. May detach according to lower-level gripper behavior; use explicit attachment bookkeeping.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `arm` | `str` | `'right'` | Arm selector: `"left"` or `"right"`; collider helpers additionally accept `"both"`. No length unit. |
| `opened_width` | `float` | `1.0` | Normalized gripper opening: 1.0 fully open, 0.0 closed. This is not the meter-valued custom grasp_width. |

### `RobotAPI.close_gripper`

```python
RobotAPI.close_gripper(arm: str='right') -> None
```

Returns None, not a success bool. Closing does not prove a physical grasp.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `arm` | `str` | `'right'` | Arm selector: `"left"` or `"right"`; collider helpers additionally accept `"both"`. No length unit. |

### `RobotAPI.get_ee_pose`

```python
RobotAPI.get_ee_pose(arm: str='right') -> np.ndarray
```

Returns a world `(4,4)` transform from the robot getter for the chosen arm. Confirm the profile EE convention when combining with custom contact poses.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `arm` | `str` | `'right'` | Arm selector: `"left"` or `"right"`; collider helpers additionally accept `"both"`. No length unit. |

### `RobotAPI.get_joint_positions`

```python
RobotAPI.get_joint_positions() -> Dict[str, float]
```

Returns a joint-name-to-position dict from server states. Revolute units are radians and prismatic units follow the runtime.

### `RobotAPI.set_joint_positions`

```python
RobotAPI.set_joint_positions(target_joint_position: List[float], target_joint_names: Optional[List[str]]=None, is_trajectory: bool=False) -> bool
```

Returns `response.errmsg == "Move Joint"`; forwards joint values/names and trajectory flag. This response-string test is implementation-specific, not a task checker.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `target_joint_position` | `List[float]` | `required` | Joint values in runtime order or matched to target_joint_names. Revolute radians/prismatic length units. |
| `target_joint_names` | `Optional[List[str]]` | `None` | Names corresponding to target_joint_position; None uses the runtime ordering. |
| `is_trajectory` | `bool` | `False` | Flag forwarded to set_joint_positions to execute in trajectory mode. |

### `RobotAPI.add_object`

```python
RobotAPI.add_object(usd_path: str, prim_path: str, label_name: str, target_position: List[float], target_quaternion: List[float], target_scale: List[float], color: List[float], material: str, mass: float=0.01, static_friction: float=0.5, dynamic_friction: float=0.5, add_rigid_body: bool=True, model_type: str='convexDecomposition') -> str
```

Returns server label_name after spawning the USD object with material/physics settings. Does not automatically update scene_info/interactions/cache; low-level dynamic-spawn tasks must manage their object contract.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `usd_path` | `str` | `required` | Resolvable asset USD path for spawning an object. |
| `prim_path` | `str` | `required` | Absolute target prim path; spawning requires the /World/Objects hierarchy. Attribute changes use the exact intended prim. |
| `label_name` | `str` | `required` | Spawned object label returned by the server; not automatically a scene_info object role. |
| `target_position` | `List[float]` | `required` | World xyz `(3,)` in meters for the spawned object. |
| `target_quaternion` | `List[float]` | `required` | Spawned object orientation `(4,)` in wxyz order. |
| `target_scale` | `List[float]` | `required` | Per-axis dimensionless scale `(3,)` for the spawned object. |
| `color` | `List[float]` | `required` | RGB `(3,)`, channels in `[0,1]`; debug marker None defaults to red. |
| `material` | `str` | `required` | Material name recognized by the installed server. |
| `mass` | `float` | `0.01` | Rigid object mass in kilograms, positive when dynamic. |
| `static_friction` | `float` | `0.5` | Dimensionless static friction coefficient. |
| `dynamic_friction` | `float` | `0.5` | Dimensionless dynamic friction coefficient. |
| `add_rigid_body` | `bool` | `True` | Whether to add a rigid-body API to the spawned object. |
| `model_type` | `str` | `'convexDecomposition'` | Server collision approximation name; default convexDecomposition. Supported alternatives depend on the runtime. |

### `RobotAPI.get_cloth_nodal_positions`

```python
RobotAPI.get_cloth_nodal_positions(cloth_prim: str, indices: Optional[List[int]]=None) -> np.ndarray
```

Returns world node positions `(M,3)` in provided index order, or `(N,3)` for all nodes. Raises RuntimeError if FEM view/server command is unavailable or fails.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `cloth_prim` | `str` | `required` | Absolute Mesh prim path with the active surface deformable/FEM body API; rigid object wrappers are insufficient. |
| `indices` | `Optional[List[int]]` | `None` | Optional zero-based FEM node indices. None reads all nodes; provided indices preserve their order. |

### `RobotAPI.set_fem_pin`

```python
RobotAPI.set_fem_pin(arm: Optional[str], cloth_prim: str, radius: float=0.03, gain: float=3.0, vmax: float=3.0, group_id: Optional[str]=None, anchor_link: Optional[str]=None, axis: str='xy') -> int
```

Returns the number of attached FEM nodes. Explicit anchor overrides inferred arm anchor; missing both raises ValueError. Defaults group identity to the anchor on the server. Failed/no-node attachment can raise RuntimeError. This is velocity servo pinning, not friction contact.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `arm` | `Optional[str]` | `required` | Arm selector: `"left"` or `"right"`; collider helpers additionally accept `"both"`. No length unit. |
| `cloth_prim` | `str` | `required` | Absolute Mesh prim path with the active surface deformable/FEM body API; rigid object wrappers are insufficient. |
| `radius` | `float` | `0.03` | Pin selection radius in meters around the anchor, using axis-selected distance. |
| `gain` | `float` | `3.0` | FEM velocity-servo gain, dimensionless; server applies position error/dt times gain. |
| `vmax` | `float` | `3.0` | FEM pin velocity limit in meters/second. |
| `group_id` | `Optional[str]` | `None` | Pin group identifier; release accepts "all". Re-attaching an existing group can replace it. Defaults differ by pin method. |
| `anchor_link` | `Optional[str]` | `None` | Absolute anchor prim path; overrides inferred arm anchor. FEM requires an arm or explicit anchor; rigid pin can anchor to world. |
| `axis` | `str` | `'xy'` | `"xy"` for projected pin radius or `"3d"` for Euclidean radius. |

### `RobotAPI.release_fem_pin`

```python
RobotAPI.release_fem_pin(group_id: str='all') -> List[str]
```

Returns released group IDs; server also zeros node velocities. "all" releases every group, including groups established outside the current phase.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `group_id` | `str` | `'all'` | Pin group identifier; release accepts "all". Re-attaching an existing group can replace it. Defaults differ by pin method. |

### `RobotAPI.set_rigid_pin`

```python
RobotAPI.set_rigid_pin(rigid_prim: str, arm: Optional[str]=None, anchor_link: Optional[str]=None, mode: str='pose', gain_lin: float=3.0, gain_ang: float=3.0, vmax_lin: float=3.0, vmax_ang: float=6.0, group_id: Optional[str]=None) -> str
```

Returns the active group ID, defaulting to the constrained prim. Invalid mode raises ValueError; failed prim/anchor/view setup raises RuntimeError. No arm/anchor locks the current world pose. pose mode bypasses normal dynamics; velocity mode is for rigid bodies and falls back to pose for Xforms.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `rigid_prim` | `str` | `required` | Valid prim to constrain; rigid body or Xform. Non-rigid prim velocity mode falls back to pose mode. |
| `arm` | `Optional[str]` | `None` | Arm selector: `"left"` or `"right"`; collider helpers additionally accept `"both"`. No length unit. |
| `anchor_link` | `Optional[str]` | `None` | Absolute anchor prim path; overrides inferred arm anchor. FEM requires an arm or explicit anchor; rigid pin can anchor to world. |
| `mode` | `str` | `'pose'` | `"pose"` sets pose each physics step; `"velocity"` limits pose changes and injects velocities for rigid bodies. |
| `gain_lin` | `float` | `3.0` | Dimensionless linear servo gain for rigid-pin velocity mode. |
| `gain_ang` | `float` | `3.0` | Dimensionless angular servo gain for rigid-pin velocity mode. |
| `vmax_lin` | `float` | `3.0` | Rigid-pin maximum linear speed and displacement limiter, meters/second. |
| `vmax_ang` | `float` | `6.0` | Rigid-pin maximum angular speed, radians/second. |
| `group_id` | `Optional[str]` | `None` | Pin group identifier; release accepts "all". Re-attaching an existing group can replace it. Defaults differ by pin method. |

### `RobotAPI.release_rigid_pin`

```python
RobotAPI.release_rigid_pin(group_id: str='all') -> List[str]
```

Returns released group IDs and clears rigid velocities. "all" affects all groups, not only this task phase.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `group_id` | `str` | `'all'` | Pin group identifier; release accepts "all". Re-attaching an existing group can replace it. Defaults differ by pin method. |

### `RobotAPI.set_gripper_colliders`

```python
RobotAPI.set_gripper_colliders(arm: str='both', enabled: bool=True, links: Optional[List[str]]=None) -> int
```

Returns the number of mesh colliders toggled. Changes physical collisions, unlike planner obstacle exclusion. Restore the prior intended state after the contact/pin phase.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `arm` | `str` | `'both'` | Arm selector: `"left"` or `"right"`; collider helpers additionally accept `"both"`. No length unit. |
| `enabled` | `bool` | `True` | True enables physical gripper colliders, false disables them. |
| `links` | `Optional[List[str]]` | `None` | Optional robot link suffix list for collider toggling; None selects the profile default base/finger set. |

### `RobotAPI.disable_gripper_colliders`

```python
RobotAPI.disable_gripper_colliders(arm: str='both') -> int
```

Returns collider count from set_gripper_colliders(arm, enabled=False).

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `arm` | `str` | `'both'` | Arm selector: `"left"` or `"right"`; collider helpers additionally accept `"both"`. No length unit. |

### `RobotAPI.enable_gripper_colliders`

```python
RobotAPI.enable_gripper_colliders(arm: str='both') -> int
```

Returns collider count from set_gripper_colliders(arm, enabled=True).

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `arm` | `str` | `'both'` | Arm selector: `"left"` or `"right"`; collider helpers additionally accept `"both"`. No length unit. |

### `RobotAPI.draw_debug_cube`

```python
RobotAPI.draw_debug_cube(name: str, position: Optional[List[float]]=None, orientation: Optional[List[float]]=None, pose_4x4: Optional[np.ndarray]=None, size: float=0.02, color: Optional[List[float]]=None) -> str
```

Returns server marker prim path or empty string. pose_4x4 overrides separate pose fields; repeated names update. Uses /World/_DebugCubes and changes the live debug scene.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `name` | `str` | `required` | Debug marker identifier; repeat calls update the same marker. |
| `position` | `Optional[List[float]]` | `None` | World xyz position, shape `(3,)`, meters; for arm selection transformed into the robot base frame. |
| `orientation` | `Optional[List[float]]` | `None` | Debug marker quaternion wxyz `(4,)`; None defaults to identity. |
| `pose_4x4` | `Optional[np.ndarray]` | `None` | Debug marker world transform `(4,4)`; when provided it overrides position/orientation. |
| `size` | `float` | `0.02` | Debug cube edge length in meters. |
| `color` | `Optional[List[float]]` | `None` | RGB `(3,)`, channels in `[0,1]`; debug marker None defaults to red. |

### `RobotAPI.clear_debug_cubes`

```python
RobotAPI.clear_debug_cubes() -> int
```

Returns the number of removed markers under /World/_DebugCubes; affects all such markers.

### `RobotAPI.attach_obj`

```python
RobotAPI.attach_obj(prim_paths: List[str], arm: str) -> bool
```

Returns `response.msg is not None`, not a measured physical grasp result. Attaches the listed object paths to the selected hand through the server.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `prim_paths` | `List[str]` | `required` | List of absolute object prim paths for attachment or planner obstacle updates. Effects depend on the server command. |
| `arm` | `str` | `required` | Arm selector: `"left"` or `"right"`; collider helpers additionally accept `"both"`. No length unit. |

### `RobotAPI.detach_obj`

```python
RobotAPI.detach_obj() -> bool
```

Returns `response.msg is not None`. No arm/path selector exists in this wrapper; assume broad server detach scope and inspect shared attachments before calling.

### `RobotAPI.remove_objs_from_obstacle`

```python
RobotAPI.remove_objs_from_obstacle(prim_paths: List[str]) -> bool
```

Returns `response.msg is not None`; excludes listed objects from planner obstacles, not physical colliders. Restrict to intended contact targets.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `prim_paths` | `List[str]` | `required` | List of absolute object prim paths for attachment or planner obstacle updates. Effects depend on the server command. |

### `RobotAPI.add_objs_to_obstacle`

```python
RobotAPI.add_objs_to_obstacle(prim_paths: List[str]) -> bool
```

Returns true when command response is non-None, not necessarily a server success flag. Re-add temporary planner exclusions and verify the intended planner state.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `prim_paths` | `List[str]` | `required` | List of absolute object prim paths for attachment or planner obstacle updates. Effects depend on the server command. |

### `RobotAPI.set_object_parameter_action`

```python
RobotAPI.set_object_parameter_action(prim_path: str, attribute_key: str, attribute_value, delay: float=0.0, description: Optional[Dict[str, str]]=None, reference: str='world') -> bool
```

Returns `response == "success"`. Applies a live USD attribute, optionally records start/end action state, waits the configured delay before application and **60 simulation steps after** (the inspected docstring says five, but code uses 60). World conversion is limited to supported translate/orient attributes. This changes simulation state directly, not a robot manipulation.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `prim_path` | `str` | `required` | Absolute target prim path; spawning requires the /World/Objects hierarchy. Attribute changes use the exact intended prim. |
| `attribute_key` | `str` | `required` | USD attribute name, e.g. xformOp:translate or xformOp:orient. |
| `attribute_value` | `JSON-serializable value` | `required` | USD attribute value of the attribute's required type; orientation uses wxyz. |
| `delay` | `float` | `0.0` | Pre-application simulation delay in seconds, rounded to 60 Hz steps and clamped nonnegative. |
| `description` | `Optional[Dict[str, str]]` | `None` | Optional action text mapping `zh`/`en`; supplied text records the attribute change as an action. |
| `reference` | `str` | `'world'` | `"world"` or `"local"`; world conversion applies to supported translate/orient attributes, other attributes are written directly. |

### `RobotAPI.solve_ik`

```python
RobotAPI.solve_ik(target_poses: np.ndarray, arm: str='right', ee_type: str='gripper', type: str='Simple', ObsAvoid: bool=False, output_link_pose: bool=False) -> Tuple[np.ndarray, Dict]
```

Returns `(ik_success_mask, ik_info)` with per-candidate solver results. ee_type and ObsAvoid are accepted but unused by this wrapper; type/output_link_pose are forwarded. Static IK alone does not prove a safe trajectory or grasp.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `target_poses` | `np.ndarray` | `required` | World robot-gripper IK targets `(N,4,4)`. Convert object targets through the current object-to-gripper transform first. |
| `arm` | `str` | `'right'` | Arm selector: `"left"` or `"right"`; collider helpers additionally accept `"both"`. No length unit. |
| `ee_type` | `str` | `'gripper'` | Accepted by the RobotAPI solve_ik signature but currently not forwarded to the lower-level solver. |
| `type` | `str` | `'Simple'` | IK planner selector; current stage calls use "Simple" or "AvoidObs". The wrapper forwards this string. |
| `ObsAvoid` | `bool` | `False` | Accepted by the solve_ik signature but currently unused; choose the appropriate type and validate collisions. |
| `output_link_pose` | `bool` | `False` | Request solver link-pose information in the result dictionary. |

### `RobotAPI.reset`

```python
RobotAPI.reset() -> None
```

Returns None after low-level robot.reset. Do not assume it restores every task-established planner exclusion or pin group.

## Action configuration constructors

### `GraspConfig`

```python
GraspConfig(grasp_offset: float=0.0, pre_grasp_offset: float=0.05, grasp_lower_percentile: float=0, grasp_upper_percentile: float=100, grasp_percentile_axis: str=None, disable_upside_down: bool=True, flip_grasp: bool=True, set_grasp_along_axis: str=None, set_grasp_pose_center: bool=False, is_from_up_side: bool=False, pick_up_distance: float=0.12, pick_up_direction: str='z', grasp_pose_filter: Callable=None, grasp_pose: np.ndarray=None, grasp_pose_frame: str='world', grasp_width: np.ndarray=None, active_primitive: str=None, passive_primitive: str=None, unmovable_objects: List[str]=[], extra_checkers: List[Callable]=[], action_description: Dict[str, str]=None, extra_params: Dict[str, Any]={})
```

Returns a configuration object; construction does not execute motion. Custom poses bypass stored annotations but still pass candidate processing. See grasp-selection.md for annotated/custom orientation differences.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `grasp_offset` | `float` | `0.0` | Contact offset in meters along the grasp transport/approach vector; verify its sign with the robot-gripper convention. |
| `pre_grasp_offset` | `float` | `0.05` | Pre-contact offset in meters along the candidate approach vector; check the entire finger approach for clearance. |
| `grasp_lower_percentile` | `float` | `0` | Lower percentile in `[0,100]` for contact translations on the selected local axis. |
| `grasp_upper_percentile` | `float` | `100` | Upper percentile in `[0,100]`, at least the lower percentile. |
| `grasp_percentile_axis` | `str` | `None` | Local axis `"x"`, `"y"`, or `"z"`; None uses local Y in the inspected stage. |
| `disable_upside_down` | `bool` | `True` | Enable robot-profile-dependent gripper orientation filtering. It is not a generic object upright constraint; inspect candidates after filtering. |
| `flip_grasp` | `bool` | `True` | Add candidates rotated 180 degrees around local grasp Z. Set false when this would violate a required grip direction. |
| `set_grasp_along_axis` | `str` | `None` | Align grasp orientation relative to local `"x"`, `"y"`, or `"z"`; arm/profile conventions affect the resulting orientation. None leaves the candidate unchanged. |
| `set_grasp_pose_center` | `bool` | `False` | With set_grasp_along_axis, center contact translation on the other local axes. The current fallback centers local X/Z. |
| `is_from_up_side` | `bool` | `False` | Preference for grasp candidates above the object in the stage sorting pipeline, not a guarantee of vertical approach. |
| `pick_up_distance` | `float` | `0.12` | Post-contact lift distance in meters. A list through extra_params defines a first-to-last range search, not literal candidates; verify shelf ceiling and reach. |
| `pick_up_direction` | `str` | `'z'` | Lift axis `"x"`, `"y"`, or `"z"`; cuRobo execution uses base coordinates but candidate IK composes in the grasp frame. See the extension guide for rotated-grip limitations. |
| `grasp_pose_filter` | `Callable` | `None` | Callable `(poses_world, object_pose, widths) -> (poses_filtered, widths_filtered)`; output arrays retain matching candidate order. |
| `grasp_pose` | `np.ndarray` | `None` | Custom robot-gripper contact matrix `(4,4)` or candidates `(N,4,4)`. Replaces stored grasp poses; still passes filters, flips and IK. |
| `grasp_pose_frame` | `str` | `'world'` | `"world"` or `"relative"`; relevant only for custom poses. Relative refers to the current object/part frame. |
| `grasp_width` | `np.ndarray` | `None` | Opening widths `(N,)` in meters for custom candidates. If omitted, custom loading fills 0.08 m; annotated loading has a different 0.02 m fallback. |
| `active_primitive` | `str` | `None` | Active-side interaction label actually present on the manipulated object (gripper for grasp). None lets compatible annotations be traversed. |
| `passive_primitive` | `str` | `None` | Passive-side interaction label actually present on the grasped/receiving object. None lets compatible annotations be traversed. |
| `unmovable_objects` | `List[str]` | `[]` | List of instance/cache IDs that must remain stationary. Do not include the manipulated object or objects expected to settle. Current checks use 0.01 m / 5 degree displacement thresholds with action-specific exclusions. Simple-action paths can compare stale cached poses; use live extra checkers for critical protection (see extension guide). |
| `extra_checkers` | `List[Callable]` | `[]` | List of zero-argument callables returning bool. Read live state and avoid side effects. All required checks must pass; failures/exceptions fail the stage. |
| `action_description` | `Dict[str, str]` | `None` | Nonempty action text mapping with `zh` and `en` keys. Ordinary actions add the arm prefix. Checkers use actual state, not this text. |
| `extra_params` | `Dict[str, Any]` | `{}` | Dictionary merged after serialized fields, not Config attributes. See [GraspConfig effective keys and limitations](config-extra-params.md#graspconfig); later builders/Actions can override or ignore values. |

### `PlaceConfig`

```python
PlaceConfig(disable_upside_down: bool=True, active_primitive: str=None, passive_primitive: str=None, pre_insert_offset: float=None, unmovable_objects: List[str]=[], extra_checkers: List[Callable]=[], action_description: Dict[str, str]=None, extra_params: Dict[str, Any]={})
```

Returns a configuration object; construction does not execute motion. A nonzero pre_insert_offset selects InsertStage. Use receiving labels that actually exist, or valid object goal transforms.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `disable_upside_down` | `bool` | `True` | Enable robot-profile-dependent gripper orientation filtering. It is not a generic object upright constraint; inspect candidates after filtering. |
| `active_primitive` | `str` | `None` | Active-side interaction label actually present on the manipulated object (gripper for grasp). None lets compatible annotations be traversed. |
| `passive_primitive` | `str` | `None` | Passive-side interaction label actually present on the grasped/receiving object. None lets compatible annotations be traversed. |
| `pre_insert_offset` | `float` | `None` | Meters for the pre-insertion approach. None or zero selects ordinary PlaceStage; nonzero selects InsertStage in get_place_stage. |
| `unmovable_objects` | `List[str]` | `[]` | List of instance/cache IDs that must remain stationary. Do not include the manipulated object or objects expected to settle. Current checks use 0.01 m / 5 degree displacement thresholds with action-specific exclusions. Simple-action paths can compare stale cached poses; use live extra checkers for critical protection (see extension guide). |
| `extra_checkers` | `List[Callable]` | `[]` | List of zero-argument callables returning bool. Read live state and avoid side effects. All required checks must pass; failures/exceptions fail the stage. |
| `action_description` | `Dict[str, str]` | `None` | Nonempty action text mapping with `zh` and `en` keys. Ordinary actions add the arm prefix. Checkers use actual state, not this text. |
| `extra_params` | `Dict[str, Any]` | `{}` | Dictionary merged after serialized fields, not Config attributes. See [PlaceConfig effective keys and limitations](config-extra-params.md#placeconfig); later builders/Actions can override or ignore values. |

### `MoveConfig`

```python
MoveConfig(path_constraint: List[float]=[], offset_and_constraint_in_goal_frame: bool=False, active_primitive: str=None, passive_primitive: str=None, gripper_action: str=None, unmovable_objects: List[str]=[], extra_checkers: List[Callable]=[], action_description: Dict[str, str]=None, extra_params: Dict[str, Any]={})
```

Returns a configuration object; construction does not execute motion. An ordinary held-object move preserves the grasp and does not release unless gripper_action requests it.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `path_constraint` | `List[float]` | `[]` | Empty means planner default; otherwise six weights `[rx,ry,rz,x,y,z]`. Zero frees a dimension; nonzero constrains it. These are weights, not angles or displacements. |
| `offset_and_constraint_in_goal_frame` | `bool` | `False` | False: robot base/planning-frame path constraints/offsets; not general world axes. True: target gripper/goal frame. See the extension guide for composition. |
| `active_primitive` | `str` | `None` | Active-side interaction label actually present on the manipulated object (gripper for grasp). None lets compatible annotations be traversed. |
| `passive_primitive` | `str` | `None` | Passive-side interaction label actually present on the grasped/receiving object. None lets compatible annotations be traversed. |
| `gripper_action` | `str` | `None` | `"open"`, `"close"`, or None after/during the stage action. Ordinary held-object MoveStage does not release by default. |
| `unmovable_objects` | `List[str]` | `[]` | List of instance/cache IDs that must remain stationary. Do not include the manipulated object or objects expected to settle. Current checks use 0.01 m / 5 degree displacement thresholds with action-specific exclusions. Simple-action paths can compare stale cached poses; use live extra checkers for critical protection (see extension guide). |
| `extra_checkers` | `List[Callable]` | `[]` | List of zero-argument callables returning bool. Read live state and avoid side effects. All required checks must pass; failures/exceptions fail the stage. |
| `action_description` | `Dict[str, str]` | `None` | Nonempty action text mapping with `zh` and `en` keys. Ordinary actions add the arm prefix. Checkers use actual state, not this text. |
| `extra_params` | `Dict[str, Any]` | `{}` | Dictionary merged after serialized fields, not Config attributes. See [MoveConfig effective keys and limitations](config-extra-params.md#moveconfig); later builders/Actions can override or ignore values. |

### `RotateConfig`

```python
RotateConfig(constraint_axises: List[str]=[], offset_and_constraint_in_goal_frame: bool=False, unmovable_objects: List[str]=[], extra_checkers: List[Callable]=[], action_description: Dict[str, str]=None, extra_params: Dict[str, Any]={})
```

Returns a configuration object; construction does not execute motion. Uses orientation candidates for a held object; constraint_axises keeps the specified rotation axes fixed.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `constraint_axises` | `List[str]` | `[]` | List of rotation axes that must not rotate, e.g. `["x","y"]`. The spelling axises is the actual API parameter. |
| `offset_and_constraint_in_goal_frame` | `bool` | `False` | False: robot base/planning-frame path constraints/offsets; not general world axes. True: target gripper/goal frame. See the extension guide for composition. |
| `unmovable_objects` | `List[str]` | `[]` | List of instance/cache IDs that must remain stationary. Do not include the manipulated object or objects expected to settle. Current checks use 0.01 m / 5 degree displacement thresholds with action-specific exclusions. Simple-action paths can compare stale cached poses; use live extra checkers for critical protection (see extension guide). |
| `extra_checkers` | `List[Callable]` | `[]` | List of zero-argument callables returning bool. Read live state and avoid side effects. All required checks must pass; failures/exceptions fail the stage. |
| `action_description` | `Dict[str, str]` | `None` | Nonempty action text mapping with `zh` and `en` keys. Ordinary actions add the arm prefix. Checkers use actual state, not this text. |
| `extra_params` | `Dict[str, Any]` | `{}` | Dictionary merged after serialized fields, not Config attributes. See [RotateConfig effective keys and limitations](config-extra-params.md#rotateconfig); later builders/Actions can override or ignore values. |

### `OpenConfig`

```python
OpenConfig(target_limit: str='min', trajectory_step_deg: float=1.0, max_consecutive_failures: int=3, solvable_threshold: float=0.5, unmovable_objects: List[str]=[], extra_checkers: List[Callable]=[], action_description: Dict[str, str]=None, extra_params: Dict[str, Any]={})
```

Returns a configuration object; construction does not execute motion. Controls revolute-joint limit trajectory generation/following. This is not a generic drawer-opening config.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `target_limit` | `str` | `'min'` | `"min"` or `"max"`: move the revolute joint toward its lower/upper limit. Which direction is open is asset-specific. |
| `trajectory_step_deg` | `float` | `1.0` | Positive joint-trajectory sample spacing in degrees. Smaller values create more samples. |
| `max_consecutive_failures` | `int` | `3` | Integer consecutive motion/IK failure budget; consumed by the relevant trajectory or bimanual planner. |
| `solvable_threshold` | `float` | `0.5` | Server trajectory IK solvability threshold, normally within `[0,1]`; lower is less restrictive, not a success checker. |
| `unmovable_objects` | `List[str]` | `[]` | List of instance/cache IDs that must remain stationary. Do not include the manipulated object or objects expected to settle. Current checks use 0.01 m / 5 degree displacement thresholds with action-specific exclusions. Simple-action paths can compare stale cached poses; use live extra checkers for critical protection (see extension guide). |
| `extra_checkers` | `List[Callable]` | `[]` | List of zero-argument callables returning bool. Read live state and avoid side effects. All required checks must pass; failures/exceptions fail the stage. |
| `action_description` | `Dict[str, str]` | `None` | Nonempty action text mapping with `zh` and `en` keys. Ordinary actions add the arm prefix. Checkers use actual state, not this text. |
| `extra_params` | `Dict[str, Any]` | `{}` | Dictionary merged after serialized fields, not Config attributes. See [OpenConfig effective keys and limitations](config-extra-params.md#openconfig); later builders/Actions can override or ignore values. |

### `FollowTrajectoryConfig`

```python
FollowTrajectoryConfig(noise_range: List[float]=None, noise_frame: str='local', sample_count: int=10, unmovable_objects: List[str]=None, extra_checkers: List[Callable]=None, action_description: Dict[str, str]=None, extra_params: Dict[str, Any]=None)
```

Returns a configuration object; construction does not execute motion. None noise_range becomes six zero ranges. Lists/dicts supplied here are copied; constructor None description becomes an empty mapping and the action rejects an empty description.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `noise_range` | `List[float]` | `None` | Six symmetric IK perturbation ranges `[dx,dy,dz,droll,dpitch,dyaw]`; meters then radians. None becomes six zeros. |
| `noise_frame` | `str` | `'local'` | `"local"` or `"world"` for waypoint IK noise. |
| `sample_count` | `int` | `10` | Number of fallback samples for an original waypoint whose IK is unsolvable; integer, validated by the trajectory helper. |
| `unmovable_objects` | `List[str]` | `None` | List of instance/cache IDs that must remain stationary. Do not include the manipulated object or objects expected to settle. Current checks use 0.01 m / 5 degree displacement thresholds with action-specific exclusions. Simple-action paths can compare stale cached poses; use live extra checkers for critical protection (see extension guide). |
| `extra_checkers` | `List[Callable]` | `None` | List of zero-argument callables returning bool. Read live state and avoid side effects. All required checks must pass; failures/exceptions fail the stage. |
| `action_description` | `Dict[str, str]` | `None` | Nonempty action text mapping with `zh` and `en` keys. Ordinary actions add the arm prefix. Checkers use actual state, not this text. |
| `extra_params` | `Dict[str, Any]` | `None` | Dictionary merged after serialized fields, not Config attributes. See [FollowTrajectoryConfig effective keys and limitations](config-extra-params.md#followtrajectoryconfig); later builders/Actions can override or ignore values. |

### `BimanualRigidMoveConfig`

```python
BimanualRigidMoveConfig(leader_arm: str='right', path_constraint: List[float]=[], offset_and_constraint_in_goal_frame: bool=False, goal_offset: List[float]=[0, 0, 0, 1, 0, 0, 0], from_current_pose: bool=False, ik_downsample_num: int=200, max_consecutive_failures: int=1, motion_run_ratio: float=1.0, disable_collision_links: List[str]=[], unmovable_objects: List[str]=[], extra_checkers: List[Callable]=[], action_description: Dict[str, str]=None, extra_params: Dict[str, Any]={})
```

Returns a configuration object; construction does not execute motion. Leader target is an EE pose; follower maintains the initial relative EE transform.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `leader_arm` | `str` | `'right'` | `"right"` or `"left"`; the other arm follows with the initial fixed relative end-effector transform. |
| `path_constraint` | `List[float]` | `[]` | Empty means planner default; otherwise six weights `[rx,ry,rz,x,y,z]`. Zero frees a dimension; nonzero constrains it. These are weights, not angles or displacements. |
| `offset_and_constraint_in_goal_frame` | `bool` | `False` | False: robot base/planning-frame path constraints/offsets; not general world axes. True: target gripper/goal frame. See the extension guide for composition. |
| `goal_offset` | `List[float]` | `[0, 0, 0, 1, 0, 0, 0]` | Relative xyz+wxyz offset `(7,)`, meters and quaternion. Applied by the planner according to from_current_pose and goal-frame settings. |
| `from_current_pose` | `bool` | `False` | Relative motion flag for the planner. Its pose/goal_offset composition must be checked for the chosen action; do not mix an object goal with an EE-relative offset. |
| `ik_downsample_num` | `int` | `200` | Maximum candidate pool size passed to bimanual IK; positive integer. |
| `max_consecutive_failures` | `int` | `1` | Integer consecutive motion/IK failure budget; consumed by the relevant trajectory or bimanual planner. |
| `motion_run_ratio` | `float` | `1.0` | Serialized requested fraction, but the inspected stage executor ignores this field and uses 1.0 unless error_data supplies a ratio. See the extension guide; do not rely on it for partial motion. |
| `disable_collision_links` | `List[str]` | `[]` | Robot link names excluded by the motion planner; not a physical collider toggle. |
| `unmovable_objects` | `List[str]` | `[]` | List of instance/cache IDs that must remain stationary. Do not include the manipulated object or objects expected to settle. Current checks use 0.01 m / 5 degree displacement thresholds with action-specific exclusions. Simple-action paths can compare stale cached poses; use live extra checkers for critical protection (see extension guide). |
| `extra_checkers` | `List[Callable]` | `[]` | List of zero-argument callables returning bool. Read live state and avoid side effects. All required checks must pass; failures/exceptions fail the stage. |
| `action_description` | `Dict[str, str]` | `None` | Nonempty action text mapping with `zh` and `en` keys. Ordinary actions add the arm prefix. Checkers use actual state, not this text. |
| `extra_params` | `Dict[str, Any]` | `{}` | Dictionary merged after serialized fields, not Config attributes. See [BimanualRigidMoveConfig effective keys and limitations](config-extra-params.md#bimanualrigidmoveconfig); later builders/Actions can override or ignore values. |

### `ShakeConfig`

```python
ShakeConfig(unmovable_objects: List[str]=[], extra_checkers: List[Callable]=[], action_description: Dict[str, str]=None, extra_params: Dict[str, Any]={})
```

Returns a configuration object; construction does not execute motion. Defines checks/text; wrist angle ranges and repetition count are method arguments.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `unmovable_objects` | `List[str]` | `[]` | List of instance/cache IDs that must remain stationary. Do not include the manipulated object or objects expected to settle. Current checks use 0.01 m / 5 degree displacement thresholds with action-specific exclusions. Simple-action paths can compare stale cached poses; use live extra checkers for critical protection (see extension guide). |
| `extra_checkers` | `List[Callable]` | `[]` | List of zero-argument callables returning bool. Read live state and avoid side effects. All required checks must pass; failures/exceptions fail the stage. |
| `action_description` | `Dict[str, str]` | `None` | Nonempty action text mapping with `zh` and `en` keys. Ordinary actions add the arm prefix. Checkers use actual state, not this text. |
| `extra_params` | `Dict[str, Any]` | `{}` | Dictionary merged after serialized fields, not Config attributes. See [ShakeConfig effective keys and limitations](config-extra-params.md#shakeconfig); later builders/Actions can override or ignore values. |

### `TurnBodyConfig`

```python
TurnBodyConfig(step_deg: float=30.0, unmovable_objects: List[str]=[], extra_checkers: List[Callable]=[], action_description: Dict[str, str]=None, extra_params: Dict[str, Any]={})
```

Returns a configuration object; construction does not execute motion. Defines G2 body-turn step size/checks/text; target world point is a method argument.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `step_deg` | `float` | `30.0` | Positive body-turn interpolation step in degrees; actual robot joint positions use radians for revolute joints. |
| `unmovable_objects` | `List[str]` | `[]` | List of instance/cache IDs that must remain stationary. Do not include the manipulated object or objects expected to settle. Current checks use 0.01 m / 5 degree displacement thresholds with action-specific exclusions. Simple-action paths can compare stale cached poses; use live extra checkers for critical protection (see extension guide). |
| `extra_checkers` | `List[Callable]` | `[]` | List of zero-argument callables returning bool. Read live state and avoid side effects. All required checks must pass; failures/exceptions fail the stage. |
| `action_description` | `Dict[str, str]` | `None` | Nonempty action text mapping with `zh` and `en` keys. Ordinary actions add the arm prefix. Checkers use actual state, not this text. |
| `extra_params` | `Dict[str, Any]` | `{}` | Dictionary merged after serialized fields, not Config attributes. See [TurnBodyConfig effective keys and limitations](config-extra-params.md#turnbodyconfig); later builders/Actions can override or ignore values. |

### `to_dict()` for all action configs

Returns a dictionary of the above fields, merged with `extra_params` last. This affects serialization only; builders and dispatch may ignore or overwrite a value. Read [effective Config extensions](config-extra-params.md). GraspConfig only includes custom pose/frame/width keys when `grasp_pose` is non-None. This avoids overriding annotated loading with empty custom fields. A config with no description constructs successfully, but most action/stage methods then reject it.

## RobotAPIConfig

`RobotAPIConfig()` constructs a lightweight settings object. Its defaults
include a G2 robot profile, auto_record=False, empty recording/metric maps,
an empty task identifier, and initially empty description fields.

```python
RobotAPIConfig.from_json(json_file: str)
```

`json_file` is required: a task_info.json path. Returns RobotAPIConfig;
file/JSON errors propagate. Reads `robot` into robot_config, `auto_record`,
`recording_setting`, `task_description`, `task_name`, and `task_metric`.
See [the task contract](task-contract.md#descriptions-and-recording-settings)
for each field, robot pose/noise, camera settings, and recorder ownership.
The constructor forwards metrics/settings; it does not validate arbitrary
metric rule names or recorder options. Runner recording options can override
the JSON auto_record flag.

## Dual specifications

These dataclasses are passed as left/right to dual_act; the arm is determined
by that argument position, so there is no arm constructor parameter. Each
Config gets a fresh default factory instance. Missing descriptions are filled
by DualSpec with a default dual-action description; provide meaningful task
text explicitly. Construction does not run a stage.

### `DualPick`

```python
DualPick(actor: Union[OmniObject, str], grasp_config: GraspConfig = field(default_factory=GraspConfig), part: Optional[str] = None)
```

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `actor` | `Union[OmniObject, str]` | required | Manipulated instance object ID or supported entity prim path; `"gripper"` where documented. `pick_actor` also accepts OmniObject. An arbitrary mesh path is not an indexed object. |
| `grasp_config` | `GraspConfig` | fresh Config instance | GraspConfig for contact candidates, filters, approach/lift, descriptions and checks. Pass a fresh instance per action. |
| `part` | `Optional[str]` | None | Relative actor part path such as `body/hole`; changes the manipulated reference frame. None uses the object default. |

### `DualPlaceToActor`

```python
DualPlaceToActor(actor: str, target_actor: str, place_config: PlaceConfig = field(default_factory=PlaceConfig))
```

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `actor` | `str` | required | Manipulated instance object ID or supported entity prim path; `"gripper"` where documented. `pick_actor` also accepts OmniObject. An arbitrary mesh path is not an indexed object. |
| `target_actor` | `str` | required | Receiving indexed instance ID for high-level calls; OmniObject for stage construction. Its receiving annotations must match the goal. |
| `place_config` | `PlaceConfig` | fresh Config instance | PlaceConfig for release/placement or insertion when pre_insert_offset is nonzero. Pass a fresh instance per action. |

### `DualPlaceToPose`

```python
DualPlaceToPose(actor: str, target_pose: np.ndarray, place_config: PlaceConfig = field(default_factory=PlaceConfig))
```

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `actor` | `str` | required | Manipulated instance object ID or supported entity prim path; `"gripper"` where documented. `pick_actor` also accepts OmniObject. An arbitrary mesh path is not an indexed object. |
| `target_pose` | `np.ndarray` | required | World actor target candidates `(N,4,4)`, except bimanual calls which target the leader end effector. Each matrix has a valid rotation and bottom row `[0,0,0,1]`. |
| `place_config` | `PlaceConfig` | fresh Config instance | PlaceConfig for release/placement or insertion when pre_insert_offset is nonzero. Pass a fresh instance per action. |

### `DualMoveToPose`

```python
DualMoveToPose(actor: str, target_pose: np.ndarray, move_config: MoveConfig = field(default_factory=MoveConfig), part: Optional[str] = None)
```

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `actor` | `str` | required | Manipulated instance object ID or supported entity prim path; `"gripper"` where documented. `pick_actor` also accepts OmniObject. An arbitrary mesh path is not an indexed object. |
| `target_pose` | `np.ndarray` | required | World actor target candidates `(N,4,4)`, except bimanual calls which target the leader end effector. Each matrix has a valid rotation and bottom row `[0,0,0,1]`. |
| `move_config` | `MoveConfig` | fresh Config instance | MoveConfig for holding/gripper movement, path constraints and optional gripper command. Pass a fresh instance per action. |
| `part` | `Optional[str]` | None | Relative actor part path such as `body/hole`; changes the manipulated reference frame. None uses the object default. |

### `DualRotate`

```python
DualRotate(actor: str, target_pose: np.ndarray, rotate_config: RotateConfig = field(default_factory=RotateConfig))
```

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `actor` | `str` | required | Manipulated instance object ID or supported entity prim path; `"gripper"` where documented. `pick_actor` also accepts OmniObject. An arbitrary mesh path is not an indexed object. |
| `target_pose` | `np.ndarray` | required | World actor target candidates `(N,4,4)`, except bimanual calls which target the leader end effector. Each matrix has a valid rotation and bottom row `[0,0,0,1]`. |
| `rotate_config` | `RotateConfig` | fresh Config instance | RotateConfig for orientation constraints and checks. Candidate translations are not the rotation goal. |

### `DualSpec.build_stage(api, arm)`

`api`: required RobotAPI; `arm`: required left/right selector. Returns
`(Stage, objects)` from the selected spec. The abstract DualSpec raises
NotImplementedError; use a concrete spec or implement a compatible extension.
Concrete build_stage methods construct the same stage kinds as the corresponding
single-arm action; execution and synchronized rollback belong to dual_act.

## Supported extension parameters

Read [Config extra_params](config-extra-params.md) for the supported keys of
**each** Config, types/defaults/units, nested schemas, merge precedence,
inactive branches, overwritten values, and single/dual dispatch differences.
Do not apply one action's extensions to another Config.

| Config | Relevant extension surface |
| --- | --- |
| [GraspConfig](config-extra-params.md#graspconfig) | Approach distance, lift range, candidate filters, contact/lift checking and motion dispatch |
| [PlaceConfig](config-extra-params.md#placeconfig) | Release, post-placement moves, insertion/depth/noise; stage selection uses the named offset |
| [MoveConfig](config-extra-params.md#moveconfig) | Relative offsets, motion options, completion checking; pre-place and nearby fallback limitations |
| [RotateConfig](config-extra-params.md#rotateconfig) | Named constraints/frame, motion options; generated weights override generic weights |
| [OpenConfig](config-extra-params.md#openconfig) | Serialized joint trajectory fields and lifecycle checks |
| [FollowTrajectoryConfig](config-extra-params.md#followtrajectoryconfig) | Lifecycle checks; noise must use named fields |
| [BimanualRigidMoveConfig](config-extra-params.md#bimanualrigidmoveconfig) | Rigid motion fields and gripper command; ratio/slowdown/follower limitations |
| [ShakeConfig](config-extra-params.md#shakeconfig) | Lifecycle checks; shake controls are method arguments |
| [TurnBodyConfig](config-extra-params.md#turnbodyconfig) | Serialized step size and lifecycle checks |

## Coordinate helpers

Import these from the installed `common.base_utils.transform_utils` module.
Their descriptions are included here so authoring does not depend on another
source checkout. Explicitly choose the quaternion convention at every boundary.

| Signature | Full argument meaning / result |
| --- | --- |
| `quat2mat(quaternion)` | Required xyzw array `(...,4)`; returns rotation `(...,3,3)` |
| `quat2mat_wxyz(quaternion_wxyz)` | Required wxyz `(4,)`; returns `(3,3)` rotation |
| `mat2quat_wxyz(rmat)` | Required `(3,3)` rotation; returns wxyz `(4,)` |
| `quat_wxyz_to_xyzw(quat_wxyz)` | Required wxyz `(4,)`; returns reordered xyzw |
| `pose_from_position_quaternion(position, quaternion)` | Required xyz `(3,)` meters and wxyz `(4,)`; returns `(4,4)` |
| `get_pose_wxyz(xyz, quat_wxyz)` | Required xyz `(3,)` meters and wxyz `(4,)`; returns `(4,4)` |
| `pose_inv(pose_mat)` | Required rigid `(4,4)` transform; returns inverse |
| `make_pose(translation, rotation)` | Required xyz `(3,)` and rotation `(3,3)`; returns `(4,4)` |
| `transform_points(points, transform_matrix)` | Required `(N,3)` points and `(4,4)` transform; returns transformed `(N,3)` points |
| `get_relative_position_and_orientation_from_world(a_position_world, a_quaternion_wxyz, b_position_world, b_quaternion_wxyz)` | Required xyz/wxyz poses of A and B in world; returns A expressed in B `(xyz,wxyz)`, using inv(T_world_B) @ T_world_A |
| `get_world_position_and_orientation_from_relative(a_position_in_b, a_quaternion_in_b_wxyz, b_position_world, b_quaternion_wxyz)` | Required xyz/wxyz pose of A in B and B in world; returns A in world using T_world_B @ T_B_A |
| `rotate_along_axis(target_affine, angle_degrees, rot_axis="z", use_local=True)` | Required `(4,4)` pose and angle in degrees; axis x/y/z; local or world rotation composition, returns pose with adjusted orientation |
| `rotation_matrix(angle, direction, point=None)` | Required angle in radians and nonzero axis `(3,)`; optional pivot `(3,)`; returns `(4,4)` rotation about that axis/pivot |
| `pose_difference(pose1, pose2)` | Required two `(4,4)` poses; returns translation distance in meters and angular difference in degrees |
| `check_box_containment(box1_position, box1_quaternion, box1_size, box2_position, box2_quaternion, box2_size, overlap_threshold=0.8, ignore_axes=None)` | Required box centers `(3,)`, wxyz rotations `(4,)`, local extents `(3,)` for both boxes; threshold `[0,1]`; ignore_axes None or list of world x/y/z axes. Returns `(box1_contains_box2, box2_contains_box1)` based on overlap fractions. The implementation uses world AABBs; this is not exact oriented-box or hollow-container containment |

For the containment helper, use actual box centers (including local center
offsets transformed into world), not an arbitrary prim origin. Ignoring Z
is a projected overlap test; add vertical/interior and settling checks.

## Client interfaces used by tasks

`robot.client` exposes lower-level RPC responses. Return objects are generally
protobuf responses, not Python booleans. Prefer RobotAPI methods for motion.
The following metadata/lifecycle calls and command escape hatch are documented
because task code uses them. Server-specific command payloads are not generic
RobotAPI parameters.

| Signature | Parameters and result / effect |
| --- | --- |
| `set_task_task_basic_info(task_description)` | Required JSON-serializable object with task_name, english_task_name, English init_scene_text; returns RPC response and updates current task text |
| `set_task_metric(task_metric)` | Required runtime metric object; returns RPC response and installs quality/filter rules |
| `set_frame_state(action: str, substage_id: int, active_id: str, passive_id: str, if_attached: bool, set_gripper_open=False, set_gripper_close=False, arm="default", target_pose=None, action_description={}, error_description={})` | First five required: action type, step index, active/passive IDs and attachment flag. Optional gripper flags are booleans; arm is text; target_pose is an ndarray converted to a JSON list; descriptions are JSON objects with frame text/error fields. Returns RPC response; ordinary stage execution manages these frame states |
| `start_recording(data_keys, fps, task_name)` | Required recorder data-key object, frame rate number and task identifier; returns RPC response. Runner-owned |
| `stop_recording()` | No arguments; returns RPC response. Runner-owned |
| `send_task_status(isSuccess, fail_stage_step)` | Required bool and `[stage,step]` index list; returns RPC response. Runner-owned |
| `store_current_state(playback_id: str)` | Required snapshot name; returns RPC response; stage execution manages its snapshots |
| `playback(playback_id: str)` | Required saved snapshot name; returns RPC response and restores simulation state |
| `run_step(steps)` | Required nonnegative integer simulation steps; returns RPC response |
| `detach_obj()` | No parameters; returns protobuf RPC response after the broad server detach command; track affected attachments |
| `remove_objs_from_obstacle(prim_paths)` | Required list of absolute object prim paths; returns protobuf RPC response. Changes planner obstacles, not physical colliders |
| `get_checker_status(checker_config)` | Required JSON-serializable runtime checker object; returns response with msg such as fail; stage runtime manages built-in checker configs |
| `command_any(command_config)` | Required JSON-serializable dict with command and its command-specific payload; top-level ndarray values are converted in place. Returns decoded response. Schema/success semantics vary by command |

Task-local README interface tags list every statically observed direct
robot.client call for that example. A call not in this supported subset needs
version-matched runtime inspection before copying it. Prefer high-level
wrappers or the documented server commands used by them where possible.
