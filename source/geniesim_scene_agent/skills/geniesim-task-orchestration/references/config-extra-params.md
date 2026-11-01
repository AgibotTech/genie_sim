# Config extra_params: supported keys and effective behavior

This reference follows configuration serialization, RobotAPI stage construction,
pose selection, per-action generation, and robot dispatch. A key being accepted
by `to_dict()` does not establish that it affects execution. Defaults below are
for the public Config path, unless explicitly identified as a stage fallback.
Use named constructor parameters for named fields whenever possible.
`RobotAPIConfig` has no `extra_params` field; these sections cover the nine
action Configs.

## Contents

- [Merge order and shared fields](#merge-order-and-shared-fields)
- [GraspConfig](#graspconfig)
- [PlaceConfig](#placeconfig)
- [MoveConfig](#moveconfig)
- [RotateConfig](#rotateconfig)
- [OpenConfig](#openconfig)
- [FollowTrajectoryConfig](#followtrajectoryconfig)
- [BimanualRigidMoveConfig](#bimanualrigidmoveconfig)
- [ShakeConfig](#shakeconfig)
- [TurnBodyConfig](#turnbodyconfig)
- [Nested dictionaries](#nested-dictionaries)
- [Dual dispatch differences](#dual-dispatch-differences)

## Merge order and shared fields

1. Each Config serializes its named fields, then updates the dictionary with
   `extra_params`. This overrides serialized values, not Config attributes.
2. RobotAPI reads some attributes directly before or independently of that
   dictionary: descriptions, interaction labels, insertion-stage selection,
   trajectory noise and the initial leader arm. Dictionary overrides do not
   reliably change these decisions.
3. A stage generates Actions. `RobotAPI.step` merges stage extras first and
   Action extras last; generated Action values win. Target resolution and
   default completion checks read Action extras, not this later merged dict.
4. The selected robot dispatch forwards only its implemented options.
   Unknown keys can be silently retained and unused.

For **all nine Configs**, dictionary overrides of `unmovable_objects` and
`extra_checkers` are consumed by the lifecycle. Use the same types as their
named fields: lists of instance/cache IDs and zero-argument bool callables.
Named defaults are empty lists (or None normalized to empty). Protected-object
checks use 0.01 m / 5 degree displacement thresholds with action-specific
exclusions. Failed extra checkers are rechecked after 30 simulation steps.
The simple-action lifecycle compares cached object poses without automatically
refreshing them after the callback (notably Open and FollowTrajectory). Critical
stationary-object checks must query live poses in `extra_checkers`; serialized
protected IDs alone can compare stale data in those paths.
`skip_check` does not disable these lifecycle checks.

Set `action_description` on the Config itself: high-level methods validate and
format the attribute, and generally overwrite the serialized description.
Set `arm`, actor, part, joint and target through method arguments. Do not
use extra dictionary metadata to redirect an action.

### Motion dispatch options

These are consumed by stage execution for Grasp, Place/Insert, Move and Rotate,
subject to generated Action overrides and the dispatch limitations below.
They are **not** a shared extension surface for Open, FollowTrajectory, Shake
or TurnBody.

| Key | Type / executor fallback | Effective behavior |
| --- | --- | --- |
| `goal_offset` | seven numbers, `[0,0,0,1,0,0,0]` | Relative translation xyz in meters plus quaternion wxyz; normalized quaternion. Planner composition depends on `from_current_pose` and the frame flag. Generated lift/insertion offsets override it. |
| `from_current_pose` | bool, false | Requests motion relative to current EE in the planner. Some stages still select an absolute candidate first. Grasp lift and constrained insertion override it with true. |
| `path_constraint` | empty list, or six nonnegative weights `[rx,ry,rz,x,y,z]` | Zero frees a dimension; nonzero constrains it. Weights are not tolerances or displacements. Rotate constructs its own weights; constrained insertion overrides them. |
| `offset_and_constraint_in_goal_frame` | bool, true in executor | True uses goal-gripper frame; false uses robot base/planning frame (not general world axes). MoveConfig and RotateConfig serialize false by default. Grasp lift and constrained insertion force false. See [frame composition](execution-tips.md#frames-and-measured-grasp-transforms). |
| `disable_collision_links` | list of robot link names, `[]` | Planner exclusion of robot links, not object prims or physical colliders. Requires valid names for the actual robot/planner. |
| `slowdown_factor` | positive float, 1.0 | Forwarded to matrix-motion timing. cuRobo clamps the factor to at least 1.0 and adjusts for physics timestep; values below 1 do not speed it up. Zero is coerced to 1.0 by the robot wrapper. Does not select a planner or relax task checks. Rigid bimanual dispatch drops this option. |
| `error_data` | dict, absent | Deliberate error injection; schema below. Normally omit for successful collection. |

`max_consecutive_failures` is read by `step` with fallback 1, but ordinary matrix
motion drops it in `move_pose`. It works in the rigid bimanual command and the
Open trajectory command. It does not change the lifecycle's fixed attempt
budget. Top-level `motion_run_ratio` and `gripper_action_timing` extras do not
control `step`: that method derives them from `error_data` instead.

Backend `Simple` / `Straight` motion and cuRobo motion have different support
for planning options. Passing weights or exclusions does not make a backend
trajectory collision-aware. Validate the executed path and physical result.

## GraspConfig

Consumer: `get_grasp_stage` -> `PickStage` -> generated approach/contact/lift.
The ordinary stage always enables its built-in pre-grasp sequence.

| Extension key | Type / default | Behavior and limitation |
| --- | --- | --- |
| `pre_grasp_distance` | nonnegative float, 0.05 m | Built-in pre-approach displacement along grasp-local negative Z. Distinct from named `pre_grasp_offset`, which affects candidate preparation. |
| `pick_up_distance` | float, named default 0.12 m; alternatively nonempty numeric list | Scalar lift, or range search using the list's first/last values as endpoints, with a 0.005 m grid and at least two samples. Interior list entries are not literal candidates. Supply ordered nonnegative endpoints and check ceiling clearance. |
| `skip_check` | bool, false | Copied into generated approach/contact/lift Actions; bypasses their default geometric completion check. Does not bypass protected objects or extra checkers. |
| `set_grasp_pose_xy` | bool, false | Legacy name: forces candidate contact local X **and Z** translations to zero. |
| `set_grasp_vertical` | bool, false | Legacy alias that sets `set_grasp_along_axis="y"`; not a universal world-vertical grip. |
| `humanlike_filter` | bool, false | Applies additional profile-dependent posture filtering. Can remove the required grip. |
| `filter_grasp_pose` | dict, `{}` | Direction filter; exact nested schema below. Nonempty filters need an explicit positive angular threshold. |
| `random_grasp_pose` | bool, false | Replaces candidates with a runtime-generated pool of 100. Does not infer task-valid mesh contact or substitute for Agent grasp reasoning. |
| `place_with_origin_orientation` | bool, true | Compatibility filtering when a next place stage is linked; favors the original object orientation. |
| `use_near_point` | bool, false | Allows nearby approximate IK targets. Can change contact; require a task-specific grip/position checker. |

Named candidate fields can be overridden in the serialized dictionary:
`grasp_offset`, `pre_grasp_offset`, `grasp_lower_percentile`,
`grasp_upper_percentile`, `grasp_percentile_axis`,
`disable_upside_down`, `flip_grasp`, `set_grasp_along_axis`,
`set_grasp_pose_center`, `is_from_up_side`, `pick_up_distance`,
`pick_up_direction`, `grasp_pose_filter`, and custom `grasp_pose`,
`grasp_pose_frame`, `grasp_width`. Their types/defaults are in the
[constructor reference](robot_api_reference.md#graspconfig).
Interaction selection uses the attributes `active_primitive` and
`passive_primitive`; set these as named arguments.

Lift candidate IK composes the offset in the grasp frame, whereas the generated
lift executes its offset in the robot base/planning frame. For nonvertical/rotated grips,
explicitly verify direction and use a separate measured extraction move when
necessary. `pre_grasp_vector` (three numbers) and `pick_up_type` (fallback
`"Simple"`) are read only in the inactive branch where `self.use_pre_grasp`
is false. An extra `use_pre_grasp=False` does **not** switch that attribute.
Do not use these legacy keys to select the ordinary approach or lift planner;
the active lift uses `"AvoidObs"`.

## PlaceConfig

Consumer: `get_place_stage` -> `PlaceStage` or `InsertStage`.
**The Config attribute** `pre_insert_offset` selects the stage: None/zero gives
ordinary placement; nonzero gives insertion. Putting only that key in
`extra_params` does not select InsertStage. Use a positive named offset, then
insertion extras when that branch fits the aperture geometry.

| Extension key | Type / default | Behavior and limitation |
| --- | --- | --- |
| `gripper_state` | `"open"`, `"close"`, or None; default `"open"` | Command at placement/insertion release phase. None retains the grip; it does not detach an explicit attachment. |
| `post_place_action` | None; list of dicts for Place, one dict for Insert | Additional offset moves with different schemas below. Insert's post-action branch bypasses `constraint_insert` / `insert_depth` logic. |
| `use_near_point` | bool, false | Place candidate fallback searches shifted targets up to 0.2 m. Can invalidate precise placement or insertion. |
| `constraint_insert` | bool, false | **Insert only**: constrained relative base-negative-Z descent by `pre_insert_offset`. Forces `from_current_pose=True`, base frame, weights `[0.1,0.1,0.1,0.1,0.1,0]`, and `skip_check=True` on that descent. |
| `insert_depth` | nonnegative float, 0.0 m | **Insert only**, without post action: above 0.00001 m, adds another relative base-negative-Z constrained descent, then release. Measure usable depth first. |
| `pre_pose_noise` | dict, absent | **Insert approach** perturbation; schema below. Presence changes constrained-insertion action/release sequencing even for `{}`. |
| `use_pre_place` | bool, false in Place | **Broken ordinary Place branch**: selection returns `pre_insert_pose`, but `PlaceStage.generate_action_sequence` does not accept it. Avoid enabling this key. InsertStage forces pre-place on regardless of this key. |

Serialized `disable_upside_down` is consumed. Serialized `pre_insert_offset`
affects the approach distance after stage selection; keep it consistent with
the named attribute. Interaction labels and description use Config attributes.
A non-None method `target_pose` is assigned **after** the Config dictionary, so
it wins over an extra target. For actor-to-actor placement an extra target may
be read, but use the explicit pose API for measured goals.

Ordinary Place copies `slowdown_factor` into its Actions. Generic stage
`skip_check` is **not propagated** into ordinary Place Actions, so it does not
bypass their default checker. Insertion constructs its own skip-check flags,
offsets and weights. Generic `use_world_pose` is not propagated either.
Use an explicit Move phase for an arbitrary approach/withdrawal axis.

## MoveConfig

Consumer: `get_move_stage` -> `MoveStage`. Move copies stage extras into its
Action, then explicitly sets the standard motion fields to the stage values.
The named path weights, frame flag, gripper command, protected objects and
checkers can be overridden through the serialized dictionary.

| Extension key | Type / default | Behavior and limitation |
| --- | --- | --- |
| `from_current_pose` | bool, false | Relative EE planner motion; pair with an explicit offset/frame and physical checker. False frame flag uses base axes. |
| `goal_offset` | seven numbers, identity offset | See shared motion schema. The method still expects a valid target pool even for relative motion. |
| `skip_check` | bool; defaults to `from_current_pose` | Relative moves automatically bypass default completion checking unless explicitly false. A fixed geometric target checker may not describe a relative endpoint. |
| `disable_upside_down` | bool, false | Additional candidate orientation filter; not a named MoveConfig field. |
| `strip_arm_prefix` | bool, false | Read directly from `move_config.extra_params`; omits the automatic arm prefix in the recorded description. |
| `use_near_point` | bool, false | **Broken fallback in inspected MoveStage**: calls `find_near_point_grasp_pose` without defining/importing it. Reaching this branch raises NameError. Supply valid measured candidates instead. |
| `use_pre_place` | bool, false | **Broken branch**: selection supplies `pre_insert_pose` to a generator accepting only `grasp_pose`. Avoid; split the approach into a separate Move call. |
| `pre_insert_offset` | float, stage fallback 0.1 m | Only used with the unusable `use_pre_place` branch. |
| `pre_pose_noise` | dict, absent | Only used with that pre-place branch; does not perturb ordinary moves. |
| `use_world_pose` | bool, false | Advanced Action target-resolver switch. Selected targets are normally converted into anchor-local coordinates before generation; forcing true can reinterpret them incorrectly. The high-level pose API already takes world object poses; do not enable this to label those poses. |

Shared dispatch extras apply. Method `target_pose` wins over an extra target.
Interaction labels and descriptions are read from the Config attributes.

## RotateConfig

Consumer: `get_rotate_stage` -> `ConstraintRotateStage`.
Serialized `constraint_axises` and `offset_and_constraint_in_goal_frame` are
consumed, as are lifecycle fields. The stage fixes translation and constructs
its own rotation weights; extra `path_constraint` is overwritten in the Action.
`use_world_pose=True` is forced. Completion returns true without a geometric
rotation test, so `skip_check` cannot enable one: supply an orientation checker.

Shared dispatch options such as link exclusions and slowdown are forwarded by
the ordinary matrix path. Offsets/relative flags can change the rotation goal;
prefer explicit desired object orientation. An extra `target_pose` can override
the method target in this builder's merge order, unlike Move/Place; use the
method argument and avoid this hidden override. Description uses the attribute.

## OpenConfig

Consumer: `open` builds a revolute handle trajectory, then `run_simple_action`.
The following serialized named fields are actually read, so their extras
work, though named constructor arguments are clearer:

| Key | Type / named default | Meaning |
| --- | --- | --- |
| `target_limit` | `"min"` or `"max"`, `"min"` | Joint endpoint; opening direction depends on the asset. |
| `trajectory_step_deg` | positive float, 1.0 degrees | Angular sample spacing for handle trajectory. |
| `solvable_threshold` | float in `[0,1]`, 0.5 | Forwarded trajectory IK solvability threshold; not proof of reaching the requested limit. |
| `max_consecutive_failures` | positive int, 3 | Forwarded trajectory failure budget. |

Lifecycle overrides work. Arm, joint, actor, part and action text are determined
by method arguments/description attribute. Shared motion extras (`goal_offset`,
`path_constraint`, `slowdown_factor`, `from_current_pose`, link exclusions,
`skip_check`) are not passed to this trajectory command. No arbitrary extra
joint/speed/noise option is consumed. Check live joint angle and body stability.

## FollowTrajectoryConfig

Consumer: `follow_trajectory` resolves waypoints **before** serializing extras.
Only lifecycle overrides (`extra_checkers`, `unmovable_objects`) are effective.

- Put `noise_range`, `noise_frame`, and `sample_count` in named constructor
  arguments. Same-name extras do not affect IK candidate resolution.
- The command uses `solvable_threshold=np.nextafter(1.0,0.0)` and
  `max_consecutive_failures=1`; extras do not override those values.
- Shared motion extras, `skip_check`, and arbitrary speed/interpolation keys
  are not forwarded to this command.
- Arm, actor IDs, attachment flag, action type and formatted description are
  overwritten by the method. Set action text on the Config attribute.

A waypoint list is ordered; a pool of alternative target poses is not a path.
Use small, task-valid named noise ranges for contact trajectories.

## BimanualRigidMoveConfig

Consumer: stage builds leader EE targets, then `BimanualRigidAvoidObs` dispatch.
Serialized named `path_constraint`, frame flag, `goal_offset`,
`from_current_pose`, `ik_downsample_num`, `max_consecutive_failures`, and
`disable_collision_links` are consumed (types/defaults in the constructor
reference). Lifecycle overrides work.

| Extension / tempting key | Type / default | Effective behavior |
| --- | --- | --- |
| `gripper_action` | `"open"`, `"close"`, or None; None | Optional action gripper command; normally retain both contacts during a rigid carry. |
| `follower_arm` | `"left"` or `"right"`; opposite leader | Read for stage IK selection, but execution re-derives the opposite arm. Overriding it can make selection and execution disagree; leave derived. |
| `leader_arm` | named default `"right"` | Extra override changes stage selection but initial builder decisions use the attribute. Set the named field and leave it consistent. |
| `target_pose` | `(N,4,4)` world leader EE targets | Extra can win over method argument here; prefer the explicit method argument. Not object target poses. |
| `motion_run_ratio` | named float, 1.0 | Stage serializes it, but `step` replaces execution ratio with 1.0 unless error injection supplies a ratio. Changing this named field or same-name extra has no effective partial-run control in this path. |
| `slowdown_factor` | float, executor fallback 1.0 | Passed to `move_pose` but omitted from its rigid dispatch call; helper sends its own default 1.0. Ineffective here. |
| `skip_check` | forced true in Action | An extra false cannot re-enable the default checker. Use lifecycle/final physical checkers. |
| `use_world_pose` | forced true in Action | No configurable frame switch here. |

With `from_current_pose=True`, stage selection returns current EEs without the
absolute candidate IK filtering. Execution still plans the relative rigid move;
the high-level method still requires a nonempty target pool. Establish both
contacts first and check the follower path and final object support.

## ShakeConfig

Only lifecycle extras are consumed. Wrist `angle_ranges`, `times`, and
`horizontal` are method arguments captured by the execution callback; extras
do not override them. Description is read from the Config attribute. No extra
planner constraint, collision, slowdown or timing option is forwarded.

## TurnBodyConfig

Lifecycle extras and serialized `step_deg` (positive degrees, named default
30.0) are consumed. Target joint angle is calculated from the method's target
point, then overwrites the dictionary. Description uses the attribute. Extra
`target_joint_angle`, arm, target point, joint limits, noise, slowdown and
planner options do not control this command. The inspected G2 implementation
uses fixed body-joint limits of approximately ±3.2 radians.

## Nested dictionaries

### filter_grasp_pose

| Key | Type / default | Meaning |
| --- | --- | --- |
| `gripper_up_axis` | signed axis string, `"y"` | Candidate rotation column: x/y/z, optionally prefixed with + or -. |
| `targrt_direction` | signed axis string or nonzero world `(3,)` vector, `"x"` | **Exact misspelled key** in the implementation. Correctly spelled `target_direction` is not read. |
| `threshold` | positive float in radians; code default 0.0 | Only angular deviations strictly below threshold survive. The zero default rejects all candidates. |

### pre_pose_noise

Insert approach dictionary: `position_noise` (default 0, symmetric uniform
translation half-range in **meters**) and `rotation_noise` (default 0, symmetric
uniform Euler xyz half-range in **degrees**). A scalar applies to all three
axes; NumPy arrays of shape `(3,)` allow per-axis ranges. Plain Python lists
are not supported by the helper's unary negation. Use nonnegative
ranges. Rotation perturbs Euler angles of the approach pose, not a collision
clearance guarantee. This differs from trajectory noise in radians and the
Gaussian error-injection standard deviations below.

### post_place_action

| Stage | Schema and defaults | Caveat |
| --- | --- | --- |
| Place | List of dicts: `gripper_cmd=None` (open/close/None), `distance=0.02` m, `direction=[0,0,1]` (nonzero passive-anchor-local vector, normalized) | Optional gripper action is followed by moves that also carry the ordinary release command; not a general retain-grip trajectory. |
| Insert | One dict: `distance=0.02` m, `direction=[0,0,1]` with the same frame/normalization | Does not iterate a list or read `gripper_cmd`; bypasses constrained/depth branch. Final planner spelling is `"simple"`, whose case is not normalized into backend `"Simple"`; it falls through the other dispatch path. Avoid treating it as a guaranteed backend withdrawal. |

### error_data

Normally omit. These deliberately create faulty collection actions.
`type` defaults to empty/None, `params` to `{}`. **Any nonempty type bypasses
PickStage's ordinary completion check**, including an unrecognized type.

| Type / nested field | Default and units | Effect |
| --- | --- | --- |
| `RandomPerturbations` -> `params.pos_std` | `[0.2,0.2,0.2]` m | Gaussian xyz displacement standard deviations for grasp perturbations. |
| `RandomPerturbations` -> `params.rot_std` | `[0.5,0.5,0.5]` radians | Gaussian local Euler xyz rotation standard deviations. |
| `MissGrasp` | no extra fields | Omits the ordinary close step in the Pick branch. |
| `WrongTarget` | no extra fields | Pick keeps only the first action and opens the gripper. |
| `KeepClose` | no extra fields | Pick keeps only the first action without a gripper command; Place/Insert suppress release; step adjusts obstacle exclusion in Pick. |
| `Drop` -> `params.drop_timing` | 0.2, normalized trajectory fraction (normally `[0,1]`) | Forces open, scheduling it when cuRobo command index reaches this fraction of planned samples. Not seconds. |
| `params.motion_run_ratio` | 1.0, normalized trajectory fraction (normally `(0,1]`) | Ratio used by `step` instead of the top-level Config field. Partial execution needs matching error semantics; synchronized dual cuRobo dispatch drops it. |

These options are not a recovery strategy for failed IK or an invalid grasp.

## Dual dispatch differences

DualSpecs build the same stages, but `dual_act` dispatch is different:

- Synchronized backend Simple/Straight motions send target poses, a fixed
  interpolation choice and `distance_frame=0.0008`. They do not forward stage
  offsets, weights, exclusions or slowdown. Extra `distance_frame` and
  `ee_interpolation` do not override those dispatcher choices.
- Synchronized cuRobo motions forward per-arm offsets, weights, frame flag,
  collision-link exclusions, relative flag and slowdown. They omit execution
  ratio, scheduled drop timing and consecutive-failure overrides.
- Mixed motion kinds fall back to sequential dispatch. This is not simultaneous
  motion, and preparation does not carry a configurable consecutive-failure
  budget. Do not infer identical option support from the same Config class.

Use separate phases if the task requires an option unavailable in its dual
path, and verify the resulting coordination/contacts physically.
