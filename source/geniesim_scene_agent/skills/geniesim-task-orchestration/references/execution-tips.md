# Task orchestration lessons and runtime pitfalls

These lessons come from source traversal of all 101 bundled `data_gen.py` files
and inspection of RobotAPI, action stages, trajectory resolution, robot dispatch,
planner goal composition, and server state/collision commands. They describe
code behavior and reusable decisions, not a newly measured success rate.
The [per-task review](example-review.md) links every reviewed example.

## Contents

- [Deliberate contact and collision scope](#deliberate-contact-and-collision-scope)
- [Frames and measured grasp transforms](#frames-and-measured-grasp-transforms)
- [Grasping, extraction and placement](#grasping-extraction-and-placement)
- [Precision insertion and closed-loop correction](#precision-insertion-and-closed-loop-correction)
- [Dual-arm coordination and ownership](#dual-arm-coordination-and-ownership)
- [Ordered trajectories and writing](#ordered-trajectories-and-writing)
- [Semantic selection and task variants](#semantic-selection-and-task-variants)
- [Checks, retries and cleanup](#checks-retries-and-cleanup)
- [Source traps to check before execution](#source-traps-to-check-before-execution)

## Deliberate contact and collision scope

A press, push, insertion, handle grip or writing stroke requires intended
contact. Identify the contact prim and geometry before executing that phase.

| Mechanism | Input | Actual effect | Appropriate use |
| --- | --- | --- | --- |
| `remove_objs_from_obstacle` / `add_objs_to_obstacle` | Absolute object or part prim paths | Server traverses **all descendant Mesh prims** and removes/adds them in the planner world | Temporarily permit a specific target contact while keeping physical collision. Parent exclusion can remove an entire appliance's geometry. |
| `disable_collision_links` in a supported motion config | Robot link names | Excludes those robot links from world-collision checks in the selected planner | A narrowly justified link-specific planning exception, after validating exact link names and dispatch support. |
| `disable_gripper_colliders` / `enable_gripper_colliders` | Arm selector | Changes gripper **physical** colliders | An explicitly chosen simulation model, e.g. a pin-based cloth grasp. Disabling them for a physical button press can prevent the press itself. |
| `skip_check` | Action-specific bool | Bypasses a default completion test where propagated | Does not change planning or physical collision and does not establish contact success. |

### Contact phase recipe

1. Read live target part pose, joint axis and limits; derive the contact surface,
   closed-finger/tool-tip position, approach direction and clearance.
2. Approach a safe pre-contact pose with ordinary obstacle checking.
3. Before contact IK/planning, temporarily exclude the **smallest required
   contact subtree** if the planner rejects the intended contact. If contact
   already exists in the start state, remove the target before solving that
   start/goal pair; otherwise START-state collision can reject every candidate.
4. Execute a bounded push along the measured axis. Keep unrelated walls,
   furniture and objects in the planner. Keep physical collision enabled when
   it is the mechanism that moves the target.
5. Check actual target part/joint displacement plus base stability, not just
   wrist arrival. Depending on the task, check while pressed or after release.
6. Withdraw along a clear axis, then restore this phase's planner exclusions.
   Restoring while the tool is still inside a target can invalidate the next
   start state. Track cleanup and recovery on failures as well as success.

[Button pressing](../../../examples/tasks/press_button_g2/data_gen.py) uses a
closed gripper, a relative constrained descent, planner exclusion and a live
button-part displacement checker. Its world-negative-Z constants, `part_00`,
and travel are specific to that button and robot pose. It does **not** directly
restore the planner obstacle. Use the phase structure, derive geometry and add
scoped restoration when adapting. The sequence-memory button examples also
move demonstration blocks with `set_object_parameter_action`; this is direct
scene state editing for the demonstration, not a robot manipulation.

For a door, exclude handle/door contact as needed, then follow the actual
revolute arc. For a drawer, use its prismatic axis and read its live travel.
For insertion, approach the opening before allowing interior contact. Excluding
an entire receiver can hide collisions with the opposite wall or gripper body.

## Frames and measured grasp transforms

Use explicit frame names for every transform. For a held object:

```python
T_object_ee = np.linalg.inv(T_world_object_now) @ T_world_ee_now
T_world_ee_goal = T_world_object_goal @ T_object_ee
```

Measure this after grasp closure/lift, and again after a possible slip. The
high-level object-pose APIs apply the current object-to-EE conversion; do not
pre-convert and then pass an EE pose with an object actor, which applies it
again. `actor="gripper"` targets the EE. Rigid bimanual targets are leader EEs.
For a pen, plug or straw, transform the actual tip/anchor offset, not the prim
origin. A matching object-center position can leave the mating tips misaligned.

For cuRobo `offset_and_constraint_in_goal_frame`:

- True composes `T_goal @ T_offset`: goal-gripper-local offset.
- False composes `T_offset @ T_goal` **in the robot base/planning frame**.
  Pure translation moves along base axes. A rotational offset also rotates
  the goal translation around the base origin; it is not an in-place wrist
  rotation. Use explicit absolute targets for an in-place reorientation.
- `from_current_pose=True` substitutes current EE for the goal before offset
  composition. With a rotated base, convert desired world displacement using
  `R_world_base.T @ delta_world`; world and base axes coincide only for an
  aligned base. The wrapper uses its configured initial base transform.

Path weights use `[rx,ry,rz,x,y,z]`; zero frees an axis. They are costs/weights,
not travel distances or allowable errors. Never copy `[0.1,...,0]` without
identifying the free axis and frame. Constructor reference and
[effective Config extensions](config-extra-params.md) state which paths consume
these options.

World AABB dimensions change with object orientation and do not locate hollow
interiors, shelf layers or contact planes. Use local mesh bounds/anchors with
current transforms, center offsets and instance scale. A prim origin need not
be a geometric center. Re-read furniture poses after moving a door/drawer or
holding a container; metadata describes the initial scene.

## Grasping, extraction and placement

- Inspect stored grasp annotations, then filter against contact direction,
  gripper opening and the **next action**. Percentile filtering is a local
  contact-coordinate filter; it does not directly express world height.
- For custom robot-gripper poses, use the expected runtime orientation convention;
  custom and stored annotations have different conversion paths. Check filtered
  candidates after flips/upside-down/posture filters, not only before them.
- `judge_arm_to_use` is a position heuristic. For crossed reaches, high shelves,
  wide loads or obstructed targets, test candidate IK for the relevant arms and
  review full approaches. Do not assume changing a robot suffix preserves EE
  axes, finger dimensions, grasp conversion or wrist reach.
- In a compartment, use a small lift that fits the ceiling, extract outward,
  then travel between surfaces. The default 0.12 m grasp lift can hit a shelf
  above. A zero lift can be appropriate for a handle or supported object, with
  a later explicit extraction.
- Candidate pools should vary only task-valid contact position/orientation.
  Some stages randomly downsample to 200 before full filtering and the lifecycle
  tries at most three sequences; a huge Cartesian product does not mean every
  candidate will be tried. Include a compact, diverse set of valid options.
- Build independent rotations from a fixed base per sample. Reusing an already
  rotated matrix inside nested loops accumulates angles and changes the intended
  distribution (the legacy pressing code has this pattern).
- Plan stable support and release height. Higher drops can bounce out of bowls;
  smaller candidate offsets and a measured bottom-to-rim gap help. Do not protect
  freely settling pile contents as stationary objects.
- For stacking, protect completed layers in subsequent phases and check them
  again at the end. Bowl/cup nesting needs interior/rim geometry and orientation,
  not the solid-box height recipe used for blocks.

Useful references: [shelf extraction](../../../examples/tasks/move_shelf_fruit_onto_plate_g2/data_gen.py),
[beverage shelf transfer](../../../examples/tasks/move_beverage_from_top_to_middle_shelf_g2/data_gen.py),
[three-block stacking](../../../examples/tasks/stack_three_building_blocks_g2/data_gen.py).

## Precision insertion and closed-loop correction

Use approach -> axis alignment -> entry -> measured descent -> release ->
withdrawal -> final verification. A nonzero **named** `pre_insert_offset`
selects InsertStage, whose constrained descent assumes negative base Z; it is
not a generic tilted/horizontal socket solver. Use explicit Move/path phases
for other axes. Review unsupported extras before borrowing an insertion config.

Thin or slippery parts need live tip/anchor, axial error, lateral error and
orientation checks. After each small correction, remeasure actual object
progress and the object-to-EE transform. Stop when corrections produce no
progress, slip exceeds a bound, aperture clearance is lost or the bounded
iteration budget ends. More commanded depth can merely slide the part in the
fingers or push the receiver. Derive tolerances from aperture clearance and
physical solver accuracy rather than convenient large center-distance values.

Some assembly scenes use bundled attractor/connection scripts. Read the selected
USD's script bindings and parameters: capture radius, anchor offsets and
orientation gates must agree with the task checker. A final small tip distance
alone does not prove a lasting connection. Check after releasing the manipulating
hand and an appropriate movement/settle interval. Simulation attachment changes
physics; do not infer a friction-only insertion from an attractor-assisted task.

## Dual-arm coordination and ownership

Separate independent dual actions from rigid carry. `dual_act` coordinates two
per-arm stage specs; `bimanual_rigid_move` preserves a measured relative EE
transform. Both require meaningful per-arm and shared physical checks. Always
unpack dual results; `(False, store_name)` is truthy as a Python tuple.

Before a rigid lift, establish both contacts, check both paths and choose the
leader deliberately. Candidate targets are leader EEs, not load centers.
Independent motions can stretch/rotate a shared load. Shared base-frame relative
moves also need the correct free axis. Review dual option forwarding: synchronized
backend and cuRobo dispatch do not consume the same extras as single-arm steps.

For a held bowl/cup/holder, choose holding and manipulating arms, re-read the
container pose after lifting/levelling, generate insertion in its **current**
frame and monitor tilt. Some examples attach both container and contents during
return transport. This can preserve contents but changes their free dynamics;
use only if that is the intended simulation model. Verify contents after release.

`handover_ownership_of_attached_object` opens the sending hand and requests
attachment to the other hand, then returns true without measuring reception.
Establish receiver contact first and verify load following after transfer.
`detach_obj` has no arm/object selector; do not use it while unrelated attachments
must remain. Track pin groups explicitly; `release_fem_pin("all")` is broad.

Examples: [long block across supports](../../../examples/tasks/place_long_block_across_two_supports_g2/data_gen.py),
[pot lift](../../../examples/tasks/lift_pot_g2/data_gen.py),
[held bowl](../../../examples/tasks/place_fruit_into_held_bowl_g2/data_gen.py),
[handover](../../../examples/tasks/hand_over_chips_bucket_g2/data_gen.py).

## Ordered trajectories and writing

`follow_trajectory` accepts `(N,7)` ordered xyz+wxyz waypoints or `(M,N,7)`
alternative complete paths. It converts actor goals through the measured grip.
A single-target pool `(N,4,4)` for Move is a different contract.

Trajectory noise is fallback **IK tolerance**, not batch generalization jitter:
solvable original waypoints stay unchanged; unsolvable points independently
sample bounded perturbations. The resolver scores successful complete paths and
can stop on the first zero-noise path. Independent corrections can distort a
stroke or an insertion axis. Use zero noise in contact-critical dimensions;
review continuity between samples and motion to the first waypoint. Larger
sample counts or wide angular ranges do not guarantee safe contact.

For writing, use `scene_info.target_letter` consistently in glyph and instruction.
Map strokes from the current paper frame using the actual pen tip transform;
hover, descend, draw, lift between strokes and retreat. The bundled A-Z example
returns based on motion results and reset, with no ink/glyph success checker.
Add an actual writing criterion when the requested output is a verified written
letter. Review its tolerated IK noise against the pen geometry and paper contact.

## Semantic selection and task variants

Symbolic tasks add selection rules before a shared motion recipe. Distinguish
number **value** from size, letter from asset filename, chess piece type/color
from instance ID, and a displayed reference from movable candidates. Validate
object cardinality and uniqueness. Repeated letters/digits in a word or answer
need separate instances; one selected object cannot occupy multiple final slots.

Use a task-defined local/robot frame for left/right, row order and chess squares.
Many examples assume the robot faces world +X and left is +Y, or use fixed table
bounds. A rotated table/board/base invalidates those constants. Generate goals
within usable support bounds and keep instruction ordering consistent with the
final arrangement. Sorting/sequence examples often check XY only; add support,
orientation, height and final global ordering checks.

For swaps, reserve a collision-free temporary slot or holding arm. Protect
uninvolved objects, remeasure both moved objects and check the final swapped
relationship after release. Scene variations can make the temporary area busy.

## Checks, retries and cleanup

Default stage checking is often loose (0.1 m / 70 degrees for the simple pose
check); Rotate returns true unconditionally, and relative Move defaults to
skip-check. Final task success must use a physical condition with task-scale
tolerances. Do not pass `pos_threshold`, `angle_threshold`, `checker` or
`checker_config` in Config extras expecting a generic completion checker.
Use zero-argument `extra_checkers` or an explicit final live-state check.

- Capture per-object loop values in factory closures; avoid late-bound IDs.
- Use named checker functions or lambdas: failure logging accesses `__name__`.
  A callable object or functools.partial without that attribute can itself break
  the failure path.
- Checker failures/exceptions fail a stage. False checks are sampled again after
  30 simulation steps, so keep checkers side-effect-free and define stable-state
  semantics. Do not equate one instantaneous overlap with lasting success.
- Check release/settling and the last retreat/reset; later wrist motion can pull
  out an insertion or sweep a finished structure. Retreat before reset when
  rotation near the contact target would be dangerous.
- Check both action return and achieved condition. `open_gripper`/`close_gripper`
  return None; attachment/obstacle wrapper booleans can indicate only that an RPC
  returned data. Measure the actual physical or planner effect.
- `run_stage` and `run_simple_action` return tuples, despite a bool hint on the
  former. High-level action wrappers unpack them. Use the public wrapper or
  unpack explicitly. Never turn a failure tuple into success with `bool(tuple)`.
- Simple-action lifecycle checks compare cached object poses without a post-
  callback refresh. Open/FollowTrajectory callbacks do not refresh them; use
  live pose-query closures to enforce critical stationary-object checks.
- Lifecycle retry budgets are fixed at three; a callback's terminate flag can
  stop earlier. `extra_params.max_attempts` is not a supported control. Repeating
  a physically impossible goal with increasingly wide noise is not recovery.
- Server snapshots cover rigid/articulation state, attachment state and generated
  geometry checkpoints. They do not capture a complete ledger of task-created
  FEM/rigid pin groups, collider toggles or arbitrary USD attribute edits/planner
  exclusions. Do not rely on playback/reset to restore all task side effects.
- Track resources introduced by this phase, restore them on exits, and surface
  cleanup failure. Use `try/finally` where appropriate without a return in finally
  that would hide the action result. Do not restore somebody else's exclusions.

## Source traps to check before execution

| Observed behavior | Avoidance |
| --- | --- |
| Extras serialize over named fields but attributes/Actions/dispatch may supersede them | Read the per-Config extension guide; prefer named fields. |
| Move `use_near_point` reaches an undefined helper; ordinary Move/Place `use_pre_place` passes an unsupported generator argument | Supply measured candidates and a separate explicit approach; do not enable these branches. |
| Pick pre-grasp is hardcoded on; `pre_grasp_vector` / `pick_up_type` are in the inactive alternative branch | Configure the active approach distance or write separate measured moves. |
| `filter_grasp_pose` reads misspelled `targrt_direction`, and angular default 0 rejects all | Use the exact key and explicit positive radians threshold. |
| Pick lift IK tests grasp-local offsets; execution uses base-frame offsets | Check rotated-grip extraction; use an explicit later move when necessary. |
| Plain Simple motion can retry failed endpoint IK with ten random orientation perturbations of ±10 degrees | Tight-contact tasks need a final orientation/contact gate; a nominal target alone does not constrain this fallback. |
| cuRobo slowdown is clamped to at least 1.0 and adjusted for physics timestep | Use factors >=1 for slowing; values below 1 do not speed it up. Rigid dispatch currently drops Config slowdown. |
| Rigid `motion_run_ratio` is overwritten by the stage executor; synchronized dual options differ | Do not author partial-run behavior from that field; split explicit motion phases. |
| FollowTrajectory extras do not control noise, thresholds or planner flags | Use named noise fields, ordered valid paths and real completion checks. |
| Cloth example computes `final_ok` then returns True regardless | In adapted tasks, make the final fold checker gate success and verify after releasing pins. It also disables physical gripper colliders without restoring them. |
| Writing returns motion/reset result without a glyph checker | Add ink/contact/glyph validation where writing success is required. |
| Several examples use XY-only containment, fixed coordinates or asset part names | Re-derive geometry/frames and add height/interior/support/stability checks. |
| 26 examples call planner exclusion; none directly call the matching restore wrapper | Treat exclusions as task-local state and review cleanup. This source observation alone does not describe runner-wide reset behavior. |

Use [task validation](../../geniesim-task-validation/SKILL.md) for actual fixed-scene
physical execution and variation coverage. Static source review cannot establish
successful contact or recorded-data quality.
