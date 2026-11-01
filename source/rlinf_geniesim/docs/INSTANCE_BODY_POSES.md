# Per-instance body poses

`task_loader.build_stage_manifest` captures each original instance's composed
USD rigid-body transforms before selecting shared superset subtrees. A child
`entity` translation or rotation belongs to that instance; it cannot be recovered
by applying one build instance's fixed offset to every locator.

## Bake and reset contract

1. Resolve slots and `name_alias` once.
2. Apply the existing shallow-spawn-penetration lift to each original instance.
   Work on private layers and anchor relative asset paths; source files and USD's
   cached source layers remain unchanged.
3. Bake composed body poses from those corrected scenes.
4. Assemble startup from the same corrected scenes and pairing. Preserve full
   ancestor transforms and parked p/q, including scale. Keep original transform
   properties effective for graph writers; do not move graph prims.
5. Verify assembled startup body poses against the selected instance's table.

Manifest remains **v4**. The referenced scene config adds:

```json
{
  "instance_switch": {
    "body_pose_version": 1,
    "build_instance": 0,
    "group_instances": [0, 1],
    "instance_body_poses": {
      "0": {"s0_asset/entity": [0.4, 0.2, 0.8, 1, 0, 0, 0]},
      "1": {"s0_asset/entity": [0.5, 0.2, 0.8, 1, 0, 0, 0]}
    }
  }
}
```

Keys are exact body paths below `/World/Objects`, including intermediate asset
layers. A rigid body on the slot itself uses the slot name. Values are env-local
metres and scalar-first quaternions `[x,y,z,qw,qx,qy,qz]`. Inactive slots have
complete parked body poses: tens of metres from the table, at table height.
The fused node turns their gravity off (a slot is inactive for an instance that
does not name it in `name_alias`), so they stay where they were parked. With
gravity they would fall to the floor and roll for the whole episode, which
wastes contacts and makes the rest-state bake report every instance as still
moving. Existing `instance_poses` still describes locators;
`name_alias` remains the single source of member/slot pairing.

Fused adds each env origin and writes only selected tensor indices, including
zero velocities. Newton converts body poses to FREE-joint coordinates using
`inverse(parent_anchor) * body * child_anchor`, then stores xyzw. It retains the
existing robot, seeded-randomization and solver-reset sequence. Neither backend
uses a donor-instance locator/body correction or live settled pose to reconstruct
the selected instance's body target.

Reset with no new instance (`-1`) restores the current instance. Fused reset
settling still advances all environments; the isolation assertion applies to
explicit teleport and velocity writes, before that settling. With the rest-state
bake below the settle is zero, so nothing else advances.

## Rest-state bake

The authored table comes from the scene files, so objects can start a few
millimetres (or, in scenes nobody settled, 12 cm) above their support. Without
a bake the server free-runs physics for `reset_settle_time_s` after every
reset, and every other env on the server runs physics with it.

`rlinf_geniesim/reset_bake.py` runs each instance in the live engine at task
load until its objects stop moving, and writes where they ended up into the same
`instance_body_poses` table. Both engines then reset straight into the rest
state with no engine change, and the server sets the settle after a reset to
`settle_after_s` (default 0). There is one kind of reset, and it returns with
every reset env ready to act. On the fused engine no other env's physics
advances during it; Newton's engine keeps pace with the wall clock, so there the
other envs move on only by the ~0.03 s the reset takes.

The scene config gains three keys next to the table:

| Key | Content |
|---|---|
| `instance_body_poses_authored` | The table as the stage builder wrote it. It, not the baked table, goes into the cache key. |
| `instance_layout` | Per instance: each object's env-local box and what carries it (a plate on the table, a cube on the plate), plus the box of each surface (`table`). The position randomization uses it; see below. |
| `rest_state_baked` | Bake statistics: wall time, and per instance the time it took to come to rest, the motion left at the end and how far objects moved sideways from the authored pose. |

How the bake decides an instance is done:

* **At rest**: objects move less than 0.1 mm over 0.25 s.
* **Jitter**: still moving at `cap_s` (1.5 s), but less than 2 mm per 0.25 s.
  This is persistent contact jitter, which no amount of waiting removes. The
  instance is recorded as it stands and named in the log.
* **Still settling**: moving more than 2 mm per 0.25 s at `cap_s`. It runs on,
  up to `hard_cap_s` (5 s). Something that is still moving at the hard cap is
  an unstable layout (Newton's `open_door` trash can is one); the log names it.
* Objects that end up more than 2 cm sideways from their authored pose are
  named in the log: they slid, rolled or fell off their support. Landing
  straight down is not reported; that is what the bake is for.

Each env takes the next instance as soon as its own comes to rest, so one slow
instance does not hold up the rest. Bodies the engine does not simulate (a
static table in Newton) keep their authored pose; their boxes come from the
stage USD (`usd_static_boxes`) so the layout still has the table top.

The result is cached in `RLINF_REST_BAKE_CACHE` (default
`~/.cache/rlinf_geniesim/rest_bake`), keyed by the engine, solver substeps, the
authored table, the generated scene files, the physics parameter files and the
physics environment variables. The floor's size is left out: it grows with the
number of envs, and the rest state does not depend on it, so a task baked at one
env count is a cache hit at any other. Only the first load of a task on an engine pays
for the bake. Each cache file keeps a digest of every one of those inputs; on a
miss the log names the inputs that differ from the task's last bake
(`cache miss: this task's last bake (...) differs in: <stem>_scene_env.usda`). A bake that fails is reported
loudly and the task runs on the authored poses with the configured settle.

Task config (`task.rest_state_bake`, all optional):

```yaml
task:
  rest_state_bake:
    enable: true          # false: authored poses and the configured settle, as before
    cap_s: 1.5            # soft cap: jitter stops here
    hard_cap_s: 5.0       # an instance still clearly settling runs on up to here
    settle_after_s: 0.0   # settle after a reset once the bake is in place
    force: false          # true: ignore the cache and bake again
```

The bake is skipped for a static `manifest.path` (a scene this server did not
build) and for physics backends without the needed engine calls. The render
hot-swap does not bake; a task without a cached bake goes through the cold
rebuild path, which bakes.

### Position constraints

Seeded resets move objects by a random xy offset of up to `object_xy_radius`.
Both engines draw it with the same code
(`genie_sim_engine_core.common.reset_randomization`), and when the instance has
a layout the draw keeps the layout intact:

* Objects that were stacked or touching move together, by the offset drawn for
  the lowest one: the cube stays on its plate.
* No two objects that were apart overlap after the move (2 mm clearance).
* An object stays on what carries it, 5 mm inside the carrier's edge, and does
  not move onto another surface.

When a draw breaks a rule, that group is redrawn from its own random stream, up
to 20 times; after that it stays where it was. An object that is not stacked
on or touching another and breaks no rule keeps exactly the offset the plain
draw gives it, so for such objects a seed reproduces the position it gave
before the constraints existed. `randomization.object_constraints: false` turns
the rules off.

## Existing artifacts and incompatible instances

Task initialization must rebuild legacy artifacts that have `instance_switch`
but no versioned body table. Both consumers reject those artifacts with a rebake
message. Scenes without `instance_switch` retain their existing reset behavior.
OpenUSD is required for task-manifest construction; path/group discovery
remains usable without loading USD.

Only body placement differences can share one physical structure. Different
geometry, collision/mass/scale, effective physical material parameters,
rigid-body hierarchy or joint structure require separate slots or rebuilding.
Superset slots are keyed by asset and physical signature. Signatures are
recomputed from saved source scenes and dependencies on each resolution, so
editing an instance or asset does not require restarting the server to discard
a cached variant assignment. A shared file-backed USD layer with unsaved edits
is rejected before refreshing; resolving variants never discards those edits.

Static/kinematic/world-constrained structures cannot be moved or parked by
Newton's FREE-joint reset. The baker rejects these transitions even when variant
splitting gave them separate slots. An unchanged static support present in every
instance remains valid. A floating articulation whose
joints stay inside one slot may move as a whole when its internal transforms,
joint frames and DriveAPI parameters match. The baker reports the slot, instance
and differing structure instead of silently selecting one donor's settings.
OmniGraph default enablement is unchanged.

The pen wrapper correction and variant slots allow `clean_the_desktop` to retain
each instance's physical structure. The isolated `sft_pen_001` supports distinct
per-instance body transforms:
instance 1's body x is `0.358308192`, while its locator x remains `0.452` metres.

A raw-scene superset/body-pose bake survey across every task group found one
rejection: `place_object_relative_position` instance 6, whose referenced
`_Aligned.usd` asset is missing.

## Background mechanisms in fused resets

The `instance_body_poses` table covers `/World/Objects`. The door in `open_door`
belongs to `/Background`, carries `ArticulationRootAPI`, and has separate door
and handle joint coordinates. Fused captures background articulation states
immediately after PhysX initialization, before camera updates or simulation.
Reset restores the selected environments' root poses, joint positions and
authored drive targets, and clears velocities and applied efforts. Rebuilding
the stage creates new views and snapshots; it never reuses a previous episode's
open-door state. Explicit writes preserve other environments; reset settling
still advances the shared physics scene.

Background objects can be mounted two ways: the legacy isolated mount at
`env_i/Background/objects/...`, or composed into the foreground at
`env_i/World/Background/objects/...`. Joint discovery and readback cover both
mounts.

The background views publish live joint positions in PhysX DOF order, with
env-relative USD paths identifying the owning object, so a success checker
that reads background-object joint state (for example `open_door`'s push/pull
checker) gets real data. Schema field `scene_joint_paths` and the separate
`FusedStateShm.scene_joint_positions` region carry those values through every
STEP/RESET, including steps without images. Robot action and policy-state
dimensions are unchanged; a missing joint value stays NaN rather than shifting
another joint into its index. The wire version is `FUSEDST3`; both sides must
restart together.

## Fixed robot tools and mimic reset poses

The RLinf baker consumes optional `robot.fixed_attachments` in the benchmark
robot JSON. Entries are paths relative to the robot's default prim, for example
`gripper_r_base_link/Aligned` in `G2_omnipicker_scoop.json`. It writes a robot
composition layer and points the manifest's existing `robot_usda` field to it;
manifest v4 is unchanged. Original asset files remain untouched. Rebuild the
task manifest after changing this configuration.

Configured tools retain their local transform and collision geometry but lose
their separate rigid-body and mass APIs. Their colliders belong to the existing
rigid parent link, so they follow the hand through motion, cloning and reset.
This represents a permanently attached tool using the parent link's authored
mass; it does not preserve the tool as a separate 1 kg body. Joint-connected
tools, reset-transform stacks and nested rigid bodies are rejected rather than
silently rewritten. This configuration is consumed by the RLinf baker, not the
legacy benchmark runtime.

Fused resolves authored PhysX revolute mimic relations after assigning the task's
master joint angles and before writing any joint positions. The saved robot
reset pose includes these follower angles. Restoring a master to +45 degrees
while its follower stays at zero violates the G2 gripper's mimic constraint;
the first solver tick can strike nearby objects. Synchronizing followers avoids
that impulse while retaining all scene and robot collisions.

`scoop_popcorn` also needs room between the idle left hand and the authored
bucket spawns: instance 5 overlaps the gripper base even with a settled linkage.
Its task-only `G2_STATES_6` left shoulder angle is reduced by 30 degrees
(`0.739033` to `0.215434` radians), raising the hand away from the bucket area.
The right arm and all instance object poses are retained. This changes the
task's initial robot pose in both benchmark and RLinf; previous policy success
rates are not a validation of the adjusted starting configuration.

The tool-follow probe reads Fabric's `omni:fabric:worldMatrix` after a hierarchy
update. Avoid constructing a new `XFormPrim` Fabric view just to inspect a rigid
link's child: in the installed Isaac build its first read can author an absolute
world transform from stale USD and change the child being measured.

## Hoisting child offsets safely

`tests/tools/hoist_body_local_offsets.py` only rewrites offsets it can preserve.
It refuses bodies referenced by joints, geometry/colliders on the body root,
affected transform relationships or connections, time-sampled transforms, and
reset-transform stacks. These checks precede the default-time geometry check;
matching that single frame alone cannot establish equivalent runtime behavior.
Refused candidates leave the original scene unchanged, including with `--apply`.
