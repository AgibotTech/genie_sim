# genie_sim_moveit

MoveIt 2 configuration package for the Genie G2 robot family.  Holds the
SRDF, kinematics solver configs, OMPL planning configs, joint-limit
overrides, ros2_control wiring, and the canonical launch files for
`move_group` + RViz.

The package is **cross-distro** (ROS 2 Humble and Jazzy).  Distro-specific
C++ compatibility is centralised in
`genie_sim_moveit_plugins/include/genie_sim_moveit_plugins/moveit_compat.hpp`
(field-name macros, `solve()` return-type macros, `.h` vs `.hpp` include
routing). `wbc.launch.py` also normalises the OMPL parameter schema
for Humble; the YAML stores the Jazzy plugin and adapter lists.

---

## Launching

```bash
ros2 launch genie_sim_moveit wbc.launch.py
```

Brings up `move_group` + the WBC-configured RViz with the Genie's full
mobile-base + dual-arm planning groups.  See `launch/wbc.launch.py` for
the parameter knobs (workspace bounds, start-state tolerance, etc.).

---

## Architecture: planar + prismatic floating base

This package targets **high-precision whole-body-control (WBC) and
mobile manipulation** workflows where MoveIt must agree with the
simulator on `base_link`'s exact world pose -- including the live
chassis Z, which varies during operation:

  * The robot rides at a non-zero height at spawn (typically ~4 cm
    above the floor) and continues to ride at that height in steady
    state when wheels are on flat ground.
  * The chassis bobs by a few millimetres under load when the WBC
    swings the torso forward or the gripper carries a payload.
  * Driving onto a small platform / curb adds the platform height to
    base_link's Z until the robot drives off again.

If MoveIt's RobotState is locked to `base_link.z = 0`, every grasp
planned against a world-frame collision object lands at the wrong
absolute height in the simulator.  At centimetre-scale precision this
breaks pick-and-place, peg insertion, and any task where the gripper's
Z relative to a fixed-frame target matters.

MoveIt 2's planar virtual_joint stores `(x, y, theta)` only -- it does
not represent Z.  The architecture below decouples Z from the planning
state and routes it through standard `/joint_states` plumbing:

  * **Planar virtual_joint** (`type="planar"`) anchors the
    chassis-ground projection and is the active 3-DoF planning state
    for any group that includes the chassis.  This is what OMPL,
    bio_ik, and the chassis trajectory controller all see and reason
    about.
  * **Passive prismatic Z joint** between `base_footprint` and
    `base_link` carries the live ride height.  Its position is
    synthesised from the simulator's `odom -> base_link` /tf each
    tick and published on `/joint_states`.  MoveIt's
    `robot_state_publisher` applies it like any other URDF joint --
    no special handling required, because prismatic is upstream's
    most-loved joint type.

Net effect: planning is exactly 3-DoF (the controller can only drive
the chassis in x, y, yaw anyway), and visualisation / FK / collision
checks see the chassis at its true world Z continuously.

### TF chain

```
world  (static identity, published by the simulator)
 -> map    (static identity)
   -> odom (dynamic identity, published every tick by ``fake_slam``)
     -> base_footprint  (planar virtual_joint, x/y/theta)
       -> base_link     (prismatic Z joint, value = base_link's live z)
         -> body_*, arm_*, head_*  (RSP from URDF + /joint_states)
```

The simulator publishes `odom -> base_link` directly as a single TF
edge (it doesn't know about `base_footprint` or the prismatic joint).
On the MoveIt side, the URDF declares the prismatic, the SRDF marks it
passive, and a small relay node synthesises the prismatic's
`/joint_states` value from /tf each tick.  RobotState then resolves
`world -> base_link` by composing
`world -> ... -> odom -> base_footprint (planar) -> base_link (prismatic)`,
which exactly matches the simulator's `odom -> base_link` edge.

### Where the floating-base joints are declared

The chassis abstraction lives entirely in MoveIt's URDF view.  The
canonical robot description that feeds the simulator's
`assemble_robot` pipeline (and downstream `robot.usda` cache) treats
`base_link` as the kinematic root.  MoveIt extends that root upward
with `base_footprint` and the prismatic Z joint by declaring them in
the **MoveIt-side xacro entry point**, which is separate from the
simulator's xacro tree.

`config/genie.urdf.xacro` includes the canonical robot URDF and
prepends the floating-base joints **before** the include:

```xml
<link name="base_footprint">
  <inertial>...tiny mass, ignored by planning...</inertial>
</link>
<joint name="base_footprint_to_base_link" type="prismatic">
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <parent link="base_footprint"/>
  <child  link="base_link"/>
  <axis xyz="0 0 1"/>
  <limit lower="0.0" upper="0.5" effort="0" velocity="0"/>
</joint>
<xacro:include filename="$(arg urdf_file)"/>
```

`base_footprint` becomes the URDF root (no parent joint).  The
simulator never sees this xacro -- it builds its USD from a different
xacro tree under `genie_sim_robot_model/xacro/robot.xacro`.  This
separation keeps the simulator focused on real physics (no fictional
prismatic joint to integrate) while giving MoveIt the rich enough
kinematic tree it needs for precision WBC.

| View | Sees `base_footprint`? |
|---|---|
| Simulator USD / physics | No -- canonical URDF only |
| URDF-to-USD cache | No -- canonical URDF only |
| MoveIt RobotModel | Yes -- this xacro declares it |
| `robot_state_publisher` | Yes -- publishes `base_footprint -> base_link` from /joint_states |
| RViz `RobotModel` displays | Yes -- consumes RSP's TF |

### SRDF wiring

`config/genie.srdf` declares the planar virtual_joint at
`base_footprint` (NOT `base_link`) and marks the prismatic as passive
so the planner never samples over it:

```xml
<virtual_joint name="planar_joint" type="planar"
               parent_frame="odom" child_link="base_footprint"/>
...
<passive_joint name="base_footprint_to_base_link"/>
```

### Ride-height bridge node

`scripts/moveit_joint_states_bridge.py` runs alongside `move_group`.
Two responsibilities, one process:

  1. **QoS bridge.**  The simulator publishes `/joint_states` with
     SensorData QoS (BEST_EFFORT).  MoveIt subscribes Reliable.
     Mismatched QoS silently fails the handshake, so we re-publish to
     `/moveit/joint_states` (Reliable) and remap MoveIt to subscribe
     there.  See `MOVEIT_MOVE_GROUP_REMAPPINGS` in
     `moveit_launch_utils.py`.

  2. **Ride-height synthesis.**  Looks up `odom -> base_link` on /tf
     each inbound JointState message, extracts the Z translation, and
     appends a `base_footprint_to_base_link` entry to the message
     before forwarding to `/moveit/joint_states`.  Without this entry
     MoveIt's PlanningSceneMonitor warns "Missing
     base_footprint_to_base_link" and FK puts base_link at z=0.

Cost: one Python node per launch, runs at the same rate as the
simulator's `/joint_states` (~100 Hz), microseconds of TF lookup per
tick.

---

## Other launch-side fixes

### OMPL parameters on Humble and Jazzy

`wbc.launch.py` selects the OMPL schema using `ROS_DISTRO`. On Humble,
`planning_plugin` is a string, `request_adapters` is a space-separated
string using `default_planner_request_adapters/*`, and
`AddTimeOptimalParameterization` runs in that request chain. On Jazzy,
the YAML's `planning_plugins`, `request_adapters`, and `response_adapters`
remain lists, with time parameterization in the response chain. Both
forms retain the same planner IDs and per-group tuning.

### URDF joint-limit padding

MuJoCo's joint limits are soft (Baumgarte-stabilised impulse
constraint), so simulated joints can drift past the URDF `<limit>` by
sub-mrad amounts under contact load.  MoveIt 2's
`CheckStartStateBounds` reads URDF limits directly and rejects the
request.

`wbc.launch.py` calls `pad_urdf_joint_limits` (from
`genie_sim_robot_model.urdf_utils`, a shared helper across packages)
on `moveit_config.robot_description` before `to_dict()`, widening every
revolute `<limit>` by ±0.01 rad and every prismatic by ±1 mm.

### `fix_start_state: True`

Even with the URDF pad, occasionally a joint drifts past the limit by
more than the pad.  Jazzy's `CheckStartStateBounds` adapter has a
renormalisation pass that nudges the start state back into bounds,
but only when its `fix_start_state` parameter is `true`; upstream
default is `false`.  `wbc.launch.py` flips it.

### `start_state_max_bounds_error: 0.2`

Per-pipeline OMPL knob that lets `FixStartStateBounds` nudge revolute
joints up to 0.2 rad (~11.5°) back into bounds.  Defaults to 0.05
which isn't enough for some of our sub-tree composite groups.

### `default_workspace_bounds: 200.0`

Per-pipeline OMPL knob.  Without this, `FixWorkspaceBounds` clamps
`planar_joint/x` and `planar_joint/y` to a narrow default box that
excludes the actual `odom -> base_link` pose, producing
"Skipping invalid start state" rejections on chassis groups.  200 m
matches the simulator's planar workspace.

---

## Files

| Path | Owner |
|---|---|
| `config/genie.urdf.xacro` | base_footprint + passive prismatic Z joint injection (MoveIt-only) |
| `config/genie.srdf` | virtual_joint = planar to base_footprint; passive_joint declaration |
| `config/ompl_planning.yaml` | plain RRTConnect on every group |
| `config/joint_limits.yaml` | planar `{x, y, theta}` + `base_footprint_to_base_link` (passive) |
| `config/kinematics.yaml`<br>`config/kinematics_vanilla.yaml` | bio_ik chassis_posture pins `[planar_joint/x, planar_joint/y, planar_joint/theta]` |
| `config/moveit_controllers.yaml` | controller -> joint mapping |
| `config/ros2_controllers.yaml` | hardware-side controller config |
| `launch/wbc.launch.py` | `move_group` + RViz; sets `fix_start_state`, workspace bounds, joint-limit pad |
| `launch/moveit_launch_utils.py` | `moveit_joint_states_bridge_node()` factory |
| `scripts/moveit_joint_states_bridge.py` | merged QoS bridge + ride-height JointState synthesiser |

---

## Build

```bash
cd /workspace/source/geniesim_ros/src/ros_ws

colcon build --packages-select \
    genie_sim_moveit \
    genie_sim_robot_model

source install/setup.bash
ros2 launch genie_sim_moveit wbc.launch.py
```

---

## Cross-distro maintenance

When you upgrade ROS / MoveIt, expect to refresh:

1. **`launch/wbc.launch.py`'s `fix_start_state` parameter** if jazzy's
   `CheckStartStateBounds` adapter is replaced with something whose
   parameter API changes.

2. **`genie_sim_moveit_plugins/include/genie_sim_moveit_plugins/moveit_compat.hpp`**
   if any new field rename or `solve()` signature change happens
   upstream (already covers the Humble→Jazzy delta).

3. **`scripts/moveit_joint_states_bridge.py`** is plain rclpy + tf2
   and should keep working as-is across distros.
