# genie_sim_planning

Python tooling and demo scripts for the GenieSim chassis pipeline —
teleop, navigation, follow-trajectory glue, and a small library of
helpers shared between them.

The Python helpers and demo scripts live here. The C++ chassis servo,
ServoBase strategies, MPC solver and tracking differentiator belong to
[genie_sim_controllers](../genie_sim_ros_control/genie_sim_controllers/).

## What's here

- **`genie_sim_planning/`** (Python package) — reusable helpers:
  `kinematics`, `math_utils`, `path_utils`, `planner`, `qos`,
  `tf_utils`.
- **`scripts/`** — small, runnable demos:
  - `simple_move_base.py` — `/cmd_twist` publisher in the teleop style.
  - `simple_follow_trajectory.py` — `FollowJointTrajectory` demo driver.
  - `simple_navigation.py` — `/cmd_twist` from Nav2 goals.

## When you'd touch this package

- Adding a Python helper that several chassis scripts will share —
  put it in `genie_sim_planning/`.
- Writing a quick teleop / navigation demo — drop a script in
  `scripts/`.
- **Don't** add new C++ chassis-servo code here — extend
  `genie_sim_controllers` instead.

## Mechanics

The Python helpers require NumPy and SciPy. The clothoid generator in
`planner.py` also requires `pyclothoids` in the ROS Python environment.

See [AGENTS.md](AGENTS.md) for the layout and routing rules.
