# genie_sim_controllers

ros2_control controller plugins for GenieSim — currently the
**four-wheel-steering chassis servo** that drives the mobile base.
Takes `/cmd_twist` from a teleop or planner, solves the wheel-side
control problem (MPC via OSQP), and publishes `/cmd_4ws` to the
simulator.

## Plugin

### `ChassisServoController` (`controller_interface::ControllerInterface`)

A ros2_control controller that wraps the chassis servo core. Doesn't
claim any hardware interfaces — it's purely topic-driven (same
pattern as `PlanarBaseController` in `genie_sim_control`). Sits next
to MoveIt's other controllers under the same `controller_manager`.

Internally selects a `ServoBase` strategy at runtime
(`GeneralServo` / `OptimalServo` / `ParkingServo` / `SpinServo` / `SelftestServo`). The
`optimal` variant is the MPC path: builds a 4WS twist problem and
solves it with the bundled OSQP glue.

## Two deployment forms

| Form | Entry point | When you'd use it |
|---|---|---|
| ros2_control controller (default) | `chassis_servo_controller` plugin loaded by `controller_manager` | Run alongside MoveIt under one controller_manager. |
| Standalone executable | `servo_4ws.launch.py` | Quick teleop / hardware bring-up without ros2_control overhead. |

Both forms share `servo_core.cpp` — fix the control law there, not in
either wrapper.

## Quick teleop

```bash
# Joystick → /cmd_twist
ros2 run genie_sim_controllers teleop_joy.py
```

## Platform tuning

Per-platform gains and geometry live in `config/`:

- `4wd.yaml` — full four-wheel-drive + four-wheel-steer chassis
- `2wd.yaml` — diagonal two-wheel-drive chassis (two driven+steered corners, two passive casters)

The installed `genie_g2.yaml` is an alias of `4wd.yaml` for current G2
bringup configuration lookup. The standalone launch defaults to `4wd.yaml`.
In both deployment forms, empty steer or drive joint names mark passive
corners; the optimal servo excludes them from no-slip constraints and
steering-alignment gating. The default mask keeps all four corners active.

## When you'd touch this package

- Tuning the chassis servo (MPC weights, gains, wheel geometry).
- Adding a new `ServoBase` strategy under `src/plugins/`.
- Changing the QP problem formulation in
  `src/solvers/` or `four_wheel_car_twist_problem.hpp`.

## Mechanics

See [AGENTS.md](AGENTS.md) for the layout, plugin registration, where
`/cmd_twist` comes from (`PlanarBaseController` in `genie_sim_control`),
and the strategy / solver architecture.
