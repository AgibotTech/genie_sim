# genie_sim_controllers

ros2_control **controller plugins** for GenieSim. Currently ships the
four-wheel-steering chassis servo and its supporting linear-MPC solvers
(OSQP) + tracking-differentiator smoother.

Source: [source/geniesim_ros/src/ros_ws/src/genie_sim_ros_control/genie_sim_controllers/](.)
License: [Mozilla Public License Version 2.0](../../../../../LICENSE)

**Maintenance contract**: when you add or rename a controller plugin or a
servo strategy, update `plugin_*.xml`, the
`pluginlib_export_plugin_description_file(...)` call in `CMakeLists.txt`,
and this file in the same diff. New ServoBase subclasses go under
`src/plugins/`; construct and register each strategy in `ServoCore::configure`.

---

## Layout

```
genie_sim_controllers/
├── include/genie_sim_controllers/
│   ├── chassis_servo_controller.hpp   ← ros2_control wrapper (ControllerInterface)
│   ├── servo_node.hpp                 ← standalone executable form
│   ├── servo_core.hpp                 ← shared servo loop (used by both forms)
│   ├── servo_base.hpp                 ← ServoBase pluginlib API
│   ├── four_wheel_car_twist_problem.hpp ← MPC problem definition (4ws kinematics)
│   └── tracking_differentiator.hpp    ← reference-signal smoother
├── src/
│   ├── chassis_servo_controller.cpp   ← ControllerInterface implementation
│   ├── servo_node.cpp                 ← standalone node
│   ├── servo_core.cpp                 ← /cmd_twist → /cmd_4ws solver loop
│   ├── main.cpp                       ← rclcpp main for the standalone node
│   ├── plugins/                       ← ServoBase strategies (pluginlib)
│   │   ├── general_servo.cpp
│   │   ├── optimal_servo.cpp          ← MPC (OSQP) path
│   │   ├── parking_servo.cpp
│   │   ├── selftest_servo.cpp
│   │   └── spin_servo.cpp
│   └── solvers/                       ← OSQP-backed quadratic-program glue
│       ├── osqp_solver.cpp
│       ├── problem.cpp
│       ├── quadratic_function.cpp
│       ├── relative_function.cpp
│       └── scalar_function.cpp
├── config/
│   ├── 4wd.yaml                       ← 4WD/4WS chassis tuning (wheel base, gains)
│   └── 2wd.yaml                       ← diagonal 2WD chassis tuning
├── launch/servo_4ws.launch.py         ← bring up the standalone servo_node form
├── scripts/teleop_joy.py              ← joystick driver publishing /cmd_twist
└── plugin_chassis_servo_controller.xml
```

---

## Plugins

### `genie_sim_controllers/ChassisServoController`
`base_class_type=controller_interface::ControllerInterface`

Four-wheel-steering chassis servo packaged as a **ros2_control controller**.
Doesn't claim any hardware interfaces — same topic-driven pattern as
`PlanarBaseController` (see `../genie_sim_control/`). Subscribes
`/cmd_twist` + `/joint_states`, runs the `servo_core` loop, publishes
`/cmd_4ws`.

### Internal: `ServoBase` strategies (`src/plugins/*.cpp`)

Strategies constructed directly inside the servo. `servo_core` selects one
at runtime by class name (`GeneralServo`, `OptimalServo`, `ParkingServo`,
`SpinServo`, `SelftestServo`). `optimal_servo` is the MPC variant — it builds a
`FourWheelCarTwistProblem` and solves it with the OSQP glue under
`src/solvers/`.

---

## Two deployment forms

| Form | Entry point | When to use |
|---|---|---|
| ros2_control controller (default) | `chassis_servo_controller` plugin in `ros2_controllers.yaml` | Run alongside MoveIt / other controllers under the same `controller_manager`. |
| Standalone executable | `servo_4ws.launch.py` → `servo_node` | Quick teleop / hardware bring-up without ros2_control overhead. |

Both forms share `servo_core` — fix bugs there, not in either wrapper.

Both wrappers derive `CommonParams.active_wheels` from the four steer/drive
joint-name pairs. A corner is active only when both names are non-empty.
The optimal strategy propagates that mask through its QP constraints and
alignment gate; passive corners must not constrain chassis twist.

`4wd.yaml` is also installed as `genie_g2.yaml` for the current bringup
lookup. Keep the alias sourced from the same file in CMake; do not maintain
a second tuning copy. The standalone launch uses `4wd.yaml`.

---

## Verification

| Test | Contract |
|---|---|
| `test/test_chassis_kinematics.cpp` | Diagonal 2WD and full 4WD track forward, reverse, lateral, yaw, and combined twist; passive corners do not block motion. |
| `test/test_teleop_joy.py` | Released/noisy triggers emit zero motion; forward/reverse triggers preserve sign and magnitude in optimal and general modes. |

Build with tests enabled, then run from the ROS workspace:

```bash
colcon test --packages-select genie_sim_controllers
colcon test-result --verbose
```

The package also registers ament source linters. Functional tests use an ideal
wheel plant and do not establish simulator contact or hardware performance.

---

## Routing rules

- ros2_control controller wrapper → `src/chassis_servo_controller.cpp`
- Servo loop (the actual control law) → `src/servo_core.cpp`
- ServoBase strategies → `src/plugins/*.cpp`
- OSQP problem glue → `src/solvers/*.cpp`
- Platform tuning (gains, geometry) → `config/{4wd,2wd}.yaml`
- `/cmd_twist` producer (P-controlled by base trajectory) → `../genie_sim_control/src/planar_base_controller.cpp`
