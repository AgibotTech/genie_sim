# genie_sim_planning

Python tooling + small CLI scripts for the GenieSim chassis pipeline
(teleop, navigation, follow_trajectory).

Source: [source/geniesim_ros/src/ros_ws/src/genie_sim_planning/](.)
License: [Mozilla Public License Version 2.0](../../../../LICENSE) (PythonRobotics-derived LQR helpers under MIT — see [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md))

The Python helpers and demo scripts live here. The C++ chassis servo,
ServoBase strategies, MPC solver and tracking differentiator belong to
[genie_sim_controllers](../genie_sim_ros_control/genie_sim_controllers/).

**Maintenance contract**: when you add a script or change one of the
helper modules, update this file in the same diff. Before adding new
**C++** servo code here, default to extending `genie_sim_controllers`
instead — this package is intentionally Python-only at this point.

---

## Layout

```
genie_sim_planning/
├── genie_sim_planning/             ← importable Python package
│   ├── __init__.py
│   ├── kinematics.py               ← geometry / IK helpers
│   ├── math_utils.py               ← vector / quaternion / SO(2) utilities
│   ├── path_utils.py               ← waypoint smoothing, resampling
│   ├── planner.py                  ← high-level planner glue
│   ├── qos.py                      ← shared QoS presets
│   └── tf_utils.py                 ← tf2 lookups + frame conversions
└── scripts/
    ├── simple_follow_trajectory.py ← FollowJointTrajectory demo driver
    ├── simple_navigation.py        ← /cmd_twist demo from Nav2 goals
    └── simple_move_base.py         ← bare /cmd_twist publisher (teleop-style)
```

---

## Routing rules

- Reusable Python helpers → `genie_sim_planning/`
- Demo / quick-and-dirty scripts → `scripts/`
- Real chassis servo (C++) → `../genie_sim_ros_control/genie_sim_controllers/`
- Planar-base trajectory dispatcher → `../genie_sim_ros_control/genie_sim_control/src/planar_base_controller.cpp`
