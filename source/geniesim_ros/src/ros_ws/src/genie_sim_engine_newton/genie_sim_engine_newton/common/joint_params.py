# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Scene overlay selection for the in-tree Isaac and Newton backends."""

from pathlib import Path

from genie_sim_engine_core.common.joint_params import JointParamResolver


def load_scene_joint_params(scene_cfg, *, engine, logger=None):
    robot = (scene_cfg or {}).get("robot") or {}
    paths = robot.get("joint_overlays", [])
    if not isinstance(paths, list) or any(not isinstance(p, str) or not p for p in paths):
        raise ValueError("robot.joint_overlays must be a list of nonempty package:// URIs or absolute paths")
    resolved = []
    for path in paths:
        if path.startswith("package://"):
            from ament_index_python.packages import get_package_share_directory

            package, sep, relative = path[len("package://") :].partition("/")
            if not sep or not relative:
                raise ValueError(f"Invalid joint overlay URI: {path}")
            resolved.append(Path(get_package_share_directory(package)) / relative)
        elif Path(path).is_absolute():
            resolved.append(Path(path))
        else:
            raise ValueError(f"Joint overlay must be package:// URI or absolute path: {path}")
    source = robot.get("robot_source") or {}
    description = dict(source)
    description.update(source.get("urdf") or {})
    description["ee"] = description.get("gripper", "")
    return JointParamResolver.load(resolved, engine=engine, robot=description, logger=logger)


def apply_newton_joint_params(
    model, resolver, *, mimic_followers=(), robot_prefix="", logger=None, update_target_modes=True
):
    """Apply SI values after class/mimic fallbacks, before solver construction.

    Newton 1.4 and 1.6 expose passive viscosity as ``joint_damping`` in
    N*m*s/rad (N*s/m for sliders). The deleted ``mujoco.dof_passive_damping``
    and its historical degree compensation must not be used here.
    """
    if not resolver:
        return
    from genie_sim_engine_newton.engine.newton.joint_index import JointIndex

    index = JointIndex(model)
    slices = [s for s in index.slices() if not robot_prefix or robot_prefix in s.label.split("/")]
    resolver.validate_names(s.name for s in slices)
    followers = set(mimic_followers)
    fields = {
        "stiffness": "joint_target_ke",
        "damping_drive": "joint_target_kd",
        "max_effort": "joint_effort_limit",
        "effort": "joint_effort_limit",
        "velocity": "joint_velocity_limit",
        "damping": "joint_damping",
        "frictionloss": "joint_friction",
        "armature": "joint_armature",
    }
    updates = {}
    for joint in slices:
        values = resolver.joint(joint.name)
        if not values:
            continue
        if joint.dof_count != 1:
            raise ValueError(f"[joint_params] {joint.label}: expected a movable one-DOF joint")
        if joint.name in followers and any(values.get(f, 0) != 0 for f in ("stiffness", "damping_drive")):
            raise ValueError(f"[joint_params] {joint.name}: mimic follower cannot carry a drive")
        for field, attr in fields.items():
            if field not in values or (field == "effort" and "max_effort" in values):
                continue
            array = getattr(model, attr, None)
            if array is None:
                raise RuntimeError(f"[joint_params] {joint.name}.{field}: Newton model lacks {attr}")
            if attr not in updates:
                updates[attr] = array.numpy().copy()
            updates[attr][joint.qd_start] = values[field]
    for attr, values in updates.items():
        getattr(model, attr).assign(values)
    # Select actuation from the final gains before MuJoCo compiles actuators.
    # In particular an importer/class zero must not leave a fitted Wuji joint
    # without a position actuator, or re-enable a constraint-driven follower.
    import newton

    mode = model.joint_target_mode.numpy().copy()
    ke = model.joint_target_ke.numpy()
    kd = model.joint_target_kd.numpy()
    for joint in slices:
        values = resolver.joint(joint.name)
        if not update_target_modes:
            continue
        if "stiffness" not in values and "damping_drive" not in values:
            continue
        i = joint.qd_start
        mode[i] = (
            newton.JointTargetMode.POSITION
            if ke[i] > 0
            else newton.JointTargetMode.VELOCITY if kd[i] > 0 else newton.JointTargetMode.EFFORT
        )
    model.joint_target_mode.assign(mode)
    if logger is not None:
        logger.info(f"[joint_params] applied {resolver.stack.ids} to Newton model after class defaults")
