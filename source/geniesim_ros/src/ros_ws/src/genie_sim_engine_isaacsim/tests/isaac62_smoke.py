#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Explicit 6.2 asset/backend smoke; copies assets and never runs under pytest.

Physics settings come from the provided launcher and scene YAML. This checks
an isolated robot over a blank ground, not the complete task/render/ROS graph.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-usda", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=20)
    args = parser.parse_args()
    if args.ticks <= 0:
        parser.error("--ticks must be positive")
    for path in (args.robot_usda, args.launcher, args.scene):
        if not path.is_file():
            parser.error(f"missing input: {path}")
    workspace = Path(__file__).resolve().parents[2]
    for name in ("genie_sim_engine_isaacsim", "genie_sim_engine_core", "genie_sim_engine_newton"):
        sys.path.insert(0, str(workspace / name))
    import yaml

    launcher = yaml.safe_load(args.launcher.read_text()).get("genie_sim_engine", {}).get("ros__parameters", {})
    scene_cfg = yaml.safe_load(args.scene.read_text())
    engine_name = launcher.get("physics_engine", "isaac_physx")
    if engine_name not in ("isaac_physx", "isaac_newton"):
        parser.error("launcher must select isaac_physx or isaac_newton")
    output = Path(tempfile.mkdtemp(prefix="geniesim62_smoke_"))
    robot_dir = output / "robot"
    shutil.copytree(args.robot_usda.resolve().parent, robot_dir)
    robot = robot_dir / args.robot_usda.name
    print(f"SMOKE_OUTPUT={output}", flush=True)
    from genie_sim_engine_isaacsim.kit.bootstrap import resolve_isaac_paths
    from isaacsim import SimulationApp

    experience = resolve_isaac_paths()["kit_experience"]["path"] if engine_name == "isaac_newton" else ""
    app = SimulationApp({"headless": True, "disable_viewport_updates": True}, experience=experience)
    exit_code = 1
    try:
        import numpy as np
        from pxr import Usd, UsdGeom, UsdPhysics
        from types import SimpleNamespace
        from genie_sim_engine_isaacsim.assemble.physics_variant import select_physics_variant, validate_urdf_joints
        from genie_sim_engine_isaacsim.assemble._robot_impl import (
            _apply_mimic_joint_overlay,
            _apply_post_transformer_collision_policy,
        )
        from genie_sim_engine_core.common.params import EngineNodeParams, load_physics_params, parse_init_joint_pos
        from genie_sim_engine_core.common.scene_config import parse_pin_base_to_world, parse_pin_base_mode
        from genie_sim_engine_core.engine.base import PhysicsEngine
        from genie_sim_engine_isaacsim.kit import isaac_newton, isaac_physx  # register both backends

        logger = SimpleNamespace(info=lambda m: print(m, flush=True), warn=lambda m: print(f"WARN: {m}", flush=True))
        urdf = robot_dir / "robot_raw.urdf"
        if not urdf.is_file():
            raise RuntimeError("smoke requires robot_raw.urdf beside the converted robot")
        stage = Usd.Stage.Open(str(robot))
        select_physics_variant(stage, logger=logger)
        joint_count = validate_urdf_joints(stage, urdf)
        stage.GetRootLayer().Save()
        stage = None
        _apply_post_transformer_collision_policy(str(robot), logger)
        overlays = _apply_mimic_joint_overlay(str(robot), logger=logger, urdf_path=str(urdf))
        scene_path = output / "scene.usda"
        stage = Usd.Stage.CreateNew(str(scene_path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")
        stage.GetRootLayer().Save()
        stage = None
        node_params = EngineNodeParams.from_dict(launcher)
        profile = Path(node_params.physics_params_file or "physics_params.yaml")
        if not profile.is_absolute():
            profile = workspace / "genie_sim_engine_core/config" / profile
        if not profile.is_file():
            raise RuntimeError(f"missing launcher physics profile: {profile}")
        robot_cfg = scene_cfg.get("robot", {})
        physics_params = load_physics_params(str(profile), logger)
        hz = node_params.physics_hz
        engine = PhysicsEngine.create(
            engine_name,
            robot_prefix=robot_cfg.get("robot_prefix", "genie"),
            scene_usda=str(scene_path),
            robot_usda=str(robot),
            render_layer_usda="",
            physics_hz=hz,
            render_hz=node_params.render_hz or physics_params.render_target_hz,
            simulation_app=app,
            logger=logger,
            params=physics_params,
            robot_from_urdf=True,
            init_joint_pos=parse_init_joint_pos(robot_cfg.get("init_joint_pos"), logger),
            runtime_usd_dump_path="",
            pin_base_to_world=parse_pin_base_to_world(robot_cfg),
            pin_base_mode=parse_pin_base_mode(robot_cfg),
            scene_cfg=scene_cfg,
            physics_solver=node_params.physics_solver or "mujoco-warp",
            physics_solver_substep=node_params.physics_solver_substep,
            physics_solver_iterations=node_params.physics_solver_iterations,
            render_mode=node_params.render_mode,
        )
        engine.startup(True)
        for tick in range(args.ticks):
            engine.step(1.0 / hz, tick / hz)
        summary = {"engine": engine_name, "ticks": args.ticks, "movable_joints": joint_count, "overlays": overlays}
        if engine_name == "isaac_newton":
            from isaacsim.physics.newton import get_newton_config
            from genie_sim_engine_isaacsim.kit.newton_config import verify_runtime_config
            import mujoco

            ns = engine._get_newton_stage()
            verify_runtime_config(get_newton_config(), engine._newton_config_expected)
            solver = ns.solver
            assert type(solver).__name__ == "SolverMuJoCo"

            def scalar(value):
                return np.asarray(value.numpy() if hasattr(value, "numpy") else value).reshape(-1)[0].item()

            iterations = int(scalar(solver.mjw_model.opt.iterations))
            cone = int(scalar(solver.mjw_model.opt.cone))
            contacts = bool(scalar(solver.mjw_model.opt.run_collision_detection))
            assert cone == int(mujoco.mjtCone.mjCONE_PYRAMIDAL) and contacts
            requested_iterations = int(launcher.get("physics_solver_iterations", 0))
            if requested_iterations:
                assert iterations == requested_iterations
            assert np.isfinite(ns.state_0.body_q.numpy()).all()
            summary.update(
                solver=type(solver).__name__,
                iterations=iterations,
                cone=cone,
                mujoco_contacts=contacts,
                substeps=ns.cfg.num_substeps,
                configured_njmax=ns.cfg.solver_cfg.njmax,
                configured_nconmax=ns.cfg.solver_cfg.nconmax,
            )
        else:
            poses, paths = engine.get_body_transforms()
            assert len(paths) and np.isfinite(poses).all()
            summary.update(bodies=len(paths))
        selected = engine._stage_obj._joint_params
        checked = 0
        if selected and engine_name == "isaac_physx":
            view = engine._stage_obj._articulation._articulation_view._physics_view
            friction = np.asarray(view.get_dof_friction_properties())[0]
            arrays = {
                "stiffness": np.asarray(view.get_dof_stiffnesses())[0],
                "damping_drive": np.asarray(view.get_dof_dampings())[0],
                "armature": np.asarray(view.get_dof_armatures())[0],
                "frictionloss": friction[:, 0],
                "damping": friction[:, 2],
            }
            for name, idx in engine._stage_obj._art_joint_index.items():
                for field, array in arrays.items():
                    expected = selected.joint(name).get(field)
                    if expected is not None:
                        assert np.isclose(array[idx], expected, rtol=1e-5, atol=1e-7), (
                            name,
                            field,
                            array[idx],
                            expected,
                        )
                        checked += 1
        elif selected:
            mj = solver.mj_model
            names = {label.replace("/", "_"): label.rsplit("/", 1)[-1] for label in ns.model.joint_label}
            checked_joints = set()
            for j in range(mj.njnt):
                name = names.get(mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j), "")
                values = selected.joint(name)
                if values:
                    checked_joints.add(name)
                idx = mj.jnt_dofadr[j]
                for field, array in [
                    ("damping", mj.dof_damping),
                    ("frictionloss", mj.dof_frictionloss),
                    ("armature", mj.dof_armature),
                ]:
                    if field in values:
                        assert np.isclose(array[idx], values[field], rtol=1e-5, atol=1e-7), (
                            name,
                            field,
                            array[idx],
                            values[field],
                        )
                        checked += 1
            live_bias = solver.mjw_model.actuator_biasprm.numpy().reshape(-1, mj.nu, 10)[0]
            for j in range(mj.njnt):
                name = names.get(mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j), "")
                values = selected.joint(name)
                actuators = mj.actuator_trnid[:, 0] == j
                # Isaac may compile separate position and velocity actuators.
                # Check their combined feedback in both CPU and live GPU models.
                for field, component in [("stiffness", 1), ("damping_drive", 2)]:
                    if field in values:
                        for bias in (mj.actuator_biasprm, live_bias):
                            actual = -bias[actuators, component].sum()
                            assert np.isclose(actual, values[field], rtol=1e-5, atol=1e-7), (
                                name,
                                field,
                                actual,
                                values[field],
                            )
                        checked += 1
            assert checked_joints == set(selected.stack.joint_names()), (checked_joints, selected.stack.joint_names())
        if selected:
            assert checked > 0, "selected sysid overlays were not checked against the solver"
            summary.update(sysid_overlays=selected.stack.ids, sysid_fields_checked=checked)
        (output / "RESULT.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"ISAAC62_SMOKE_PASS {json.dumps(summary)}", flush=True)
        exit_code = 0
    except Exception:
        traceback.print_exc()
        print("ISAAC62_SMOKE_FAIL", flush=True)
    finally:
        # Isaac 6.2 fast shutdown can terminate the interpreter; preserve failure.
        app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
