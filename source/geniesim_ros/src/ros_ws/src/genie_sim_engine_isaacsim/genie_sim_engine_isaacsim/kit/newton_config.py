# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Isaac 6.2 public Newton configuration, before the first World.reset.

MuJoCo constructor options remain on MuJoCoSolverConfig. Iterations moved to
MjcSceneAPI (mjc:option:iterations); ls_parallel was removed upstream. The
existing wrapper's joint-limit and contact-allocation targets are preserved.
"""

# Existing wrapper tuning, formerly ineffective dataclass-default patches.
# Provenance: kit/isaac_newton.py and docker/GENIESIM4_MIGRATION.md R1.
SOLVER_OPTIONS = {"cone": "pyramidal", "njmax": 1024, "nconmax": 512, "use_mujoco_contacts": True}
NEWTON_OPTIONS = {"joint_limit_ke": 1.0e4, "joint_limit_kd": 10.0}
SCENE_ITERATIONS = "mjc:option:iterations"


def verify_runtime_config(config, expected):
    """Refuse missing/ignored required fields rather than logging a wrong run."""
    for key, value in expected.items():
        owner, name = key.split(".")
        obj = config if owner == "NewtonConfig" else getattr(config, "solver_cfg", None)
        actual = getattr(obj, name, None)
        if actual != value:
            raise RuntimeError(f"[isaac_newton] effective {key}={actual!r}; required {value!r}")


def configure_runtime(*, physics_solver, physics_hz, substeps, logger):
    """Configure instances through the public API; never mutate class defaults."""
    if physics_solver not in ("mujoco-warp", "mujoco", "mjwarp"):
        raise ValueError(f"isaac_newton supports MuJoCo-Warp, got physics_solver={physics_solver!r}")
    if substeps < 0 or physics_hz <= 0:
        raise ValueError("Newton requires nonnegative substeps and positive physics_hz")
    try:
        from isaacsim.physics.newton import MuJoCoSolverConfig, NewtonConfig, configure_newton, get_newton_config
    except ImportError as exc:
        raise RuntimeError("isaac_newton requires the Isaac 6.2 public configure_newton API") from exc
    cfg_options = dict(NEWTON_OPTIONS, physics_frequency=float(physics_hz))
    if substeps:
        cfg_options["num_substeps"] = int(substeps)
    try:
        solver = MuJoCoSolverConfig(**SOLVER_OPTIONS)
        config = NewtonConfig(solver_cfg=solver, **cfg_options)
    except TypeError as exc:
        raise RuntimeError(
            "Isaac Newton's public configuration fields changed; refusing unconfigured defaults"
        ) from exc
    expected = {f"NewtonConfig.{key}": value for key, value in cfg_options.items()}
    expected.update({f"MuJoCoSolverConfig.{key}": value for key, value in SOLVER_OPTIONS.items()})
    expected["MuJoCoSolverConfig.solver_type"] = "mujoco"
    configure_newton(config)
    verify_runtime_config(get_newton_config(), expected)
    logger.info(f"[isaac_newton] configure_newton applied before reset: {expected}")
    return expected


def configure_physics_scene(stage, iterations, logger):
    """Author 6.2's solver selection/iteration schema before World.reset."""
    from pxr import Sdf, UsdPhysics
    from isaacsim.physics.newton import get_newton_solver, newton_solver_to_api_schema, switch_newton_solver

    if iterations < 0:
        raise ValueError("physics_solver_iterations must be nonnegative")
    prim = stage.GetPrimAtPath("/physicsScene")
    if not prim or not prim.IsA(UsdPhysics.Scene):
        scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
        if len(scenes) != 1:
            raise RuntimeError(f"Newton requires one identifiable physics scene, found {len(scenes)}")
        prim = scenes[0]
    if not switch_newton_solver("mujoco", prim):
        raise RuntimeError(f"Could not select MuJoCo on {prim.GetPath()}")
    if not prim.HasAPI("MjcSceneAPI") or get_newton_solver(prim) != "mujoco":
        raise RuntimeError(f"MuJoCo scene selection did not apply on {prim.GetPath()}")
    active_scenes = [
        p
        for p in stage.Traverse()
        if p.IsA(UsdPhysics.Scene) and any(p.HasAPI(api) for api in newton_solver_to_api_schema.values())
    ]
    if len(active_scenes) != 1:
        raise RuntimeError(f"Multiple Newton physics scenes are ambiguous: {[str(p.GetPath()) for p in active_scenes]}")
    if iterations:
        attr = prim.CreateAttribute(SCENE_ITERATIONS, Sdf.ValueTypeNames.Int, custom=False)
        if not attr.Set(int(iterations)) or attr.Get() != iterations:
            raise RuntimeError(f"Could not apply {SCENE_ITERATIONS}={iterations}")
    logger.info(
        f"[isaac_newton] {prim.GetPath()}: solver=mujoco, iterations={prim.GetAttribute(SCENE_ITERATIONS).Get()}"
    )


def verify_live_solver(solver, iterations):
    """Check required construction options on the actual first-step solver."""
    import numpy as np

    opt = getattr(getattr(solver, "mjw_model", None), "opt", None)
    if opt is None:
        raise RuntimeError("[isaac_newton] no live MuJoCo solver options after first step")
    expected = {"cone": 0, "run_collision_detection": True}  # MuJoCo's pyramidal enum is 0.
    if iterations:
        expected["iterations"] = iterations
    for name, value in expected.items():
        actual = getattr(opt, name, None)
        if hasattr(actual, "numpy"):
            actual = actual.numpy()
        actual = np.asarray(actual).reshape(-1)[0]
        if actual != value:
            raise RuntimeError(f"[isaac_newton] live solver {name}={actual!r}; required {value!r}")
