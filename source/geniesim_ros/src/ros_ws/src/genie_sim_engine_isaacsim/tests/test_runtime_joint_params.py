# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Regression coverage for sysid precedence through actual USD/Newton models.

PhysX schema tests additionally run inside a 6.2 SimulationApp; plain USD
does not register PhysxJointAxisAPI. Newton tests build CPU models, no GPU.
"""

from pathlib import Path
from types import SimpleNamespace
import math
import sys

import numpy as np
import pytest
import yaml

from genie_sim_engine_core.common.joint_params import JointParamResolver
from genie_sim_engine_core.common.params import default_physics_params
from genie_sim_engine_core.schema.joint_params import load_joint_overlays
from genie_sim_engine_newton.common.joint_params import apply_newton_joint_params, load_scene_joint_params

SRC = Path(__file__).resolve().parents[2]
MODEL = SRC / "genie_sim_robot_model"
CONFIG = SRC / "genie_sim_bringup" / "config"
G2 = MODEL / "robots/genie/g2/overlays"
LOG = SimpleNamespace(info=lambda *_: None, warn=lambda *_: None, debug=lambda *_: None)


def resolver(*files):
    return JointParamResolver(load_joint_overlays(files).for_engine("newton"), LOG)


@pytest.fixture
def g2():
    return resolver(G2 / "g2_crsB.sysid-2026-09-22.yaml", G2 / "g2_swiftpicker.sysid-2026-09-18.yaml")


@pytest.mark.parametrize("engine", ["physx", "newton"])
def test_shipped_scene_selections(monkeypatch, engine):
    monkeypatch.setitem(
        sys.modules, "ament_index_python.packages", SimpleNamespace(get_package_share_directory=lambda _: str(MODEL))
    )
    count = 0
    for path in sorted(CONFIG.glob("scene_*.yaml")):
        scene = yaml.safe_load(path.read_text())
        selected = load_scene_joint_params(scene, engine=engine, logger=LOG)
        if selected:
            count += 1
            assert selected.stack.ids
    assert count == 8


def test_sparse_values_and_follower_zeros(g2):
    assert g2.joint("idx21_arm_l_joint1")["damping"] == pytest.approx(4.15966580248)
    assert g2.drive("idx21_arm_l_joint1", 500, 20, 50) == (500, 20, 50)
    assert g2.drive("idx31_gripper_l_inner_joint1", 10000, 10, 50) == (202.026337, 13.345311, 50)
    assert g2.drive("idx32_gripper_l_outer_joint1", 10000, 10, 50) == (0, 0, 50)
    assert g2.drive("idx01_body_joint1", 500, 20, 50) == (500, 20, 50)


def test_wrong_robot_rejected():
    with pytest.raises(ValueError, match="applies_to.arm"):
        JointParamResolver.load(
            [G2 / "g2_crsB.sysid-2026-09-22.yaml"],
            engine="newton",
            robot={"robot_model": "genie", "body": "g2", "arm": "crs"},
        )


@pytest.mark.parametrize("paths", ["file.yaml", ["relative.yaml"], [17]])
def test_bad_overlay_selection_rejected(paths):
    with pytest.raises(ValueError):
        load_scene_joint_params({"robot": {"joint_overlays": paths}}, engine="newton")


def _newton_model(names, ke=8000):
    newton = pytest.importorskip("newton")
    import warp as wp

    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    for name in names:
        body = builder.add_link(mass=1, inertia=wp.mat33(np.eye(3)))
        joint = builder.add_joint_revolute(
            -1,
            body,
            label=f"/genie/{name}",
            target_ke=ke,
            target_kd=0 if ke == 0 else 80,
            effort_limit=50,
            damping=0.05,
            friction=0.3,
            armature=0.03,
        )
        builder.add_articulation([joint])
    return builder.finalize(device="cpu")


@pytest.mark.parametrize("adapter_name", ["mujoco-warp", "mujoco-warp-multi-env", "mjvbd", "mjxpbd"])
def test_newton_prepare_preserves_sysid_after_all_class_writes(g2, adapter_name):
    pytest.importorskip("newton")
    from genie_sim_engine_newton.engine.newton.adapters import make_adapter
    from genie_sim_engine_newton.engine.newton.joint_index import JointIndex

    model = _newton_model([*g2.stack.joint_names(), "idx01_body_joint1"])
    adapter = make_adapter(adapter_name, physics_params=default_physics_params(), joint_params=g2)
    followers = {
        master: [(f, 1, 0) for f in g2.stack.joint_names() if f.startswith(prefix) and f != master]
        for master, prefix in [("idx31_gripper_l_inner_joint1", "idx3"), ("idx71_gripper_r_inner_joint1", "idx7")]
    }
    adapter.prepare_model(model, LOG, mimic_followers=followers, robot_prefix="genie")
    index = JointIndex(model)
    for name in g2.stack.joint_names():
        i = index.get(name).qd_start
        for field, attr in {
            "stiffness": "joint_target_ke",
            "damping_drive": "joint_target_kd",
            "damping": "joint_damping",
            "frictionloss": "joint_friction",
            "armature": "joint_armature",
        }.items():
            if field in g2.joint(name):
                assert getattr(model, attr).numpy()[i] == pytest.approx(g2.joint(name)[field], rel=1e-6)
    i = index.get("idx01_body_joint1").qd_start
    baseline = _newton_model([*g2.stack.joint_names(), "idx01_body_joint1"])
    make_adapter(adapter_name, physics_params=default_physics_params()).prepare_model(
        baseline, LOG, mimic_followers=followers, robot_prefix="genie"
    )
    assert model.joint_target_ke.numpy()[i] == baseline.joint_target_ke.numpy()[i]


def test_newton_empty_selection_is_noop():
    model = _newton_model(["idx21_arm_l_joint1"])
    before = {a: getattr(model, a).numpy().copy() for a in ["joint_target_ke", "joint_damping", "joint_armature"]}
    apply_newton_joint_params(model, JointParamResolver())
    for attr, expected in before.items():
        assert np.array_equal(getattr(model, attr).numpy(), expected)


def test_missing_joint_does_not_silently_fall_back(g2):
    with pytest.raises(ValueError, match="absent from robot"):
        apply_newton_joint_params(_newton_model(["idx21_arm_l_joint1"]), g2)


def test_wuji_explicit_zero_passives_replace_defaults():
    selected = resolver(G2 / "g2_wuji.sysid-2026-09-24.yaml")
    model = _newton_model(selected.stack.joint_names(), ke=0)
    apply_newton_joint_params(model, selected)
    assert np.all(model.joint_friction.numpy() == 0)
    assert np.all(model.joint_damping.numpy() == 0)
    assert np.allclose(model.joint_target_ke.numpy(), 0.65)
    assert np.allclose(model.joint_target_kd.numpy(), 0.05)
    assert np.all(model.joint_target_mode.numpy() == 1)  # POSITION, not importer EFFORT


def test_already_compiled_wrapper_keeps_split_actuator_modes(g2):
    model = _newton_model(g2.stack.joint_names())
    modes = model.joint_target_mode.numpy()
    modes[:] = 3  # Isaac's POSITION_VELOCITY topology is already compiled.
    model.joint_target_mode.assign(modes)
    apply_newton_joint_params(model, g2, update_target_modes=False)
    assert np.array_equal(model.joint_target_mode.numpy(), modes)


def test_runtime_stack_order_engine_selection_and_zero_caps(tmp_path):
    paths = []
    for n, fields in enumerate(
        [
            {"damping": 2, "armature": 1, "max_effort": 15},
            {"damping": {"physx": 3, "newton": 0}, "armature": 0, "max_effort": 0},
        ]
    ):
        path = tmp_path / f"override{n}.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "schema": "genie_sim.joint_params/1",
                    "id": f"operator{n}",
                    "applies_to": {"robot_model": "genie"},
                    "meta": {"kind": "operator"},
                    "joints": {"idx21_arm_l_joint1": fields},
                }
            )
        )
        paths.append(path)
    selected = JointParamResolver.load(paths, engine="newton", robot={"robot_model": "genie"})
    model = _newton_model(["idx21_arm_l_joint1"])
    apply_newton_joint_params(model, selected)
    assert model.joint_damping.numpy()[0] == 0
    assert model.joint_armature.numpy()[0] == 0
    assert model.joint_effort_limit.numpy()[0] == 0
    assert (
        JointParamResolver.load(paths, engine="physx", robot={"robot_model": "genie"}).joint("idx21_arm_l_joint1")[
            "damping"
        ]
        == 3
    )


def test_nonzero_mimic_drive_rejected(g2):
    with pytest.raises(ValueError, match="mimic follower cannot carry a drive"):
        apply_newton_joint_params(
            _newton_model(g2.stack.joint_names()), g2, mimic_followers=["idx31_gripper_l_inner_joint1"]
        )


def test_newton_compiled_solver_keeps_si_passives(g2):
    newton = pytest.importorskip("newton")
    import importlib.metadata
    from packaging.requirements import Requirement

    version = importlib.metadata.version("mujoco")
    requirements = [Requirement(r) for r in importlib.metadata.requires("newton") or []]
    if any(r.name == "mujoco" and version not in r.specifier for r in requirements):
        pytest.skip("Newton/MuJoCo dependency mismatch; run with the pinned container dependencies")
    model = _newton_model(g2.stack.joint_names())
    apply_newton_joint_params(model, g2)
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)
    import mujoco

    names = {label.replace("/", "_"): label.rsplit("/", 1)[-1] for label in model.joint_label}
    checked = set()
    for j in range(solver.mj_model.njnt):
        full = mujoco.mj_id2name(solver.mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
        name = names.get(full, "")
        values = g2.joint(name)
        if not values:
            continue
        i = solver.mj_model.jnt_dofadr[j]
        for field, attr in [
            ("damping", "dof_damping"),
            ("frictionloss", "dof_frictionloss"),
            ("armature", "dof_armature"),
        ]:
            assert getattr(solver.mj_model, attr)[i] == pytest.approx(values[field], rel=1e-6)
        checked.add(name)
    assert checked == set(g2.stack.joint_names())


@pytest.mark.parametrize("angular", [False, True])
def test_physx_native_passives_and_tensor_gains(g2, angular):
    from pxr import Usd, UsdPhysics
    from genie_sim_engine_isaacsim.kit.joint_params import apply_articulation_joint_params, apply_usd_joint_params

    if not Usd.SchemaRegistry().GetTypeFromSchemaTypeName("PhysxJointAxisAPI"):
        pytest.skip("PhysxJointAxisAPI requires an initialized Isaac 6.2 runtime")
    stage = Usd.Stage.CreateInMemory()
    paths = {}
    for name in g2.stack.joint_names():
        joint = (UsdPhysics.RevoluteJoint if angular else UsdPhysics.PrismaticJoint).Define(stage, f"/Robot/{name}")
        paths[name] = str(joint.GetPath())
        UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular" if angular else "linear")
        if g2.joint(name).get("stiffness") == 0:
            joint.GetPrim().CreateRelationship("newton:mimicJoint").SetTargets(["/Robot/idx31_gripper_l_inner_joint1"])
    apply_usd_joint_params(stage, paths, g2, physics_engine="isaac_physx", logger=LOG)
    index = {name: i for i, name in enumerate(paths)}
    kp, kd, effort = (np.full(len(index), n, dtype=float) for n in [10000, 10, 50])
    apply_articulation_joint_params(stage, paths, index, g2, kp, kd, effort)
    scale = math.pi / 180 if angular else 1
    token = "angular" if angular else "linear"
    for name, path in paths.items():
        p = stage.GetPrimAtPath(path)
        values = g2.joint(name)
        assert p.GetAttribute(f"physxJointAxis:{token}:viscousFrictionCoefficient").Get() == pytest.approx(
            values["damping"] * scale, rel=1e-6
        )
        for attr in ["staticFrictionEffort", "dynamicFrictionEffort"]:
            assert p.GetAttribute(f"physxJointAxis:{token}:{attr}").Get() == pytest.approx(values["frictionloss"])
        assert p.GetAttribute(f"physxJointAxis:{token}:armature").Get() == pytest.approx(values["armature"])
        if "stiffness" in values:
            assert kp[index[name]] == values["stiffness"]
            assert kd[index[name]] == values["damping_drive"]
    # Passive arm damping must not be folded into the actuator's kd.
    assert kd[index["idx21_arm_l_joint1"]] == 10


def test_physx_live_tensor_readback(g2):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    if not Usd.SchemaRegistry().GetTypeFromSchemaTypeName("PhysxJointAxisAPI"):
        pytest.skip("PhysX readback requires an initialized Isaac 6.2 runtime")
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from genie_sim_engine_isaacsim.kit.joint_params import apply_usd_joint_params

    World.clear_instance()
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    root = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(root)
    previous = None
    paths = {}
    for i, name in enumerate(g2.stack.joint_names()):
        body = UsdGeom.Xform.Define(stage, f"/Robot/link{i}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        mass = UsdPhysics.MassAPI.Apply(body)
        mass.CreateMassAttr(1)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1))
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/Robot/{name}")
        if previous is not None:
            joint.CreateBody0Rel().SetTargets([previous])
        joint.CreateBody1Rel().SetTargets([body.GetPath()])
        joint.CreateAxisAttr("Z")
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(1000 * math.pi / 180)
        drive.CreateDampingAttr(10 * math.pi / 180)
        drive.CreateMaxForceAttr(50)
        paths[name] = str(joint.GetPath())
        previous = body.GetPath()
    # This fixture has ordinary joints so all 26 passive values can be
    # inspected in one articulation; mimic constraints are covered above.
    apply_usd_joint_params(stage, paths, g2, physics_engine="isaac_physx", logger=LOG)
    world = World(backend="numpy", device="cpu")
    articulation = world.scene.add(SingleArticulation(prim_path="/Robot", name="test_robot"))
    world.reset()
    view = articulation._articulation_view._physics_view
    friction = np.asarray(view.get_dof_friction_properties())[0]
    armature = np.asarray(view.get_dof_armatures())[0]
    kp, kd = view.get_dof_stiffnesses(), view.get_dof_dampings()
    for i, name in enumerate(articulation.dof_names):
        values = g2.joint(name)
        assert friction[i, 0] == pytest.approx(values["frictionloss"], rel=1e-5)
        assert friction[i, 1] == pytest.approx(values["frictionloss"], rel=1e-5)
        assert friction[i, 2] == pytest.approx(values["damping"], rel=1e-5)
        assert armature[i] == pytest.approx(values["armature"], rel=1e-5)
        if "stiffness" in values:
            assert np.asarray(kp).reshape(-1)[i] == pytest.approx(values["stiffness"], rel=1e-5)
            assert np.asarray(kd).reshape(-1)[i] == pytest.approx(values["damping_drive"], rel=1e-5)
    world.stop()
    World.clear_instance()


@pytest.mark.parametrize(
    "joint_type, dof_prefix, supported",
    [
        ("spherical", "idx54_gripper_l_inner_joint2", True),
        ("spherical", "unknown_joint", False),
        ("revolute", "idx54_gripper_l_inner_joint2", False),
    ],
)
def test_newton_spherical_axes_preserve_authored_gains(monkeypatch, joint_type, dof_prefix, supported):
    from pxr import Usd, UsdGeom, UsdPhysics
    from genie_sim_engine_isaacsim.kit.stage import _init_articulation

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/genie").GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(root)
    joint_class = UsdPhysics.SphericalJoint if joint_type == "spherical" else UsdPhysics.RevoluteJoint
    joint_class.Define(stage, "/genie/idx54_gripper_l_inner_joint2")
    kp, kd, effort = [np.array([values], dtype=np.float32) for values in ([0, 2, 0], [0, 0, 3], [0, 4, 5])]
    written = {}
    physics_view = SimpleNamespace(
        set_dof_stiffnesses=lambda values, indices: written.update(kp=values.copy()),
        set_dof_dampings=lambda values, indices: written.update(kd=values.copy()),
        set_dof_max_forces=lambda values, indices: written.update(effort=values.copy()),
    )
    view = SimpleNamespace(
        get_gains=lambda: (kp, kd),
        get_max_efforts=lambda: effort,
        set_joint_position_targets=lambda **kwargs: None,
        set_joint_velocity_targets=lambda **kwargs: None,
        _physics_view=physics_view,
    )
    articulation = SimpleNamespace(
        initialize=lambda: None,
        dof_names=[f"{dof_prefix}:{i}" for i in range(3)],
        num_dof=3,
        _articulation_view=view,
        get_joint_positions=lambda: np.zeros(3, dtype=np.float32),
    )
    monkeypatch.setitem(
        sys.modules, "isaacsim.core.prims", SimpleNamespace(SingleArticulation=lambda **kwargs: articulation)
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            as_tensor=lambda values, **kwargs: np.asarray(values),
            tensor=lambda values, **kwargs: np.asarray(values),
            int32=np.int32,
        ),
    )

    warnings = []
    logger = SimpleNamespace(info=LOG.info, debug=LOG.debug, warn=warnings.append)
    result, _, _, _ = _init_articulation(
        stage, "genie", logger, default_physics_params(), False, physics_engine="isaac_newton"
    )
    if supported:
        assert result is articulation, warnings
        for key, expected in (("kp", kp), ("kd", kd), ("effort", effort)):
            np.testing.assert_array_equal(written[key], expected)
    else:
        assert result is None
        assert written == {}
        assert any("could not bucket" in message for message in warnings)
