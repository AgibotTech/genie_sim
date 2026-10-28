# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Hand contacts must enter the pair list before mujoco-warp uploads it."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest

_PACKAGE = Path(__file__).resolve().parents[1]
_HELPER = _PACKAGE / "common" / "hand_collision.py"
_ADAPTERS = _PACKAGE / "engine" / "newton" / "adapters"
_MODULE = "genie_sim_engine_newton.common.hand_collision"


@pytest.fixture
def policy(monkeypatch):
    # Importing the package normally also initializes the ROS-facing engine.
    spec = importlib.util.spec_from_file_location(_MODULE, _HELPER)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, _MODULE, module)
    spec.loader.exec_module(module)
    return module


def _body(name, children="", *, mask="1 -2", extra=""):
    contype, conaffinity = mask.split()
    return (
        f'<body name="{name}"><joint type="hinge" axis="0 0 1"/>'
        f'<geom name="{name}_collision" type="sphere" size="0.02" '
        f'contype="{contype}" conaffinity="{conaffinity}"/>{extra}{children}</body>'
    )


def _model(*, hands=True):
    if not hands:
        return mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody>"
            + _body("arm_l_link1", _body("arm_l_end_link"))
            + '</worldbody><contact><exclude body1="arm_l_link1" body2="arm_l_end_link"/></contact></mujoco>'
        )
    # Distinct original bits allow native MuJoCo's compiled body-level broadphase
    # to exercise excludes too; MuJoCo Warp builds its pair list from geom masks.
    thumb = _body("hand_l_thumb_1", _body("hand_l_thumb_2", _body("hand_l_thumb_3", mask="4 -5")), mask="2 -3")
    index = _body("hand_l_index_1", _body("hand_l_index_2", _body("hand_l_index_3")))
    visual = '<geom name="hand_l_visual" type="sphere" size="0.025" contype="0" conaffinity="0"/>'
    palm = _body("hand_l_base_link", thumb + index, extra=visual)
    arm = _body("arm_l_link1", _body("arm_l_end_link", palm))
    # Converter and replicated-model names can carry slash/underscore prefixes.
    right = _body("robot_1/hand_r_base_link", _body("robot_1_hand_r_index_1"), mask="0 4")
    excludes = [
        ("arm_l_link1", "arm_l_end_link"),  # Non-hand policy stays untouched.
        ("arm_l_end_link", "hand_l_base_link"),  # Parent.
        ("arm_l_link1", "hand_l_base_link"),  # Grandparent at the arm boundary.
        ("hand_l_base_link", "hand_l_thumb_1"),  # Parent.
        ("hand_l_base_link", "hand_l_thumb_2"),  # Grandparent.
        ("hand_l_thumb_1", "hand_l_thumb_3"),  # Grandparent within a finger.
        ("hand_l_base_link", "hand_l_thumb_3"),  # Fingertip vs palm opens.
        ("hand_l_thumb_1", "hand_l_index_1"),  # Different fingers open.
        ("arm_l_link1", "hand_l_thumb_1"),  # Distant upstream arm opens.
        ("hand_l_index_3", "robot_1_hand_r_index_1"),  # Opposite hands open.
    ]
    contact = "".join(f'<exclude body1="{a}" body2="{b}"/>' for a, b in excludes)
    return mujoco.MjModel.from_xml_string(
        f'<mujoco><option gravity="0 0 0"/><worldbody>{arm}{right}</worldbody><contact>{contact}</contact></mujoco>'
    )


def _signature(model, name1, name2):
    body1, body2 = sorted(int(model.body(name).id) for name in (name1, name2))
    return (body1 << 16) + body2


def _geom(model, name):
    return int(model.geom(name).id)


def _pair_allowed(model, geom1, geom2):
    return bool(
        (int(model.geom_contype[geom1]) & int(model.geom_conaffinity[geom2]))
        | (int(model.geom_contype[geom2]) & int(model.geom_conaffinity[geom1]))
    )


def test_only_hand_collision_masks_are_opened_and_visual_stays_disabled(policy):
    model = _model()
    types_before = model.geom_contype.copy()
    affinity_before = model.geom_conaffinity.copy()

    geom_count, cleared, kept = policy.enable_hand_self_collision(model)

    assert geom_count == 9  # Seven left and two right collision geoms; no visual.
    assert (cleared, kept) == (4, 6)
    np.testing.assert_array_equal(model.geom_contype, types_before)
    for geom_id in range(model.ngeom):
        body_name = model.body(int(model.geom_bodyid[geom_id])).name
        if "hand_" in body_name and geom_id != _geom(model, "hand_l_visual"):
            assert model.geom_conaffinity[geom_id] == -1
        else:
            assert model.geom_conaffinity[geom_id] == affinity_before[geom_id]
    visual = _geom(model, "hand_l_visual")
    assert (model.geom_contype[visual], model.geom_conaffinity[visual]) == (0, 0)
    # Existing contype=0 / nonzero affinity collision geoms remain collidable.
    assert model.geom_contype[_geom(model, "robot_1/hand_r_base_link_collision")] == 0


def test_excludes_clear_in_place_but_keep_nonhand_parent_and_grandparent(policy):
    model = _model()
    signatures = model.exclude_signature
    before = signatures.copy()
    removed = {
        _signature(model, "hand_l_base_link", "hand_l_thumb_3"),
        _signature(model, "hand_l_thumb_1", "hand_l_index_1"),
        _signature(model, "arm_l_link1", "hand_l_thumb_1"),
        _signature(model, "hand_l_index_3", "robot_1_hand_r_index_1"),
    }

    policy.enable_hand_self_collision(model)

    assert np.shares_memory(signatures, model.exclude_signature)
    assert model.nexclude == len(before)
    assert (
        set(int(signature) for signature in signatures if signature)
        == set(int(signature) for signature in before) - removed
    )
    assert np.count_nonzero(signatures == 0) == len(removed)
    # MuJoCo Warp uses membership over this original array, not an XML rewrite.
    assert not np.isin(list(removed), signatures).any()


def test_cross_finger_and_hand_arm_masks_open_without_enabling_arm_arm(policy):
    model = _model()
    thumb = _geom(model, "hand_l_thumb_1_collision")
    index = _geom(model, "hand_l_index_1_collision")
    arm1 = _geom(model, "arm_l_link1_collision")
    arm2 = _geom(model, "arm_l_end_link_collision")
    assert not _pair_allowed(model, index, arm1)
    assert not _pair_allowed(model, arm1, arm2)

    policy.enable_hand_self_collision(model)

    assert _pair_allowed(model, thumb, index)
    assert _pair_allowed(model, index, arm1)
    assert not _pair_allowed(model, arm1, arm2)


def test_mujoco_generates_previously_excluded_cross_finger_contacts(policy):
    model = _model()
    data = mujoco.MjData(model)
    thumb = _geom(model, "hand_l_thumb_1_collision")
    index = _geom(model, "hand_l_index_1_collision")

    def contacts():
        mujoco.mj_forward(model, data)
        return {frozenset((int(contact.geom1), int(contact.geom2))) for contact in data.contact[: data.ncon]}

    pair = frozenset((thumb, index))
    assert pair not in contacts()
    policy.enable_hand_self_collision(model)
    actual = contacts()
    assert pair in actual
    palm = _geom(model, "hand_l_base_link_collision")
    middle = _geom(model, "hand_l_thumb_2_collision")
    tip = _geom(model, "hand_l_thumb_3_collision")
    assert frozenset((palm, middle)) not in actual
    assert frozenset((thumb, tip)) not in actual
    assert frozenset((palm, tip)) in actual


def test_no_hand_scene_is_unchanged(policy):
    model = _model(hands=False)
    before = [array.copy() for array in (model.geom_contype, model.geom_conaffinity, model.exclude_signature)]

    assert policy.enable_hand_self_collision(model) == (0, 0, 0)

    for old, current in zip(before, (model.geom_contype, model.geom_conaffinity, model.exclude_signature)):
        np.testing.assert_array_equal(old, current)


def test_reapplying_policy_preserves_masks_and_excludes(policy):
    model = _model()
    assert policy.enable_hand_self_collision(model) == (9, 4, 6)
    affinity = model.geom_conaffinity.copy()
    signatures = model.exclude_signature.copy()

    assert policy.enable_hand_self_collision(model) == (9, 0, 6)

    np.testing.assert_array_equal(model.geom_conaffinity, affinity)
    np.testing.assert_array_equal(model.exclude_signature, signatures)


def test_mjcf_dump_reloads_with_the_same_hand_collision_policy(policy, tmp_path):
    model = _model()
    path = tmp_path / "robot_runtime.xml"
    mujoco.mj_saveLastXML(str(path), model)

    expected_counts = policy.enable_hand_self_collision(model)
    assert policy.enable_hand_self_collision_in_mjcf(path) == expected_counts

    reloaded = mujoco.MjModel.from_xml_path(str(path))
    np.testing.assert_array_equal(reloaded.geom_contype, model.geom_contype)
    np.testing.assert_array_equal(reloaded.geom_conaffinity, model.geom_conaffinity)
    assert set(int(signature) for signature in reloaded.exclude_signature) == {
        int(signature) for signature in model.exclude_signature if signature
    }
    assert reloaded.nexclude == 6


@pytest.mark.parametrize("construction_fails", [False, True])
def test_context_updates_generated_dump_only_after_success(policy, monkeypatch, tmp_path, construction_fails):
    model = _model()
    path = tmp_path / "robot_runtime.xml"
    mujoco.mj_saveLastXML(str(path), model)
    original_xml = path.read_bytes()

    def upload(mj_model):
        return object()

    module = SimpleNamespace(put_model=upload)
    monkeypatch.setitem(sys.modules, "mujoco_warp", module)
    logger = SimpleNamespace(info=Mock(), warn=Mock(), warning=Mock())

    def construct():
        with policy.hand_self_collision_at_put_model(logger, mjcf_path=path):
            module.put_model(model)
            # Newton serializes its unchanged MjSpec after put_model returns.
            path.write_bytes(original_xml)
            if construction_fails:
                raise RuntimeError("solver construction failed")

    if construction_fails:
        with pytest.raises(RuntimeError, match="solver construction failed"):
            construct()
        assert path.read_bytes() == original_xml
    else:
        construct()
        reloaded = mujoco.MjModel.from_xml_path(str(path))
        assert reloaded.geom_conaffinity[_geom(reloaded, "hand_l_index_1_collision")] == -1
        assert _signature(reloaded, "hand_l_thumb_1", "hand_l_index_1") not in reloaded.exclude_signature
        visual = _geom(reloaded, "hand_l_visual")
        assert (reloaded.geom_contype[visual], reloaded.geom_conaffinity[visual]) == (0, 0)
    assert module.put_model is upload


@pytest.mark.parametrize("upload_fails", [False, True])
def test_context_applies_before_upload_and_restores_put_model(policy, monkeypatch, upload_fails):
    model = _model()
    sentinel, device = object(), object()

    def upload(mj_model, positional, *, device):
        assert positional is sentinel
        assert device is requested_device
        assert mj_model is model
        assert model.geom_conaffinity[_geom(model, "hand_l_index_1_collision")] == -1
        assert _signature(model, "hand_l_thumb_1", "hand_l_index_1") not in model.exclude_signature
        if upload_fails:
            raise RuntimeError("upload failed")
        return sentinel

    requested_device = device
    module = SimpleNamespace(put_model=upload)
    monkeypatch.setitem(sys.modules, "mujoco_warp", module)
    logger = SimpleNamespace(info=Mock(), warn=Mock(), warning=Mock())
    if upload_fails:
        with pytest.raises(RuntimeError, match="upload failed"):
            with policy.hand_self_collision_at_put_model(logger):
                module.put_model(model, sentinel, device=device)
    else:
        with policy.hand_self_collision_at_put_model(logger):
            assert module.put_model(model, sentinel, device=device) is sentinel
    assert module.put_model is upload


@pytest.mark.parametrize("filename", ["mujoco_warp.py", "mujoco_warp_multi_env.py"])
def test_real_adapter_build_solver_opens_collision_before_upload(policy, monkeypatch, filename):
    """Run the production constructor boundary with its expensive GPU work stubbed."""
    path = _ADAPTERS / filename
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "build_solver")
    mj_model = _model()
    observations = []

    def upload(model):
        observations.append(model.geom_conaffinity[_geom(model, "hand_l_index_1_collision")])
        assert _signature(model, "hand_l_thumb_1", "hand_l_index_1") not in model.exclude_signature
        return object()

    module = SimpleNamespace(put_model=upload)
    monkeypatch.setitem(sys.modules, "mujoco_warp", module)

    def solver(model, **kwargs):
        module.put_model(mj_model)
        return SimpleNamespace(mj_model=mj_model)

    namespace = {
        "Any": Any,
        "newton": SimpleNamespace(solvers=SimpleNamespace(SolverMuJoCo=solver)),
        "hand_self_collision_at_put_model": policy.hand_self_collision_at_put_model,
        "resolve_mjc_iterations": lambda params: None,
        "resolve_mjc_impratio_cone": lambda params: (None, None),
        "resolve_constraint_capacity": lambda *args: (1024, 512),
        "_compute_body_aabb_local": lambda *args: None,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    adapter = SimpleNamespace(
        _physics_params=None,
        _enable_multiccd=None,
        _njmax=1024,
        _nconmax=512,
        _save_to_mjcf="",
        **{
            name: Mock()
            for name in (
                "_report_collision_filter_budget",
                "_apply_mimic_dof_damping",
                "_apply_mimic_eq_solref",
                "_apply_global_contact_options",
                "_cap_gripper_master_force",
                "_cap_arm_actuator_force",
                "_fix_gripper_loop_constraints",
                "_build_base_pin",
                "_snapshot_base_mocap",
            )
        },
    )
    logger = SimpleNamespace(info=Mock(), warn=Mock(), warning=Mock())
    result = namespace["build_solver"](adapter, SimpleNamespace(body_count=10, world_count=1), 2, 8, logger)

    assert result is adapter._solver
    assert observations == [-1]
    assert module.put_model is upload
