#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pure geometry regression for USDA authored-vertex/tet binding."""

from __future__ import annotations

import importlib.util
import copy
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

_HERE = Path(__file__).resolve().parent
_CHEF = _HERE.parents[2] / "genie_sim_bringup" / "config" / "scene_flat_g2_sp_chef.py"
_WUJI_SCENE = _CHEF.with_name("scene_flat_g2_sp_chef_wuji.yaml")


def _load_chef():
    spec = importlib.util.spec_from_file_location("chef_usda_noodle_mapping", _CHEF)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


chef = _load_chef()


def _unit_cube_proxy():
    particles = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    # Newton 1.4 add_soft_grid's five-tet decomposition for one cell.
    tets = np.array(
        [
            [1, 5, 3, 0],
            [4, 0, 6, 5],
            [2, 6, 0, 3],
            [7, 3, 5, 6],
            [3, 5, 6, 0],
        ],
        dtype=np.int32,
    )
    return particles, tets


def test_authored_points_reconstruct_in_original_order():
    particles, tets = _unit_cube_proxy()
    authored = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.2, 0.3, 0.4],
            [0.8, 0.1, 0.7],
            [0.5, 0.5, 0.5],
        ]
    )

    indices, weights = chef._bind_points_to_tets(authored, particles, tets)
    reconstructed = chef._reconstruct_tet_bound_points(particles, indices, weights)

    assert indices.shape == (len(authored), 4)
    assert weights.shape == (len(authored), 4)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert np.allclose(reconstructed, authored, atol=1.0e-7)


def test_boundary_binding_and_hash_are_deterministic():
    particles, tets = _unit_cube_proxy()
    authored = np.array([[0.5, 0.5, 0.5], [1.0, 0.25, 0.75]])

    first = chef._bind_points_to_tets(authored, particles, tets)
    second = chef._bind_points_to_tets(authored, particles, tets)

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert chef._tet_binding_sha256(*first) == chef._tet_binding_sha256(*second)


def test_point_outside_proxy_is_rejected():
    particles, tets = _unit_cube_proxy()
    with np.testing.assert_raises_regex(ValueError, "outside the soft proxy"):
        chef._bind_points_to_tets([[1.01, 0.5, 0.5]], particles, tets)


def test_sparse_voxel_proxy_preserves_hole_and_has_positive_tets():
    occupied_centers = [[0.5, y + 0.5, z + 0.5] for y in range(4) for z in range(4) if y in (0, 3) or z in (0, 3)]
    points = np.asarray(occupied_centers, dtype=np.float64)
    triangles = np.repeat(np.arange(len(points), dtype=np.int32)[:, None], 3, axis=1)

    vertices, tets, occupied = chef._sparse_voxel_tet_mesh(
        points,
        triangles,
        bbox_min=[0.0, 0.0, 0.0],
        bbox_size=[1.0, 4.0, 4.0],
        resolution=[1, 4, 4],
    )

    assert len(occupied) == 12
    assert (0, 1, 1) not in occupied
    assert (0, 2, 2) not in occupied
    p, q, r, s = vertices[tets].transpose(1, 0, 2)
    volumes = np.linalg.det(np.stack((q - p, r - p, s - p), axis=2)) / 6.0
    assert np.all(volumes > 0.0)
    indices, weights = chef._bind_points_to_tets(points, vertices, tets)
    assert np.allclose(
        chef._reconstruct_tet_bound_points(vertices, indices, weights),
        points,
        atol=1.0e-7,
    )


def test_selecting_instance_ids_preserves_procedural_spawn_samples():
    class Builder:
        def __init__(self):
            self.particle_count = 0
            self.calls = []

        def add_soft_grid(self, **kwargs):
            self.calls.append(kwargs)
            p = params.n_len
            self.particle_count += 4 * (p + 1)

    params = chef.NoodleParams(n_noodles=3, n_len=2)
    full = Builder()
    selected = Builder()
    chef._emit_noodles(full, params, 1.0, (2.0, 3.0))
    chef._emit_noodles(selected, params, 1.0, (2.0, 3.0), instance_ids=[1])

    assert len(selected.calls) == 1
    assert np.allclose(tuple(selected.calls[0]["pos"]), tuple(full.calls[1]["pos"]))
    assert np.allclose(tuple(selected.calls[0]["rot"]), tuple(full.calls[1]["rot"]))


def test_wuji_scene_pins_complete_noodle_family_and_topology():
    config = yaml.safe_load(_WUJI_SCENE.read_text())
    noodles = config["newton"]["chef"]["usd_noodles"]
    assert noodles["enabled"] is True
    assert noodles["object_name_prefix"] == "benchmark_noodles_001_"
    assert noodles["expected_instance_count"] == 16
    assert noodles["expected_vertex_count"] == 1100
    assert noodles["expected_topology_sha256"] == "cc12e7d653fb61af0178055723c0cb2644f4fb997a0776d1073950b292253049"


def test_wuji_scene_pins_only_retained_side_dish_families():
    config = yaml.safe_load(_WUJI_SCENE.read_text())
    usd_objects = config["newton"]["chef"]["usd_objects"]
    noodles = config["newton"]["chef"]["usd_noodles"]
    side_dishes = config["newton"]["chef"]["usd_side_dishes"]
    families = side_dishes["families"]
    assert side_dishes["enabled"] is True
    assert side_dishes["expected_instance_count"] == 26
    assert set(families) == set(side_dishes["migrated_families"])
    assert sum(item["expected_instance_count"] for item in families.values()) == 26
    assert len({item["expected_topology_sha256"] for item in families.values()}) == 6
    assert (
        usd_objects["expected_food_count"]
        == (noodles["expected_instance_count"] + side_dishes["expected_instance_count"])
        == 42
    )

    semantic_counts = {}
    for item in families.values():
        semantic_counts[item["semantic_kind"]] = (
            semantic_counts.get(item["semantic_kind"], 0) + item["expected_instance_count"]
        )
    assert semantic_counts == {
        "carrot": 11,
        "meat": 4,
        "pepper": 11,
    }
    assert families["benchmark_side_dish_004"]["proxy_geometry"] == "strip_grid"
    assert families["benchmark_side_dish_005"]["proxy_geometry"] == "strip_grid"
    assert families["benchmark_side_dish_000"]["proxy_resolution"] == [2, 6, 3]
    assert families["benchmark_side_dish_009"]["proxy_resolution"] == [1, 6, 3]
    expected_meat_topology = {
        "benchmark_side_dish_000": (
            402,
            "04738e93dd5a1ad237b1fd9dde08600fb47672f8830b0a7204fac4be2fa634bc",
        ),
        "benchmark_side_dish_008": (
            318,
            "3df25802daaa21a49865912172ec60fd7ad65dc18596a105e4622b8407916ddd",
        ),
        "benchmark_side_dish_009": (
            268,
            "bfdd819b07bb5125ead874d79e272dc7d7dd65a9cd17d05ecdb81c9c010aa55e",
        ),
        "benchmark_side_dish_010": (
            296,
            "dff929e024eaa87f4a6ed7e2154528734ead7941f0379a16b4284bc33cdb100f",
        ),
    }
    assert {
        family: (
            families[family]["expected_vertex_count"],
            families[family]["expected_topology_sha256"],
        )
        for family in expected_meat_topology
    } == expected_meat_topology
    assert side_dishes["migrated_families"] == [
        "benchmark_side_dish_000",
        "benchmark_side_dish_004",
        "benchmark_side_dish_005",
        "benchmark_side_dish_008",
        "benchmark_side_dish_009",
        "benchmark_side_dish_010",
    ]
    assert side_dishes["procedural_fallback"] is False
    assert sum(families[family]["expected_instance_count"] for family in side_dishes["migrated_families"]) == 26
    assert families["benchmark_side_dish_005"]["physics_profile"] == "pepper"
    assert all(item["physics_profile"] != "pending" for item in families.values())


def _imported_food_shapes(monkeypatch, layout):
    class ShapeFlags:
        VISIBLE = 1
        COLLIDE_SHAPES = 2
        COLLIDE_PARTICLES = 4

    monkeypatch.setitem(sys.modules, "newton", types.SimpleNamespace(ShapeFlags=ShapeFlags))
    paths = tuple(
        [f"/World/objects/benchmark_noodles_001_{index:02d}" for index in range(16)]
        + [f"/World/objects/benchmark_side_dish_000_{index:02d}" for index in range(26)]
    )
    labels = ["/World/objects/benchmark_pot_013/entity/body/visual"]
    food_indices = []
    for index, path in enumerate(paths):
        if layout == "single":
            suffixes = ("entity/body/visual",)
        elif layout == "visual_and_bbox":
            suffixes = ("entity/body/visual", "lowpoly/bbox")
        else:
            suffixes = ("entity/body/visual", "lowpoly/bbox") + tuple(
                f"entity/body/visual/material_{subset}" for subset in range(index % 3)
            )
        for suffix in suffixes:
            food_indices.append(len(labels))
            labels.append(f"{path}/{suffix}")
    labels.extend(("/genie/hand_r_index_link1/collision", "/World/background/table/visual"))
    plugin = chef.ScenePlugin()
    plugin.cfg = copy.deepcopy(chef.CONFIG)
    plugin._usd_food_object_paths = paths
    builder = types.SimpleNamespace(shape_label=labels, shape_flags=[15] * len(labels))
    return plugin, builder, food_indices


@pytest.mark.parametrize("layout", ["single", "visual_and_bbox", "variable_subsets"])
def test_imported_food_shape_cleanup_counts_exact_entities_and_disables_all_their_shapes(monkeypatch, layout):
    plugin, builder, food_indices = _imported_food_shapes(monkeypatch, layout)

    disabled = plugin._disable_usda_food_shapes(builder)

    assert disabled == food_indices
    assert len(plugin._usd_food_object_paths) == plugin.cfg.usd_objects.expected_food_count == 42
    for index, flags in enumerate(builder.shape_flags):
        assert flags == (8 if index in food_indices else 15)
    if layout == "visual_and_bbox":
        assert len(disabled) == 84
    elif layout == "variable_subsets":
        assert len(disabled) > 84


@pytest.mark.parametrize("inventory_change", ["missing", "extra", "replacement", "duplicate_replaces_missing"])
def test_imported_food_shape_cleanup_rejects_wrong_entity_inventory_before_mutating_flags(
    monkeypatch, inventory_change
):
    plugin, builder, _food_indices = _imported_food_shapes(monkeypatch, "single")
    last_food_index = len(builder.shape_label) - 3
    if inventory_change == "missing":
        del builder.shape_label[last_food_index]
        del builder.shape_flags[last_food_index]
    elif inventory_change == "extra":
        builder.shape_label.append("/World/objects/benchmark_side_dish_999/entity/body/visual")
        builder.shape_flags.append(15)
    elif inventory_change == "replacement":
        builder.shape_label[last_food_index] = "/World/objects/benchmark_side_dish_999/entity/body/visual"
    else:
        # The total shape count is still 42, but one entity has two shapes and
        # another has none: counting shape entries must not accept this case.
        builder.shape_label[last_food_index] = f"{plugin._usd_food_object_paths[0]}/lowpoly/bbox"
    previous_flags = builder.shape_flags.copy()

    with pytest.raises(RuntimeError, match="food"):
        plugin._disable_usda_food_shapes(builder)

    assert builder.shape_flags == previous_flags


def test_stage_food_inventory_captures_exact_object_paths_before_import(monkeypatch):
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    plugin, _builder, _food_indices = _imported_food_shapes(monkeypatch, "single")
    expected_paths = plugin._usd_food_object_paths
    plugin._usd_food_object_paths = ()
    plugin.cfg.usd_objects.enabled = True
    plugin.cfg.usd_objects.procedural_food_transition = True
    stage = Usd.Stage.CreateInMemory()
    pot = UsdGeom.Xform.Define(stage, plugin.cfg.usd_objects.pot_body_prim_path).GetPrim()
    UsdPhysics.MassAPI.Apply(pot).CreateMassAttr(plugin.cfg.usd_objects.expected_pot_mass)
    mesh = UsdGeom.Mesh.Define(stage, plugin.cfg.usd_objects.pot_mesh_prim_path)
    mesh.GetPointsAttr().Set([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.GetFaceVertexCountsAttr().Set([3])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])
    for path in expected_paths:
        body = UsdGeom.Xform.Define(stage, f"{path}/entity/body").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body).CreateRigidBodyEnabledAttr(True)
        food_mesh = UsdGeom.Mesh.Define(stage, f"{path}/entity/body/visual").GetPrim()
        UsdPhysics.CollisionAPI.Apply(food_mesh).CreateCollisionEnabledAttr(True)
        # A distinct auxiliary visual prim is not another food entity.
        UsdGeom.Cube.Define(stage, f"{path}/lowpoly/bbox")
    monkeypatch.setattr(plugin, "_apply_scene_overrides", lambda _ctx: None)
    monkeypatch.setattr(plugin, "_prepare_usda_noodle_sources", lambda _foods, _xforms: None)
    monkeypatch.setattr(plugin, "_prepare_usda_side_dish_sources", lambda _foods, _xforms: None)
    monkeypatch.setattr(
        chef,
        "_fit_usda_wok_bowl",
        lambda *_args: {"rim_radius": 0.1, "depth": 0.05, "component_count": 1},
    )
    ctx = types.SimpleNamespace(logger=types.SimpleNamespace(info=lambda _message: None))

    plugin.on_stage(stage, ctx)

    assert plugin._usd_food_object_paths == expected_paths
    for path in expected_paths:
        assert stage.GetPrimAtPath(path).IsValid()
        assert stage.GetPrimAtPath(f"{path}/entity/body/visual").IsValid()
        assert stage.GetPrimAtPath(f"{path}/lowpoly/bbox").IsValid()


def test_scene_override_loads_nested_side_dish_classification():
    class Logger:
        def info(self, _message):
            pass

        def warn(self, message):
            raise AssertionError(message)

    class Context:
        logger = Logger()
        scene_cfg = yaml.safe_load(_WUJI_SCENE.read_text())

    plugin = chef.ScenePlugin()
    plugin.cfg = copy.deepcopy(chef.CONFIG)
    plugin._apply_scene_overrides(Context())

    assert plugin.cfg.usd_side_dishes.enabled is True
    assert plugin.cfg.usd_side_dishes.expected_instance_count == 26
    assert len(plugin.cfg.usd_side_dishes.families) == 6
    assert plugin.cfg.food_state.hz == 30.0
    assert plugin.cfg.startup_hand_motion.enabled is True
    assert plugin.cfg.startup_hand_motion.delay_s == 1.0
    assert plugin.cfg.startup_hand_motion.duration_s == 2.0
    assert len(plugin.cfg.startup_hand_motion.target_joint_pos) == 20


def test_usda_background_keeps_rigid_collision_but_not_particle_collision(monkeypatch):
    class ShapeFlags:
        VISIBLE = 1
        COLLIDE_SHAPES = 2
        COLLIDE_PARTICLES = 4

    monkeypatch.setitem(sys.modules, "newton", types.SimpleNamespace(ShapeFlags=ShapeFlags))
    builder = types.SimpleNamespace(
        shape_label=[
            "/World/background/benchmark_gas_stove_001/entity/body/visual_convex_00",
            "/World/background/scene/Cube",
            "/World/objects/benchmark_pot_013/entity/body/visual",
            "/genie/base_link/collisions/mesh",
        ],
        shape_flags=[7, 7, 7, 7],
    )
    messages = []
    ctx = types.SimpleNamespace(
        logger=types.SimpleNamespace(info=messages.append),
    )

    changed = chef.ScenePlugin._disable_usda_background_particle_collisions(builder, ctx)

    assert changed == 2
    assert builder.shape_flags == [3, 3, 7, 7]
    assert all(flag & ShapeFlags.VISIBLE for flag in builder.shape_flags)
    assert all(flag & ShapeFlags.COLLIDE_SHAPES for flag in builder.shape_flags)
    assert "shapes=2" in messages[0]


def test_usda_stove_grater_collision_is_disabled_only_for_gas_stove(monkeypatch):
    class ShapeFlags:
        VISIBLE = 1
        COLLIDE_SHAPES = 2
        COLLIDE_PARTICLES = 4

    monkeypatch.setitem(sys.modules, "newton", types.SimpleNamespace(ShapeFlags=ShapeFlags))
    builder = types.SimpleNamespace(
        shape_label=[
            "/World/background/benchmark_gas_stove_001/entity/body/collider_000",
            "/World/background/benchmark_gas_stove_001/entity/body/visual_convex_00",
            "/World/background/benchmark_table_019/entity/body/collider_00",
            "/World/background/scene/Cube",
        ],
        shape_flags=[7, 7, 7, 7],
    )
    messages = []
    ctx = types.SimpleNamespace(logger=types.SimpleNamespace(info=messages.append))

    changed = chef.ScenePlugin._disable_usda_stove_rigid_collisions(builder, ctx)

    assert changed == 2
    assert builder.shape_flags == [1, 1, 7, 7]
    assert "shapes=2" in messages[0]


def test_usda_stove_support_uses_imported_wok_pose(monkeypatch):
    class ShapeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class ShapeFlags:
        VISIBLE = 1

    monkeypatch.setitem(
        sys.modules,
        "newton",
        types.SimpleNamespace(
            ShapeFlags=ShapeFlags,
            ModelBuilder=types.SimpleNamespace(ShapeConfig=ShapeConfig),
        ),
    )

    class Builder:
        body_q = [chef.wp.transform(chef.wp.vec3(1.2, -0.4, 0.7), chef.wp.quat_identity())]

        def __init__(self):
            self.shape_flags = []
            self.calls = []

        def add_shape_sphere(self, body, **kwargs):
            self.calls.append((body, kwargs))
            self.shape_flags.append(1)
            return len(self.calls) - 1

    class Logger:
        def info(self, _message):
            pass

    plugin = chef.ScenePlugin()
    plugin.cfg = copy.deepcopy(chef.CONFIG)
    plugin.cfg.stove.ring_beads = 3
    plugin.cfg.stove.bead_radius = 0.01
    plugin.cfg.wok.rim_r = 0.16
    plugin.cfg.wok.depth = 0.09
    plugin.pan_id = 0
    plugin._usd_wok_fit = {
        "center": np.array([0.1, 0.2, 0.3]),
        "up": np.array([0.0, 0.0, 1.0]),
        "radial_0": np.array([1.0, 0.0, 0.0]),
        "handle_axis": np.array([1.0, 0.0, 0.0]),
    }
    # The helper only needs the fitted axes used by the runtime method.
    plugin._usd_wok_fit["radial_1"] = np.array([0.0, 1.0, 0.0])
    builder = Builder()

    plugin._add_usda_stove_support(builder, types.SimpleNamespace(logger=Logger()))

    assert len(builder.calls) == 3
    points = [call[1]["xform"].p for call in builder.calls]
    centre = np.mean(np.asarray([[p[0], p[1], p[2]] for p in points]), axis=0)
    assert np.allclose(centre, [1.3, -0.2, 0.93203044], atol=1.0e-6)
    assert all((flag & ShapeFlags.VISIBLE) == 0 for flag in builder.shape_flags)


def test_viewer_keeps_native_particles_when_custom_render_fails(monkeypatch):
    plugin = chef.ScenePlugin()
    plugin._render_ready = True
    plugin.tube_groups = [
        {
            "K": 1,
            "M": 2,
            "start": 0,
            "centroids_dev": object(),
            "out_pts": 2,
            "cr_W_dev": object(),
            "smooth_dev": object(),
            "radius": 0.001,
            "verts": object(),
            "norms": object(),
            "tube_idx": object(),
            "name": "chef_noodles",
            "color": (1.0, 1.0, 1.0),
        }
    ]
    viewer = types.SimpleNamespace(show_particles=True, show_triangles=True)
    state = types.SimpleNamespace(
        particle_q=types.SimpleNamespace(device="cpu"),
    )
    messages = []
    ctx = types.SimpleNamespace(logger=types.SimpleNamespace(warn=messages.append))

    monkeypatch.setattr(chef.wp, "launch", lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError()))
    plugin.on_render(viewer, state, 0.0, ctx)

    assert viewer.show_particles is True
    assert viewer.show_triangles is True
    assert plugin._native_render_hidden is False
    assert len(messages) == 1


def test_wuji_scene_starts_pregrasp_then_targets_configured_grasp():
    config = yaml.safe_load(_WUJI_SCENE.read_text())
    initial = {name: value for name, value in config["robot"]["init_joint_pos"].items() if name.startswith("hand_r_")}
    motion = config["newton"]["chef"]["startup_hand_motion"]
    target = motion["target_joint_pos"]

    assert motion["enabled"] is True
    assert motion["delay_s"] > 0.0
    assert motion["duration_s"] > 0.0
    assert set(initial) == set(target)
    assert len(target) == 20
    assert initial["hand_r_thumb_joint1"] == 22.97
    assert initial["hand_r_index_joint1"] == 22.35
    assert target["hand_r_thumb_joint1"] == 6.798686
    assert target["hand_r_index_joint1"] == 78.855531


def test_startup_hand_motion_waits_then_uses_smoothstep_in_radians():
    start = {"hand_r_index_joint1": 10.0}
    target = {"hand_r_index_joint1": 70.0}

    assert chef._startup_joint_targets_radians(start, target, 1.0, 2.0, 0.5) is None
    at_start = chef._startup_joint_targets_radians(start, target, 1.0, 2.0, 1.0)
    halfway = chef._startup_joint_targets_radians(start, target, 1.0, 2.0, 2.0)
    at_end = chef._startup_joint_targets_radians(start, target, 1.0, 2.0, 3.0)

    assert np.isclose(at_start["hand_r_index_joint1"], np.deg2rad(10.0))
    assert np.isclose(halfway["hand_r_index_joint1"], np.deg2rad(40.0))
    assert np.isclose(at_end["hand_r_index_joint1"], np.deg2rad(70.0))


def test_side_dish_pilots_replace_matching_procedural_instances():
    class Builder:
        def __init__(self):
            self.particle_count = 24
            self.tri_indices = []

        def add_soft_grid(self, **kwargs):
            self.particle_count += (kwargs["dim_x"] + 1) * (kwargs["dim_y"] + 1) * (kwargs["dim_z"] + 1)

    plugin = chef.ScenePlugin()
    plugin.cfg = copy.deepcopy(chef.CONFIG)
    plugin.cfg.noodle.n_noodles = 0
    plugin.cfg.ham.n_cubes = 2
    plugin.cfg.ham.dim = 1
    plugin.cfg.pepper.n = 2
    plugin.cfg.pepper.n_len = 1
    plugin.cfg.carrot.n = 2
    plugin.cfg.carrot.n_len = 1

    bindings = []
    for food_type, kind, start in ((2, "ham", 0), (3, "pepper", 8), (4, "carrot", 16)):
        instance_path = f"/World/objects/pilot_{kind}"
        binding = {
            "food_type": food_type,
            "kind": kind,
            "semantic_kind": "meat" if kind == "ham" else kind,
            "instance_id": 0,
            "family": f"family_{kind}",
            "family_instance_id": 0,
            "instance_prim_path": instance_path,
            "object_prim_path": f"{instance_path}/entity",
            "parent_prim_path": f"{instance_path}/entity",
            "mesh_prim_path": f"{instance_path}/entity/body/visual",
            "recording_prim_path": f"{instance_path}/entity/body/visual",
            "parent_pose_world": np.eye(4).tolist(),
            "mesh_pose_parent": np.eye(4).tolist(),
            "points_coordinate_space": "mesh_local",
            "particle_start": start,
            "particle_count": 8,
            "vertex_count": 10,
            "proxy_geometry": "solid_grid",
            "proxy_resolution": [1, 1, 1],
            "proxy_occupied_cell_count": 1,
            "mapping_kind": "tet_barycentric_v1",
            "topology_sha256": "topology",
            "mapping_sha256": "mapping",
            "rest_reconstruction_max_error_m": 1.0e-7,
        }
        bindings.append(binding)

    builder = Builder()
    plugin._add_procedural_food(
        builder,
        plugin.cfg,
        (1.0, 0.0),
        1.1,
        authored_side_dishes=bindings,
    )

    for kind in ("ham", "pepper", "carrot"):
        items = sorted(
            (item for item in plugin.food_instances if item["kind"] == kind),
            key=lambda item: item["instance_id"],
        )
        assert [item["instance_id"] for item in items] == [0, 1]
        assert "recording_prim_path" in items[0]
        assert "proposed_usd_prim" not in items[0]
        assert "proposed_usd_prim" in items[1]


def test_disabled_side_dish_fallback_keeps_only_authored_bindings():
    class Builder:
        def __init__(self):
            self.particle_count = 24
            self.tri_indices = []

        def add_soft_grid(self, **_kwargs):
            raise AssertionError("procedural side dish must not be emitted")

    plugin = chef.ScenePlugin()
    plugin.cfg = copy.deepcopy(chef.CONFIG)
    plugin.cfg.noodle.n_noodles = 0
    plugin.cfg.usd_side_dishes.procedural_fallback = False

    bindings = []
    for food_type, kind, instance_id, start in (
        (2, "ham", 3, 0),
        (3, "pepper", 15, 8),
        (4, "carrot", 10, 16),
    ):
        instance_path = f"/World/objects/authored_{kind}"
        bindings.append(
            {
                "food_type": food_type,
                "kind": kind,
                "semantic_kind": "meat" if kind == "ham" else kind,
                "instance_id": instance_id,
                "family": f"family_{kind}",
                "family_instance_id": 0,
                "instance_prim_path": instance_path,
                "object_prim_path": f"{instance_path}/entity",
                "parent_prim_path": f"{instance_path}/entity",
                "mesh_prim_path": f"{instance_path}/entity/body/visual",
                "recording_prim_path": f"{instance_path}/entity/body/visual",
                "parent_pose_world": np.eye(4).tolist(),
                "mesh_pose_parent": np.eye(4).tolist(),
                "points_coordinate_space": "mesh_local",
                "particle_start": start,
                "particle_count": 8,
                "vertex_count": 10,
                "proxy_geometry": "solid_grid",
                "proxy_resolution": [1, 1, 1],
                "proxy_occupied_cell_count": 1,
                "mapping_kind": "tet_barycentric_v1",
                "topology_sha256": "topology",
                "mapping_sha256": "mapping",
                "rest_reconstruction_max_error_m": 1.0e-7,
            }
        )

    builder = Builder()
    plugin._add_procedural_food(
        builder,
        plugin.cfg,
        (1.0, 0.0),
        1.1,
        authored_side_dishes=bindings,
    )

    assert builder.particle_count == 24
    assert plugin.food_instances == [
        {key: value for key, value in binding.items() if key not in {"tet_particle_indices", "barycentric_weights"}}
        for binding in bindings
    ]
    assert plugin.ham_tri_range == (0, 0)
    assert plugin.pepper_render_count == 0
    assert plugin.carrot_render_count == 0


def test_all_authored_noodles_replace_procedural_instances_with_exact_paths():
    class Builder:
        def __init__(self):
            self.particle_count = 16 * 324
            self.calls = []

        def add_soft_grid(self, **kwargs):
            self.calls.append(kwargs)
            self.particle_count += 324

    plugin = chef.ScenePlugin()
    plugin.cfg = copy.deepcopy(chef.CONFIG)
    plugin.cfg.ham.n_cubes = 0
    plugin.cfg.pepper.n = 0
    plugin.cfg.carrot.n = 0
    bindings = []
    for instance_id in range(16):
        instance_path = f"/World/objects/benchmark_noodles_001_{instance_id:02d}"
        mesh_path = f"{instance_path}/entity/body/visual"
        bindings.append(
            {
                "instance_id": instance_id,
                "particle_start": instance_id * 324,
                "particle_count": 324,
                "instance_prim_path": instance_path,
                "object_prim_path": f"{instance_path}/entity",
                "mesh_prim_path": mesh_path,
                "recording_prim_path": mesh_path,
                "parent_prim_path": f"{instance_path}/entity",
                "parent_pose_world": np.eye(4).tolist(),
                "mesh_pose_parent": np.eye(4).tolist(),
                "points_coordinate_space": "mesh_local",
                "vertex_count": 1100,
                "mapping_kind": "tet_barycentric_v1",
                "topology_sha256": "topology",
                "mapping_sha256": "shared-mapping",
                "rest_reconstruction_max_error_m": 1.0e-7,
            }
        )

    builder = Builder()
    plugin._add_procedural_food(
        builder,
        plugin.cfg,
        (1.0, 0.0),
        1.1,
        authored_noodles=bindings,
    )

    noodles = [item for item in plugin.food_instances if item["kind"] == "noodle"]
    assert builder.calls == []
    assert builder.particle_count == 16 * 324
    assert len(noodles) == 16
    assert [item["instance_id"] for item in noodles] == list(range(16))
    assert [item["particle_start"] for item in noodles] == list(range(0, 16 * 324, 324))
    assert all(item["particle_count"] == 324 for item in noodles)
    assert all(item["mapping_kind"] == "tet_barycentric_v1" for item in noodles)
    assert {item["mapping_sha256"] for item in noodles} == {"shared-mapping"}
    assert all("proposed_usd_prim" not in item for item in noodles)
    assert all(item["parent_prim_path"] == item["object_prim_path"] for item in noodles)
    assert all(item["points_coordinate_space"] == "mesh_local" for item in noodles)
    assert all(np.array(item["parent_pose_world"]).shape == (4, 4) for item in noodles)
    assert all(np.array(item["mesh_pose_parent"]).shape == (4, 4) for item in noodles)
    assert [item["recording_prim_path"] for item in noodles] == [
        f"/World/objects/benchmark_noodles_001_{instance_id:02d}/entity/body/visual" for instance_id in range(16)
    ]


def test_food_state_publisher_emits_static_authored_mapping_and_tick_aligned_state(monkeypatch):
    class Stamp:
        sec = 0
        nanosec = 0

    class Header:
        def __init__(self):
            self.stamp = Stamp()
            self.frame_id = ""

    class PointCloud2:
        def __init__(self):
            self.header = Header()

    class PointField:
        FLOAT32 = 7
        UINT8 = 2
        UINT16 = 4
        UINT32 = 6

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class String:
        def __init__(self):
            self.data = ""

    class Publisher:
        def __init__(self, topic):
            self.topic = topic
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    class Node:
        def __init__(self):
            self.publishers = {}

        def create_publisher(self, _msg_type, topic, _qos):
            publisher = Publisher(topic)
            self.publishers[topic] = publisher
            return publisher

        def get_namespace(self):
            return "/genie_sim_engine"

    class QoSProfile:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    rclpy = types.ModuleType("rclpy")
    rclpy.ok = lambda: True
    node = Node()
    rclpy.create_node = lambda *_args, **_kwargs: node
    qos = types.ModuleType("rclpy.qos")
    qos.DurabilityPolicy = types.SimpleNamespace(TRANSIENT_LOCAL=1)
    qos.HistoryPolicy = types.SimpleNamespace(KEEP_LAST=1)
    qos.ReliabilityPolicy = types.SimpleNamespace(RELIABLE=1)
    qos.QoSProfile = QoSProfile
    qos.qos_profile_sensor_data = object()
    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.PointCloud2 = PointCloud2
    sensor_msgs_msg.PointField = PointField
    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.String = String
    for name, module in {
        "rclpy": rclpy,
        "rclpy.qos": qos,
        "sensor_msgs": sensor_msgs,
        "sensor_msgs.msg": sensor_msgs_msg,
        "std_msgs": std_msgs,
        "std_msgs.msg": std_msgs_msg,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    cfg = copy.deepcopy(chef.CONFIG)
    cfg.food_state.hz = 30.0
    instances = []
    bindings = []
    for binding_index, start in enumerate((0, 4)):
        instance_path = f"/World/objects/food_{binding_index}"
        mesh_path = f"{instance_path}/entity/body/visual"
        parent_path = f"{instance_path}/entity"
        item = {
            "food_type": binding_index + 1,
            "kind": "noodle",
            "instance_id": binding_index,
            "instance_prim_path": instance_path,
            "parent_prim_path": parent_path,
            "mesh_prim_path": mesh_path,
            "recording_prim_path": mesh_path,
            "parent_pose_world": np.eye(4).tolist(),
            "mesh_pose_parent": np.eye(4).tolist(),
            "points_coordinate_space": "mesh_local",
            "particle_start": start,
            "particle_count": 4,
            "vertex_count": 2,
            "mapping_kind": "tet_barycentric_v1",
            "topology_sha256": f"topology-{binding_index}",
            "mapping_sha256": f"mapping-{binding_index}",
            "rest_reconstruction_max_error_m": 0.0,
        }
        binding = dict(item)
        binding["tet_particle_indices"] = np.array([[0, 1, 2, 3], [3, 2, 1, 0]])
        binding["barycentric_weights"] = np.array([[1.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]], dtype=np.float32)
        instances.append(item)
        bindings.append(binding)

    publisher = chef._ChefFoodStatePublisher(
        cfg,
        instances,
        bindings,
        particle_count=8,
        physics_hz=60.0,
        logger=types.SimpleNamespace(info=lambda *_args: None),
    )

    info = json.loads(node.publishers[cfg.food_state.info_topic].messages[0].data)
    assert info["schema_version"] == 2
    assert info["points_coordinate_space"] == "world"
    assert info["mapping_point_count"] == 4
    assert info["instances"][0]["object_key"] == "World_objects_food_0_entity"
    assert info["instances"][0]["mapping_offset"] == 0
    assert info["instances"][1]["mapping_offset"] == 2

    mapping_msg = node.publishers[cfg.food_state.mapping_topic].messages[0]
    assert len(mapping_msg.data) == 4 * chef._ChefFoodStatePublisher._MAPPING_DTYPE.itemsize
    mapping = np.frombuffer(mapping_msg.data, dtype=chef._ChefFoodStatePublisher._MAPPING_DTYPE)
    assert mapping["binding_index"].tolist() == [0, 0, 1, 1]
    assert mapping["vertex_index"].tolist() == [0, 1, 0, 1]
    assert mapping["particle_0"].tolist() == [0, 3, 4, 7]

    positions = np.arange(24, dtype=np.float32).reshape(8, 3)
    rigid = np.array([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    body_paths = ["/World/objects/pot/entity"]
    publisher.publish(None, None, body_paths, 0.01)
    publisher.publish(positions, rigid, body_paths, 0.02)
    state_messages = node.publishers[cfg.food_state.state_topic].messages
    assert len(state_messages) == 1
    points = np.frombuffer(state_messages[0].data, dtype=chef._ChefFoodStatePublisher._POINT_DTYPE)
    assert np.array_equal(np.column_stack((points["x"], points["y"], points["z"])), positions)
    rigid_messages = node.publishers[cfg.food_state.rigid_state_topic].messages
    assert len(rigid_messages) == 1
    rigid_rows = np.frombuffer(
        rigid_messages[0].data,
        dtype=chef._ChefFoodStatePublisher._RIGID_DTYPE,
    )
    assert np.array_equal(
        np.column_stack(tuple(rigid_rows[name] for name in ("x", "y", "z", "qw", "qx", "qy", "qz"))),
        rigid,
    )
    latest_info = json.loads(node.publishers[cfg.food_state.info_topic].messages[-1].data)
    assert latest_info["body_paths"] == body_paths


def test_food_recording_requests_snapshots_only_while_publisher_is_live():
    plugin = chef.ScenePlugin()
    assert not plugin.record_snapshot_enabled()
    plugin._food_state_pub = object()
    assert plugin.record_snapshot_enabled()
    plugin._food_state_pub = None
    assert not plugin.record_snapshot_enabled()


def test_food_publisher_failure_turns_off_snapshot_requests():
    from unittest.mock import Mock

    plugin = chef.ScenePlugin()
    plugin._food_state_pub = types.SimpleNamespace(publish=Mock(side_effect=RuntimeError("publisher failed")))
    ctx = types.SimpleNamespace(robot_prefix="genie", logger=types.SimpleNamespace(warn=Mock()))
    assert plugin.record_snapshot_enabled()
    plugin.on_record_snapshot(None, np.zeros((0, 7)), [], 0.01, ctx)
    assert not plugin.record_snapshot_enabled()
    ctx.logger.warn.assert_called_once()
