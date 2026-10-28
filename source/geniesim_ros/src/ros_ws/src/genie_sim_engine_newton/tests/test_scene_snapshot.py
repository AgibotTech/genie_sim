# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Snapshot boundary tests with no renderer, ROS context or GPU."""

from abc import ABC, abstractmethod
import builtins
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "genie_sim_engine_newton/engine/newton/visualizers/scene_snapshot.py"
spec = spec_from_file_location("tested_scene_snapshot", SOURCE)
snapshot = module_from_spec(spec)
sys.modules[spec.name] = snapshot
spec.loader.exec_module(snapshot)


class Array:
    """A Warp-like host readback that still aliases the engine's storage."""

    def __init__(self, value):
        self.value = np.asarray(value)

    def numpy(self):
        return self.value

    def __len__(self):
        return len(self.value)


POINTS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
FACES = np.array([0, 1, 2], dtype=np.int32)


class Base(ABC):
    """Exercise the actual base/sink dispatch contract, including delta colors."""

    constructed = []

    def __init__(self):
        self.events = []
        self.show_triangles = True
        self.show_particles = False
        self.clear_model()
        self.constructed.append(self)

    def clear_model(self):
        self.model = None

    def set_model(self, model):
        self.events.append("set_model")
        self.model = model
        self.log_mesh("/prototype", Array(POINTS), Array(FACES), hidden=True)
        if getattr(model, "fail", False):
            raise RuntimeError("construction failed")

    def begin_frame(self, time):
        self.events.append(("begin", time))

    def log_state(self, state):
        self.events.append("log_state")
        first = not hasattr(self, "logged")
        self.log_instances(
            "/wok",
            "/prototype",
            state.poses,
            None,
            Array([[0.3, 0.4, 0.5]]) if first else None,
            Array([[0.6, 0.7, 0.8, 0.9]]) if first else None,
            opacities=Array([0.4]) if first else None,
        )
        self.log_instances("/hidden_collider", "/prototype", state.poses, None, None, None, hidden=True)
        self.log_mesh(
            "/model/triangles", state.food, Array(FACES), hidden=not self.show_triangles, dynamic=True, opacity=0.6
        )
        self.log_points("/model/particles", state.food, hidden=not self.show_particles)
        self.logged = True

    @abstractmethod
    def log_mesh(self, *args, **kwargs): ...

    @abstractmethod
    def log_instances(self, *args, **kwargs): ...

    @abstractmethod
    def log_points(self, *args, **kwargs): ...

    @abstractmethod
    def log_lines(self, *args, **kwargs): ...

    @abstractmethod
    def log_array(self, *args, **kwargs): ...

    @abstractmethod
    def log_scalar(self, *args, **kwargs): ...

    @abstractmethod
    def apply_forces(self, *args, **kwargs): ...

    @abstractmethod
    def end_frame(self): ...

    @abstractmethod
    def close(self): ...


def make_sim():
    state = SimpleNamespace(
        poses=Array([[1, 2, 3, 0.1, 0.2, 0.3, 0.9]]),
        food=Array(POINTS.copy()),
    )
    sim = SimpleNamespace(_model=SimpleNamespace(), _state_0=state, hooks=[], closed=False)
    sim._plugin_ctx = lambda: "original context"

    def call_plugin(hook, viewer, observed_state, sim_time, ctx):
        assert observed_state is state
        assert ctx == "original context"
        sim.hooks.append((hook, sim_time))
        viewer.events.append("plugin")
        viewer.show_triangles = False
        viewer.show_particles = False
        viewer.log_mesh(
            "/chef_noodles",
            state.food,
            Array(FACES),
            normals=Array([[0, 0, 1]] * 3),
            color=(0.9, 0.8, 0.4),
            backface_culling=False,
            roughness=0.4,
            metallic=0.1,
        )

    sim._call_plugin = call_plugin
    return sim


@pytest.fixture(autouse=True)
def fake_base(monkeypatch):
    Base.constructed.clear()
    monkeypatch.setattr(snapshot, "_load_viewer_base", lambda: Base)


def test_capture_uses_original_base_and_plugin_without_stepping_and_keeps_visibility():
    sim = make_sim()
    scene = snapshot.SceneSnapshot(sim)
    frame = scene.capture(1.25)
    viewer = Base.constructed[-1]
    assert viewer.events == ["set_model", ("begin", 1.25), "log_state", "plugin"]
    assert sim.hooks == [("on_render", 1.25)]
    assert frame["meshes"]["/prototype"]["hidden"] is True
    assert frame["instances"]["/wok"]["hidden"] is False
    assert frame["instances"]["/hidden_collider"]["hidden"] is True
    # Chef suppresses raw FEM after log_state: even the first frame obeys it.
    assert frame["meshes"]["/model/triangles"]["hidden"] is True
    assert frame["meshes"]["/model/triangles"]["opacity"] == 0.6
    custom = frame["meshes"]["/chef_noodles"]
    assert custom["backface_culling"] is False
    assert custom["roughness"] == 0.4 and custom["metallic"] == 0.1
    np.testing.assert_array_equal(custom["normals"], [[0, 0, 1]] * 3)
    np.testing.assert_allclose(frame["instances"]["/wok"]["xforms"], sim._state_0.poses.value)
    assert custom["points"].dtype == np.float32
    assert custom["indices"].dtype == np.int32


def test_later_frames_send_mesh_deltas_but_full_instances_with_retained_appearance():
    sim = make_sim()
    scene = snapshot.SceneSnapshot(sim)
    first = scene.capture(0)
    assert "/prototype" in first["meshes"]
    second = scene.capture(0)
    assert second["meshes"] == {}
    assert set(second["instances"]) == set(first["instances"])
    np.testing.assert_array_equal(second["instances"]["/wok"]["colors"], first["instances"]["/wok"]["colors"])
    np.testing.assert_array_equal(second["instances"]["/wok"]["materials"], first["instances"]["/wok"]["materials"])
    np.testing.assert_allclose(first["instances"]["/wok"]["opacities"], [[0.4]])
    np.testing.assert_array_equal(second["instances"]["/wok"]["opacities"], first["instances"]["/wok"]["opacities"])
    sim._state_0.food.value[0, 2] = 0.5
    third = scene.capture(0.1)
    assert set(third["meshes"]) == {"/model/triangles", "/chef_noodles"}
    np.testing.assert_array_equal(first["meshes"]["/chef_noodles"]["points"], POINTS)


def test_returned_arrays_do_not_alias_engine_or_retained_comparison_cache():
    sim = make_sim()
    scene = snapshot.SceneSnapshot(sim)
    frame = scene.capture(0)
    frame["meshes"]["/chef_noodles"]["points"][:] = 99
    frame["instances"]["/wok"]["xforms"][:] = 99
    frame["instances"]["/wok"]["colors"][:] = 99
    following = scene.capture(0)
    assert following["meshes"] == {}
    assert np.max(following["instances"]["/wok"]["xforms"]) < 99
    np.testing.assert_allclose(following["instances"]["/wok"]["colors"], [[0.3, 0.4, 0.5]])
    np.testing.assert_array_equal(sim._state_0.food.value, POINTS)


def test_construction_failure_releases_viewer_caches():
    sim = make_sim()
    sim._model.fail = True
    with pytest.raises(RuntimeError, match="construction failed"):
        snapshot.SceneSnapshot(sim)
    viewer = Base.constructed[-1]
    assert viewer.model is None
    assert not viewer._snapshot_meshes


def test_close_is_idempotent_and_does_not_close_the_engine():
    sim = make_sim()
    scene = snapshot.SceneSnapshot(sim)
    scene.capture(0)
    scene.close()
    scene.close()
    assert sim.closed is False
    assert sim._model is not None
    with pytest.raises(RuntimeError, match="closed"):
        scene.capture(0)


def test_invalid_or_nonfinite_geometry_fails_instead_of_sending_a_partial_snapshot():
    sim = make_sim()
    scene = snapshot.SceneSnapshot(sim)
    sim._state_0.food.value[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        scene.capture(0)
    sim._state_0.food.value[0, 0] = 0
    # Initial static prototypes remain pending after a failed capture.
    assert "/prototype" in scene.capture(0)["meshes"]


def test_module_import_does_not_import_newton_or_a_renderer(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"newton", "warp", "pyglet", "omni", "isaacsim"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    fresh_spec = spec_from_file_location("fresh_cpu_scene_snapshot", SOURCE)
    fresh = module_from_spec(fresh_spec)
    fresh_spec.loader.exec_module(fresh)
    assert fresh.SceneSnapshot


def test_required_mesh_must_be_computed_this_frame_even_when_unchanged():
    sim = make_sim()
    scene = snapshot.SceneSnapshot(sim, required_meshes=["/chef_noodles"])
    scene.capture(0)
    assert scene.capture(0)["meshes"] == {}
    # A plugin can catch its own exception and return normally: an old mesh
    # in the cache cannot satisfy the required-mesh barrier for the new frame.
    sim._call_plugin = lambda *args: None
    with pytest.raises(RuntimeError, match="missing required meshes.*chef_noodles"):
        scene.capture(0)


def test_required_mesh_check_rejects_partial_food_and_can_retry_complete_capture():
    sim = make_sim()
    scene = snapshot.SceneSnapshot(sim, required_meshes=["/chef_noodles", "/chef_ham"])
    original = sim._call_plugin
    with pytest.raises(RuntimeError, match="chef_ham"):
        scene.capture(0)

    def complete(*args):
        original(*args)
        viewer = args[1]
        viewer.log_mesh("/chef_ham", sim._state_0.food, Array(FACES))

    sim._call_plugin = complete
    frame = scene.capture(0)
    assert {"/prototype", "/chef_noodles", "/chef_ham"} <= frame["meshes"].keys()


def test_authored_robot_uses_exact_body_poses_and_filters_only_robot_rows(monkeypatch):
    original = Base.set_model

    def set_model(self, model):
        original(self, model)
        # Robot and scene share one primitive batch, as real Newton can do.
        self._shape_instances = {0: SimpleNamespace(name="/wok", model_shapes=[0, 1])}

    monkeypatch.setattr(Base, "set_model", set_model)
    sim = make_sim()
    sim._body_paths = ["/genie/base_link", "/chef_wok"]
    sim._model.shape_body = Array([0, 1])
    sim._state_0.body_q = Array([[0.0, 0, 0, 0, 0, 0, 1], [1, 2, 3, 0, 0, 0, 1]])
    sim._state_0.poses = Array(sim._state_0.body_q.value.copy())
    # Omit appearance deltas here, as the real base does after first submission.
    original_log = Base.log_state

    def log_state(self, state):
        self.logged = True
        original_log(self, state)

    monkeypatch.setattr(Base, "log_state", log_state)
    scene = snapshot.SceneSnapshot(
        sim, authored_robot={"asset_path": "/assets/robot_visual.usda"}, robot_bodies=["base_link"]
    )
    frame = scene.capture(0)
    assert frame["authored_robot"] == {"asset_path": "/assets/robot_visual.usda"}
    np.testing.assert_array_equal(frame["body_poses"]["base_link"], sim._state_0.body_q.value[0])
    np.testing.assert_array_equal(frame["instances"]["/wok"]["xforms"], sim._state_0.body_q.value[1:])
    assert "/chef_noodles" in frame["meshes"]
    # The message owns its storage and each fresh capture follows Newton.
    frame["body_poses"]["base_link"][:] = 99
    sim._state_0.body_q.value[0, 0] = 0.2
    assert scene.capture(0)["body_poses"]["base_link"][0] == pytest.approx(0.2)


def test_authored_robot_rejects_missing_or_ambiguous_bodies_and_closes_viewer():
    sim = make_sim()
    sim._body_paths = ["/genie/base_link", "/duplicate/base_link"]
    for names, pattern in ((["base_link"], "ambiguous"), (["missing"], "no Newton body"), ([], "nonempty")):
        with pytest.raises(ValueError, match=pattern):
            snapshot.SceneSnapshot(sim, authored_robot={"asset_path": "/asset.usda"}, robot_bodies=names)
        assert Base.constructed[-1].model is None


def visual_config(asset="scene/background.usda"):
    return {
        "asset_path": asset,
        "root_prim": "/World",
        "hide_prims": ["/World/objects"],
        "use_asset_lights": True,
        "body_visuals": [
            {
                "body": "chef_wok",
                "source_prim": "/World/objects/benchmark_pot_013",
                "reference_pose": [0.9949, 0, 1.09, 0, 0, 0, 1],
            }
        ],
        "hide_shape_prefixes": ["chef_stove_"],
    }


def test_authored_scene_config_resolves_assets_root_and_returns_owned_descriptor(tmp_path):
    asset = tmp_path / "scene/background.usda"
    asset.parent.mkdir()
    asset.write_text("#usda 1.0\n")
    config = visual_config()
    descriptor, prefixes = snapshot.SceneSnapshot.authored_visuals_from_config(config, assets_root=tmp_path)
    assert descriptor["asset_path"] == str(asset)
    assert prefixes == ("chef_stove_",)
    assert "hide_shape_prefixes" not in descriptor
    config["hide_prims"].clear()
    config["body_visuals"][0]["reference_pose"][0] = 99
    assert descriptor["hide_prims"] == ["/World/objects"]
    assert descriptor["body_visuals"][0]["reference_pose"][0] == 0.9949
    absolute = visual_config(str(asset))
    assert snapshot.SceneSnapshot.authored_visuals_from_config(absolute, assets_root=None)[0] == descriptor


@pytest.mark.parametrize(
    "change,pattern",
    [
        ({"unknown": True}, "unknown authored_visuals"),
        ({"root_prim": "World"}, "absolute prim path"),
        ({"hide_prims": "/World/objects"}, "list of nonempty"),
        ({"use_asset_lights": "true"}, "boolean"),
        ({"hide_shape_prefixes": [""]}, "nonempty"),
        ({"body_visuals": {}}, "must be a list"),
        ({"body_visuals": [{"body": "chef_wok"}]}, "requires body"),
        (
            {"body_visuals": [{"body": "chef_wok", "source_prim": "/World/pot", "reference_pose": [0] * 7}]},
            "nonzero quaternion",
        ),
        (
            {"body_visuals": [{"body": "chef_wok", "source_prim": "/World/pot", "reference_pose": [np.nan] * 7}]},
            "non-finite",
        ),
    ],
)
def test_authored_scene_config_rejects_invalid_descriptor_before_constructing_viewer(tmp_path, change, pattern):
    asset = tmp_path / "background.usda"
    asset.touch()
    config = visual_config(str(asset))
    config.update(change)
    with pytest.raises(ValueError, match=pattern):
        snapshot.SceneSnapshot.authored_visuals_from_config(config, assets_root=tmp_path)
    assert Base.constructed == []


def test_authored_scene_config_rejects_missing_asset_or_relative_path_without_root(tmp_path):
    with pytest.raises(ValueError, match="requires manifest.base_path"):
        snapshot.SceneSnapshot.authored_visuals_from_config(visual_config(), assets_root=None)
    with pytest.raises(ValueError, match="does not exist"):
        snapshot.SceneSnapshot.authored_visuals_from_config(visual_config(), assets_root=tmp_path)


def test_authored_scene_filters_mixed_batch_by_shape_ids_and_sends_fresh_body_pose(monkeypatch):
    original = Base.set_model

    def set_model(self, model):
        original(self, model)
        # Batch row order differs from model shape order; the wok has two
        # shapes, and the static stove shares its prototype with kept shapes.
        self._shape_instances = {0: SimpleNamespace(name="/wok", model_shapes=[5, 4, 2, 0, 3, 1])}

    monkeypatch.setattr(Base, "set_model", set_model)
    original_log = Base.log_state

    def log_state(self, state):
        self.logged = True
        original_log(self, state)

    monkeypatch.setattr(Base, "log_state", log_state)
    sim = make_sim()
    sim._body_paths = ["/genie/base_link", "/chef_wok"]
    sim._model.shape_body = Array([0, 1, -1, -1, 1, -1])
    sim._model.shape_label = ["robot", "chef_wok_visual", "scene/chef_stove_base", "table", "grip", "other"]
    sim._state_0.body_q = Array([[0, 0, 0, 0, 0, 0, 1], [1, 2, 3, 0, 0, 0, 1]])
    sim._state_0.poses = Array([[i, 0, 0, 0, 0, 0, 1] for i in (5, 4, 2, 0, 3, 1)])
    config = visual_config("/assets/background.usda")
    config.pop("hide_shape_prefixes")
    expected = deepcopy(config)
    scene = snapshot.SceneSnapshot(
        sim,
        authored_robot={"asset_path": "/assets/robot.usda"},
        robot_bodies=["base_link"],
        authored_scene=config,
        hide_shape_prefixes=["chef_stove_"],
    )
    config["hide_prims"].clear()
    frame = scene.capture(0)
    assert frame["authored_scene"] == expected
    np.testing.assert_array_equal(frame["instances"]["/wok"]["xforms"][:, 0], [5, 3])
    np.testing.assert_array_equal(frame["visual_body_poses"]["chef_wok"], [1, 2, 3, 0, 0, 0, 1])
    assert "/chef_noodles" in frame["meshes"]
    # The capture filter never changes physical shape assignments or body state.
    np.testing.assert_array_equal(sim._model.shape_body.value, [0, 1, -1, -1, 1, -1])
    frame["visual_body_poses"]["chef_wok"][:] = 99
    frame["authored_scene"]["hide_prims"].clear()
    sim._state_0.body_q.value[1, 2] = 4
    following = scene.capture(0.1)
    assert following["authored_scene"] == expected
    assert following["visual_body_poses"]["chef_wok"][2] == 4
    assert following["meshes"] == {}


@pytest.mark.parametrize(
    "paths,prefixes,pattern",
    [
        (["/other"], [], "no Newton body"),
        (["/chef_wok", "/duplicate/chef_wok"], [], "ambiguous"),
        (["/chef_wok"], ["missing_"], "matches no Newton shape"),
    ],
)
def test_authored_scene_configuration_failure_closes_viewer(paths, prefixes, pattern):
    sim = make_sim()
    sim._body_paths = paths
    sim._model.shape_label = ["chef_stove_base"]
    config = visual_config("/assets/background.usda")
    config.pop("hide_shape_prefixes")
    with pytest.raises(ValueError, match=pattern):
        snapshot.SceneSnapshot(sim, authored_scene=config, hide_shape_prefixes=prefixes)
    assert Base.constructed[-1].model is None
    assert sim.closed is False


@pytest.mark.parametrize("opacity", [-0.1, 1.1, float("nan")])
def test_instances_reject_invalid_opacity(opacity):
    sim = make_sim()
    scene = snapshot.SceneSnapshot(sim)
    viewer = Base.constructed[-1]
    with pytest.raises(ValueError):
        viewer.log_instances("/wok", "/prototype", sim._state_0.poses, None, None, None, opacities=[opacity])


def test_sink_signatures_accept_installed_newton_viewer_keywords():
    import inspect

    native = pytest.importorskip("newton.viewer").ViewerBase
    for name in ("log_mesh", "log_instances", "log_points", "log_lines", "log_scalar", "log_array"):
        expected = inspect.signature(getattr(native, name)).parameters
        actual = inspect.signature(getattr(snapshot._SnapshotSink, name)).parameters
        assert expected.keys() <= actual.keys(), name
