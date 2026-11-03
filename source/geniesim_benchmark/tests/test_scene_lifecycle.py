"""SDK contract tests without a live simulator or model service."""
import ast
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from pxr import Usd, UsdGeom

assets = os.environ.get("GENIESIM_ASSETS_PATH")
if assets:
    package = ModuleType("geniesim_assets")
    package.__path__ = [assets]
    sys.modules.setdefault("geniesim_assets", package)
from geniesim_assets.behavior_lifecycle import SceneOwner, retire_scope

path = Path(__file__).resolve().parents[1] / "src/geniesim_benchmark/scene_lifecycle.py"
spec = importlib.util.spec_from_file_location("scene_lifecycle_under_test", path)
lifecycle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lifecycle)


class Manager:
    def __init__(self, instances):
        self._prim_to_scripts = instances
        self.removed = []

    def _destroy_script(self, path, script):
        instance = self._prim_to_scripts[path].pop(script)
        if instance is not None:
            instance.on_destroy()
        self.removed.append((path, script))
        if not self._prim_to_scripts[path]:
            del self._prim_to_scripts[path]

    def module(self):
        return SimpleNamespace(ScriptManager=SimpleNamespace(get_instance=lambda: self))


class SceneLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.stage = Usd.Stage.CreateInMemory()

    def tearDown(self):
        retire_scope(self.stage, "/")

    def test_sdk_forgets_old_instance_while_original_prim_is_alive(self):
        prim = UsdGeom.Xform.Define(self.stage, "/Workspace/Owner").GetPrim()
        events = []
        owner = SceneOwner(prim, lambda: events.append(("retire", prim.IsValid())))
        instance = SimpleNamespace(on_destroy=lambda: events.append(("destroy", owner.closed)))
        manager = Manager({str(prim.GetPath()): {"asset.py": instance},
                           "/WorkspaceOther/Owner": {"other.py": None},
                           "/genie": {"parking.py": None}})
        with patch.dict(sys.modules, {"omni.kit.scripting": manager.module()}):
            lifecycle.retire_asset_scripts(self.stage, "/Workspace")
        self.assertEqual(events, [("retire", True), ("destroy", True)])
        self.assertNotIn(str(prim.GetPath()), manager._prim_to_scripts)
        self.assertIn("/WorkspaceOther/Owner", manager._prim_to_scripts)
        self.assertIn("/genie", manager._prim_to_scripts)

    def test_sdk_aliases_do_not_destroy_same_manager_twice(self):
        manager = Manager({"/World/Owner": {"asset.py": None}})
        with patch.dict(sys.modules, {"omni.kit.scripting": manager.module(),
                                     "omni.behavior.scripting.core": manager.module()}):
            lifecycle.retire_asset_scripts(self.stage, "/World")
        self.assertEqual(manager.removed, [("/World/Owner", "asset.py")])

    def test_unsupported_sdk_fails_before_retiring_live_objects(self):
        prim = UsdGeom.Xform.Define(self.stage, "/Workspace/Owner").GetPrim()
        owner = SceneOwner(prim, lambda: None)
        with patch.dict(sys.modules, {"omni.kit.scripting": SimpleNamespace()}):
            with self.assertRaisesRegex(RuntimeError, "Unsupported script manager"):
                lifecycle.retire_asset_scripts(self.stage, "/Workspace")
        self.assertTrue(owner.current)
        self.assertTrue(prim.IsValid())

    def test_failed_sdk_unassignment_prevents_scene_replacement(self):
        manager = Manager({"/Workspace/Owner": {"asset.py": None}})
        manager._destroy_script = lambda *args: None
        with patch.dict(sys.modules, {"omni.kit.scripting": manager.module()}):
            with self.assertRaisesRegex(RuntimeError, "instances remain"):
                lifecycle.retire_asset_scripts(self.stage, "/Workspace")

    def test_no_active_sdk_still_retires_standalone_asset_owners(self):
        prim = UsdGeom.Xform.Define(self.stage, "/Workspace/Owner").GetPrim()
        owner = SceneOwner(prim, lambda: None)
        lifecycle.retire_asset_scripts(self.stage, "/Workspace")
        self.assertTrue(owner.closed)

    def check_scene_switch_boundary(self, same_background):
        source = path.parent / "app/controllers/api_core.py"
        tree = ast.parse(source.read_text())
        core = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "APICore")
        fn = next(n for n in core.body if isinstance(n, ast.FunctionDef) and n.name == "_init_robot_cfg")
        ns = {"system_utils": SimpleNamespace(assets_path=lambda: "assets"),
              "retire_asset_scripts": lifecycle.retire_asset_scripts}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), ns)
        root = UsdGeom.Xform.Define(self.stage,"/World/Lock").GetPrim()
        robot = UsdGeom.Xform.Define(self.stage,"/old_robot").GetPrim()
        background_owner = SceneOwner(root,lambda:None)
        robot_owner = SceneOwner(robot,lambda:None)
        class BoundaryReached(Exception):
            pass
        def reload(*args):
            self.assertEqual(background_owner.current, same_background)
            self.assertEqual(robot_owner.current, same_background)
            raise BoundaryReached()
        api = SimpleNamespace(_stage=self.stage, _loaded_scene_key="old",
                              robot_cfg=SimpleNamespace(robot_prim_path="/old_robot"),
                              _robot_scene_key=lambda *a:"old" if same_background else "new",
                              _validate_floor_collision_request=lambda *a:None,
                              _clear_wheel_parking=lambda:None, _reload_scenes=reload)
        with self.assertRaises(BoundaryReached):
            ns["_init_robot_cfg"](api,"robot.json","scene.usda",[0,0,0],[1,0,0,0],"next.usda",False)

    def test_full_scene_retires_background_before_new_workspace_load(self):
        self.check_scene_switch_boundary(False)

    def test_workspace_only_switch_preserves_background_and_robot_owners(self):
        self.check_scene_switch_boundary(True)

    def test_embedded_keycap_lifecycle_matches_canonical_source(self):
        from pxr import Sdf
        asset_source = Path(assets) / "objects/benchmark/keyboard/benchmark_keyboard_002/Aligned.usd"
        layer = Sdf.Layer.FindOrOpen(str(asset_source))
        script = layer.GetPrimAtPath("/World/part_00/AutoDetachGraph/Release").attributes["inputs:script"].default
        canonical = ast.parse(script)
        def methods(tree):
            return {(c.name, f.name): ast.dump(f, include_attributes=False)
                    for c in tree.body if isinstance(c,ast.ClassDef)
                    for f in c.body if isinstance(f,ast.FunctionDef)
                    and c.name in {"SimulationResetState","KeycapAutoDetach"}
                    and f.name != "release_allowed"}
        expected = methods(canonical)
        root = path.parent / "benchmark/config/llm_task"
        from pxr import Sdf
        count = 0
        for name in ["keyboard_keycap_swap_agilex","keyboard_keycap_swap_arxone",
                     "keyboard_keycap_swap_g1","keyboard_keycap_swap_g2_omnipicker"]:
            for scene in sorted((root/name).glob("*/scene.usda")):
                layer = Sdf.Layer.FindOrOpen(str(scene))
                found = []
                def visit(prim):
                    for attr in prim.attributes.values():
                        if attr.name == "inputs:script" and isinstance(attr.default,str) and "class KeycapAutoDetach" in attr.default:
                            found.append(attr.default)
                    for child in prim.nameChildren.values():visit(child)
                for prim in layer.rootPrims:visit(prim)
                self.assertEqual(len(found),1,str(scene))
                self.assertEqual(methods(ast.parse(found[0])),expected,str(scene))
                self.assertIn("if str(self.cap.GetPath()) in self._installed_caps:", found[0])
                count += 1
        self.assertEqual(count,80)

    def test_inactive_stage_cannot_destroy_current_sdk_instances(self):
        other = Usd.Stage.CreateInMemory()
        manager = Manager({"/Workspace/Owner": {"asset.py": None}})
        usd = SimpleNamespace(get_context=lambda: SimpleNamespace(get_stage=lambda: other))
        with patch.dict(sys.modules, {"omni.kit.scripting": manager.module(), "omni.usd": usd}):
            with self.assertRaisesRegex(RuntimeError, "inactive USD stage"):
                lifecycle.retire_asset_scripts(self.stage,"/Workspace")
        self.assertFalse(manager.removed)

    def test_already_closed_stage_needs_no_second_sdk_cleanup(self):
        lifecycle.retire_asset_scripts(None,"/")


if __name__ == "__main__":
    unittest.main(verbosity=2)
