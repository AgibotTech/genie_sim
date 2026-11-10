# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Isaac Sim batch render worker; cameras and lights exist only in memory."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
import sys
from pathlib import Path


def prepare_launch_config(config: dict) -> dict:
    """Use legacy tick scheduling for explicit captures on Isaac Sim 6.2."""
    result = dict(config)
    try:
        root = os.environ.get("ISAAC_PATH")
        if root and Path(sys.executable).resolve().is_relative_to(Path(root).resolve() / "kit/python"):
            # Bundled Python can inherit wheel metadata from another SDK.
            installed_version = (Path(root) / "VERSION").read_text().strip()
        else:
            installed_version = metadata.version("isaacsim")
        version = tuple(int(part) for part in installed_version.split(".")[:2])
    except (metadata.PackageNotFoundError, OSError, ValueError):
        return result
    if version == (6, 2):
        # The 6.2 multi-tick renderer stalls explicit captures after replacing a scene.
        result["extra_args"] = [*result.get("extra_args", []), "--/rtx/hydra/supportMultiTickRate=false"]
    return result


def camera_matrix(eye, target):
    import numpy as np
    from pxr import Gf

    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    matrix = Gf.Matrix4d(1.0)
    matrix.SetTranslateOnly(Gf.Vec3d(*eye))
    matrix.SetRow3(0, Gf.Vec3d(*right))
    matrix.SetRow3(1, Gf.Vec3d(*camera_up))
    matrix.SetRow3(2, Gf.Vec3d(*-forward))
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

    from isaacsim import SimulationApp
    app = SimulationApp(prepare_launch_config({"headless": True, "width": args.width, "height": args.height}))
    try:
        import numpy as np
        from PIL import Image
        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
        from isaacsim.core.utils.prims import delete_prim, get_prim_at_path
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux
        import omni.replicator.core as rep

        for _ in range(3):
            app.update()
        stage = get_current_stage()
        for item in manifest:
            if get_prim_at_path("/World").IsValid():
                delete_prim("/World")
            if get_prim_at_path("/SceneRender").IsValid():
                delete_prim("/SceneRender")
            add_reference_to_stage(usd_path=item["scene"], prim_path="/World")
            stage.Load(Sdf.Path("/World"), Usd.LoadWithDescendants)
            for _ in range(8):
                app.update()
            objects = stage.GetPrimAtPath("/World/Objects")
            if not objects:
                objects = stage.GetPrimAtPath("/World")
            if not objects:
                raise RuntimeError(f"No /World in {item['scene']}")
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
            box = cache.ComputeWorldBound(objects).ComputeAlignedRange()
            if box.IsEmpty():
                raise RuntimeError(f"Empty bounds in {item['scene']}")
            minimum = np.asarray(box.GetMin(), dtype=float)
            maximum = np.asarray(box.GetMax(), dtype=float)
            center = (minimum + maximum) / 2
            radius = max(float(np.linalg.norm(maximum - minimum)), 0.08)

            if not get_prim_at_path("/SceneRender/DomeLight").IsValid():
                dome = UsdLux.DomeLight.Define(stage, "/SceneRender/DomeLight")
                dome.CreateIntensityAttr().Set(1200)
                dome.CreateColorTemperatureAttr().Set(6500)
                dome.CreateEnableColorTemperatureAttr().Set(True)
            views = {
                "front": center + np.array([radius * 0.8, -radius * 1.3, radius * 0.6]),
                "side": center + np.array([-radius * 1.5, 0, radius * 0.4]),
                "top": center + np.array([0, -0.01, radius * 2.0]),
            }
            output = Path(item["output"])
            output.mkdir(parents=True, exist_ok=True)
            annotators = {}
            products = []
            try:
                for view, eye in views.items():
                    path = f"/SceneRender/{view}_camera"
                    camera = UsdGeom.Camera.Define(stage, path)
                    camera.CreateFocalLengthAttr().Set(35.0)
                    camera.CreateHorizontalApertureAttr().Set(36.0)
                    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.001, 10000.0))
                    xform = UsdGeom.Xformable(camera.GetPrim())
                    xform.ClearXformOpOrder()
                    xform.AddTransformOp().Set(camera_matrix(eye, center))
                    product = rep.create.render_product(path, (args.width, args.height))
                    products.append(product)
                    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                    annotator.attach([product])
                    annotators[view] = (annotator, product)
                for _ in range(3):
                    rep.orchestrator.step(delta_time=0.0)
                for view, (annotator, _) in annotators.items():
                    pixels = annotator.get_data()
                    if pixels is None or len(pixels) == 0:
                        raise RuntimeError(f"No RGB frame for {item['scene']} {view}")
                    Image.fromarray(np.asarray(pixels)[:, :, :3].astype("uint8")).save(output / f"{view}.png")
            finally:
                # Detach before removing products or replacing their scene/camera prims.
                try:
                    for annotator, product in annotators.values():
                        annotator.detach([product])
                finally:
                    for product in products:
                        product.destroy()
                app.update()
            print(f"Rendered {item['scene']} -> {output}", flush=True)
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
