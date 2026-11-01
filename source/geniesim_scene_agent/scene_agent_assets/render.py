# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Render asset candidates with the scene agent's bundled Isaac Sim renderer."""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, ImageDraw


def prepare_render_stage(asset_usd: str | Path, cache_dir: str | Path) -> Path:
    from pxr import Sdf, Usd, UsdGeom

    source = Path(asset_usd).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    wrapper = Path(cache_dir) / source.parent.name / "Aligned.usda"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    if wrapper.exists():
        wrapper.unlink()
    stage = Usd.Stage.CreateNew(str(wrapper))
    asset = UsdGeom.Xform.Define(stage, Sdf.Path("/Asset"))
    asset.GetPrim().GetPayloads().AddPayload(str(source))
    stage.SetDefaultPrim(asset.GetPrim())
    stage.GetRootLayer().Save()
    return wrapper


def _contact_sheet(images: list[tuple[str, Path]], target: Path, columns: int = 4):
    if not images:
        return None
    width, height = 240, 270
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (asset_id, image_path) in enumerate(images):
        with Image.open(image_path) as source:
            tile = source.convert("RGB")
            tile.thumbnail((width - 12, height - 35))
            x = (index % columns) * width
            y = (index // columns) * height
            sheet.paste(tile, (x + (width - tile.width) // 2, y))
            draw.text((x + 6, y + height - 28), asset_id[:33], fill="black")
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target)
    return target


def render_records(records: list[dict], output_dir: str | Path, python_executable: str,
                   force: bool = False) -> dict:
    from scene_agent_scene.builder import SceneBuilder
    from scene_agent_scene.render import render_scenes

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    images = []
    failures = []
    pending = []
    for record in records:
        asset_id = record["asset_id"]
        existing = [Path(path) for path in record.get("snapshots", []) if Path(path).is_file()]
        local = output / asset_id / "front.png"
        if not force and local.is_file():
            image_path = local
        elif not force and existing:
            image_path = existing[0]
        else:
            try:
                scene_dir = output / "_stages" / asset_id
                if scene_dir.exists():
                    shutil.rmtree(scene_dir)
                builder = SceneBuilder(scene_dir, "asset_preview", seed=0)
                builder.add_asset("asset", record["usd"])
                builder.save()
                pending.append((asset_id, scene_dir / "scene.usda", local.parent))
                continue
            except (OSError, ValueError, RuntimeError) as exc:
                failures.append({"asset_id": asset_id, "error": str(exc)})
                continue
        images.append((asset_id, image_path))
    if pending:
        render_scenes([(scene, target) for _, scene, target in pending], isaac_python=python_executable)
        for asset_id, _, target in pending:
            images.append((asset_id, target / "front.png"))
    sheet = _contact_sheet(images, output / "contact_sheet.png")
    return {"contact_sheet": str(sheet) if sheet else None,
            "shown": len(images), "rendered": len(pending), "failed": failures,
            "images": [{"asset_id": asset_id, "path": str(path)} for asset_id, path in images]}
