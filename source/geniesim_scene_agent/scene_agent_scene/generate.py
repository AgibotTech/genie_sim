# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Generate reviewed scenes and deterministic variant batches."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from .builder import SceneBuilder, validate_scene_id
from .dsl import build_from_dsl, load_python_file
from .validate import validate_scene


class SceneRejected(ValueError):
    """A sampled variant violates a scene-specific constraint and may be resampled."""


def sample_parameters(spec: dict, seed: int, instance_id: int, attempt: int = 0) -> dict:
    dimensions = spec.get("dimensions", {})
    rng = random.Random(seed + instance_id * 100003 + attempt)
    result = {}
    for name, rule in dimensions.items():
        kind = rule.get("type")
        if instance_id == 0 and "base" in rule:
            value = rule["base"]
        elif kind == "uniform":
            value = rng.uniform(float(rule["min"]), float(rule["max"]))
        elif kind == "choice":
            value = rng.choice(rule["values"])
        elif kind == "int":
            value = rng.randint(int(rule["min"]), int(rule["max"]))
        else:
            raise ValueError(f"Unsupported dimension {name}: {kind}")
        result[name] = value
    return result


def _generate_one(
    mode: str,
    recipe_path: Path,
    folder: Path,
    scene_id: str,
    assets_root: Path | None,
    seed: int,
    params: dict,
    patch_path: Path | None,
) -> None:
    builder = SceneBuilder(folder, scene_id, seed)
    recipe = None
    rng = random.Random(seed)
    if mode == "dsl":
        if assets_root is None:
            raise ValueError("assets_root is required for DSL scenes")
        build_from_dsl(builder, recipe_path, assets_root, seed, params)
    elif mode == "direct":
        recipe = load_python_file(recipe_path, f"scene_direct_{seed}")
        if not hasattr(recipe, "build_scene"):
            raise ValueError("Direct recipe must define build_scene(builder, rng, params)")
        recipe.build_scene(builder, rng, params)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    if patch_path is not None:
        patch = load_python_file(patch_path, f"scene_patch_{seed}")
        if not hasattr(patch, "patch_scene"):
            raise ValueError("Patch script must define patch_scene(builder, rng, params)")
        patch.patch_scene(builder, rng, params)
    if recipe is not None and hasattr(recipe, "check_scene"):
        recipe.check_scene(builder, params)
    builder.save()


def generate_instances(
    mode: str,
    recipe: str | Path,
    output_root: str | Path,
    scene_id: str,
    *,
    count: int = 1,
    seed: int = 0,
    generalization: dict | None = None,
    patch: str | Path | None = None,
    assets_root: str | Path | None = None,
    render: bool = True,
    isaac_python: str | Path | None = None,
) -> dict:
    """Create numeric scene instances; batch output is committed per valid instance."""
    validate_scene_id(scene_id)
    if count < 1:
        raise ValueError("count must be positive")
    if count > 1 and generalization is None:
        raise ValueError("A reviewed generalization spec is required when count > 1")
    recipe_path = Path(recipe).resolve()
    if not recipe_path.is_file():
        raise FileNotFoundError(recipe_path)
    patch_path = Path(patch).resolve() if patch else None
    if patch_path and not patch_path.is_file():
        raise FileNotFoundError(patch_path)
    from scene_agent_assets.location import resolve_assets_root

    assets_path = resolve_assets_root(assets_root) if mode == "dsl" or assets_root else None
    scene_root = Path(output_root).resolve() / scene_id
    scene_root.mkdir(parents=True, exist_ok=True)
    spec = generalization or {"dimensions": {}}
    spec_path = scene_root / "generalization.json"
    if not spec_path.exists():
        spec_path.write_text(json.dumps({"count": count, "seed": seed, **spec}, indent=2), encoding="utf-8")
    elif json.loads(spec_path.read_text(encoding="utf-8")) != {"count": count, "seed": seed, **spec}:
        raise ValueError("Existing generalization.json differs; use a new output root")
    pending: list[tuple[int, Path, Path]] = []
    rejections: list[dict] = []
    max_attempts = int(spec.get("max_attempts", 5))
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    for index in range(count):
        folder = scene_root / str(index)
        if folder.exists():
            raise FileExistsError(folder)
        staging = scene_root / f".staging_{index}"
        accepted = False
        for attempt in range(1 if index == 0 else max_attempts):
            if staging.exists():
                shutil.rmtree(staging)
            params = sample_parameters(spec, seed, index, attempt)
            instance_seed = seed + index * 100003 + attempt
            try:
                _generate_one(mode, recipe_path, staging, scene_id, assets_path, instance_seed, params, patch_path)
            except SceneRejected as exc:
                rejections.append({"instance_id": index, "attempt": attempt, "reason": str(exc)})
                continue
            accepted = True
            break
        if not accepted:
            raise ValueError(f"Instance {index} exhausted {max_attempts} attempts: {rejections}")
        (staging / "renders").mkdir()
        static_report = validate_scene(staging)
        if not static_report["ok"]:
            (staging / "validation.json").write_text(json.dumps(static_report, indent=2), encoding="utf-8")
            raise ValueError(f"Scene {index} failed USD validation: {static_report['errors']}")
        (staging / "generation.json").write_text(
            json.dumps(
                {
                    "scene_id": scene_id,
                    "instance_id": index,
                    "mode": "hybrid" if patch_path else mode,
                    "seed": instance_seed,
                    "attempt": attempt,
                    "parameters": params,
                    "recipe": str(recipe_path),
                    "patch": str(patch_path) if patch_path else None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        pending.append((index, staging, folder))
    if pending:
        first_info = json.loads((pending[0][1] / "scene_info.json").read_text(encoding="utf-8"))
        expected_roles = set(first_info["layout"])
        expected_background = set(first_info.get("background", {}))
        for index, staging, _ in pending[1:]:
            info = json.loads((staging / "scene_info.json").read_text(encoding="utf-8"))
            if set(info["layout"]) != expected_roles or set(info.get("background", {})) != expected_background:
                raise ValueError(f"Object roles differ in instance {index}; see {staging}")
    if render:
        from .render import render_scenes

        render_scenes(
            [(staging / "scene.usda", staging / "renders") for _, staging, _ in pending], isaac_python=isaac_python
        )
    for index, staging, _ in pending:
        report = validate_scene(staging, require_renders=render)
        (staging / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not report["ok"]:
            raise ValueError(f"Scene {index} failed final validation: {report['errors']}")
    results = []
    for index, staging, folder in pending:
        staging.rename(folder)
        results.append({"instance_id": index, "path": str(folder), "ok": True})
    summary = {"scene_id": scene_id, "count": len(results), "instances": results, "rejections": rejections}
    (scene_root / "batch_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
