# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Launch one Isaac Sim process to render one or many final scene stages."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


def render_scenes(items: list[tuple[str | Path, str | Path]], *, isaac_python: str | Path | None = None,
                  timeout: int | None = None) -> None:
    if not items:
        return
    executable = isaac_python or os.environ.get("ISAACSIM_PYTHON")
    if not executable:
        raise ValueError("Set ISAACSIM_PYTHON or pass --isaac-python pointing to Isaac Sim's python.sh")
    executable = str(executable)
    manifest = [{"scene": str(Path(scene).resolve()), "output": str(Path(output).resolve())}
                for scene, output in items]
    for item in manifest:
        if not Path(item["scene"]).is_file():
            raise FileNotFoundError(item["scene"])
        Path(item["output"]).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="scene-render-") as temporary:
        manifest_path = Path(temporary) / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        command = [executable, str(Path(__file__).with_name("render_isaac.py")), "--manifest", str(manifest_path)]
        result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(f"Isaac Sim render failed ({result.returncode}):\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")
    for item in manifest:
        for view in ("front", "side", "top"):
            image = Path(item["output"]) / f"{view}.png"
            if not image.is_file():
                raise RuntimeError(f"Isaac Sim did not produce {image}")


def render_scene(scene: str | Path, output: str | Path, *, isaac_python: str | Path | None = None) -> None:
    render_scenes([(scene, output)], isaac_python=isaac_python)
