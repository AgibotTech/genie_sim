#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Build the scene-agent image with shared mirror presets from a source checkout."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

# This build-only tool uses checkout sources; scene authoring has no CLI dependency.
MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT.parent / "geniesim_cli/src"))

from geniesim_cli import _module_image
from geniesim_cli._style import BOLD, CYAN, DIM, MAGENTA, RED, RST, YELLOW


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Other arguments are forwarded to docker build. Explicit --build-arg values override presets.",
    )
    parser.add_argument("--image", help="Output image tag (default: latest, or vl for the VL target)")
    parser.add_argument("--target", choices=["runtime", "vl", "mirrors"], default="runtime")
    parser.add_argument("--dry-run", action="store_true", help="Print the build command without invoking Docker")
    mirrors = parser.add_mutually_exclusive_group()
    mirrors.add_argument("--china", action="store_true", help="Alias for --mirrors china")
    mirrors.add_argument(
        "--mirrors", metavar="NAME", help="Preset from docker/mirrors.toml; omitted inherits the parent"
    )
    options, extra = parser.parse_known_args(argv)
    image = (
        options.image
        or f"registry.agibot.com/genie-sim/geniesim3-scene-agent:{'vl' if options.target == 'vl' else 'latest'}"
    )
    if options.china:
        extra = ["--china", *extra]
    elif options.mirrors is not None:
        extra = ["--mirrors", options.mirrors, *extra]
    try:
        command, preset = _module_image.prepare_command(
            MODULE_ROOT, "docker/Dockerfile.6.2", image,
            ["--build-arg", f"SCENE_AGENT_VERSION={(MODULE_ROOT / 'VERSION').read_text().strip()}",
             "--target", options.target, *extra],
        )
    except (OSError, ValueError) as exc:
        print(f"{RED}❌ {exc}{RST}", file=sys.stderr)
        return 2
    print(f"{BOLD}{MAGENTA}🔨 Build scene-agent image{RST}")
    print(f"   {DIM}Mirrors:{RST} {CYAN}{preset or 'inherit parent image (official base: upstream)'}{RST}")
    print(f"   {YELLOW}$ (cd {shlex.quote(str(MODULE_ROOT))} && {shlex.join(command)}){RST}")
    if options.dry_run:
        return 0
    try:
        return subprocess.run(command, cwd=str(MODULE_ROOT), env={**os.environ, "DOCKER_BUILDKIT": "1"}).returncode
    except FileNotFoundError:
        print(f"{RED}❌ Docker is required to build this image.{RST}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
