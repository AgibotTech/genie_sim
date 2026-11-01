# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Prepare independent module builds with the checkout's shared mirror resources."""

from __future__ import annotations

from pathlib import Path

from geniesim_cli import _mirrors


def prepare_command(module_root: Path, recipe: str, image: str, args: list[str]) -> tuple[list[str], str | None]:
    """Consume mirror options, preserving parent defaults and explicit Docker arguments."""
    preset = None
    forwarded = []
    tokens = iter(args)
    for token in tokens:
        if token == "--build-arg":
            argument = next(tokens, "")
            if not argument or argument.startswith("--"):
                raise ValueError("--build-arg needs NAME or NAME=VALUE")
            forwarded.extend([token, argument])
        elif token == "--china" or token == "--mirrors" or token.startswith("--mirrors="):
            selected = (
                "china"
                if token == "--china"
                else (next(tokens, "") if token == "--mirrors" else token.partition("=")[2])
            )
            if not selected or selected.startswith("--") or (preset is not None and preset != selected):
                raise ValueError("Select one mirror preset with --mirrors NAME or --china")
            preset = selected
        else:
            forwarded.append(token)

    shared = next(
        (
            parent / "docker"
            for parent in module_root.resolve().parents
            if (parent / "docker/configure_mirrors.py").is_file() and (parent / "docker/mirrors.toml").is_file()
        ),
        None,
    )
    if shared is None:
        raise ValueError("Build from a checkout containing docker/configure_mirrors.py and docker/mirrors.toml")
    supported = _mirrors.declared_args(module_root / recipe)
    settings = _mirrors.load_preset(shared / "mirrors.toml", preset) if preset else {}
    defaults = {name: value for name, value in settings.items() if name in supported}
    values, forwarded, _ = _mirrors.merge_build_args(defaults, forwarded)
    command = [
        "docker",
        "build",
        "-f",
        recipe,
        "-t",
        image,
        "--build-context",
        f"geniesim_mirrors={shared}",
        *_mirrors.as_flags(values),
        *forwarded,
        ".",
    ]
    return command, preset
