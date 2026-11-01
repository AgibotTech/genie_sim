# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Resolve built ROS overlays before exec, without importing ROS in the CLI."""

from pathlib import Path
import os
import subprocess
import sys


def _has_client(prefix: Path) -> bool:
    marker = Path("share/ament_index/resource_index/packages/genie_sim_client")
    return (prefix / marker).is_file() or (prefix / "genie_sim_client" / marker).is_file()


def _with_prepared_python(environment: dict[str, str]) -> dict[str, str]:
    for value in environment.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        if not value:
            continue
        prefix = Path(value)
        for root in (prefix, prefix.parent):
            virtualenv = root / "rt_python"
            config = virtualenv / "pyvenv.cfg"
            python = virtualenv / "bin/python"
            if not config.is_file() or not python.is_file():
                continue
            settings = dict(line.split(" = ", 1) for line in config.read_text().splitlines() if " = " in line)
            if not settings.get("version", "").startswith(f"{sys.version_info.major}.{sys.version_info.minor}."):
                continue
            return {
                **environment,
                "VIRTUAL_ENV": str(virtualenv),
                "PATH": str(virtualenv / "bin") + os.pathsep + environment.get("PATH", ""),
            }
    return environment


def runtime_environment(config_path: Path) -> dict[str, str]:
    """Keep an active overlay, otherwise source this checkout's standard build."""
    environment = dict(os.environ)
    prefixes = environment.get("AMENT_PREFIX_PATH", "").split(os.pathsep)
    if any(_has_client(Path(value)) for value in prefixes if value):
        return _with_prepared_python(environment)

    from geniesim_cli.commands.ros import _discover_opt_ros_distro

    distro = _discover_opt_ros_distro()
    setup_files = [Path("/opt/ros") / distro / "setup.bash"] if distro else []
    for parent in config_path.resolve().parents:
        if not (parent / "source/geniesim_ros/src/ros_ws/src/genie_sim_bringup").is_dir():
            continue
        for name in ("install", "devel"):
            prefix = parent / name
            if _has_client(prefix) and (prefix / "setup.bash").is_file():
                setup_files.append(prefix / "setup.bash")
                break
        break
    if not setup_files:
        raise RuntimeError("No ROS runtime found. Prepare the runtime with geniesim ros build dev, then retry.")
    # Paths are positional arguments, never interpolated shell code. Setup output
    # goes to stderr so it cannot corrupt the NUL-separated environment records.
    result = subprocess.run(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            'set -e; for setup in "$@"; do source "$setup" >&2; done; env -0',
            "benchmark-runtime",
            *map(str, setup_files),
        ],
        env=environment,
        capture_output=True,
        check=True,
    )
    sourced = dict(record.decode().split("=", 1) for record in result.stdout.split(b"\0") if record)
    if not any(_has_client(Path(value)) for value in sourced.get("AMENT_PREFIX_PATH", "").split(os.pathsep) if value):
        raise RuntimeError(
            "ROS benchmark runtime is not built. Run geniesim ros build dev from the checkout root, then retry."
        )
    return _with_prepared_python(sourced)
