#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Align the container's ROS build SDKs with the checked-out package metadata."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import sysconfig

try:
    import tomllib
except ImportError:
    import tomli as tomllib
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement

PYPROJECT = Path(__file__).resolve().parents[1] / "source/geniesim_ros/pyproject.toml"


def unsatisfied_requirements() -> list[str]:
    """Use the build contract directly; do not duplicate SDK pins in the installer."""
    project = tomllib.loads(PYPROJECT.read_text())
    missing = []
    for text in project["build-system"]["requires"]:
        requirement = Requirement(text)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        try:
            installed = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            installed = None
        if installed is None or not requirement.specifier.contains(installed):
            missing.append(text)
    return missing


def ensure_dependencies(*, check_only: bool) -> None:
    missing = unsatisfied_requirements()
    if not missing:
        return
    if check_only:
        raise RuntimeError("Unsatisfied ROS build dependencies: " + ", ".join(missing))
    if os.geteuid() != 0 or sys.prefix != sys.base_prefix:
        raise RuntimeError("Run the installer as root with the container's system Python 3.12")
    # Refresh only the declared SDKs. Do not let a full project dependency resolve
    # replace the already validated shared Newton physics baseline.
    for text in missing:
        requirement = Requirement(text)
        options = []
        old_metadata = []
        if requirement.name == "ovrtx":
            # Main mounts a persistent volume at ovrtx/bin/cache. pip's uninstall
            # removes the whole package directory and fails with EBUSY there.
            # Overlay the new wheel, then retire only superseded dist-info so
            # importlib.metadata cannot keep reporting the previous version.
            options.append("--ignore-installed")
            for path in Path(sysconfig.get_path("purelib")).glob("*.dist-info"):
                distribution = metadata.PathDistribution(path)
                if distribution.metadata.get("Name") == "ovrtx" and not requirement.specifier.contains(
                    distribution.version
                ):
                    old_metadata.append(path)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--break-system-packages",
                "--root-user-action=ignore",
                "--no-deps",
                *options,
                text,
            ],
            check=True,
        )
        for path in old_metadata:
            shutil.rmtree(path)
    missing = unsatisfied_requirements()
    if missing:
        raise RuntimeError("ROS build dependencies remain unsatisfied: " + ", ".join(missing))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check versions without installing")
    args = parser.parse_args()
    try:
        ensure_dependencies(check_only=args.check)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"[ros-build-deps] {error}", file=sys.stderr)
        return 1
    print("[ros-build-deps] Declared ROS build dependencies verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
