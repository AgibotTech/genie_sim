# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Expose the container's bundled libraries and preserve their installed versions."""

from __future__ import annotations

import argparse
from importlib import metadata
from pathlib import Path
import re
import sys
import sysconfig


def discover_sdk_paths(root: Path) -> dict[str, Path]:
    """Resolve the SDK that supplies pre-Application planner and ROS imports."""
    result = {
        "torch": root / "extsDeprecated/omni.isaac.ml_archive/pip_prebundle",
        "nvidia": root / "exts/isaacsim.pip.nv/pip_prebundle",
    }
    for name, pattern in (
        ("warp", "extscache/omni.warp.core-*"),
        ("usd", "extscache/omni.usd.libs-*"),
        ("validation", "extscache/omni.asset_validator.core-*/pip_prebundle"),
    ):
        matches = sorted(path for path in root.glob(pattern) if path.is_dir())
        if len(matches) != 1:
            raise RuntimeError(f"Expected one bundled {name} extension, found {matches}")
        result[name] = matches[0]
    result["ros"] = root / "exts/isaacsim.ros2.core/jazzy/rclpy"
    # Extension prebundle directories are not processed as site directories, so
    # their cmeel.pth does not expose Pinocchio before SimulationApp starts.
    result["pinocchio"] = (
        root / "exts/isaacsim.robot_motion.pink/pip_prebundle/cmeel.prefix/lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    )
    for name, path in result.items():
        if not path.is_dir():
            raise RuntimeError(f"Missing bundled {name} directory: {path}")
    return result


def configure_imports(root: Path, links: Path, site: Path) -> dict[str, Path]:
    """Make SDK packages importable before SimulationApp without copying them."""
    paths = discover_sdk_paths(root)
    links.mkdir(parents=True, exist_ok=True)
    site.mkdir(parents=True, exist_ok=True)
    for name, target in paths.items():
        link = links / name
        if link.is_symlink() and link.resolve() == target.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise RuntimeError(f"Refusing to replace an existing SDK path: {link}")
        link.symlink_to(target, target_is_directory=True)
    # cuda.pathfinder searches site.getsitepackages(), not .pth additions, for
    # wheel headers. Expose the SDK component there without replacing its code,
    # libraries or distribution metadata with a second CUDA installation.
    nvrtc = paths["nvidia"] / "nvidia/cuda_nvrtc"
    if not (nvrtc / "include/nvrtc.h").is_file():
        raise RuntimeError(f"Missing bundled NVRTC headers: {nvrtc}")
    native_nvrtc = site / "nvidia/cuda_nvrtc"
    if native_nvrtc.is_symlink() and native_nvrtc.resolve() == nvrtc.resolve():
        pass
    elif native_nvrtc.exists() or native_nvrtc.is_symlink():
        raise RuntimeError(f"Refusing to replace an existing NVRTC provider: {native_nvrtc}")
    else:
        native_nvrtc.parent.mkdir(parents=True, exist_ok=True)
        native_nvrtc.symlink_to(nvrtc, target_is_directory=True)
    (site / "geniesim_bundled_sdk.pth").write_text(
        "".join(f"{links / name}\n" for name in paths), encoding="utf-8"
    )
    return paths


def freeze_sdk_constraints(distributions, warp_version: str) -> list[str]:
    """Use Python's first visible distribution when bundles contain duplicates."""
    versions = {}
    for dist in distributions:
        name = re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
        versions.setdefault(name, dist.version)
    # The extension provides Warp code but no wheel metadata. Its matching wheel
    # supplies metadata required by cuRobo without upgrading the SDK's kernels.
    versions["warp-lang"] = warp_version
    # The SDK bundles coverage 7.4.4 as test tooling. Numba's optional coverage
    # integration imports Tracer, introduced in 7.6.1, even during normal use.
    # Preserve numerical/native libraries while allowing this one tooling fix.
    has_coverage = versions.pop("coverage", None) is not None
    constraints = [f"{name}=={version}" for name, version in sorted(versions.items())]
    if has_coverage:
        constraints.append("coverage>=7.6.1")
    return constraints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", type=Path, default=Path("/isaac-sim"))
    parser.add_argument("--links", type=Path, default=Path("/opt/isaac-sdk"))
    parser.add_argument("--constraints", type=Path, required=True)
    args = parser.parse_args()
    paths = configure_imports(args.sdk_root, args.links, Path(sysconfig.get_path("purelib")))
    # This process predates the new .pth file. Match the next interpreter's paths.
    sys.path.extend(str(path) for path in paths.values())
    import warp

    constraints = freeze_sdk_constraints(metadata.distributions(), warp.__version__)
    args.constraints.write_text("\n".join(constraints) + "\n", encoding="utf-8")
    print(f"Preserved {len(constraints)} bundled SDK package versions")


if __name__ == "__main__":
    main()
