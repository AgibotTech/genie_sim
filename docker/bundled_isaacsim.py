#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Register the existing 6.2 SDK with system Python, without copying SDK files.

The local wheel owns only the lazy import adapter and its metadata. NVIDIA's
SDK remains under /isaac-sim. Generic Python/ROS/Newton processes do not activate
Kit paths; importing isaacsim activates the SDK's own environment setup.
"""

from __future__ import annotations

import argparse
import base64
import csv
import ctypes
import hashlib
from email.parser import Parser
from importlib import metadata
from importlib.machinery import PathFinder
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED


def sdk_environment(root: Path) -> dict[str, str]:
    """Read NVIDIA's generated paths; retain the caller's ROS environment."""
    result = subprocess.run(
        ["bash", "-c", 'source "$1/setup_python_env.sh" && env -0', "bundled-isaac", str(root)],
        check=True,
        capture_output=True,
    )
    environment = dict(item.split("=", 1) for item in result.stdout.decode().split("\0") if "=" in item)
    return {name: environment[name] for name in ("PYTHONPATH", "LD_LIBRARY_PATH")}


class BundledIsaacFinder:
    """Activate only when the application asks for the Isaac package."""

    def __init__(self, root: Path):
        self.root = root

    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"torch", "torchgen", "functorch", "torchvision", "torchaudio"}:
            return PathFinder.find_spec(
                fullname, [str(self.root / "extsDeprecated/omni.isaac.ml_archive/pip_prebundle")]
            )
        if fullname != "isaacsim":
            return None
        if str(self.root / "python_packages") in sys.path and os.environ.get("CARB_APP_PATH") == str(self.root / "kit"):
            return None
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError("The bundled Isaac 6.2 SDK requires CPython 3.12")
        environment = sdk_environment(self.root)
        os.environ.update(environment)
        os.environ.update(
            ISAAC_PATH=str(self.root), CARB_APP_PATH=str(self.root / "kit"), EXP_PATH=str(self.root / "apps")
        )
        sdk_paths = [
            entry
            for entry in environment["PYTHONPATH"].split(os.pathsep)
            if entry and Path(entry).is_relative_to(self.root)
        ]
        sys.path[:] = sdk_paths + [entry for entry in sys.path if entry not in sdk_paths]
        ctypes.CDLL(str(self.root / "kit/libcarb.so"), mode=ctypes.RTLD_GLOBAL)
        spec = PathFinder.find_spec(fullname, [str(self.root / "python_packages")])
        if spec is None:
            raise ImportError(f"Bundled Isaac SDK is missing at {self.root}")
        return spec


def register() -> None:
    """Install a cheap finder; do not import Isaac or change native paths yet."""
    config = Path(__file__).with_suffix(".json")
    root = Path(json.loads(config.read_text())["root"])
    # Torch discovers its CUDA shared libraries by searching sys.path. Append
    # only the SDK's CUDA archive; do not expose Kit or the whole ML archive.
    cuda_path = str(root / "exts/isaacsim.pip.nv/pip_prebundle")
    if cuda_path not in sys.path:
        sys.path.append(cuda_path)
    if not any(isinstance(finder, BundledIsaacFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, BundledIsaacFinder(root))


def dependency_inventory(root: Path) -> tuple[list[str], list[str]]:
    """Pin SDK-selected external packages, including standalone Newton's sim extra.

    Follow the generated startup path order rather than scanning unrelated,
    version-conflicting extension archives. Newton, the converter and Torch
    archives additionally serve the standalone workflows outside Kit.
    """
    startup_paths = [Path(value) for value in sdk_environment(root)["PYTHONPATH"].split(os.pathsep) if value]
    paths = [
        root / leaf
        for leaf in (
            "exts/omni.isaac.core_archive/pip_prebundle",
            "exts/omni.pip.compute/pip_prebundle",
            "exts/isaacsim.pip.newton/pip_prebundle",
            "exts/isaacsim.asset.importer.urdf/urdf_prebundle",
            "extsDeprecated/omni.isaac.ml_archive/pip_prebundle",
        )
    ]
    numpy = next(
        (
            dist
            for path in startup_paths
            if path.is_relative_to(root)
            for dist in metadata.distributions(path=[str(path)])
            if dist.metadata["Name"].lower() == "numpy"
        ),
        None,
    )
    if numpy is None:
        raise RuntimeError("The bundled SDK has no NumPy dependency metadata")
    selected = {"numpy": numpy.version}
    dependency_paths = []
    for path in paths:
        if not path.is_relative_to(root) or not path.is_dir():
            continue
        distributions = list(metadata.distributions(path=[str(path)]))
        if distributions:
            dependency_paths.append(str(path))
        for dist in distributions:
            name = re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
            if name not in {"pip", "setuptools", "wheel"}:
                selected.setdefault(name, dist.version)
    required = {"numpy", "scipy", "newton", "mujoco", "mujoco-warp", "torch", "urdf-usd-converter"}
    if not required <= selected.keys():
        raise RuntimeError(f"Incomplete bundled dependency inventory: {sorted(required - selected.keys())}")
    requirements = [
        f"{name}{'[sim]' if name == 'newton' else '[epath]' if name == 'etils' else ''}=={version}"
        for name, version in sorted(selected.items())
    ]
    return requirements, dependency_paths


def build_adapter(root: Path, version: str, output: Path) -> Path:
    """Build a pip-managed local wheel describing the existing SDK provider."""
    root = root.resolve()
    sdk_version = (root / "VERSION").read_text().strip()
    release = re.match(r"(\d+\.\d+\.\d+)", sdk_version)
    if not release or not sdk_version.startswith("6.2.") or version.split(".")[:3] != release[1].split("."):
        raise RuntimeError(f"Unsupported SDK/version pair: {sdk_version!r}, {version!r}")
    if not re.fullmatch(r"\d+(?:\.\d+){3}", version):
        raise RuntimeError(f"Invalid SDK compatibility version: {version!r}")
    for leaf in ("python_packages/isaacsim/__init__.py", "kit/libcarb.so", "setup_python_env.sh"):
        if not (root / leaf).is_file():
            raise RuntimeError(f"Incomplete bundled SDK: {root / leaf}")
    bundled_abi = subprocess.check_output(
        [str(root / "kit/python/bin/python3"), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"], text=True
    ).strip()
    if bundled_abi != "3.12" or sys.version_info[:2] != (3, 12):
        raise RuntimeError("Both the bundled SDK and system Python must use CPython 3.12")
    requirements, paths = dependency_inventory(root)
    manifest = json.dumps({"root": str(root), "sdk_version": sdk_version, "dependency_paths": paths}, indent=2) + "\n"
    info = f"isaacsim-{version}.dist-info"
    files = {
        "_geniesim_bundled_isaacsim.py": Path(__file__).read_text(),
        "_geniesim_bundled_isaacsim.json": manifest,
        "geniesim_bundled_isaacsim.pth": "import _geniesim_bundled_isaacsim; _geniesim_bundled_isaacsim.register()\n",
        f"{info}/METADATA": (
            f"Metadata-Version: 2.1\nName: isaacsim\nVersion: {version}\n"
            "Summary: Genie Sim system-Python adapter for the bundled NVIDIA Isaac Sim SDK\n"
            "Requires-Python: ==3.12.*\nProvides-Extra: all\nProvides-Extra: extscache\n"
            + "".join(f"Requires-Dist: {requirement}\n" for requirement in requirements)
        ),
        f"{info}/WHEEL": "Wheel-Version: 1.0\nGenerator: geniesim-bundled-sdk\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        f"{info}/geniesim-bundled.json": manifest,
    }
    output.mkdir(parents=True, exist_ok=True)
    wheel = output / f"isaacsim-{version}-py3-none-any.whl"
    write_wheel(wheel, info, files)
    return wheel


def write_wheel(wheel: Path, info: str, files: dict[str, str | bytes]) -> None:
    """Record only files this local metadata/adapter wheel actually owns."""
    record = io.StringIO()
    writer = csv.writer(record)
    for name, data in files.items():
        payload = data.encode() if isinstance(data, str) else data
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
        writer.writerow((name, f"sha256={digest}", len(payload)))
    writer.writerow((f"{info}/RECORD", "", ""))
    files[f"{info}/RECORD"] = record.getvalue()
    with ZipFile(wheel, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)


def build_torch_providers(root: Path, output: Path) -> list[Path]:
    """Expose the SDK's Torch/CUDA metadata without copying its native payload.

    Pass these local wheels to the initial pip solve, including when that solve
    uses --ignore-installed to overlay Ubuntu's RECORD-less apt distributions.
    The finder serves Torch modules; the appended CUDA path serves bindings and
    Torch's native-library discovery. Uninstall owns metadata, never SDK files.
    """
    root = root.resolve()
    ml = root / "extsDeprecated/omni.isaac.ml_archive/pip_prebundle"
    cuda = root / "exts/isaacsim.pip.nv/pip_prebundle"
    distributions = [
        dist
        for dist in metadata.distributions(path=[str(ml)])
        if dist.metadata["Name"].lower() in {"torch", "torchvision", "torchaudio"}
    ] + list(metadata.distributions(path=[str(cuda)]))
    names = {dist.metadata["Name"].lower().replace("_", "-") for dist in distributions}
    required = {"torch", "torchvision", "torchaudio", "cuda-bindings", "nvidia-cuda-runtime-cu12"}
    if not required <= names:
        raise RuntimeError(f"Incomplete bundled Torch/CUDA metadata: {sorted(required - names)}")
    output.mkdir(parents=True, exist_ok=True)
    wheels = []
    for dist in distributions:
        name = re.sub(r"[-_.]+", "_", dist.metadata["Name"]).lower()
        info = f"{name}-{dist.version}.dist-info"
        source_info = Path(dist._path).resolve()
        if not source_info.is_relative_to(root):
            raise RuntimeError(f"Bundled metadata escapes SDK: {source_info}")
        files = {}
        for path in source_info.rglob("*"):
            if path.is_file() and path.name not in {"RECORD", "WHEEL", "INSTALLER", "direct_url.json"}:
                files[f"{info}/{path.relative_to(source_info)}"] = path.read_bytes()
        message = Parser().parsestr(dist.read_text("METADATA"))
        if "Summary" in message:
            del message["Summary"]
        message["Summary"] = f"System-Python provider for bundled NVIDIA SDK {dist.metadata['Name']}"
        files[f"{info}/METADATA"] = message.as_string()
        files[f"{info}/WHEEL"] = (
            "Wheel-Version: 1.0\nGenerator: geniesim-bundled-sdk\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        )
        files[f"{info}/geniesim-bundled.json"] = (
            json.dumps({"root": str(root), "metadata_source": str(source_info)}) + "\n"
        )
        wheel = output / f"{name}-{dist.version}-py3-none-any.whl"
        write_wheel(wheel, info, files)
        wheels.append(wheel)
    return wheels


def clean_bytecode(root: Path) -> int:
    """Drop regenerable bytecode only, keeping sourceless/vendor SDK files."""
    from importlib.util import source_from_cache

    root = root.resolve()
    removed = 0
    for directory, _, names in os.walk(root, followlinks=False):
        for name in names:
            path = Path(directory) / name
            if path.suffix != ".pyc" or path.is_symlink():
                continue
            try:
                source = Path(source_from_cache(str(path)))
            except ValueError:
                source = path.with_suffix(".py")
            if source.is_file() and source.resolve().is_relative_to(root):
                removed += path.stat().st_size
                path.unlink()
    print(f"[bundled-sdk] Removed {removed} bytes of regenerable bytecode from {root}", flush=True)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/isaac-sim"))
    parser.add_argument("--version")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--torch-providers", action="store_true")
    parser.add_argument("--clean-bytecode", type=Path)
    args = parser.parse_args()
    if args.clean_bytecode:
        clean_bytecode(args.clean_bytecode)
        return
    if not args.output:
        parser.error("--output is required when building wheels")
    if args.torch_providers:
        for wheel in build_torch_providers(args.root, args.output):
            print(wheel)
        return
    if not args.version:
        parser.error("--version is required when building the Isaac adapter")
    print(build_adapter(args.root, args.version, args.output))


if __name__ == "__main__":
    main()
