#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Update or validate the system-Python 6.2 dependency lock against its SDK."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit
from zipfile import ZipFile

from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

try:
    import tomllib
except ModuleNotFoundError:  # CPU tests may run on Python 3.10 hosts.
    import tomli as tomllib

from bundled_isaacsim import build_torch_providers
import collect_deps
from collect_runtime_constraints import collect_constraints

HERE = Path(__file__).resolve().parent
IK_WHEEL = "ik_solver-0.4.3-cp312-cp312-linux_x86_64.whl"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_hashes(source, lock_dir):
    collect_deps.SOURCE_DIR = source
    peers = sorted(set(collect_deps.discover_tier1_packages()) | {"geniesim_teleop"})
    files = {f"source/{peer}/pyproject.toml": source / peer / "pyproject.toml" for peer in peers}
    files.update(
        {
            f"docker/{name}": HERE / name
            for name in (
                "collect_deps.py",
                "collect_runtime_constraints.py",
                "bundled_isaacsim.py",
                "lock_isaac62.py",
                "libuipc-artifacts.json",
            )
        }
    )
    files["docker/isaac62/requirements.in"] = lock_dir / "requirements.in"
    files["docker/isaac62/nvidia.in"] = lock_dir / "nvidia.in"
    return {name: digest(path) for name, path in sorted(files.items()) if path.is_file()}


def provider_inventory(providers):
    result = {}
    for wheel in sorted(providers.glob("*.whl")):
        with ZipFile(wheel) as archive:
            name = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
            result[wheel.name] = hashlib.sha256(archive.read(name)).hexdigest()
        if wheel.name == IK_WHEEL:
            result[wheel.name] = digest(wheel)
    if IK_WHEEL not in result:
        raise RuntimeError(f"Missing local IK wheel: {providers / IK_WHEEL}")
    return result


def fingerprint(args, constraints):
    return {
        "schema": 2,
        "uv_version": subprocess.check_output(["uv", "--version"], text=True).split()[1],
        "isaacsim_tag": args.isaacsim_tag,
        "isaacsim_version": metadata.version("isaacsim"),
        "sdk_version": (args.sdk_root / "VERSION").read_text().strip(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "machine": platform.machine(),
        "wuji_sdk_version": args.wuji_sdk_version,
        "inputs": input_hashes(args.source, args.lock_dir),
        "constraints_sha256": hashlib.sha256(constraints.encode()).hexdigest(),
        "providers": provider_inventory(args.providers),
    }


def validate(expected, actual, requirements, vendor_requirements=None):
    """Fail before installation when source, runtime or lock bytes drift."""
    changed = [key for key, value in actual.items() if expected.get(key) != value]
    for key, path in (("requirements_sha256", requirements), ("nvidia_requirements_sha256", vendor_requirements)):
        if path is not None and expected.get(key) != digest(path):
            changed.append(path.name)
    if changed:
        raise RuntimeError(f"Stale Isaac 6.2 lock ({', '.join(changed)}); run lock_isaac62.py --update")


def index_options(args):
    if args.extra_index_url:
        # The SDK pins releases split across both indexes (including NVIDIA
        # OpenCV prereleases). Match pip's combined candidate set; committed
        # artifact hashes still govern installation.
        return [
            "--default-index",
            args.index_url,
            "--index",
            args.extra_index_url,
            "--index-strategy",
            "unsafe-best-match",
        ]
    return ["--default-index", args.index_url]


def split_vendor_lock(text, names):
    """Route explicitly owned vendor packages without changing their hashes."""
    blocks = []
    for line in text.splitlines(keepends=True):
        if line and not line[0].isspace():
            blocks.append([])
        blocks[-1].append(line)
    common, vendor, found = [], [], set()
    for block in blocks:
        name = canonicalize_name(block[0].split("==")[0])
        (vendor if name in names else common).extend(block)
        found.add(name)
    if not names <= found:
        raise RuntimeError(f"Vendor packages absent from resolved lock: {sorted(names - found)}")
    return "".join(common), "".join(vendor)


def compact_hashes(text, packages):
    """Keep UV's target-compatible artifacts, without admitting any new hash."""
    tags = set(sys_tags())
    artifacts = {}
    for package in packages:
        hashes = set()
        for wheel in package.get("wheels", []):
            filename = unquote(urlsplit(wheel["url"]).path.rsplit("/", 1)[-1])
            if parse_wheel_filename(filename)[3] & tags:
                hashes.add(wheel["hashes"]["sha256"])
        # Preserve source-build fallback only when its bytes were already locked.
        if "sdist" in package:
            hashes.add(package["sdist"]["hashes"]["sha256"])
        artifacts[canonicalize_name(package["name"]), Version(package["version"])] = hashes
    result = []
    for block in re.split(r"(?m)(?=^\S)", text):
        if not block.strip() or block.startswith("#"):
            continue
        pin = block.splitlines()[0].rstrip(" \\")
        name, version = pin.split("==")
        allowed = artifacts.get((canonicalize_name(name), Version(version)), set())
        hashes = sorted(set(re.findall(r"--hash=sha256:([0-9a-f]{64})", block)) & allowed)
        if not hashes:
            raise RuntimeError(f"No already-locked compatible artifact for {pin}")
        result.append(pin + " \\\n    " + " \\\n    ".join(f"--hash=sha256:{value}" for value in hashes) + "\n")
    return "".join(result)


def update(args, actual, constraints):
    """Resolve only on an explicit update; ordinary image builds never resolve."""
    output = args.lock_dir / "requirements.txt"
    vendor_output = args.lock_dir / "nvidia-requirements.txt"
    vendor_names = {
        canonicalize_name(line.strip())
        for line in (args.lock_dir / "nvidia.in").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    collected = subprocess.check_output(
        [sys.executable, str(HERE / "collect_deps.py"), str(args.source), "--include-peer", "geniesim_teleop"],
        text=True,
    )
    with tempfile.TemporaryDirectory(prefix="isaac62-lock-") as directory:
        temporary = Path(directory)
        inputs = temporary / "requirements.in"
        inputs.write_text(
            collected
            + constraints
            + (args.lock_dir / "requirements.in").read_text()
            + f"\nwuji-sdk=={args.wuji_sdk_version}\n"
            + "\n".join(str(p) for p in sorted(args.providers.glob("*.whl")))
            + "\n"
        )
        constraint_file = temporary / "constraints.txt"
        constraint_file.write_text(constraints)
        compiled = temporary / "requirements.txt"
        if output.is_file():
            shutil.copyfile(output, compiled)
            if vendor_output.is_file():
                with compiled.open("a") as stream:
                    stream.write(vendor_output.read_text())
        command = [
            "uv",
            "--no-config",
            "pip",
            "compile",
            str(inputs),
            "--python",
            sys.executable,
            "--no-python-downloads",
            "--constraint",
            str(constraint_file),
            "--generate-hashes",
            "--no-header",
            "--no-annotate",
            "--output-file",
            str(compiled),
            *index_options(args),
        ]
        for wheel in sorted(args.providers.glob("*.whl")):
            command += ["--no-emit-package", wheel.name.split("-")[0]]
        if args.seed_installed:
            pins = {}
            for dist in metadata.distributions():
                name = dist.metadata["Name"]
                if not name:
                    continue
                try:
                    pins[name] = f"{name}=={Version(metadata.version(name))}"
                except InvalidVersion:
                    continue
            seed = temporary / "installed.txt"
            seed.write_text("\n".join(sorted(pins.values())) + "\n")
            command += ["--constraint", str(seed)]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        # UV's pylock output lists target-compatible wheels with their filenames;
        # requirements.txt output instead hashes every platform's release files.
        artifacts = temporary / "pylock.toml"
        subprocess.run(
            [
                "uv",
                "--no-config",
                "pip",
                "compile",
                str(compiled),
                "--python",
                sys.executable,
                "--no-python-downloads",
                "--no-deps",
                "--format",
                "pylock.toml",
                "--output-file",
                str(artifacts),
                *index_options(args),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        actual["base_image"] = args.base_image
        compact = compact_hashes(compiled.read_text(), tomllib.loads(artifacts.read_text())["packages"])
        common, vendor = split_vendor_lock(compact, vendor_names)
        actual["requirements_sha256"] = hashlib.sha256(common.encode()).hexdigest()
        actual["nvidia_requirements_sha256"] = hashlib.sha256(vendor.encode()).hexdigest()
        # Publish only after a successful solve. The manifest detects an interrupted update.
        output.write_text(common)
        vendor_output.write_text(vendor)
        (args.lock_dir / "manifest.json").write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
    print(f"Updated {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--lock-dir", type=Path, default=HERE / "isaac62")
    parser.add_argument("--sdk-root", type=Path, default=Path("/isaac-sim"))
    parser.add_argument("--providers", type=Path, default=HERE / "providers")
    parser.add_argument("--isaacsim-tag", required=True)
    parser.add_argument("--wuji-sdk-version", required=True)
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--seed-installed", action="store_true", help="Constrain the first lock to the qualified image")
    parser.add_argument("--base-image", help="SDK base repository@sha256 provenance (required for updates)")
    parser.add_argument("--index-url", default=os.environ.get("PIP_INDEX_URL", "https://pypi.org/simple"))
    parser.add_argument("--extra-index-url", default="https://pypi.nvidia.com")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12) or sys.platform != "linux" or platform.machine() != "x86_64":
        parser.error("The Isaac 6.2 lock requires system CPython 3.12 / Linux x86_64")
    if args.update and (not args.base_image or "@sha256:" not in args.base_image):
        parser.error("--update requires --base-image repository@sha256:digest")
    if args.seed_installed and not args.update:
        parser.error("--seed-installed is only valid with --update")
    build_torch_providers(args.sdk_root, args.providers)
    constraints = "\n".join(collect_constraints()) + "\n"
    actual = fingerprint(args, constraints)
    if args.update:
        update(args, actual, constraints)
    else:
        validate(
            json.loads((args.lock_dir / "manifest.json").read_text()),
            actual,
            args.lock_dir / "requirements.txt",
            args.lock_dir / "nvidia-requirements.txt",
        )
        print("Isaac 6.2 dependency lock matches source, SDK and local providers", flush=True)
    # Keep current SDK constraints available for native installers and deliberate additions.
    (args.providers.parent / "isaac-constraints.txt").write_text(constraints)


if __name__ == "__main__":
    main()
