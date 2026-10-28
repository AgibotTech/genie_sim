#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Extract third-party dependencies from **tier-1** geniesim pyproject.toml files.

Outputs a flat list of pip-installable package specs that the docker
image needs to install at build time. Tier-2 peers are deliberately
**excluded** — they're opt-in via
``pip install -e "source/geniesim/[teleop|scene_agent|world|all]"``
and their heavy stacks (VR / LLM / CUDA-ML) shouldn't be baked into
the default container.

**Single source of truth:** ``source/geniesim/pyproject.toml``.

- ``[project].dependencies``               → tier 1
- ``[project.optional-dependencies]``      → tier 2 (skipped here)

This script re-implements the same parse as
``geniesim_cli._tiers`` (it can't import the module — runs at Docker
build time before ``geniesim_cli`` is pip-installed). Adding a new
tier-1 peer is a one-line edit in the umbrella's pyproject; this
script picks it up automatically on the next ``geniesim docker
build``.

The Dockerfile ``COPY`` lines also need to include the new peer's
pyproject.toml so this script can read its third-party deps — they
remain a manual list by Docker's static-COPY constraint. Keep the
COPY block in each ``docker/Dockerfile*`` in sync with the tier-1
list, and document the contract there.

Skips, per-package:
- internal ``geniesim_*`` deps (they're installed editable from source)
- ``isaacsim``, ``cv_bridge``, ``numpy``, ``scipy`` — provided by the
  runtime base image, not by pip
"""

import pathlib
import re
import sys

SKIP_PREFIXES = ("geniesim",)
SKIP_EXACT = {"isaacsim", "cv_bridge", "numpy", "scipy"}

SOURCE_DIR = (
    pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parent.parent / "source"
)


def _strip_specifier(req: str) -> str:
    """``geniesim_cli>=3.2.0`` → ``geniesim_cli``. Matches
    ``geniesim_cli._tiers._strip_specifier`` — keep in sync."""
    for op in (">=", "<=", "==", "~=", ">", "<", "!=", "[", " ", ";"):
        idx = req.find(op)
        if idx != -1:
            req = req[:idx]
    return req.strip()


def parse_dependencies_array(pyproject: pathlib.Path) -> list[str]:
    """Read a TOML-y ``dependencies = [...]`` array as a list of
    quoted strings. Intentionally regex-based (not a full TOML parser)
    so the script stays stdlib-only with no ``tomli`` install at
    Docker build time."""
    text = pyproject.read_text()
    in_deps = False
    deps: list[str] = []
    for line in text.splitlines():
        if re.match(r"^dependencies\s*=\s*\[", line):
            in_deps = True
            continue
        if in_deps:
            if line.strip().startswith("]"):
                break
            m = re.match(r'\s*"([^"]+)"', line)
            if m:
                deps.append(m.group(1))
    return deps


def discover_tier1_packages() -> list[str]:
    """Derive the tier-1 package list from
    ``source/geniesim/pyproject.toml``'s ``[project].dependencies``.

    Falls back to the original hardcoded canon if the umbrella
    pyproject can't be read — this should never happen in a real
    Docker build (the Dockerfile always COPYs it first), but the
    fallback keeps the script self-contained for ad-hoc invocations."""
    umbrella = SOURCE_DIR / "geniesim" / "pyproject.toml"
    if not umbrella.is_file():
        # Fallback canon — matches the umbrella's `[project].dependencies`
        # as of the tier model's introduction. Update if the umbrella
        # adds a new tier-1 peer AND the source-tree path is genuinely
        # unavailable (rare).
        return ["geniesim_cli", "geniesim_assets", "geniesim_benchmark", "geniesim_ros"]
    peers: list[str] = []
    # Always include the umbrella itself as a tier-1 source for deps
    # parsing — its `[project].dependencies` may include third-party
    # specs in the future, and reading them is harmless today.
    peers.append("geniesim")
    for raw in parse_dependencies_array(umbrella):
        name = _strip_specifier(raw)
        if name.startswith("geniesim"):
            peers.append(name)
    return peers


def main() -> None:
    # Two-mode CLI:
    #   collect_deps.py            → third-party pip deps (default, image-build use)
    #   collect_deps.py --peers    → tier-1 peer source paths, one per line,
    #                                relative to ``SOURCE_DIR``. Used by
    #                                ``docker/entrypoint.sh`` to derive the
    #                                editable-install loop without hardcoding
    #                                the peer list.
    #
    # ``--peers`` honors the same canonical source: ``source/geniesim/pyproject.toml``
    # (via ``discover_tier1_packages``). Each emitted line is a directory name
    # under ``SOURCE_DIR``; the caller is responsible for confirming
    # ``${SOURCE_DIR}/<name>/pyproject.toml`` actually exists before pip-installing.
    if "--peers" in sys.argv[2:]:
        for pkg in discover_tier1_packages():
            print(pkg)
        return

    all_deps: set[str] = set()
    tier1_packages = discover_tier1_packages()
    for pkg in tier1_packages:
        pyproject = SOURCE_DIR / pkg / "pyproject.toml"
        if not pyproject.exists():
            # Tier-1 peers without a pyproject in source/ (e.g.
            # geniesim_assets — distributed out-of-band) are silently
            # skipped: their pip deps aren't on the geniesim contract.
            continue
        for dep in parse_dependencies_array(pyproject):
            name = re.split(r"[<>=!~\[]", dep)[0].strip().lower().replace("-", "_")
            if name in SKIP_EXACT or any(name.startswith(p) for p in SKIP_PREFIXES):
                continue
            all_deps.add(dep)

    for dep in sorted(all_deps):
        print(dep)


if __name__ == "__main__":
    main()
